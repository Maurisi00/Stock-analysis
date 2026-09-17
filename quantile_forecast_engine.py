import numpy as np
import pandas as pd
import logging
from typing import Dict, Any
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_pinball_loss

logger = logging.getLogger("QuantSystem.QuantileForecastEngine")

QUANTILES = [0.025, 0.05, 0.10, 0.50, 0.90, 0.95, 0.975]
CONFIDENCE_BANDS = {"80": (0.10, 0.90), "90": (0.05, 0.95), "95": (0.025, 0.975)}
# El 99% (cuantiles 0.005/0.995) se eliminó deliberadamente: estimar un
# cuantil tan extremo con las ~500 observaciones solapadas típicas de este
# engine equivale a ~2.5 observaciones EFECTIVAS en la cola (ver
# run_holding_period_backtest.n_independent_signals para el mismo problema
# de solapamiento) — no es una banda estimable con esta muestra, solo un
# número que aparenta precisión. El 80% (nuevo) sí es estimable y más útil
# para decidir una compra a 6 meses que un 99% ilusorio.


def rearrange_quantiles(valores_por_cuantil: Dict[float, float]) -> Dict[float, float]:
    """
    Rearrangement de Chernozhukov, Fernández-Val y Galichon (2010) para un
    único vector de predicciones: ordena los NIVELES de cuantil de forma
    ascendente, ordena los VALORES predichos también ascendentemente, y los
    empareja.

    Hace falta porque los modelos por cuantil son INDEPENDIENTES (uno por
    nivel) y nada les impone un orden entre sí: el p05 puede predecir por
    encima del p50. El rearrangement garantiza monotonía por construcción, sin
    reentrenar ni restringir los modelos.

    Extraído a función pura en la Fase 3.2 para poder aplicarlo en los DOS
    sitios que lo necesitan (la predicción publicada y la validación fuera de
    muestra) y poder testearlo aisladamente.
    """
    niveles = sorted(valores_por_cuantil)
    valores = sorted(valores_por_cuantil[q] for q in niveles)
    return dict(zip(niveles, valores))


def rearrange_quantile_matrix(preds: Dict[float, np.ndarray]) -> Dict[float, np.ndarray]:
    """
    Igual que `rearrange_quantiles` pero sobre PREDICCIONES VECTORIALES: cada
    cuantil trae un array con una predicción por observación de test, y el
    rearrangement se aplica FILA A FILA (por observación), no a lo largo del
    tiempo.

    Ordenar por columnas sería un error grave: mezclaría la predicción del p05
    de una fecha con la del p95 de otra. Lo que se ordena son los 7 valores que
    los 7 modelos dan PARA LA MISMA fecha.
    """
    niveles = sorted(preds)
    matriz = np.column_stack([preds[q] for q in niveles])   # (n_obs, n_cuantiles)
    matriz_ordenada = np.sort(matriz, axis=1)                # monótona por fila
    return {q: matriz_ordenada[:, i] for i, q in enumerate(niveles)}


class QuantileForecastEngine:
    """
    Engine encargado de estimar la distribución de precios futura mediante regresión
    cuantílica (Gradient Boosting), validando honestamente fuera de muestra antes de
    reentrenar sobre el histórico completo para la predicción "de hoy".
    """
    def __init__(self, feature_matrix: pd.DataFrame, price_series: pd.Series, horizon_days: int = 126,
                 train_size: float = 0.8, random_state: int = 42):
        self.horizon_days = horizon_days
        self.train_size = train_size
        self.random_state = random_state

        y_reg = (price_series.shift(-horizon_days) / price_series) - 1
        valid_idx = feature_matrix.dropna().index.intersection(y_reg.dropna().index)

        self.X = feature_matrix.loc[valid_idx]
        self.y = y_reg.loc[valid_idx]
        self.models: Dict[float, GradientBoostingRegressor] = {}
        self.validation_metrics: Dict[str, Any] = {}

    def _purged_train_test_split(self):
        """
        Split cronológico con purga (embargo) de horizon_days filas en el borde del
        entrenamiento: como el target de cada fila mira horizon_days hacia adelante,
        sin este recorte las últimas filas de train tendrían un objetivo que ya
        incluye información de precios dentro de la ventana de test.
        """
        split_idx = int(len(self.X) * self.train_size)
        train_end = max(0, split_idx - self.horizon_days)

        X_train, y_train = self.X.iloc[:train_end], self.y.iloc[:train_end]
        X_test, y_test = self.X.iloc[split_idx:], self.y.iloc[split_idx:]
        return X_train, y_train, X_test, y_test

    def evaluate_out_of_sample(self) -> Dict[str, Any]:
        """
        Entrena un modelo por cuantil solo sobre el tramo de entrenamiento purgado,
        y evalúa sobre el tramo de test genuinamente fuera de muestra: pinball loss
        por cuantil y cobertura empírica de los intervalos de confianza 80/90/95%.

        APLICA EL REARRANGEMENT (Fase 3.2). Antes no lo hacía y
        `predict_price_quantiles` sí, así que **se validaba un objeto distinto
        del que se publica**: la cobertura y el pinball loss que el informe
        imprime describían unas predicciones crudas, potencialmente cruzadas,
        mientras las bandas publicadas venían de las reordenadas. Se reportan
        también las métricas SIN rearrangement (`*_sin_rearrange`) y cuántas
        filas venían cruzadas, para poder ver de cuánto es la diferencia en vez
        de tener que suponerla.
        """
        X_train, y_train, X_test, y_test = self._purged_train_test_split()

        preds_crudas: Dict[float, np.ndarray] = {}
        for q in QUANTILES:
            logger.info(f"Validando cuantil {q} fuera de muestra...")
            gbr = GradientBoostingRegressor(loss='quantile', alpha=q, n_estimators=100, random_state=self.random_state)
            gbr.fit(X_train, y_train)
            preds_crudas[q] = gbr.predict(X_test)

        preds = rearrange_quantile_matrix(preds_crudas)

        # Cuántas observaciones de test tenían los cuantiles cruzados: es la
        # magnitud real del problema que el rearrangement corrige.
        niveles = sorted(QUANTILES)
        matriz_cruda = np.column_stack([preds_crudas[q] for q in niveles])
        filas_cruzadas = int((np.diff(matriz_cruda, axis=1) < 0).any(axis=1).sum()) \
            if len(y_test) > 0 else 0

        def _metricas(p: Dict[float, np.ndarray]) -> Dict[str, Any]:
            pinball = {q: float(mean_pinball_loss(y_test, p[q], alpha=q)) for q in QUANTILES}
            cov = {}
            for label, (lo, hi) in CONFIDENCE_BANDS.items():
                covered = (y_test.values >= p[lo]) & (y_test.values <= p[hi])
                cov[label] = float(covered.mean()) if len(covered) > 0 else np.nan
            return {"pinball_loss": pinball, "coverage": cov}

        con_rearrange = _metricas(preds)
        sin_rearrange = _metricas(preds_crudas) if len(y_test) > 0 else con_rearrange

        self.validation_metrics = {
            "pinball_loss": con_rearrange["pinball_loss"],
            "coverage": con_rearrange["coverage"],
            "n_test_obs": int(len(y_test)),
            # --- añadido en la Fase 3.2 -----------------------------------
            "pinball_loss_sin_rearrange": sin_rearrange["pinball_loss"],
            "coverage_sin_rearrange": sin_rearrange["coverage"],
            "n_filas_cruzadas": filas_cruzadas,
            "pct_filas_cruzadas": (float(filas_cruzadas / len(y_test))
                                    if len(y_test) > 0 else float("nan")),
        }
        return self.validation_metrics

    def refit_on_full_data(self) -> None:
        """
        Reentrena un modelo por cuantil usando el 100% del histórico disponible,
        para producir la mejor predicción posible del día de hoy.
        """
        for q in QUANTILES:
            gbr = GradientBoostingRegressor(loss='quantile', alpha=q, n_estimators=100, random_state=self.random_state)
            gbr.fit(self.X, self.y)
            self.models[q] = gbr

    def predict_price_quantiles(self, latest_features: pd.DataFrame, current_price: float) -> Dict[str, Any]:
        """
        Traduce los retornos cuantílicos predichos por los modelos reentrenados
        en precios nominales para el horizonte configurado.

        Corrige quantile crossing antes de convertir a precio: los 7
        GradientBoostingRegressor son modelos INDEPENDIENTES (uno por
        cuantil), así que nada les impide predecir, por ejemplo, un p05 por
        encima del p50 — no hay restricción de orden entre ellos. Se aplica
        el rearrangement de Chernozhukov, Fernández-Val y Galichon (2010):
        se ordenan los NIVELES de cuantil de forma ascendente y se emparejan
        con los VALORES predichos también ordenados ascendentemente, lo que
        garantiza monotonía (ret[q_a] <= ret[q_b] para q_a < q_b) por
        construcción, sin necesitar reentrenar ni restringir los modelos.
        """
        if not self.models:
            raise ValueError("Ejecuta refit_on_full_data primero.")

        latest_clean = latest_features[self.X.columns].ffill().bfill()
        raw_ret = {q: float(self.models[q].predict(latest_clean)[0]) for q in QUANTILES}

        # Misma corrección que antes, ahora vía la función pura compartida con
        # evaluate_out_of_sample (Fase 3.2).
        ret = rearrange_quantiles(raw_ret)

        return {
            "central": current_price * (1 + ret[0.50]),
            "p80": (current_price * (1 + ret[0.10]), current_price * (1 + ret[0.90])),
            "p90": (current_price * (1 + ret[0.05]), current_price * (1 + ret[0.95])),
            "p95": (current_price * (1 + ret[0.025]), current_price * (1 + ret[0.975])),
        }


class QuantileFanForecaster:
    """
    ABANICO multi-horizonte de bandas cuantílicas (PLAN_REFACTOR.md 3.2).

    Mantiene un `QuantileForecastEngine` independiente por horizonte, porque el
    target de cada uno es distinto: el retorno forward a 21 días no es el mismo
    problema de aprendizaje que el retorno a 504 días, así que compartir
    modelos entre horizontes no tendría sentido. El embargo de cada split se
    dimensiona al horizonte propio de ese target, como exige el invariante de
    purga del repo.

    Por qué existe: el sistema anterior proyectaba a UN horizonte fijo (126
    días) elegido de antemano, no porque los datos dijeran que es donde el
    modelo funciona. La estrategia real del propietario no tiene plazo de
    mantenimiento fijo, así que un abanico responde su pregunta y un punto
    único no.

    ADITIVO: no reemplaza a `QuantileForecastEngine`, lo compone. El pipeline
    actual sigue usando el engine de horizonte único exactamente igual; el
    abanico queda disponible para el informe nuevo (Fase 3.4).
    """

    def __init__(self, feature_matrix: pd.DataFrame, price_series: pd.Series,
                 horizons_days: Dict[int, int], train_size: float = 0.8,
                 random_state: int = 42):
        """
        :param horizons_days: mapa {etiqueta_meses: dias}, tal como lo produce
            `entry_context_engine.horizontes_en_dias()`.
        """
        self.horizons_days = dict(horizons_days)
        self.engines: Dict[int, QuantileForecastEngine] = {}
        self.skipped: Dict[int, str] = {}

        for meses, dias in self.horizons_days.items():
            engine = QuantileForecastEngine(
                feature_matrix=feature_matrix, price_series=price_series,
                horizon_days=dias, train_size=train_size, random_state=random_state,
            )
            # Un horizonte largo puede dejar sin filas utilizables un histórico
            # corto (el target mira `dias` hacia adelante y se pierde esa cola).
            # Se registra el motivo en vez de fallar: el abanico debe poder
            # entregar los horizontes que sí son estimables.
            if len(engine.X) <= dias:
                self.skipped[meses] = (
                    f"histórico insuficiente: {len(engine.X)} filas utilizables para un "
                    f"horizonte de {dias} días"
                )
                continue
            self.engines[meses] = engine

    def evaluate_out_of_sample(self) -> Dict[int, Dict[str, Any]]:
        """Valida cada horizonte por separado (pinball loss + cobertura)."""
        return {meses: eng.evaluate_out_of_sample() for meses, eng in self.engines.items()}

    def refit_on_full_data(self) -> None:
        for eng in self.engines.values():
            eng.refit_on_full_data()

    def predict_price_fan(self, latest_features: pd.DataFrame,
                           current_price: float) -> Dict[str, Any]:
        """
        Bandas 80/90/95% para todos los horizontes evaluables, cada una ya
        libre de quantile crossing.

        Devuelve también `omitidos` con el motivo por horizonte: un abanico al
        que le falta el tramo de 24 meses tiene que decir por qué falta, no
        presentar cuatro horizontes como si fueran los cinco pedidos.
        """
        bandas: Dict[int, Dict[str, Any]] = {}
        for meses, eng in self.engines.items():
            bandas[meses] = eng.predict_price_quantiles(latest_features, current_price)
            bandas[meses]["horizonte_dias"] = self.horizons_days[meses]
        return {
            "bandas": bandas,
            "omitidos": dict(self.skipped),
            "precio_actual": float(current_price),
        }
