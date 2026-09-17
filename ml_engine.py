import numpy as np
import pandas as pd
import logging
from typing import Tuple, Dict, Any, List
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, balanced_accuracy_score
from xgboost import XGBClassifier

logger = logging.getLogger("QuantSystem.MLEngine")

# Balanced-Accuracy mínima (walk-forward, agregada) del modelo ganador para
# que su probabilidad sustente una recomendación. 0.50 es lo que da una
# moneda al aire en un problema binario balanceado — un modelo por debajo de
# eso no tiene poder predictivo demostrado sobre ese ticker, así que su
# probabilidad no es más informativa que el azar (o peor). Mismo patrón que
# `reliable=False` en FundamentalFactorEngine o `statistically_conclusive` en
# el event study: preferimos que el consumidor de esta señal sepa "no hay
# base" a que reciba una recomendación construida sobre una probabilidad sin
# poder predictivo.
MIN_BALANCED_ACCURACY_FOR_SIGNAL = 0.50


class MachineLearningEngine:
    """
    Engine encargado del preprocesamiento de características (features),
    entrenamiento de modelos competitivos y selección automatizada del mejor clasificador.
    """
    def __init__(self, feature_matrix: pd.DataFrame, target_series: pd.Series, embargo_days: int = 126,
                 train_size: float = 0.8, random_state: int = 42):
        """
        :param feature_matrix: DataFrame con todas las variables predictoras (X)
        :param target_series: Serie binaria con la variable objetivo (Y)
        :param embargo_days: Nº de filas a purgar del final de CADA ventana de entrenamiento
            walk-forward. El target es forward-looking (mira embargo_days hacia adelante), así
            que sin esta purga las últimas filas de entrenamiento tendrían una etiqueta que ya
            incorpora información de precios dentro del bloque de test siguiente (fuga por
            ventana solapada en el borde del split).
        :param train_size: Proporción cronológica del histórico usada como ventana de
            entrenamiento INICIAL antes del primer bloque walk-forward.
        :param random_state: Semilla compartida por los 3 clasificadores, para reproducibilidad.
        """
        # Alinear X e Y eliminando registros donde falten etiquetas (como el horizonte futuro)
        combined = pd.concat([feature_matrix, target_series], axis=1).dropna()

        self.Y = combined.iloc[:, -1]
        self.X = combined.iloc[:, :-1]
        self.embargo_days = embargo_days
        self.train_size = train_size
        self.random_state = random_state
        # Escalador SEPARADO del walk-forward de evaluación/selección (que usa
        # un StandardScaler local por bloque — ver _walkforward_predict): este
        # es exclusivamente el del modelo de PRODUCCIÓN (refit_best_model_on_full_data
        # + predict_probability_now), para que nada más pueda mutarlo entre medio.
        self.production_scaler = StandardScaler()
        self.best_model_name: str = ""
        self.best_model: Any = None
        # Balanced-Accuracy walk-forward AGREGADA del ganador (ver
        # evaluate_and_select_best_model) — NaN hasta que ese método corre,
        # igual que best_model_name empieza en "" antes de la selección.
        self.best_balanced_accuracy: float = float("nan")
        # n_jobs=1 / nthread=1 NO son una optimización: son un requisito de
        # REPRODUCIBILIDAD. Con varios hilos, el reparto de filas/features entre
        # workers varía entre corridas y con él el ORDEN de las sumas de
        # gradientes y de los conteos de impureza; en punto flotante la suma no
        # es asociativa, así que el modelo ajustado difiere en los últimos
        # decimales y ocasionalmente voltea una hoja. Es la causa que explica
        # que cache/ml_prob_history.json tenga JD con 0.66208637 y 0.66246547
        # la MISMA fecha teniendo random_state=42 fijado.
        self.models: Dict[str, Any] = {
            "Logistic_Regression": LogisticRegression(max_iter=1000, random_state=random_state),
            "Random_Forest": RandomForestClassifier(n_estimators=100, max_depth=6, random_state=random_state,
                                                    n_jobs=1),
            "XGBoost": XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, eval_metric='logloss',
                                      random_state=random_state, n_jobs=1, nthread=1)
        }
        # Cachea los resultados walk-forward por (nombre_modelo, block_days) para
        # que generate_walkforward_probabilities no tenga que recalcular desde
        # cero lo que evaluate_and_select_best_model ya computó para el ganador.
        self._walkforward_cache: Dict[Tuple[str, int], Dict[str, np.ndarray]] = {}

    def _walkforward_predict(self, model_name: str, block_days: int) -> Dict[str, np.ndarray]:
        """
        Camina hacia adelante en bloques de block_days filas: entrena con todo
        lo disponible hasta el inicio del bloque (menos embargo_days purgado
        del borde), predice ESE bloque (genuinamente fuera de muestra),
        avanza al bloque siguiente incorporando el anterior al entrenamiento
        (ventana EXPANSIVA), y repite hasta agotar la serie.

        Cada bloque se entrena con un clon SIN ajustar de self.models[model_name]
        (mismos hiperparámetros, sin arrastrar estado entre iteraciones) y un
        StandardScaler local, ajustado solo con los datos de ESE bloque de
        entrenamiento — igual que el split único anterior, pero repetido en
        cada iteración walk-forward.

        Devuelve las fechas/etiquetas/predicciones/probabilidades CONCATENADAS
        de todos los bloques, listas para métricas agregadas (selección de
        modelo) o para construir la serie walk-forward de probabilidades.
        """
        cache_key = (model_name, block_days)
        if cache_key in self._walkforward_cache:
            return self._walkforward_cache[cache_key]

        model_template = self.models[model_name]
        split_idx = int(len(self.X) * self.train_size)
        n = len(self.X)

        records_dates: List[pd.Timestamp] = []
        records_y_true: List[float] = []
        records_preds: List[float] = []
        records_probs: List[float] = []

        block_start = split_idx
        while block_start < n:
            block_end = min(block_start + block_days, n)
            train_end = max(0, block_start - self.embargo_days)

            X_train_raw = self.X.iloc[:train_end]
            y_train = self.Y.iloc[:train_end].values
            X_block_raw = self.X.iloc[block_start:block_end]
            y_block = self.Y.iloc[block_start:block_end].values

            if len(X_train_raw) == 0 or len(X_block_raw) == 0:
                block_start = block_end
                continue

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train_raw)
            X_block = scaler.transform(X_block_raw)

            model = clone(model_template)
            model.fit(X_train, y_train)
            preds_block = model.predict(X_block)
            try:
                probs_block = model.predict_proba(X_block)[:, 1]
            except AttributeError:
                probs_block = preds_block

            records_dates.extend(X_block_raw.index)
            records_y_true.extend(y_block)
            records_preds.extend(preds_block)
            records_probs.extend(probs_block)

            block_start = block_end

        result = {
            "dates": pd.DatetimeIndex(records_dates),
            "y_true": np.array(records_y_true, dtype=float),
            "preds": np.array(records_preds, dtype=float),
            "probs": np.array(records_probs, dtype=float),
        }
        self._walkforward_cache[cache_key] = result
        return result

    def evaluate_and_select_best_model(self, block_days: int = 63) -> Dict[str, Dict[str, float]]:
        """
        Evalúa los 3 modelos con ventana EXPANSIVA walk-forward (bloques de
        block_days filas — ver _walkforward_predict) y selecciona el ganador
        por Balanced-Accuracy AGREGADA sobre TODOS los bloques concatenados,
        no sobre un único split 80/20.

        Un solo split deja una ventana de test minúscula en términos de
        observaciones independientes: con un lookback de pocos años, un target
        que mira horizon_days hacia adelante, y ~20% de test, el tramo de test
        puede equivaler a ~1 observación independiente (ver
        config.MIN_LOOKBACK_YEARS_FOR_INTERPRETABLE_BACKTEST) — elegir el
        mejor de 3 modelos sobre eso es selección sobre ruido. Caminar hacia
        adelante en varios bloques (distintos períodos/regímenes de mercado) y
        agregar sus predicciones antes de medir Balanced-Accuracy da una señal
        de selección mucho más robusta.

        Se usa Balanced-Accuracy (no F1) por la misma razón que en el split
        único: el conjunto agregado puede terminar sesgado hacia una clase si
        varios bloques comparten régimen, y Balanced-Accuracy no degenera en
        ese caso mientras que F1/ROC-AUC sí.
        """
        performance_metrics = {}
        best_balanced_acc = -1.0
        metric_keys = ["Accuracy", "Precision", "Recall", "F1-Score", "Balanced-Accuracy", "ROC-AUC"]

        for name in self.models:
            logger.info(f"Evaluando (walk-forward, bloques de {block_days} días): {name}...")
            wf = self._walkforward_predict(name, block_days)
            y_true, preds, probs = wf["y_true"], wf["preds"], wf["probs"]

            if len(y_true) == 0:
                logger.warning(f"{name}: sin bloques walk-forward evaluables (histórico insuficiente "
                                f"para train_size/embargo_days/block_days actuales).")
                performance_metrics[name] = {k: 0.0 for k in metric_keys}
                continue

            if len(np.unique(y_true)) == 1:
                logger.warning(f"{name}: las predicciones walk-forward agregadas quedaron compuestas por "
                                f"una sola clase ({y_true[0]:.0f}) — F1-Score y ROC-AUC no son informativos "
                                f"en este caso; la selección se basa en Balanced-Accuracy.")

            acc = accuracy_score(y_true, preds)
            prec = precision_score(y_true, preds, zero_division=0)
            rec = recall_score(y_true, preds, zero_division=0)
            f1 = f1_score(y_true, preds, zero_division=0)
            balanced_acc = balanced_accuracy_score(y_true, preds)
            auc = roc_auc_score(y_true, probs) if len(np.unique(y_true)) > 1 else 0.5

            performance_metrics[name] = {
                "Accuracy": float(acc),
                "Precision": float(prec),
                "Recall": float(rec),
                "F1-Score": float(f1),
                "Balanced-Accuracy": float(balanced_acc),
                "ROC-AUC": float(auc),
            }

            if balanced_acc > best_balanced_acc:
                best_balanced_acc = balanced_acc
                self.best_model_name = name
                # Clon SIN ajustar: refit_best_model_on_full_data (o
                # generate_walkforward_probabilities, vía self.models) se
                # encargan de entrenarlo antes de usarlo para predecir.
                self.best_model = clone(self.models[name])

        self.best_balanced_accuracy = float(best_balanced_acc)
        logger.info(f"Ganador seleccionado (walk-forward agregado): {self.best_model_name} "
                    f"con Balanced-Accuracy: {best_balanced_acc:.4f}")
        if self.best_balanced_accuracy < MIN_BALANCED_ACCURACY_FOR_SIGNAL:
            logger.warning(f"{self.best_model_name}: Balanced-Accuracy {self.best_balanced_accuracy:.4f} "
                            f"< {MIN_BALANCED_ACCURACY_FOR_SIGNAL:.2f} — el modelo acierta menos que el "
                            f"azar en la validación walk-forward; su probabilidad no tiene poder "
                            f"predictivo demostrado sobre este ticker.")
        return performance_metrics

    def generate_walkforward_probabilities(self, block_days: int = 63) -> pd.Series:
        """
        Genera la serie de probabilidades OUT-OF-SAMPLE walk-forward del
        modelo ganador (self.best_model_name), para usar como
        'historical_probs' en el backtest y en evaluate_probability_calibration
        — en vez del único split 80/20 anterior, que da una ventana de test
        demasiado chica y solapada con su propio target para ser una base
        honesta de backtest o de calibración.

        Reutiliza (vía _walkforward_predict, con el mismo cacheo por
        (modelo, block_days)) el cálculo ya hecho durante
        evaluate_and_select_best_model si block_days coincide, en vez de
        reentrenar el ganador desde cero.
        """
        if not self.best_model_name or self.best_model_name not in self.models:
            raise ValueError("Ejecuta evaluate_and_select_best_model primero.")

        wf = self._walkforward_predict(self.best_model_name, block_days)
        if len(wf["dates"]) == 0:
            return pd.Series(dtype=float)

        return pd.Series(wf["probs"], index=wf["dates"]).sort_index()

    def evaluate_probability_calibration(self, walkforward_probs: pd.Series, n_bins: int = 5) -> Dict[str, Any]:
        """
        Brier score y curva de fiabilidad (n_bins bins de igual ancho en
        [0,1]) sobre las probabilidades walk-forward: para una decisión de
        compra, que el modelo esté CALIBRADO (cuando dice 63%, acierta ~63%
        de las veces) importa más que su accuracy pura — un modelo con buena
        accuracy pero mal calibrado invita a apostar con una confianza que no
        está justificada.

        mean_calibration_error es el error de calibración esperado (ECE):
        el desvío |predicho - observado| de cada bin, ponderado por cuántas
        observaciones cayeron en ese bin. well_calibrated es False si ese
        desvío supera 10 puntos porcentuales (0.10) — el informe debe
        advertir explícitamente en ese caso que la probabilidad no debe
        leerse como una probabilidad literal.
        """
        common_idx = self.Y.index.intersection(walkforward_probs.index)
        y_true = self.Y.loc[common_idx].to_numpy(dtype=float)
        probs = walkforward_probs.loc[common_idx].to_numpy(dtype=float)

        if len(y_true) == 0:
            return {
                "brier_score": float("nan"),
                "reliability_bins": [],
                "mean_calibration_error": float("nan"),
                "well_calibrated": False,
            }

        brier = float(np.mean((probs - y_true) ** 2))

        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        # right=True: el bin i cubre (edge[i], edge[i+1]], salvo el primero
        # que además incluye 0.0 (probabilidades exactamente en 0 no quedan huérfanas).
        bin_indices = np.clip(np.digitize(probs, bin_edges[1:-1], right=True), 0, n_bins - 1)

        reliability_bins = []
        weighted_abs_error_sum = 0.0
        for b in range(n_bins):
            mask = bin_indices == b
            if not mask.any():
                continue
            mean_predicted = float(probs[mask].mean())
            observed_frequency = float(y_true[mask].mean())
            n_obs = int(mask.sum())
            reliability_bins.append({
                "bin": f"{bin_edges[b]:.1f}-{bin_edges[b + 1]:.1f}",
                "mean_predicted": mean_predicted,
                "observed_frequency": observed_frequency,
                "n_obs": n_obs,
            })
            weighted_abs_error_sum += abs(mean_predicted - observed_frequency) * n_obs

        mean_calibration_error = float(weighted_abs_error_sum / len(y_true))
        well_calibrated = mean_calibration_error <= 0.10

        return {
            "brier_score": brier,
            "reliability_bins": reliability_bins,
            "mean_calibration_error": mean_calibration_error,
            "well_calibrated": well_calibrated,
        }

    def refit_best_model_on_full_data(self) -> None:
        """
        Reentrena el modelo ganador usando el 100% del histórico disponible,
        con su propio production_scaler (nunca el usado internamente por el
        walk-forward de evaluación/selección). evaluate_and_select_best_model
        deja al modelo sin ajustar; sin este paso, la predicción "de hoy" no
        existe todavía.
        """
        if self.best_model is None:
            raise ValueError("Ejecuta evaluate_and_select_best_model primero.")
        X_full_scaled = self.production_scaler.fit_transform(self.X)
        self.best_model.fit(X_full_scaled, self.Y.values)

    def predict_probability_now(self, current_features: pd.DataFrame) -> float:
        """
        Toma el vector de características más reciente (el día de hoy)
        y devuelve la probabilidad matemática de que el activo supere al mercado.

        Alinea columnas y aplica ffill/bfill como QuantileForecastEngine.
        predict_price_quantiles (mismo tratamiento de NaN en los dos motores
        que predicen "hoy"), pero si tras eso persiste algún NaN (ej. todas
        las features de esa columna vinieron vacías), falla explícitamente en
        vez de dejar que el NaN se propague silenciosamente al scaler/modelo
        y produzca una probabilidad sin sentido (o un error críptico de sklearn).
        """
        if self.best_model is None:
            raise ValueError("El motor de ML no ha sido entrenado. Ejecuta evaluate_and_select_best_model primero.")

        current_features = current_features[self.X.columns].ffill().bfill()

        if current_features.isna().any().any():
            missing_cols = current_features.columns[current_features.isna().any()].tolist()
            raise ValueError(f"El vector de features de hoy tiene NaN irrecuperables (ffill/bfill no "
                             f"alcanzó) en: {missing_cols}. No se puede predecir la probabilidad de hoy.")

        scaled_features = self.production_scaler.transform(current_features)

        # Extraer probabilidad de la clase 1 (superar al mercado)
        prob = self.best_model.predict_proba(scaled_features)[-1, 1]
        return float(prob)
