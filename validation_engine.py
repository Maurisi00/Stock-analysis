import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew

from ml_engine import MachineLearningEngine
from scoring_backtest_engine import BacktestAndScoringEngine

logger = logging.getLogger("QuantSystem.ValidationEngine")

# Número de tramos cronológicos en que se divide el histórico para la
# validación por sub-períodos (Prompt 11-A).
N_SUBPERIODS = 4

# Si el hit_rate del grupo señal varía más de esto (puntos porcentuales) entre
# el mejor y el peor sub-período, el modelo se marca como régimen-dependiente:
# funciona en algunos regímenes de mercado y no en otros, no un patrón estable.
HIT_RATE_REGIME_DEPENDENT_THRESHOLD = 0.20

# Umbral de Sharpe Deflactado (probabilidad de que el Sharpe verdadero del
# modelo ganador sea positivo, ya corregido por haber probado N modelos) por
# debajo del cual se considera que la ventaja de Sharpe podría ser producto
# de la selección múltiple, no de una habilidad real. 0.95 replica la
# convención habitual de un test a una cola con 95% de confianza.
DSR_CONFIDENCE_THRESHOLD = 0.95

# Constante de Euler-Mascheroni, usada en la aproximación de Bailey & López de
# Prado al valor esperado del máximo de N Sharpe ratios bajo la null.
EULER_MASCHERONI = 0.5772156649015329

# sqrt(trading_days_per_year) — factor para pasar un Sharpe ANUALIZADO (el
# que devuelve run_strategy_backtest) al Sharpe DIARIO que exige la
# Probabilistic Sharpe Ratio, cuyo T se cuenta en observaciones diarias. Ver
# _deflate_sharpe: sin esta conversión el numerador quedaba inflado ~16x y el
# DSR salía saturado en 1.0000 siempre.
#
# Fase 3.6: el 252 pasa a ser un parámetro. TRADING_DAYS_SQRT se conserva como
# el valor de acciones (el default) porque es lo que consumen los tests de
# unidades del DSR; las funciones aceptan trading_days_per_year para que un
# activo que cotiza 365 días al año no se de-anualice con la raíz equivocada.
TRADING_DAYS_PER_YEAR_DEFAULT = 252
TRADING_DAYS_SQRT = float(np.sqrt(TRADING_DAYS_PER_YEAR_DEFAULT))


class ValidationEngine:
    """
    Chequeos de sobreajuste sobre la señal ML: ¿descubre un patrón real o solo
    captura el período concreto en que se la backtesteó?

    FASE 3.3 — ESTE ENGINE YA NO SE EJECUTA EN EL PIPELINE. Sus chequeos
    validan un modelo que salió de la ruta de decisión, así que validar su
    robustez es responder una pregunta que ya no se hace. El fichero se
    conserva —con sus tests intactos— porque lo que hay dentro (event study
    por sub-períodos independientes, purga, embargo, la matemática de la
    Probabilistic Sharpe Ratio) es la infraestructura que haría falta entera
    si algún día se construye el panel transversal de la Fase 6, y es la vara
    con la que la Fase 2 tumbó al ML. Se reutiliza, no se reinventa.

    Quedan dos chequeos:

      1. run_subperiod_validation — ¿el modelo funciona igual en distintos
         tramos cronológicos, o solo en uno (régimen-dependencia)?
      2. compute_deflated_sharpe_ratio — ¿el Sharpe de la estrategia sobrevive
         la corrección por haber probado 3 modelos y quedarse con el mejor
         (selección múltiple), o es en gran parte ese sesgo?

    SE ELIMINARON `run_sensitivity_analysis` (con todos sus `sensitivity_*`) y
    `generate_robustness_verdict`. El primero medía cuánto se movía el Score
    Multifactorial al variar una decisión de modelado a la vez: sin score no
    hay nada que mover. El segundo comprimía los tres chequeos en una etiqueta
    ALTA/MEDIA/BAJA — exactamente el tipo de resumen categórico de varias
    dimensiones en una palabra que la Fase 3 elimina del informe, así que no
    tiene sentido reponerlo aquí bajo otro nombre.
    """

    # --- 1. Validación por sub-períodos -------------------------------------

    def run_subperiod_validation(self, market_data: pd.DataFrame, benchmark_data: pd.DataFrame,
                                  features_df: pd.DataFrame, target_series: pd.Series, ticker: str,
                                  horizon_days: int, buy_threshold: float, ml_train_size: float,
                                  ml_block_days: int, random_state: int,
                                  n_blocks: int = N_SUBPERIODS) -> Dict[str, Any]:
        """
        Divide el histórico ALINEADO (intersección de features y target, tras
        el burn-in de momentum/GARCH/ARIMA) en n_blocks tramos cronológicos
        CONSECUTIVOS y NO solapados. Para cada tramo: reentrena un
        MachineLearningEngine SOLO con datos de ese tramo (su propia
        selección walk-forward y sus propias probabilidades históricas, no
        las de la corrida sobre el histórico completo) y corre
        run_holding_period_backtest restringido a los precios de ese mismo
        tramo — un event study genuinamente independiente por sub-período.

        Un tramo se marca "skipped" (en vez de hacer fallar toda la
        validación) si es demasiado corto para producir siquiera un bloque
        walk-forward evaluable — un sub-período de un histórico ya acotado
        por lookback_years puede fácilmente no alcanzar el mínimo.

        Si el hit_rate del grupo señal varía más de
        HIT_RATE_REGIME_DEPENDENT_THRESHOLD (20 puntos porcentuales) entre el
        mejor y el peor tramo evaluable, 'regime_dependent' queda en True: el
        modelo funciona en algunos períodos de mercado y no en otros, no es
        un patrón estable a través de regímenes.
        """
        target_named = target_series.rename("target")
        combined = pd.concat([features_df, target_named], axis=1).dropna()

        if combined.empty:
            return {"blocks": [], "n_valid_blocks": 0, "hit_rate_spread": float("nan"),
                    "regime_dependent": None,
                    "note": "Histórico alineado vacío; no se pudo dividir en sub-períodos."}

        date_chunks = np.array_split(combined.index, n_blocks)

        blocks: List[Dict[str, Any]] = []
        for i, block_dates in enumerate(date_chunks, start=1):
            block_dates = pd.DatetimeIndex(block_dates)
            block_info: Dict[str, Any] = {
                "block": i,
                "start_date": str(block_dates.min().date()) if len(block_dates) else None,
                "end_date": str(block_dates.max().date()) if len(block_dates) else None,
            }
            try:
                block_X = combined.loc[block_dates, features_df.columns]
                block_y = combined.loc[block_dates, "target"]

                block_ml_engine = MachineLearningEngine(
                    feature_matrix=block_X, target_series=block_y, embargo_days=horizon_days,
                    train_size=ml_train_size, random_state=random_state,
                )
                block_ml_engine.evaluate_and_select_best_model(block_days=ml_block_days)
                block_probs = block_ml_engine.generate_walkforward_probabilities(block_days=ml_block_days)
                if block_probs.empty:
                    raise ValueError("walk-forward vacío (tramo demasiado corto)")

                block_scoring_engine = BacktestAndScoringEngine(
                    market_data=market_data.loc[block_dates], benchmark_data=benchmark_data.loc[block_dates],
                    ml_probabilities=block_probs,
                )
                hp_block = block_scoring_engine.run_holding_period_backtest(
                    ticker=ticker, horizon_days=horizon_days, buy_threshold=buy_threshold,
                )

                block_info.update({
                    "skipped": False,
                    "n_signals": hp_block["n_signals"],
                    "n_independent_signals": hp_block["n_independent_signals"],
                    "hit_rate": hp_block["signal_group"]["hit_rate"],
                    "mean_excess": hp_block["signal_group"]["mean"],
                    "statistically_conclusive": hp_block["statistically_conclusive"],
                })
            except Exception as e:
                logger.warning(f"Sub-período {i}: se omite ({e}).")
                block_info.update({"skipped": True, "reason": str(e)})

            blocks.append(block_info)

        n_valid_blocks, hit_rate_spread, regime_dependent = self._assess_regime_dependence(blocks)

        return {
            "blocks": blocks,
            "n_valid_blocks": n_valid_blocks,
            "hit_rate_spread": hit_rate_spread,
            "regime_dependent": regime_dependent,
        }

    @staticmethod
    def _assess_regime_dependence(blocks: List[Dict[str, Any]]) -> Tuple[int, float, Optional[bool]]:
        """
        Lógica pura (sin reentrenamiento) extraída de run_subperiod_validation
        para que sea testeable con bloques fabricados a mano: con menos de 2
        bloques con señales evaluables no hay base para juzgar régimen-
        dependencia (ni True ni False, sino 'no evaluable' -> None).

        Un bloque solo entra al cálculo del spread si es 'statistically_conclusive'
        (n_independent_signals >= MIN_INDEPENDENT_SIGNALS_FOR_CONCLUSIVE, ver
        scoring_backtest_engine.py) — no basta con tener al menos 1 señal. Con
        1 señal por bloque, un hit_rate de 0% en un tramo y 100% en otro son
        ambos degenerados (una sola observación cada uno): comparar esos dos
        números produce un spread del 100% y una alarma de "régimen-dependiente"
        sin ninguna base estadística real, aunque cada bloque individualmente
        ya se reporta como no concluyente.
        """
        valid_hit_rates = [
            b["hit_rate"] for b in blocks
            if not b.get("skipped")
            and b.get("n_signals", 0) > 0
            and not np.isnan(b["hit_rate"])
            and b.get("statistically_conclusive", False)
        ]

        if len(valid_hit_rates) < 2:
            return len(valid_hit_rates), float("nan"), None

        hit_rate_spread = float(max(valid_hit_rates) - min(valid_hit_rates))
        regime_dependent = bool(hit_rate_spread > HIT_RATE_REGIME_DEPENDENT_THRESHOLD)
        return len(valid_hit_rates), hit_rate_spread, regime_dependent

    # --- 2. Deflated Sharpe Ratio --------------------------------------------

    def compute_deflated_sharpe_ratio(self, ml_engine: MachineLearningEngine, market_data: pd.DataFrame,
                                       benchmark_data: pd.DataFrame, ticker: str, buy_threshold: float,
                                       transaction_cost_bps: float, slippage_bps: float,
                                       risk_free_rate: float, block_days: int,
                                       ml_selection_metrics: Optional[Dict[str, Dict[str, float]]] = None,
                                       trading_days_per_year: int = TRADING_DAYS_PER_YEAR_DEFAULT,
                                       ) -> Dict[str, Any]:
        """
        Bailey & López de Prado (2014), "The Deflated Sharpe Ratio": probar
        varios modelos/configuraciones y quedarse con el de mejor desempeño
        (aquí: 3 clasificadores, elegidos por Balanced-Accuracy walk-forward
        en evaluate_and_select_best_model) infla el Sharpe reportado del
        ganador por selección múltiple — el máximo de N variables aleatorias
        tiene un valor esperado positivo bajo la null de que NINGUNA tenga
        ventaja real, solo por buscar el máximo entre N.

        Se calcula el Sharpe (run_strategy_backtest, bruto) de la estrategia
        que resultaría de CADA uno de los 3 modelos (reutilizando el
        walk-forward ya cacheado en ml_engine._walkforward_predict — no hay
        reentrenamiento adicional), y se evalúa la Probabilistic Sharpe Ratio
        del modelo GANADOR contra el Sharpe esperado del máximo de esos 3
        (no contra 0): esa PSR deflactada es la probabilidad de que el
        Sharpe verdadero del ganador sea positivo, ya corregida por la
        búsqueda entre 3 configuraciones (Mertens 2002 / Bailey-LdP 2012/2014).

        ml_selection_metrics es el dict que devuelve
        ml_engine.evaluate_and_select_best_model() (mismas claves que
        sharpe_by_model, {nombre_modelo: {"Balanced-Accuracy": ..., ...}}) —
        el CRITERIO REAL con el que se eligió al ganador, distinto del Sharpe
        que se calcula acá. Se reexpone en el resultado como
        'balanced_accuracy_by_model' porque el Sharpe walk-forward de cada
        modelo y su Balanced-Accuracy miden objetivos distintos (retorno
        ajustado por riesgo vs. acierto de dirección) y pueden discrepar — sin
        este dato, el informe podía mostrar un ganador con Sharpe muy por
        debajo de otro modelo sin dar ninguna pista de por qué.

        `n_trials` ES UN SUELO, NO EL NÚMERO REAL DE ESPECIFICACIONES PROBADAS
        (Fase 1.10 / AUDITORIA_2026-09.md C2). Vale `len(ml_engine.models)` = 3,
        los tres clasificadores entre los que elige
        evaluate_and_select_best_model. Pero la búsqueda real que produjo la
        configuración actual del sistema es de DECENAS de especificaciones:
        ventana de momentum, umbrales de calidad, train_size, spec del GARCH,
        el horizonte, el buy_threshold, drift_mode, los pesos del score que
        hubo hasta la Fase 3.3, la elección de Balanced-Accuracy como criterio
        de selección, el embargo/purga, block_days... Nada de eso entra en el
        conteo. (El análisis de sensibilidad que recorría las cuatro primeras
        dimensiones se borró en la Fase 3.3 junto con el score que medía; el
        argumento sobre n_trials no depende de él.)

        Consecuencia directa: `expected_max_sharpe` crece con n_trials, así
        que subestimarlo hace que el umbral a superar sea DEMASIADO BAJO y el
        DSR resultante sea OPTIMISTA POR CONSTRUCCIÓN. Un DSR alto acá no
        prueba habilidad: prueba que el ganador supera lo que cabría esperar
        de buscar entre 3 alternativas, cuando en realidad se buscó entre
        muchas más. Leerlo como un límite superior de la confianza, nunca como
        una medida calibrada.
        """
        model_names = list(ml_engine.models.keys())
        n_trials = len(model_names)
        ml_selection_metrics = ml_selection_metrics or {}
        balanced_accuracy_by_model = {
            name: metrics.get("Balanced-Accuracy", float("nan"))
            for name, metrics in ml_selection_metrics.items()
        }

        sharpe_by_model: Dict[str, float] = {}
        strat_returns_by_model: Dict[str, pd.Series] = {}
        for name in model_names:
            wf = ml_engine._walkforward_predict(name, block_days)
            if len(wf["dates"]) == 0:
                continue
            probs = pd.Series(wf["probs"], index=wf["dates"]).sort_index()
            temp_engine = BacktestAndScoringEngine(
                market_data=market_data, benchmark_data=benchmark_data, ml_probabilities=probs,
            )
            df_bt, bt_metrics = temp_engine.run_strategy_backtest(
                ticker=ticker, buy_threshold=buy_threshold, transaction_cost_bps=transaction_cost_bps,
                slippage_bps=slippage_bps, risk_free_rate=risk_free_rate,
                # Fase 3.6: el backtest anualiza con las MISMAS sesiones por año
                # con las que _deflate_sharpe va a de-anualizar más abajo. Si
                # las dos no coinciden, vuelve el desajuste de unidades que la
                # Fase 1.10 arregló.
                trading_days_per_year=trading_days_per_year,
            )
            sharpe_by_model[name] = bt_metrics["strat_sharpe"]
            strat_returns_by_model[name] = df_bt["strat_ret"]

        if not sharpe_by_model:
            return {
                "n_trials": n_trials, "n_evaluable_trials": 0, "sharpe_by_model": {},
                "balanced_accuracy_by_model": balanced_accuracy_by_model,
                "selected_model": None, "nominal_sharpe": float("nan"), "sharpe_threshold": float("nan"),
                "deflated_sharpe_ratio": float("nan"),
                "note": "Histórico insuficiente para evaluar ningún modelo walk-forward.",
            }

        selected_name = ml_engine.best_model_name if ml_engine.best_model_name in sharpe_by_model \
            else max(sharpe_by_model, key=sharpe_by_model.get)

        result = self._deflate_sharpe(
            sharpe_by_model=sharpe_by_model, selected_name=selected_name,
            strat_ret=strat_returns_by_model[selected_name],
            trading_days_per_year=trading_days_per_year,
        )
        result["n_trials"] = n_trials
        result["n_evaluable_trials"] = len(sharpe_by_model)
        result["balanced_accuracy_by_model"] = balanced_accuracy_by_model
        return result

    @staticmethod
    def _deflate_sharpe(sharpe_by_model: Dict[str, float], selected_name: str,
                         strat_ret: pd.Series,
                         trading_days_per_year: int = TRADING_DAYS_PER_YEAR_DEFAULT) -> Dict[str, Any]:
        """
        Matemática pura de Bailey & López de Prado, extraída de
        compute_deflated_sharpe_ratio para que sea testeable con Sharpes y
        series de retorno fabricadas a mano, sin reentrenar nada:
          1. sigma_sr = desvío de los Sharpe de las n_trials configuraciones
             probadas -> expected_max_sharpe, el Sharpe esperado del MEJOR de
             ellas bajo la null de que ninguna tenga ventaja real (aproximación
             de Bailey-LdP vía la distribución de Gumbel del máximo de N
             normales).
          2. Probabilistic Sharpe Ratio (Mertens 2002 / Bailey-López de Prado)
             del modelo ganador, evaluada en expected_max_sharpe (no en 0):
             γ3 = skewness, γ4 = kurtosis NO-excedente (normal = 3.0) de sus
             retornos de estrategia.

        UNIDADES (Fase 1.10 / AUDITORIA_2026-09.md C2). La PSR exige que el
        Sharpe y T estén en la MISMA frecuencia. Los Sharpe que llegan acá
        vienen ANUALIZADOS de run_strategy_backtest, mientras T se mide en
        observaciones DIARIAS: el numerador quedaba inflado ~sqrt(252) ≈ 16x y
        el DSR salía saturado en 1.0000 siempre que el ganador superara el
        umbral. El chequeo más sofisticado del sistema era un sello de goma.

        La conversión se aplica a TODO el vector de Sharpes, no solo al del
        ganador: `sigma_sr` —y por tanto `expected_max_sharpe`— se derivan de
        ese vector, así que desanualizar únicamente `sr_hat` compararía un
        Sharpe diario contra un umbral anualizado y hundiría el DSR a ~0
        siempre, que es el espejo exacto del bug original. La skewness y la
        kurtosis ya se calculan sobre `strat_ret` DIARIO, así que son
        coherentes con la frecuencia diaria.

        Los valores que se devuelven para el informe (`nominal_sharpe`,
        `sharpe_threshold`) se reportan ANUALIZADOS, que es la escala en la que
        un Sharpe es interpretable; el cálculo interno es el diario. Se
        devuelven también las versiones diarias para poder auditarlo.
        """
        # Sharpe DIARIO = Sharpe anualizado / sqrt(252), la inversa exacta de
        # la anualización que aplica _perf_metrics en run_strategy_backtest.
        # Fase 3.6: la raíz del número de sesiones por año, no el literal
        # sqrt(252). La PSR exige que el Sharpe y el T estén en la MISMA
        # frecuencia, así que de-anualizar con la raíz equivocada es el mismo
        # error de unidades que la Fase 1.10 tuvo que arreglar, solo que
        # desplazado a los activos que no cotizan 252 días al año.
        raiz_dias = float(np.sqrt(trading_days_per_year))
        sharpe_diario_by_model = {
            name: sharpe / raiz_dias for name, sharpe in sharpe_by_model.items()
        }

        sr_hat_anual = sharpe_by_model[selected_name]
        sr_hat = sharpe_diario_by_model[selected_name]
        T = len(strat_ret)

        sharpe_values = list(sharpe_diario_by_model.values())
        sigma_sr = float(np.std(sharpe_values, ddof=1)) if len(sharpe_values) > 1 else 0.0

        if sigma_sr <= 0.0 or T < 3:
            return {
                "sharpe_by_model": sharpe_by_model, "selected_model": selected_name,
                "nominal_sharpe": sr_hat_anual, "sharpe_threshold": float("nan"),
                "nominal_sharpe_daily": sr_hat, "sharpe_threshold_daily": float("nan"),
                "deflated_sharpe_ratio": float("nan"),
                "note": "Histórico insuficiente o sin variación de Sharpe entre modelos para deflactar de forma significativa.",
            }

        n_trials = len(sharpe_values)
        z_n = norm.ppf(1.0 - 1.0 / n_trials)
        z_ne = norm.ppf(1.0 - 1.0 / (n_trials * np.e))
        expected_max_sharpe = sigma_sr * ((1 - EULER_MASCHERONI) * z_n + EULER_MASCHERONI * z_ne)

        skewness = float(skew(strat_ret))
        kurt = float(kurtosis(strat_ret, fisher=False))
        denom = np.sqrt(max(1e-12, 1 - skewness * sr_hat + ((kurt - 1) / 4) * sr_hat ** 2))
        z_stat = (sr_hat - expected_max_sharpe) * np.sqrt(T - 1) / denom
        dsr = float(norm.cdf(z_stat))

        return {
            "sharpe_by_model": sharpe_by_model,
            "selected_model": selected_name,
            # Anualizados: la escala en la que un Sharpe se lee.
            "nominal_sharpe": sr_hat_anual,
            "sharpe_threshold": float(expected_max_sharpe * raiz_dias),
            # Diarios: los que realmente entran en la PSR, para poder auditarla.
            "nominal_sharpe_daily": sr_hat,
            "sharpe_threshold_daily": float(expected_max_sharpe),
            "deflated_sharpe_ratio": dsr,
            "note": None,
        }
