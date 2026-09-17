import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

# Por debajo de este número de observaciones INDEPENDIENTES (no solapadas), el
# event study de run_holding_period_backtest se marca como estadísticamente no
# concluyente en el informe, sin importar qué tan grande se vea la diferencia
# de exceso entre el grupo señal y el grupo control.
MIN_INDEPENDENT_SIGNALS_FOR_CONCLUSIVE = 10

# FASE 3.3 — EL SCORE MULTIFACTORIAL Y LA RECOMENDACIÓN SE ELIMINARON.
#
# Se fueron de este módulo: DEFAULT_FACTOR_WEIGHTS, DEFAULT_RECOMMENDATION_BANDS,
# NO_EVALUABLE_TAG, _RISK_COMPONENT_THRESHOLDS, _bounded_linear_score,
# calculate_multifactor_score, generate_recommendation, calculate_risk_factor,
# calibrate_recommendation_bands y load_recommendation_bands (con su fichero
# persistido: sin score no hay nada que cortar en bandas).
#
# Por qué, y por qué no volver a ponerlo: la Fase 2 midió el poder predictivo
# de la capa que el score ponderaba, con la vara del propio sistema (ventanas
# INDEPENDIENTES, no fechas solapadas). De 49 celdas evaluables solo 10 tienen
# un Information Coefficient con muestra suficiente, las diez al horizonte de 1
# mes, y ahí sale -0,009 de media con el signo repartido 5-5. Al horizonte de
# producción (6 meses) NINGUNO de los 10 tickers tiene los dos grupos del event
# study estimables. Un promedio ponderado de cinco factores cuyo componente
# predictivo mide cero no deja de medir cero por promediarse: solo se vuelve
# más difícil de auditar, porque una etiqueta ("COMPRA") esconde de qué
# medición salió y con cuántas observaciones. Ver PLAN_REFACTOR.md 3.3 y el
# bloque FASE 2/FASE 3 de "Ajustes y Arreglos.txt".
#
# LO QUE SÍ QUEDA en este fichero son los dos backtests. Ya no son secciones
# del informe, pero siguen siendo la infraestructura que reutilizan
# `ValidationEngine` (event study por sub-períodos, Sharpe deflactado) y
# `scripts/diagnostico_multihorizonte.py` (IC, Newey-West) — y esa es la parte
# mejor construida del repo, la que haría falta entera si algún día se
# construye el panel transversal de la Fase 6.


class BacktestAndScoringEngine:
    """
    Backtesting histórico de una señal de probabilidad sobre un activo.

    Hasta la Fase 3.3 este engine también calculaba el Score Multifactorial y
    emitía la recomendación COMPRAR/MANTENER/EVITAR; las dos cosas se
    eliminaron (ver la nota de cabecera del módulo). Lo que queda son los dos
    backtests:

      - `run_holding_period_backtest`: event study "compro cuando la señal
        dice comprar y aguanto H días", con la corrección por solapamiento de
        ventanas (`n_independent_signals`) al lado de cada métrica.
      - `run_strategy_backtest`: rebalanceo diario long/flat, que NADIE
        ejecuta en la práctica — insumo interno del Sharpe deflactado.

    Ninguno de los dos se imprime ya en el informe: validan una señal ML que
    salió de la ruta de decisión. Se conservan porque `ValidationEngine` y
    `scripts/diagnostico_multihorizonte.py` los consumen, y porque son la vara
    con la que se mediría cualquier señal futura.
    """
    def __init__(self, market_data: pd.DataFrame, benchmark_data: pd.DataFrame, ml_probabilities: pd.Series):
        self.prices = market_data
        self.bench_prices = benchmark_data
        self.ml_probs = ml_probabilities
        
        # Calcular retornos logarítmicos diarios para el backtest
        self.asset_returns = np.log(self.prices / self.prices.shift(1))
        self.bench_returns = np.log(self.bench_prices / self.bench_prices.shift(1))

    def run_strategy_backtest(self, ticker: str, buy_threshold: float = 0.50,
                               trading_days_per_year: int = 252,
                               transaction_cost_bps: float = 5.0, slippage_bps: float = 5.0,
                               risk_free_rate: float = 0.04
                               ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Simula una estrategia donde se mantiene el activo si la probabilidad del modelo
        está por encima del umbral establecido, de lo contrario se refugia en liquidez (0%).

        Aplica un coste fijo (transaction_cost_bps + slippage_bps) cada vez que la posición
        cambia (entrada o salida), y reporta métricas tanto brutas (sin fricciones) como
        netas de costes, para dejar explícito cuánto retorno se erosiona en la práctica.
        """
        # Alinear series temporales
        df_bt = pd.concat([self.asset_returns[ticker], self.bench_returns], axis=1).dropna()
        df_bt.columns = ['asset_ret', 'bench_ret']

       # Mapear las señales del modelo de Machine Learning (retardadas un día para evitar lookahead bias)
        df_bt['ml_prob'] = self.ml_probs

        # Restringir el backtest SOLO a las fechas donde el modelo realmente generó
        # una probabilidad. Sin esto, los días sin ml_prob quedan forzados a
        # "señal=0", mezclando "no hay dato" con "el modelo predijo bajista".
        df_bt = df_bt.dropna(subset=['ml_prob'])

        df_bt['signal'] = (df_bt['ml_prob'].shift(1) > buy_threshold).astype(int).fillna(0)

        # Calcular retornos de la estrategia
        df_bt['strat_ret'] = df_bt['asset_ret'] * df_bt['signal']

        # Costes de fricción: se paga cada vez que la posición cambia (entra o sale),
        # incluida la primera entrada (diff() da NaN en la primera fila).
        df_bt['trade'] = df_bt['signal'].diff().fillna(df_bt['signal']).abs()
        cost_per_trade = (transaction_cost_bps + slippage_bps) / 10000.0
        df_bt['cost'] = df_bt['trade'] * cost_per_trade
        df_bt['strat_ret_net'] = df_bt['strat_ret'] - df_bt['cost']

        # Curvas de rendimiento acumulado compuesto (convertido de logarítmico a aritmético para visualización)
        df_bt['cum_bench'] = np.exp(df_bt['bench_ret'].cumsum()) - 1
        df_bt['cum_strat'] = np.exp(df_bt['strat_ret'].cumsum()) - 1
        df_bt['cum_strat_net'] = np.exp(df_bt['strat_ret_net'].cumsum()) - 1

        # Métricas de desempeño de la estrategia
        total_days = len(df_bt)
        # Fase 3.6: el factor de anualización y la sigma anual salen de
        # trading_days_per_year, no del literal 252 (ver
        # config.PipelineConfig.trading_days_per_year).
        ann_factor = trading_days_per_year / total_days
        rf = risk_free_rate

        def _perf_metrics(ret_series: pd.Series, cum_series: pd.Series) -> Tuple[float, float, float, float]:
            cum_return = float(cum_series.iloc[-1])
            ann_return = float(np.exp(ret_series.sum() * ann_factor) - 1)
            vol = ret_series.std() * np.sqrt(trading_days_per_year)
            sharpe = float((ann_return - rf) / vol) if vol > 0 else 0.0
            cum_prices = np.exp(ret_series.cumsum())
            roll_max = cum_prices.cummax()
            drawdown = (cum_prices - roll_max) / roll_max
            max_dd = float(drawdown.min())
            return cum_return, ann_return, sharpe, max_dd

        strat_cum_return, strat_ann_return, strat_sharpe, max_dd = _perf_metrics(df_bt['strat_ret'], df_bt['cum_strat'])
        strat_cum_return_net, strat_ann_return_net, strat_sharpe_net, max_dd_net = _perf_metrics(df_bt['strat_ret_net'], df_bt['cum_strat_net'])

        bench_cum_return = float(df_bt['cum_bench'].iloc[-1])

        metrics = {
            "strat_cum_return": strat_cum_return,
            "bench_cum_return": bench_cum_return,
            "strat_ann_return": strat_ann_return,
            "strat_sharpe": strat_sharpe,
            "strat_max_drawdown": max_dd,
            "strat_cum_return_net": strat_cum_return_net,
            "strat_ann_return_net": strat_ann_return_net,
            "strat_sharpe_net": strat_sharpe_net,
            "strat_max_drawdown_net": max_dd_net,
            "n_trades": int(df_bt['trade'].sum()),
            "total_cost_drag": float(df_bt['cost'].sum()),
        }

        return df_bt, metrics

    def run_holding_period_backtest(self, ticker: str, horizon_days: int,
                                     buy_threshold: float = 0.50) -> Dict[str, Any]:
        """
        Event study de la estrategia REAL del usuario: "compro hoy si la señal
        dice comprar, y aguanto horizon_days días" — no el rebalanceo diario
        que simula run_strategy_backtest, que nunca se va a ejecutar en la
        práctica y cuyo Sharpe/drawdown/n_trades no describen esta decisión.

        Para cada fecha t con ml_prob[t] disponible, calcula el retorno
        SIMPLE (no logarítmico — a diferencia del resto del engine, esto
        responde una pregunta de "cuánto más ganaste", que es aritmética por
        naturaleza) del activo y del benchmark entre t y t+horizon_days, y
        separa esas fechas en dos grupos:
          - señal: ml_prob[t] > buy_threshold
          - control: el resto (mismo período, sin señal de compra)
        La diferencia de exceso medio entre ambos grupos es la señal real:
        ¿el modelo distingue algo, o el "exceso" del grupo señal es
        indistinguible del que se obtendría comprando en una fecha al azar?

        Las ventanas de horizon_days días se SOLAPAN entre fechas
        consecutivas, así que NO son observaciones independientes: con
        horizon_days=126, 500 señales solapadas equivalen a ~500/126 ≈ 4
        observaciones independientes, no 500. n_independent_signals expone
        esa cuenta ajustada; por debajo de MIN_INDEPENDENT_SIGNALS_FOR_CONCLUSIVE,
        'statistically_conclusive' queda en False y el informe debe presentar
        el bloque como no concluyente sin importar cuán grande se vea la diferencia.
        """
        asset_prices = self.prices[ticker].dropna()
        bench_col = self.bench_prices.columns[0]
        bench_prices = self.bench_prices[bench_col].dropna()

        combined = pd.DataFrame({"asset_price": asset_prices, "bench_price": bench_prices}).dropna()
        combined["asset_fwd_ret"] = combined["asset_price"].shift(-horizon_days) / combined["asset_price"] - 1
        combined["bench_fwd_ret"] = combined["bench_price"].shift(-horizon_days) / combined["bench_price"] - 1
        combined["excess_fwd_ret"] = combined["asset_fwd_ret"] - combined["bench_fwd_ret"]
        combined["ml_prob"] = self.ml_probs

        # Solo fechas con señal conocida Y retorno forward ya realizado (las
        # últimas horizon_days fechas de la serie aún no tienen desenlace).
        combined = combined.dropna(subset=["ml_prob", "excess_fwd_ret"])

        signal_mask = combined["ml_prob"] > buy_threshold
        signal_excess = combined.loc[signal_mask, "excess_fwd_ret"]
        control_excess = combined.loc[~signal_mask, "excess_fwd_ret"]

        n_signals = int(len(signal_excess))
        n_independent_signals = float(n_signals / horizon_days) if horizon_days > 0 else float(n_signals)

        def _describe(excess: pd.Series) -> Dict[str, float]:
            if excess.empty:
                return {"n": 0, "hit_rate": np.nan, "mean": np.nan, "median": np.nan, "std": np.nan,
                        "p5": np.nan, "p25": np.nan, "p50": np.nan, "p75": np.nan, "p95": np.nan}
            quantiles = excess.quantile([0.05, 0.25, 0.50, 0.75, 0.95])
            return {
                "n": int(len(excess)),
                "hit_rate": float((excess > 0).mean()),
                "mean": float(excess.mean()),
                "median": float(excess.median()),
                "std": float(excess.std()) if len(excess) > 1 else 0.0,
                "p5": float(quantiles.loc[0.05]),
                "p25": float(quantiles.loc[0.25]),
                "p50": float(quantiles.loc[0.50]),
                "p75": float(quantiles.loc[0.75]),
                "p95": float(quantiles.loc[0.95]),
            }

        signal_stats = _describe(signal_excess)
        control_stats = _describe(control_excess)

        excess_difference = (
            float(signal_stats["mean"] - control_stats["mean"])
            if signal_stats["n"] > 0 and control_stats["n"] > 0 else float("nan")
        )

        # Comprar y aguantar TODO el período de estudio (mismo tramo que las
        # señales/control), como tercera referencia junto al benchmark.
        buy_and_hold_asset_return = float(combined["asset_price"].iloc[-1] / combined["asset_price"].iloc[0] - 1)
        buy_and_hold_bench_return = float(combined["bench_price"].iloc[-1] / combined["bench_price"].iloc[0] - 1)

        return {
            "horizon_days": horizon_days,
            "buy_threshold": buy_threshold,
            "n_signals": n_signals,
            "n_independent_signals": n_independent_signals,
            "signal_group": signal_stats,
            "control_group": control_stats,
            "excess_difference": excess_difference,
            "buy_and_hold_asset_return": buy_and_hold_asset_return,
            "buy_and_hold_bench_return": buy_and_hold_bench_return,
            "statistically_conclusive": bool(n_independent_signals >= MIN_INDEPENDENT_SIGNALS_FOR_CONCLUSIVE),
        }