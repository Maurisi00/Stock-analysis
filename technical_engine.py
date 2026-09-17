import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional

class TechnicalFactorEngine:
    """
    Engine encargado del cálculo de factores de Momentum Académico, 
    Tendencia y Métricas de Riesgo Estadístico Avanzado.
    """
    def __init__(self, market_data: pd.DataFrame, benchmark_data: pd.DataFrame,
                 trading_days_per_year: int = 252):
        """
        :param market_data: DataFrame con precios de cierre ajustados (Columnas = Tickers)
        :param benchmark_data: DataFrame con precio de cierre ajustado del Benchmark
        """
        self.market_data = market_data
        self.benchmark_data = benchmark_data
        self.log_returns = np.log(market_data / market_data.shift(1))
        self.bench_returns = np.log(benchmark_data / benchmark_data.shift(1))
        # Fase 3.6: días de negociación por año del activo, para la
        # anualización de la volatilidad y para la ventana de la beta rolling
        # (que es una ventana de UN AÑO, no de "252 días"). Las ventanas de
        # FORMA del momentum académico 12-1 (21/252) siguen siendo literales a
        # propósito: definen el factor, no una conversión de unidades — ver
        # config.PipelineConfig.trading_days_per_year.
        self.trading_days_per_year = int(trading_days_per_year)

    def compute_momentum_factors(self) -> Dict[str, pd.DataFrame]:
        """
        Calcula el Momentum Académico "X-1" (lookback de X meses EXCLUYENDO el
        último mes, para mitigar reversión de corto plazo/ruido de
        microestructura) para X = 3, 6 y 12 meses — la misma convención
        (excluir shift(21)) se aplica a los TRES, no solo al de 12 meses.

        IMPORTANTE — los nombres son la convención académica del lookback
        (12-1, 6-1, 3-1 momentum), NO la ventana de precios efectivamente
        medida, que es 21 días hábiles (~1 mes) más corta en los tres casos:
          - mom_3m  = ln(P[t-21] / P[t-63])  -> mide P entre t-63 y t-21:
                      una ventana de 42 días hábiles (~2 meses), no 3 meses.
          - mom_6m  = ln(P[t-21] / P[t-126]) -> mide entre t-126 y t-21:
                      105 días hábiles (~5 meses), no 6.
          - mom_12m = ln(P[t-21] / P[t-252]) -> mide entre t-252 y t-21:
                      231 días hábiles (~11 meses) — el "12-1 momentum"
                      académico estándar (Jegadeesh-Titman), ya documentado
                      así en la literatura bajo ese nombre.
        No renombrar sin actualizar todos los consumidores (features_df en
        main_pipeline.py, ml_engine, quantile_forecast_engine, tests).
        """
        momentum_dict = {}

        # 1 mes bursátil ≈ 21 días
        momentum_dict['mom_3m'] = np.log(self.market_data.shift(21) / self.market_data.shift(63))
        momentum_dict['mom_6m'] = np.log(self.market_data.shift(21) / self.market_data.shift(126))
        # Académico: 12 meses excluyendo el último mes (12-1 Mom)
        momentum_dict['mom_12m'] = np.log(self.market_data.shift(21) / self.market_data.shift(252))

        return momentum_dict

    def compute_custom_momentum(self, lookback_months: int) -> pd.DataFrame:
        """
        Generaliza compute_momentum_factors a un lookback arbitrario (X-1
        meses, misma convención de excluir el último mes): usado por
        ValidationEngine para el análisis de sensibilidad del Score frente a
        la elección de ventana de Momentum (9/12/15 meses en vez del 12
        académico fijo) — no reemplaza a compute_momentum_factors, que sigue
        fijando 3/6/12 para las features de ML y el resto del pipeline.
        """
        # LITERAL A PROPÓSITO (Fase 3.6): las ventanas del momentum académico
        # son de FORMA, no una conversión de unidades — ver
        # config.PipelineConfig.trading_days_per_year para por qué esto NO se
        # enhebra con trading_days_per_year.
        lookback_days = lookback_months * 21
        return np.log(self.market_data.shift(21) / self.market_data.shift(lookback_days))

    @staticmethod
    def score_momentum(mom_12m_series: pd.Series, absolute_floor: float = -0.40,
                        absolute_ceiling: float = 0.60) -> float:
        """
        Combina, para el factor de Momentum, la MEDIA de dos señales:
          (a) el percentil histórico del momentum 12-1 de la propia empresa, y
          (b) un score de NIVEL ABSOLUTO del retorno en sí, mapeando
              linealmente [absolute_floor, absolute_ceiling] a [0, 1]
              (acotado fuera de ese rango).

        Solo percentil histórico produce lecturas engañosas: una acción con un
        pico extraordinario en su propio pasado (ej. META o UBER en 2023)
        puntúa bajo HOY aunque su momentum actual sea bueno en términos
        absolutos, simplemente porque no alcanza ese pico — el score queda
        ciego al nivel real de retorno. Los anclajes [-40%, 60%] son una
        heurística de referencia general (retornos 12 meses típicos de renta
        variable), calibrable por activo/sector.

        :param mom_12m_series: serie de mom_12m (puede traer NaNs de burn-in).
        :return: score en [0,1], o NaN si no hay ningún dato disponible.
        """
        series = mom_12m_series.dropna()
        if series.empty:
            return np.nan

        percentile_score = float(series.rank(pct=True).iloc[-1])
        latest_value = float(series.iloc[-1])
        absolute_score = float(np.clip(
            (latest_value - absolute_floor) / (absolute_ceiling - absolute_floor), 0.0, 1.0
        ))
        return float(np.mean([percentile_score, absolute_score]))

    def compute_trend_signals(self) -> pd.DataFrame:
        """
        Calcula medias móviles (50 y 200 días), su cruce (Golden/Death Cross) y el RSI.
        """
        signals = {}
        
        for ticker in self.market_data.columns:
            prices = self.market_data[ticker]
            ma50 = prices.rolling(window=50).mean()
            ma200 = prices.rolling(window=200).mean()
            
            # Señal de cruce estructural: 1 si MA50 > MA200, 0 de lo contrario
            crossover = (ma50 > ma200).astype(int)
            
            # RSI con suavizado de Wilder (estándar de mercado: EMA con alpha=1/14)
            delta = prices.diff()
            gain = delta.where(delta > 0, 0.0)
            loss = -delta.where(delta < 0, 0.0)
            avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
            avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
            rs = avg_gain / (avg_loss + 1e-10)
            rsi = 100 - (100 / (1 + rs))
            
            signals[f'{ticker}_ma50'] = ma50
            signals[f'{ticker}_ma200'] = ma200
            signals[f'{ticker}_crossover'] = crossover
            signals[f'{ticker}_rsi'] = rsi
            
        return pd.DataFrame(signals, index=self.market_data.index)

    def compute_statistical_risk(self, beta_window: Optional[int] = None) -> Dict[str, Any]:
        """
        Calcula la Volatilidad Histórica Anualizada (trading_days_per_year),
        el Máximo Drawdown (ambos de muestra completa) y la Beta ROLLING
        (ventana de beta_window días) frente al Benchmark.

        Beta rolling = Cov(retorno activo, retorno benchmark) / Var(retorno
        benchmark) sobre una ventana móvil — algebraicamente equivalente al
        coeficiente de una OLS de un solo regresor (con constante) en esa
        misma ventana, pero mucho más barato de recalcular día a día que
        reajustar sm.OLS ventana por ventana. Antes se calculaba una ÚNICA
        OLS de muestra completa (una beta congelada para todo el histórico),
        pese a que este docstring ya decía "Beta móvil": un activo puede
        cambiar de perfil de riesgo con el tiempo (ej. una empresa que
        madura de crecimiento a value), y una beta de muestra completa no
        lo captura — por eso 'beta' devuelve una serie TEMPORAL (un valor
        por fecha, NaN en el burn-in de beta_window días), a diferencia de
        'volatility_ann'/'max_drawdown', que siguen siendo un escalar por
        ticker.
        """
        if beta_window is None:
            beta_window = self.trading_days_per_year

        vol_anualizada = self.log_returns.std() * np.sqrt(self.trading_days_per_year)

        # Máximo Drawdown Histórico
        roll_max = self.market_data.cummax()
        drawdown = (self.market_data - roll_max) / roll_max
        max_drawdown = drawdown.min()

        # Beta ROLLING contra el Benchmark
        combined_returns = pd.concat([self.log_returns, self.bench_returns], axis=1).dropna()
        bench_col = self.bench_returns.columns[0]
        bench_ret = combined_returns[bench_col]
        rolling_bench_var = bench_ret.rolling(window=beta_window).var()

        betas = {}
        for ticker in self.market_data.columns:
            asset_ret = combined_returns[ticker]
            rolling_cov = asset_ret.rolling(window=beta_window).cov(bench_ret)
            betas[ticker] = rolling_cov / rolling_bench_var

        return {
            'volatility_ann': vol_anualizada,
            'max_drawdown': max_drawdown,
            'beta': pd.DataFrame(betas)
        }