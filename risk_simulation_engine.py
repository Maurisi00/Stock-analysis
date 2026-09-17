import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Dict, Any, Optional

# Encogimiento aplicado en drift_mode='shrunk': cuánto peso se le da a la tasa
# libre de riesgo frente a la deriva histórica (0.5 = 50/50).
SHRINKAGE_TOWARD_RISK_FREE = 0.5


class RiskSimulationEngine:
    """
    Engine encargado de ejecutar simulaciones estocásticas Monte Carlo (MBG)
    y calcular métricas de riesgo de cola y ratios de performance para la gestión de capital.
    """
    def __init__(self, price_series: pd.Series, log_returns: pd.Series, predicted_volatility: float,
                 drift_mode: str = "shrunk", risk_free_rate: float = 0.04,
                 random_state: Optional[int] = None, trading_days_per_year: int = 252):
        """
        :param drift_mode: cómo estimar la deriva anualizada del GBM:
            - 'historical': media histórica de log-retornos anualizada. Con
              pocos años de historia su error estándar es enorme (~15%
              anual para una acción típica) — el "mejor estimador insesgado"
              es, en la práctica, ruido.
            - 'zero': deriva nula (movimiento browniano sin tendencia). Útil
              como referencia neutral que no apuesta ninguna dirección.
            - 'shrunk' (default): la media histórica encogida un 50% hacia la
              tasa libre de riesgo (SHRINKAGE_TOWARD_RISK_FREE), un estimador
              bayesiano simple que reduce la varianza de la estimación a
              costa de un sesgo hacia un ancla razonable.
        :param random_state: semilla del generador de números aleatorios
            (np.random.default_rng) usado en run_monte_carlo. Sin esto, dos
            corridas seguidas con los mismos inputs dan resultados distintos
            (el generador global de numpy no está fijado por config.random_state).
        """
        self.prices = price_series.dropna()
        self.returns = log_returns.dropna()
        self.current_price = float(self.prices.iloc[-1])
        # Días de negociación por año del activo (Fase 3.6). Antes era el
        # literal 252 repetido en cinco sitios de este fichero: la deriva
        # anualizada, la sigma anualizada, el dt del Monte Carlo, y la
        # volatilidad total y a la baja de Sharpe/Sortino. Para un activo que
        # cotiza 365 días al año el 252 anualiza de menos en un factor
        # sqrt(365/252) = 1,20, y ese error se propaga a TODO lo que proyecta
        # este motor.
        self.trading_days_per_year = int(trading_days_per_year)

        # DOS derivas, para DOS usos distintos que no deben confundirse:
        #
        # - self.historical_mu: la media histórica anualizada de log-retornos,
        #   SIN encoger. Es lo que mide el rendimiento que el activo REALMENTE
        #   tuvo, así que es la única base válida para un ratio que se reporta
        #   como "histórico" (Sharpe, Sortino) y que el lector va a comparar
        #   contra bandas estándar de la literatura.
        #
        # - self.mu: esa misma media ajustada según drift_mode, que es lo
        #   correcto para PROYECTAR (Monte Carlo, bandas lognormales): la media
        #   histórica tiene un error estándar enorme (~15% anual para una acción
        #   típica) y encogerla hacia rf reduce la varianza del estimador.
        #
        # Antes solo existía self.mu y calculate_risk_metrics la usaba para el
        # Sharpe. Con el default 'shrunk' el álgebra daba exactamente
        # 0.5*(mu_hist - rf)/sigma, o sea la MITAD del Sharpe real, y se
        # imprimía como "Sharpe Ratio (histórico)" (AUDITORIA_2026-09.md A10).
        self.historical_mu = float(self.returns.mean() * self.trading_days_per_year)
        self.drift_mode = drift_mode

        if drift_mode == "historical":
            self.mu = self.historical_mu
        elif drift_mode == "zero":
            self.mu = 0.0
        elif drift_mode == "shrunk":
            self.mu = (SHRINKAGE_TOWARD_RISK_FREE * risk_free_rate
                       + (1 - SHRINKAGE_TOWARD_RISK_FREE) * self.historical_mu)
        else:
            raise ValueError(f"drift_mode desconocido: {drift_mode!r} (usar 'historical', 'zero' o 'shrunk')")

        # Volatilidad anualizada de última generación (GARCH)
        self.sigma = float(predicted_volatility * np.sqrt(self.trading_days_per_year))
        self.rng = np.random.default_rng(random_state)

    def run_monte_carlo(self, horizon_days: int = 126, n_simulations: int = 5000, target_return: float = 0.10) -> Dict[str, Any]:
        """
        Simula trayectorias futuras del precio usando Movimiento Browniano Geométrico.
        Calcula distribuciones, intervalos de confianza y probabilidades de éxito/pérdida.

        self.mu es la media de LOG-retornos anualizada (ya en escala
        logarítmica, no aritmética), así que la solución de la SDE se aplica
        SIN la corrección de convexidad "-0.5*sigma^2": esa corrección solo
        hace falta para convertir una deriva ARITMÉTICA (E[dS/S]) en deriva
        logarítmica vía el lema de Itô. Restarla de una deriva que YA es
        logarítmica la resta dos veces y desplaza toda la distribución hacia
        abajo — verificado empíricamente: con la resta duplicada, la mediana
        simulada salía exp(-0.5*sigma^2*horizonte) por debajo de la correcta.
        """
        dt = 1 / self.trading_days_per_year  # Paso temporal de una sesión

        # Matriz de choques aleatorios gaussianos: [Días, Simulaciones]
        Z = self.rng.standard_normal((horizon_days, n_simulations))

        # Inicializar matriz de precios
        price_simulations = np.zeros((horizon_days + 1, n_simulations))
        price_simulations[0] = self.current_price

        # Simulación vectorial (sin bucles for lentos)
        for t in range(1, horizon_days + 1):
            price_simulations[t] = price_simulations[t-1] * np.exp(self.mu * dt + self.sigma * np.sqrt(dt) * Z[t-1])

        final_prices = price_simulations[-1]
        simulated_returns = (final_prices / self.current_price) - 1
        
        # Cálculos estadísticos sobre los resultados de la simulación
        prob_perdidda = float(np.mean(simulated_returns < 0))
        prob_objetivo = float(np.mean(simulated_returns >= target_return))
        
        intervalo_95 = np.percentile(final_prices, [2.5, 97.5])
        
        return {
            "final_prices": final_prices,
            "expected_mean_price": float(np.mean(final_prices)),
            "prob_loss": prob_perdidda,
            "prob_target_reached": prob_objetivo,
            "confidence_interval_95": (float(intervalo_95[0]), float(intervalo_95[1]))
        }

    def calculate_risk_metrics(self, confidence_level: float = 0.95, risk_free_rate: float = 0.04) -> Dict[str, float]:
        """
        Calcula métricas de riesgo de mercado: VaR, Expected Shortfall, Sharpe, Sortino y Max Drawdown.

        Sharpe y Sortino se calculan con la deriva HISTÓRICA sin encoger
        (self.historical_mu), NO con self.mu. Son métricas descriptivas de lo
        que el activo hizo, no proyecciones: encogerlas hacia rf no reduce
        ningún error de estimación, solo divide el ratio reportado. Con el
        default drift_mode='shrunk' (encogimiento del 50%) el Sharpe salía
        exactamente a la mitad del real y se imprimía como "histórico", donde
        el lector lo compara contra las bandas habituales (>1 bueno, >2 muy
        bueno) — ver AUDITORIA_2026-09.md A10. La deriva encogida sigue siendo
        la correcta para run_monte_carlo, que es un uso distinto.
        """
        # Volatilidad anualizada estándar
        vol_ann = float(self.returns.std() * np.sqrt(self.trading_days_per_year))
        
        # Value at Risk (VaR) Histórico diario
        var_pct = float(np.percentile(self.returns, (1 - confidence_level) * 100))
        var_monetario = self.current_price * (1 - np.exp(var_pct))
        
        # Expected Shortfall (ES) Histórico diario
        tail_losses = self.returns[self.returns <= var_pct]
        es_pct = float(tail_losses.mean()) if not tail_losses.empty else var_pct
        
        # Ratios de rendimiento: se usa la media histórica SIN encoger (ver el
        # docstring y __init__). self.mu queda reservada para proyectar.
        rf = risk_free_rate
        excess_return = self.historical_mu - rf
        sharpe = excess_return / vol_ann if vol_ann > 0 else 0.0
        
        # Sortino: semi-desviación respecto al MAR (0), calculada sobre TODA la muestra
        # (no solo sobre los retornos negativos, y no respecto a su propia media)
        mar = 0.0
        downside_sq = np.minimum(0.0, self.returns - mar) ** 2
        downside_vol_ann = float(np.sqrt(downside_sq.mean()) * np.sqrt(self.trading_days_per_year))
        sortino = excess_return / downside_vol_ann if downside_vol_ann > 0 else 0.0
        
        # Maximum Drawdown Histórico
        roll_max = self.prices.cummax()
        drawdown = (self.prices - roll_max) / roll_max
        max_dd = float(drawdown.min())
        
        return {
            "volatility_ann": vol_ann,
            "var_daily_95_pct": float(var_pct),
            "var_daily_95_monetary": var_monetario,
            "expected_shortfall_daily": float(es_pct),
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_dd
        }