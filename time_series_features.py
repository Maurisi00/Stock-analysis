import time
import numpy as np
import pandas as pd
import logging
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
import warnings
from statsmodels.tools.sm_exceptions import ValueWarning
warnings.filterwarnings("ignore", category=ValueWarning, module="statsmodels")
# El walk-forward reajusta ARIMA docenas/cientos de veces por corrida; sin este
# filtro, los avisos benignos de arranque del optimizador (parámetros AR/MA
# iniciales no estacionarios/invertibles, con fallback automático a ceros)
# inundarían el log en cada reajuste.
warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")

# Igual que arriba pero para el walk-forward GJR-GARCH: el optimizador de arch
# emite ConvergenceWarning ("Inequality constraints incompatible", "Positive
# directional derivative for linesearch") y RuntimeWarnings de división por
# cero/valor inválido (distribution.py/volatility.py) en cientos de reajustes
# por corrida de cartera — no son errores (hay fallback y el pipeline sigue),
# pero inundan la consola. Se filtra por módulo, no globalmente, para no
# ocultar RuntimeWarnings numéricos legítimos de numpy/pandas en el resto del
# pipeline. La no-convergencia en sí se sigue contando (ver
# _log_garch_convergence_summary) para no perder la señal diagnóstica.
try:
    from arch.utility.exceptions import ConvergenceWarning as ArchConvergenceWarning
    warnings.filterwarnings("ignore", category=ArchConvergenceWarning)
except ImportError:
    warnings.filterwarnings("ignore", message=".*[Ii]nequality constraints.*")
    warnings.filterwarnings("ignore", message=".*[Ll]inesearch.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="arch")

logger = logging.getLogger("QuantSystem.TimeSeriesEngine")

# Mínimo de observaciones históricas exigido antes de producir la primera
# estimación walk-forward (GARCH o ARIMA). Por debajo de este mínimo no hay
# historia suficiente para un ajuste confiable -> NaN (o fallback si el
# histórico total de la serie es menor a esto).
MIN_OBSERVATIONS_FOR_FIRST_ESTIMATE = 252

# Umbral (segundos) a partir del cual se loguea una advertencia de que el
# walk-forward completo (todos los reajustes periódicos de la serie) tardó
# más de lo esperado.
WALK_FORWARD_SLOW_WARNING_SECONDS = 60.0

# Proporción de reajustes GJR-GARCH no convergidos (scipy convergence_flag != 0)
# por encima de la cual el resumen de walk-forward se loguea como WARNING en
# vez de INFO -- por debajo de esto, la no-convergencia ocasional es normal
# (el optimizador la reintenta implícitamente en el siguiente reajuste) y no
# amerita elevar la severidad.
GARCH_NON_CONVERGENCE_WARNING_RATIO = 0.25


class TimeSeriesFeatureExtractor:
    """
    Ajusta modelos ARIMA y GARCH sobre los retornos pasados para extraer variables
    predictoras estructurales (Media y Varianza condicionales avanzadas).

    extract_garch_volatility y extract_arima_trend usan una ventana EXPANSIVA
    walk-forward (reajuste periódico cada refit_every días, propagando los
    parámetros vigentes entre reajustes) en vez de ajustar el modelo una única
    vez sobre toda la serie: un ajuste único estima omega/alpha/beta (o los
    coeficientes AR/MA) usando datos del FUTURO respecto a cada fecha pasada de
    la serie, y esas features alimentan el entrenamiento del motor de ML y el
    backtest — el mismo lookahead bias que el resto del pipeline evita
    explícitamente en todos los demás puntos (ver CLAUDE.md).
    """
    def __init__(self, log_returns: pd.DataFrame):
        self.log_returns = log_returns

    def extract_garch_volatility(self, ticker: str, refit_every: int = 21,
                                  window_size: int = 252, o: int = 1) -> pd.Series:
        """
        Volatilidad condicional GJR-GARCH(1,1) con distribución t de Student,
        estimada con ventana expansiva walk-forward: cada refit_every días se
        REAJUSTA el modelo (MLE completo) usando solo datos hasta esa fecha;
        entre reajustes, se propaga con los MISMOS parámetros — se re-filtra
        la recursión de varianza sobre los datos ya conocidos (arch_model.fix),
        sin reoptimizar — hasta el siguiente reajuste.

        GJR-GARCH (o=1, default) en vez de GARCH simétrico: las acciones tienen
        efecto apalancamiento (una caída sube la volatilidad futura más que una
        subida de igual magnitud), que un GARCH simétrico no captura y que
        distorsiona directamente el VaR y el Monte Carlo aguas abajo. La
        distribución t (en vez de normal) captura las colas gruesas típicas de
        retornos diarios. `o` es parametrizable (o=0 = GARCH simétrico simple)
        únicamente para que ValidationEngine pueda medir la sensibilidad del
        Score a esta decisión de especificación — el resto del pipeline usa
        siempre el default o=1.

        Exige un mínimo de window_size observaciones antes de la primera
        estimación walk-forward — NaN antes de eso (o el fallback rolling std
        si el histórico TOTAL de la serie ni siquiera alcanza ese mínimo).
        """
        returns = self.log_returns[ticker].dropna()
        n = len(returns)
        result = pd.Series(np.nan, index=returns.index)

        if n < MIN_OBSERVATIONS_FOR_FIRST_ESTIMATE:
            logger.warning(f"{ticker}: histórico insuficiente ({n} obs, mínimo "
                            f"{MIN_OBSERVATIONS_FOR_FIRST_ESTIMATE}) para arrancar el walk-forward "
                            f"GJR-GARCH; se usa el fallback de volatilidad rolling.")
            return returns.rolling(window=window_size).std()

        logger.info(f"Calculando GJR-GARCH(1,1)-t walk-forward (refit cada {refit_every} días) para {ticker}...")
        start_time = time.monotonic()

        try:
            position = MIN_OBSERVATIONS_FOR_FIRST_ESTIMATE - 1
            current_params = None
            n_refits = 0
            n_non_converged = 0

            while position < n:
                is_refit = current_params is None
                block_len = 1 if is_refit else min(refit_every - 1, n - position)
                block_end = position + block_len - 1

                sample = returns.iloc[:block_end + 1] * 100
                am = arch_model(sample, vol='GARCH', p=1, o=o, q=1, dist='t')

                if is_refit:
                    res = am.fit(disp='off')
                    n_refits += 1
                    if res.convergence_flag != 0:
                        n_non_converged += 1
                    current_params = res.params
                    cond_vol = res.conditional_volatility
                else:
                    # Mismos parámetros del último reajuste: re-filtra la
                    # recursión de varianza con los datos reales ya conocidos
                    # hasta block_end (walk-forward genuino, no una re-optimización).
                    fixed_res = am.fix(current_params)
                    cond_vol = fixed_res.conditional_volatility

                result.iloc[position:block_end + 1] = cond_vol.iloc[position:block_end + 1].to_numpy() / 100

                position = block_end + 1
                if not is_refit:
                    current_params = None  # el siguiente bloque vuelve a reajustar

        except Exception as e:
            logger.error(f"Error en el walk-forward GJR-GARCH para {ticker}: {str(e)}")
            return returns.rolling(window=window_size).std()

        elapsed = time.monotonic() - start_time
        if elapsed > WALK_FORWARD_SLOW_WARNING_SECONDS:
            logger.warning(f"{ticker}: el walk-forward GJR-GARCH tardó {elapsed:.1f}s "
                            f"(> {WALK_FORWARD_SLOW_WARNING_SECONDS:.0f}s) — considerar aumentar refit_every.")

        self._log_garch_convergence_summary(ticker, n_non_converged, n_refits)

        return result

    @staticmethod
    def _log_garch_convergence_summary(ticker: str, n_non_converged: int, n_refits: int) -> None:
        """
        Resume en UNA línea por ticker cuántos reajustes GJR-GARCH del
        walk-forward no convergieron (scipy convergence_flag != 0), en vez de
        dejar que cada ConvergenceWarning individual (silenciada más arriba)
        se pierda sin dejar rastro diagnóstico. Por encima de
        GARCH_NON_CONVERGENCE_WARNING_RATIO se eleva a WARNING: una fracción
        alta de reajustes fallidos sugiere que la volatilidad estimada para
        ese ticker puede no ser confiable.
        """
        if n_refits == 0:
            return
        ratio = n_non_converged / n_refits
        message = (f"{ticker}: {n_non_converged} de {n_refits} reajustes GJR-GARCH "
                    f"no convergieron ({ratio:.1%})")
        if ratio > GARCH_NON_CONVERGENCE_WARNING_RATIO:
            logger.warning(f"{message} — la volatilidad GARCH estimada para {ticker} "
                            f"puede ser poco fiable.")
        else:
            logger.info(message)

    def extract_garch_forecast(self, ticker: str, horizon_days: int, window_size: int = 252) -> float:
        """
        Volatilidad diaria ESPERADA (promedio) sobre los próximos horizon_days,
        proyectada con res.forecast(horizon=horizon_days) sobre un GJR-GARCH(1,1)-t
        ajustado con el 100% del histórico disponible.

        A diferencia de extract_garch_volatility, este método SÍ usa toda la
        muestra: es una proyección hacia el FUTURO desde hoy (la predicción "de
        hoy", no un feature histórico de entrenamiento), así que no hay
        lookahead que evitar.

        Reemplaza el uso ingenuo de garch_vol.iloc[-1] * sqrt(horizon_days) en
        main_pipeline.py: esa aproximación ignora la reversión a la media del
        GARCH — si hoy la volatilidad condicional está anormalmente alta o
        baja, sobre/subestima el horizonte completo en vez de reflejar hacia
        dónde se espera que revierta la varianza proyectada.
        """
        returns = self.log_returns[ticker].dropna()
        n = len(returns)

        if n < window_size:
            logger.warning(f"{ticker}: histórico insuficiente ({n} obs, mínimo {window_size}) para "
                            f"proyectar el forecast GARCH; se usa la volatilidad histórica (rolling std).")
            return self._historical_vol_fallback(returns, window_size)

        try:
            am = arch_model(returns * 100, vol='GARCH', p=1, o=1, q=1, dist='t')
            res = am.fit(disp='off')
            forecast = res.forecast(horizon=horizon_days, reindex=False)
            projected_variances = forecast.variance.iloc[-1].to_numpy()
            expected_daily_vol = np.sqrt(projected_variances.mean()) / 100
            return float(expected_daily_vol)
        except Exception as e:
            logger.error(f"Error proyectando el forecast GARCH para {ticker}: {str(e)}. "
                         f"Se usa la volatilidad histórica (rolling std) como fallback.")
            return self._historical_vol_fallback(returns, window_size)

    @staticmethod
    def _historical_vol_fallback(returns: pd.Series, window_size: int) -> float:
        fallback = returns.rolling(window=window_size).std().dropna()
        return float(fallback.iloc[-1]) if not fallback.empty else float(returns.std())

    def extract_arima_trend(self, ticker: str, refit_every: int = 21,
                             window_size: int = 252) -> pd.Series:
        """
        Componente autorregresivo ARIMA(1,0,1), estimado con ventana expansiva
        walk-forward: cada refit_every días se REAJUSTA el modelo (MLE
        completo) usando solo datos hasta esa fecha; entre reajustes, se
        propaga con los MISMOS parámetros extendiendo el resultado anterior
        con los datos ya conocidos (res.append(nuevos_datos, refit=False)),
        sin reoptimizar, hasta el siguiente reajuste.

        Exige un mínimo de window_size observaciones antes de la primera
        estimación walk-forward — NaN antes de eso (o el fallback de media
        móvil si el histórico TOTAL de la serie ni siquiera alcanza ese mínimo).
        """
        returns = self.log_returns[ticker].dropna()
        n = len(returns)
        result = pd.Series(np.nan, index=returns.index)

        if n < MIN_OBSERVATIONS_FOR_FIRST_ESTIMATE:
            logger.warning(f"{ticker}: histórico insuficiente ({n} obs, mínimo "
                            f"{MIN_OBSERVATIONS_FOR_FIRST_ESTIMATE}) para arrancar el walk-forward "
                            f"ARIMA; se usa el fallback de media móvil.")
            return returns.rolling(window=5).mean()

        logger.info(f"Calculando ARIMA(1,0,1) walk-forward (refit cada {refit_every} días) para {ticker}...")
        start_time = time.monotonic()

        # ARIMAResults.append() exige que el índice de los nuevos datos
        # "extienda" el índice del modelo original — algo que statsmodels solo
        # puede validar con una frecuencia explícita. Los precios reales de
        # mercado (feriados irregulares) casi nunca tienen un DatetimeIndex con
        # freq inferible (a diferencia de un pd.bdate_range sintético), lo que
        # hacía fallar el walk-forward completo en producción. Se trabaja
        # internamente con un RangeIndex posicional (0..n-1) y se reasignan los
        # resultados al índice real de returns solo al final, vía .iloc.
        returns_positional = returns.reset_index(drop=True)

        try:
            position = MIN_OBSERVATIONS_FOR_FIRST_ESTIMATE - 1
            current_res = None

            while position < n:
                is_refit = current_res is None
                block_len = 1 if is_refit else min(refit_every - 1, n - position)
                block_end = position + block_len - 1

                if is_refit:
                    model = ARIMA(returns_positional.iloc[:block_end + 1], order=(1, 0, 1))
                    current_res = model.fit()
                else:
                    # Mismos parámetros del último reajuste: extiende el
                    # resultado con las observaciones ya conocidas, sin
                    # reoptimizar (walk-forward genuino).
                    new_obs = returns_positional.iloc[position:block_end + 1]
                    current_res = current_res.append(new_obs, refit=False)

                fitted = current_res.fittedvalues
                result.iloc[position:block_end + 1] = fitted.iloc[position:block_end + 1].to_numpy()

                position = block_end + 1
                if not is_refit:
                    current_res = None  # el siguiente bloque vuelve a reajustar

        except Exception as e:
            logger.error(f"Error en el walk-forward ARIMA para {ticker}: {str(e)}")
            return returns.rolling(window=5).mean()

        elapsed = time.monotonic() - start_time
        if elapsed > WALK_FORWARD_SLOW_WARNING_SECONDS:
            logger.warning(f"{ticker}: el walk-forward ARIMA tardó {elapsed:.1f}s "
                            f"(> {WALK_FORWARD_SLOW_WARNING_SECONDS:.0f}s) — considerar aumentar refit_every.")

        return result
