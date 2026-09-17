import hashlib
import json
import logging
import os
import pickle
import threading
import time
from typing import Dict, List, Optional, Tuple
import datetime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import requests
import yfinance as yf

# Configuración del Logger institucional
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("QuantSystem.DataEngine")

# --- Congelado del snapshot de datos ----------------------------------------
# Constantes de calendario del mercado US (NYSE/Nasdaq). Describen una FORMA
# fija del mercado, no un parámetro operativo, así que viven acá y no en
# PipelineConfig — mismo criterio que el 252 de anualización (ver CLAUDE.md).
_MARKET_TIMEZONE = ZoneInfo("America/New_York")
_MARKET_CLOSE_HOUR = 16  # 16:00 ET


def last_closed_session(now: Optional[datetime.datetime] = None) -> datetime.date:
    """
    Devuelve la fecha de la última sesión bursátil CERRADA.

    Motivo (PLAN_REFACTOR.md 1.2 / AUDITORIA_2026-09.md C4): usar
    `datetime.date.today()` como end_date hace que, corriendo en horario de
    mercado, la última barra descargada sea un precio VIVO que cambia entre una
    corrida y la siguiente. Con modelos de árbol eso a veces no mueve nada y a
    veces voltea una hoja — es una de las causas de que
    cache/ml_prob_history.json tenga dos probabilidades distintas para JD en la
    misma fecha teniendo random_state fijado.

    Regla: se retrocede hasta el último día HÁBIL cuyo cierre (16:00 ET) ya
    pasó. Los festivos no se modelan (haría falta un calendario de mercado como
    dependencia): pedirle a yfinance un end_date festivo simplemente devuelve
    la sesión anterior, así que el efecto es el correcto — nunca se incluye una
    barra en formación, que es lo que importa para la reproducibilidad.

    `now` se inyecta en los tests; en producción se toma la hora actual en ET.
    """
    if now is None:
        now = datetime.datetime.now(tz=_MARKET_TIMEZONE)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=_MARKET_TIMEZONE)
    else:
        now = now.astimezone(_MARKET_TIMEZONE)

    candidate = now.date()
    # Si hoy es hábil pero el cierre todavía no pasó, la sesión de hoy está en
    # curso: la última CERRADA es anterior.
    if now.hour < _MARKET_CLOSE_HOUR:
        candidate -= datetime.timedelta(days=1)
    # weekday(): 5 = sábado, 6 = domingo.
    while candidate.weekday() >= 5:
        candidate -= datetime.timedelta(days=1)
    return candidate

# Rangos plausibles por columna fundamental normalizada, usados por
# _validate_fundamentals para anular (NaN) valores que un fallback mal
# mapeado de yfinance podría haber colado silenciosamente (ver fetch_fundamental_data).
#
# 'total_equity' NO lleva rango (Fase 1.5 / AUDITORIA_2026-09.md, sección
# MEDIO): el patrimonio neto NEGATIVO es un estado contable REAL y relevante
# —Boeing, McDonald's, Starbucks y Home Depot llevan años así por recompras
# agresivas financiadas con deuda— y anularlo ponía ROE, D/E y P/B a NaN
# justo en los casos donde más importan. La regla correcta no es censurar el
# dato de entrada, sino que cada RATIO devuelva NaN cuando no tenga sentido:
# eso lo hace ahora fundamental_engine._positivo_o_nan sobre el denominador.
_VALID_FUNDAMENTAL_RANGES = {
    "tax_rate": (0.0, 1.0),
    "total_revenue": (0.0, np.inf),
    "shares_outstanding": (0.0, np.inf),
}

# --- SEC EDGAR: acceso HTTP -------------------------------------------------
# La SEC exige un User-Agent identificable (organización + contacto) y pide no
# superar 10 req/s; sin cabecera, bloquea la petición con 403.
_SEC_USER_AGENT = "QuantSystem Research juanfiguelos@gmail.com"
_SEC_HEADERS = {"User-Agent": _SEC_USER_AGENT}
_SEC_MAX_REQUESTS_PER_SECOND = 10.0
_SEC_MIN_REQUEST_INTERVAL = 1.0 / _SEC_MAX_REQUESTS_PER_SECOND
_sec_rate_lock = threading.Lock()
_sec_last_request_monotonic = [0.0]

_CIK_MAP_CACHE_TTL_SECONDS = 7 * 24 * 3600
_COMPANYFACTS_CACHE_TTL_SECONDS = 24 * 3600
_SEC_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_SEC_COMPANYFACTS_URL_TEMPLATE = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

# --- SEC EDGAR: mapeo de conceptos US-GAAP al esquema plano de FundamentalFactorEngine ---
# Conceptos "duration" (partidas de resultados/flujo de caja: miran un período,
# necesitan 'start' y 'end'). Se filtran a duración ~trimestral (80-100 días)
# para descartar acumulados YTD/anuales — ver _extract_concept_quarterly_series.
# NOTA: las partidas del estado de flujos de caja (D&A, operating cash flow,
# capex, dividendos) suelen reportarse SOLO acumuladas (YTD) en los 10-Q de
# muchos emisores; el filtro estricto de duración las deja más dispersas que
# las partidas del estado de resultados (que sí suelen taggearse trimestre a
# trimestre). Limitación conocida, no una des-acumulación completa.
_EDGAR_DURATION_CONCEPTS: Dict[str, List[str]] = {
    "net_income": ["NetIncomeLoss", "ProfitLoss", "NetIncomeLossAvailableToCommonStockholdersBasic"],
    "ebit": ["OperatingIncomeLoss"],
    "total_revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                       "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueNet"],
    "depreciation_amortization": ["DepreciationDepletionAndAmortization",
                                   "DepreciationAmortizationAndAccretionNet", "DepreciationAndAmortization"],
    "interest_expense": ["InterestExpense", "InterestExpenseNonoperating", "InterestExpenseDebt"],
    "tax_expense": ["IncomeTaxExpenseBenefit"],
    "pretax_income": ["IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
                       "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments"],
    "operating_cash_flow": ["NetCashProvidedByUsedInOperatingActivities",
                             "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsForCapitalImprovements",
              "PaymentsToAcquireProductiveAssets"],
    "total_dividends": ["PaymentsOfDividendsCommonStock", "PaymentsOfDividends"],
}

# Conceptos "instant" (partidas de balance: una fecha, un valor — solo 'end').
_EDGAR_INSTANT_CONCEPTS: Dict[str, List[str]] = {
    "total_equity": ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "long_term_debt": ["LongTermDebtNoncurrent", "LongTermDebt"],
    "current_debt": ["LongTermDebtCurrent", "DebtCurrent", "ShortTermBorrowings"],
    "cash_and_equiv": ["CashAndCashEquivalentsAtCarryingValue",
                        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"],
    "shares_outstanding": ["CommonStockSharesOutstanding"],
}
# 'shares_outstanding' también se busca en la taxonomía 'dei' (cover page),
# donde muchos emisores solo reportan EntityCommonStockSharesOutstanding.
_EDGAR_SHARES_OUTSTANDING_DEI_TAGS = ["EntityCommonStockSharesOutstanding"]


def _sec_rate_limited_get(url: str, timeout: float = 30.0) -> requests.Response:
    """
    Espaciamos las peticiones a la SEC a un mínimo de 100ms entre sí (<=10 req/s),
    con un lock para que sea seguro incluso si en el futuro se llama desde
    varios hilos/tickers en paralelo.
    """
    with _sec_rate_lock:
        elapsed = time.monotonic() - _sec_last_request_monotonic[0]
        if elapsed < _SEC_MIN_REQUEST_INTERVAL:
            time.sleep(_SEC_MIN_REQUEST_INTERVAL - elapsed)
        response = requests.get(url, headers=_SEC_HEADERS, timeout=timeout)
        _sec_last_request_monotonic[0] = time.monotonic()
    return response


class DataEngine:
    """
    Engine encargado de la ingesta, limpieza y alineación temporal de datos de mercado
    y datos macroeconómicos, minimizando el lookahead bias.
    """
    def __init__(self, tickers: List[str], benchmark: str = "^GSPC", cache_dir: str = "cache",
                 trading_days_per_year: int = 252):
        # dict.fromkeys deduplica PRESERVANDO EL ORDEN de entrada. Antes esto
        # era list(set(tickers)), y el orden de iteración de un set de strings
        # cambia entre PROCESOS por la randomización de PYTHONHASHSEED
        # (verificado: list(set(['META','^GSPC'])) alterna entre corridas). Eso
        # reordena las columnas del panel de precios y, con ellas, el orden de
        # las sumas en punto flotante aguas abajo -> dos corridas idénticas del
        # mismo día podían diferir en los últimos decimales. Ningún test
        # in-process puede detectarlo, porque dentro de un proceso la semilla
        # de hash es fija.
        self.tickers = list(dict.fromkeys(tickers))
        self.benchmark = benchmark
        self.cache_dir = cache_dir
        # Fase 3.6: días de negociación por año del activo (252 acciones, 365
        # cripto). Lo consumen el cálculo del histórico EFECTIVO en años y el
        # horizonte de calculate_forward_returns.
        self.trading_days_per_year = int(trading_days_per_year)
        self.market_data: pd.DataFrame = pd.DataFrame()
        self.benchmark_data: pd.DataFrame = pd.DataFrame()
        # Histórico REALMENTE obtenido tras recortar al primer precio real (ver
        # _recortar_al_primer_precio_real). Puede ser mucho menor que
        # lookback_years si el ticker cotiza desde hace poco, y el informe tiene
        # que avisarlo: antes el bfill() lo enmascaraba rellenando el hueco.
        # None hasta que fetch_market_data corra.
        self.effective_history_years: Optional[float] = None
        self.effective_history_start: Optional[pd.Timestamp] = None

    def _price_cache_path(self, start_date: str, end_date: str) -> str:
        """
        Ruta del caché del panel de precios, con clave (tickers, start, end).

        La clave incluye los tickers EN ORDEN (self.tickers ya está deduplicado
        preservando el orden, ver __init__) más el benchmark, de forma que dos
        corridas del mismo día con la misma petición leen el MISMO fichero y no
        vuelven a pegarle a la red. Se hashea porque un ticker puede traer
        caracteres no válidos en un nombre de fichero (ej. '^GSPC', 'BRK-B').
        """
        clave = "|".join(self.tickers + [self.benchmark, start_date, end_date])
        digest = hashlib.sha256(clave.encode("utf-8")).hexdigest()[:16]
        return os.path.join(self.cache_dir, f"prices_{digest}.pkl")

    def _recortar_al_primer_precio_real(self, df_close: pd.DataFrame, start_date: str) -> pd.DataFrame:
        """
        Recorta el panel a partir de la primera fecha en la que TODAS las
        columnas (tickers + benchmark) tienen un precio real, y registra el
        histórico efectivo obtenido frente al solicitado.

        Al quitar el bfill() (ver fetch_market_data), un ticker que cotiza desde
        2020 dentro de una ventana que arranca en 2013 deja NaN en todas las
        filas previas. Esas filas no aportan nada —los factores técnicos las
        descartan igual— pero sí desplazan el burn-in de cada ventana móvil y
        hacen creer que hay 12 años de historia cuando hay 5. Recortar al
        primer dato real deja el panel diciendo la verdad sobre su propio
        tamaño, que es lo que después consume la advertencia de
        MIN_LOOKBACK_YEARS_FOR_INTERPRETABLE_BACKTEST en el informe.

        Se exige que TODAS las columnas tengan dato (no solo el ticker) porque
        el beta y el target relativo se calculan CONTRA el benchmark: una fila
        con precio del activo pero sin benchmark no es utilizable de todos modos.
        """
        if df_close.empty:
            return df_close

        # Primera fila sin ningún NaN en ninguna columna.
        filas_completas = df_close.dropna(how="any")
        if filas_completas.empty:
            logger.warning("Ninguna fecha tiene precio para todos los tickers y el benchmark a la vez; "
                            "se devuelve el panel sin recortar y los motores descartarán los NaN.")
            return df_close

        primera_real = filas_completas.index[0]
        recortado = df_close.loc[primera_real:]

        descartadas = len(df_close) - len(recortado)
        anios_efectivos = len(recortado) / float(self.trading_days_per_year)
        if descartadas > 0:
            # Se nombra el ticker que llegó más tarde, que es el que manda.
            primeros = {col: df_close[col].first_valid_index() for col in df_close.columns}
            limitante = max((c for c in primeros if primeros[c] is not None),
                            key=lambda c: primeros[c])
            logger.warning(
                f"Histórico efectivo recortado: se descartaron {descartadas} fechas previas a "
                f"{primera_real.date()} sin precio real (solicitado desde {start_date}). "
                f"Limita '{limitante}', cuyo primer precio es {primeros[limitante].date()}. "
                f"Histórico efectivo: {len(recortado)} sesiones (~{anios_efectivos:.1f} años)."
            )
        else:
            logger.info(f"Histórico efectivo: {len(recortado)} sesiones "
                        f"(~{anios_efectivos:.1f} años), sin recorte.")

        self.effective_history_years = anios_efectivos
        self.effective_history_start = primera_real
        return recortado

    def _persist_price_cache(self, df_close: pd.DataFrame, cache_path: str) -> None:
        """
        Persiste el panel de precios de forma ATÓMICA (.tmp + os.replace): una
        interrupción a mitad de escritura dejaría un pickle truncado que la
        siguiente corrida leería como caché válido. Un fallo al cachear no debe
        tumbar una corrida que ya tiene los datos, así que solo se loguea.
        """
        try:
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            tmp_path = cache_path + ".tmp"
            with open(tmp_path, "wb") as f:
                pickle.dump(df_close, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_path, cache_path)
        except Exception as e:
            logger.warning(f"No se pudo escribir el caché de precios en {cache_path} ({e}); se continúa sin cachear.")

    def fetch_market_data(self, start_date: str, end_date: str, use_cache: bool = True) -> pd.DataFrame:
        """
        Descarga precios de cierre ajustados para los tickers seleccionados.
        Versión robusta adaptada a los cambios de esquema estructural de yfinance.

        El panel resultante se persiste en cache_dir con clave
        (tickers, start_date, end_date) — ver _price_cache_path. Dos corridas
        del mismo día con la misma petición leen el mismo fichero, así que el
        informe es reproducible aunque yfinance devuelva algo distinto entre
        medio (revisión de precios ajustados, barra en formación, etc.).
        Se usa pickle y no CSV para garantizar round-trip EXACTO de los floats:
        una diferencia en el último decimal es suficiente para voltear una hoja
        de un árbol y cambiar la probabilidad ML.

        use_cache=False fuerza la descarga (útil para refrescar a mano).
        """
        all_tickers = self.tickers + [self.benchmark]
        cache_path = self._price_cache_path(start_date, end_date)

        if use_cache and os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    df_close = pickle.load(f)
                self.benchmark_data = df_close[[self.benchmark]].copy()
                self.market_data = df_close[self.tickers].copy()
                # El panel cacheado ya viene recortado al primer precio real,
                # así que el histórico efectivo se deriva de su tamaño (no se
                # vuelve a recortar, que sería idempotente pero redundante).
                if not df_close.empty:
                    self.effective_history_years = len(df_close) / float(self.trading_days_per_year)
                    self.effective_history_start = df_close.index[0]
                logger.info(f"Panel de precios leído del caché ({os.path.basename(cache_path)}): "
                            f"{len(self.market_data)} filas, {start_date} a {end_date}.")
                return self.market_data
            except Exception as e:
                logger.warning(f"No se pudo leer el caché de precios {cache_path} ({e}); se descarga de nuevo.")

        logger.info(f"Descargando datos de mercado desde {start_date} hasta {end_date}...")

        try:
            # Forzamos que yfinance mantenga siempre la estructura MultiIndex de columnas limpia
            raw_data = yf.download(all_tickers, start=start_date, end=end_date, group_by='ticker', progress=False)
            
            if raw_data.empty:
                raise ValueError("No se descargaron datos de yfinance. Verifica los tickers o la conexión.")
                
            adjusted_closes = {}
            
            # Si solo hay un ticker + benchmark, yfinance a veces aplana la estructura. 
            # Esta lógica normaliza CUALQUIER estructura que nos devuelva la API.
            for ticker in all_tickers:
                if isinstance(raw_data.columns, pd.MultiIndex):
                    if ticker in raw_data.columns.levels[0]:
                        # Si tiene 'Adj Close' lo toma, si no, cae defensivamente en 'Close'
                        if 'Adj Close' in raw_data[ticker].columns:
                            adjusted_closes[ticker] = raw_data[ticker]['Adj Close']
                        else:
                            adjusted_closes[ticker] = raw_data[ticker]['Close']
                else:
                    # Estructura aplanada de emergencia
                    if ticker in raw_data.columns:
                        adjusted_closes[ticker] = raw_data[ticker]
            
            df_close = pd.DataFrame(adjusted_closes)

            # SOLO ffill (arrastra el último cierre conocido sobre un hueco:
            # festivo local, suspensión de cotización). NUNCA bfill.
            #
            # bfill() replicaba el PRIMER precio conocido hacia ATRÁS sobre todo
            # el período previo a la cotización del ticker: una acción que salió
            # a bolsa en 2020 con lookback_years=12 recibía su precio de IPO
            # copiado hasta 2013, produciendo retornos exactamente 0 durante
            # años -> volatilidad artificialmente baja, momentum 0, beta 0, y
            # miles de filas de entrenamiento inventadas que el pipeline no
            # podía distinguir de datos reales. Es lookahead puro: información
            # del precio de 2020 metida en fechas de 2013.
            df_close = df_close.ffill()
            df_close = self._recortar_al_primer_precio_real(df_close, start_date)

            self._persist_price_cache(df_close, cache_path)

            # Separar Benchmark de los activos bajo análisis
            self.benchmark_data = df_close[[self.benchmark]].copy()
            self.market_data = df_close[self.tickers].copy()

            logger.info(f"Datos descargados con éxito. Filas de tiempo procesadas: {len(self.market_data)}")
            return self.market_data
            
        except Exception as e:
            logger.error(f"Error crítico en la descarga de datos: {str(e)}")
            raise e

    def fetch_fundamental_data(self, ticker: str, source: str = "edgar") -> pd.DataFrame:
        """
        Descarga los estados financieros trimestrales y los normaliza a las columnas
        que espera FundamentalFactorEngine. Dos fuentes disponibles:

        - 'edgar' (default): companyfacts de SEC EDGAR, indexado por fecha de
          PUBLICACIÓN (point-in-time) — típicamente 40-80 trimestres limpios,
          con manejo de restatements. Ver _fetch_fundamental_data_edgar.
        - 'yfinance': ~5 trimestres, sin restatements ni point-in-time. Es el
          fallback automático si 'edgar' falla o no devuelve datos utilizables
          por cualquier motivo (ticker sin CIK mapeado, EDGAR caído, etc.), y
          también puede pedirse explícitamente.

        A diferencia de fetch_market_data, los fallos aquí NO se relanzan: los factores
        fundamentales son solo 2 de 5 inputs del score, y el resto del informe (técnico,
        ML, riesgo, backtest) no depende de ellos. Ante cualquier fallo de descarga o
        ausencia de datos, se devuelve un DataFrame vacío para que el pipeline siga
        corriendo; FundamentalFactorEngine detecta ese caso y marca los factores
        resultantes como no confiables en vez de fallar la ejecución completa.
        """
        if source not in ("edgar", "yfinance"):
            raise ValueError(f"fundamentals_source desconocido: {source!r} (usar 'edgar' o 'yfinance')")

        if source == "edgar":
            try:
                data = self._fetch_fundamental_data_edgar(ticker)
            except Exception as e:
                logger.error(f"Error obteniendo datos fundamentales de SEC EDGAR para {ticker}: {e}. "
                             f"Se usará el fallback yfinance.")
                data = pd.DataFrame()

            if not data.empty:
                return data

            logger.warning(f"SEC EDGAR no devolvió datos fundamentales utilizables para {ticker}; "
                            f"usando fallback yfinance.")

        return self._fetch_fundamental_data_yfinance(ticker)

    def _fetch_fundamental_data_yfinance(self, ticker: str) -> pd.DataFrame:
        """
        Descarga los estados financieros trimestrales reales desde yfinance y los
        normaliza a las columnas que espera FundamentalFactorEngine. Solo expone
        ~5 trimestres (yfinance no da más histórico), sin point-in-time ni
        restatements — ver _fetch_fundamental_data_edgar para la fuente preferida.
        """
        logger.info(f"Descargando datos fundamentales trimestrales para {ticker}...")
        try:
            tk = yf.Ticker(ticker)
            fin = tk.quarterly_financials
            bs = tk.quarterly_balance_sheet
            cf = tk.quarterly_cashflow
        except Exception as e:
            logger.error(f"Error descargando datos fundamentales para {ticker}: {e}. "
                         f"Se devolverán datos vacíos (factores de Calidad/Valoración quedarán "
                         f"marcados como no confiables).")
            return pd.DataFrame()

        if (fin is None or fin.empty) and (bs is None or bs.empty):
            logger.warning(f"yfinance no devolvió estados financieros para {ticker}. "
                            f"Se devolverán datos vacíos.")
            return pd.DataFrame()

        # Nueva función _row más robusta que acepta una lista de posibles nombres
        def _row(df: pd.DataFrame, possible_labels: list, columns) -> pd.Series:
            if df is None or df.empty:
                return pd.Series(np.nan, index=columns)
            for label in possible_labels:
                if label in df.index:
                    return df.loc[label]
            return pd.Series(np.nan, index=columns)

        cols = fin.columns if fin is not None and not fin.empty else bs.columns
        data = pd.DataFrame(index=cols)

        # Buscamos con los nombres reales de la API (con espacios y variantes)
        data['net_income'] = _row(fin, ['Net Income', 'Net Income Common Stockholders'], cols)
        data['ebit'] = _row(fin, ['EBIT', 'Operating Income'], cols)
        data['total_revenue'] = _row(fin, ['Total Revenue', 'Operating Revenue'], cols)
        data['ebitda'] = _row(fin, ['EBITDA', 'Normalized EBITDA'], cols)
        data['interest_expense'] = _row(fin, ['Interest Expense', 'Interest Expense Non Operating'], cols).abs()

        # 'Tax Provision' es un IMPORTE en dólares, no una tasa — usarlo como
        # fallback directo de tax_rate corrompía el ROIC (nopat = ebit * (1 - tax_rate)
        # con tax_rate en cientos de millones). Si falta 'Tax Rate For Calcs', se
        # calcula la tasa EFECTIVA como Tax Provision / Pretax Income y se valida
        # que sea plausible antes de confiar en ella.
        tax_rate_direct = _row(fin, ['Tax Rate For Calcs'], cols)
        tax_provision = _row(fin, ['Tax Provision'], cols)
        pretax_income = _row(fin, ['Pretax Income'], cols)

        with np.errstate(divide='ignore', invalid='ignore'):
            effective_tax_rate = tax_provision / pretax_income
        effective_tax_rate = effective_tax_rate.replace([np.inf, -np.inf], np.nan)

        implausible = effective_tax_rate.notna() & ~effective_tax_rate.between(0.0, 0.60)
        if implausible.any():
            logger.warning(f"{ticker}: tasa efectiva de impuestos calculada (Tax Provision / Pretax "
                            f"Income) fuera de [0, 0.60] en {int(implausible.sum())} trimestre(s); "
                            f"se usará el default 0.21 para esos casos.")
            effective_tax_rate = effective_tax_rate.where(~implausible, np.nan)

        data['tax_rate'] = tax_rate_direct.fillna(effective_tax_rate).fillna(0.21)

        data['total_equity'] = _row(bs, ['Stockholders Equity', 'Total Equity Gross Minority Interest', 'Common Stock Equity'], cols)
        # 'Total Liabilities Net Minority Interest' incluye pasivos comerciales y
        # deferred revenue, no solo deuda financiera — infla D/E, invested capital
        # y enterprise value. El fallback debe ser deuda financiera real (o NaN).
        long_term_debt = _row(bs, ['Long Term Debt'], cols)
        current_debt = _row(bs, ['Current Debt'], cols)
        total_debt_direct = _row(bs, ['Total Debt'], cols)
        total_debt_fallback = long_term_debt.add(current_debt, fill_value=0.0)
        # add(fill_value=0) solo sustituye NaN cuando el OTRO lado tiene dato; si
        # ambos faltan para un trimestre, el resultado queda en NaN (no en 0).
        data['total_debt'] = total_debt_direct.fillna(total_debt_fallback)
        data['cash_and_equiv'] = _row(bs, ['Cash And Cash Equivalents', 'Cash Cash Equivalents And Short Term Investments'], cols)

        shares = _row(bs, ['Ordinary Shares Number', 'Share Issued'], cols)
        if shares.isna().all():
            # Aislado en su propio try/except: un fallo de este endpoint no debe
            # descartar los datos de fin/bs/cf que sí se descargaron correctamente.
            try:
                shares_outstanding = tk.info.get('sharesOutstanding', np.nan)
            except Exception as e:
                logger.warning(f"No se pudo obtener sharesOutstanding para {ticker}: {e}")
                shares_outstanding = np.nan
            shares = pd.Series(shares_outstanding, index=cols)
        data['shares_outstanding'] = shares

        data['free_cash_flow'] = _row(cf, ['Free Cash Flow'], cols)
        data['total_dividends'] = _row(cf, ['Cash Dividends Paid', 'Common Stock Dividend Paid'], cols).abs()

        data = self._validate_fundamentals(data, ticker=ticker)
        return data.sort_index()

    def _validate_fundamentals(self, data: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Última línea de defensa contra fallbacks de yfinance mal mapeados (ej. una
        tasa de impuestos que en realidad era un importe en dólares): valida rangos
        plausibles por columna sobre los datos ya normalizados y anula (NaN) —
        nunca deja pasar — cualquier valor infinito o fuera de rango, logueando
        cuántos trimestres se vieron afectados por columna.
        """
        data = data.copy()
        for column in data.columns:
            values = data[column].to_numpy(dtype="float64")
            inf_mask = np.isinf(values)
            if inf_mask.any():
                logger.warning(f"{ticker}: {int(inf_mask.sum())} valor(es) infinito(s) en '{column}'; "
                                f"se anulan (NaN).")
                data.loc[inf_mask, column] = np.nan

            if column in _VALID_FUNDAMENTAL_RANGES:
                lo, hi = _VALID_FUNDAMENTAL_RANGES[column]
                series = data[column]
                out_of_range = series.notna() & ~series.between(lo, hi)
                if out_of_range.any():
                    logger.warning(f"{ticker}: {int(out_of_range.sum())} valor(es) de '{column}' fuera "
                                    f"del rango plausible [{lo}, {hi}]; se anulan (NaN) para evitar "
                                    f"contaminar los factores derivados.")
                    data.loc[out_of_range, column] = np.nan

        return data

    # --- SEC EDGAR: fuente preferida de fundamentales (point-in-time, 40-80 trimestres) ---

    def _cached_get_json(self, url: str, cache_path: str, ttl_seconds: float) -> dict:
        """
        Descarga JSON con cache en disco: si el archivo cacheado existe y no ha
        expirado (ttl_seconds), lo reutiliza sin pegarle a la red. Si la caché
        está corrupta se re-descarga. Cualquier fallo de red se propaga (el
        caller decide el fallback).
        """
        if os.path.exists(cache_path):
            age = time.time() - os.path.getmtime(cache_path)
            if age < ttl_seconds:
                try:
                    with open(cache_path, "r", encoding="utf-8") as f:
                        return json.load(f)
                except (json.JSONDecodeError, OSError) as e:
                    logger.warning(f"Caché SEC corrupta en {cache_path}, se re-descargará: {e}")

        response = _sec_rate_limited_get(url)
        response.raise_for_status()
        payload = response.json()

        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

        return payload

    def _resolve_cik(self, ticker: str) -> Optional[str]:
        """
        Traduce ticker -> CIK de 10 dígitos usando el mapeo público de la SEC
        (cacheado en disco). Devuelve None si el ticker no aparece en el mapeo.
        """
        cache_path = os.path.join(self.cache_dir, "company_tickers.json")
        payload = self._cached_get_json(_SEC_COMPANY_TICKERS_URL, cache_path, _CIK_MAP_CACHE_TTL_SECONDS)

        ticker_upper = ticker.upper()
        for entry in payload.values():
            if str(entry.get("ticker", "")).upper() == ticker_upper:
                return str(entry["cik_str"]).zfill(10)
        return None

    def _fetch_companyfacts(self, cik: str) -> dict:
        """Descarga (con cache en disco) el JSON completo de companyfacts para un CIK."""
        cache_path = os.path.join(self.cache_dir, f"companyfacts_CIK{cik}.json")
        url = _SEC_COMPANYFACTS_URL_TEMPLATE.format(cik=cik)
        return self._cached_get_json(url, cache_path, _COMPANYFACTS_CACHE_TTL_SECONDS)

    # Bandas de duración (días) usadas para clasificar hechos XBRL "duration".
    _QUARTER_DURATION_DAYS = (80, 100)
    _NINE_MONTH_DURATION_DAYS = (250, 300)
    _ANNUAL_DURATION_DAYS = (350, 380)

    @staticmethod
    def _extract_concept_quarterly_series(
        facts: dict, tags: List[str], is_duration: bool, taxonomies: Tuple[str, ...] = ("us-gaap",),
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Junta los hechos XBRL de TODOS los tags alternativos de un concepto en
        una única serie por fecha de FIN de período contable.

        Fusionar tags alternativos en vez de usar solo el primero con datos es
        necesario porque muchas empresas migran de tag con el tiempo (ej.
        'Revenues' -> 'RevenueFromContractWithCustomerExcludingAssessedTax' tras
        adoptar ASC 606): quedarse con uno solo trunca el histórico exactamente
        en el punto que se busca extender.

        Para conceptos 'duration' (resultados/flujo de caja), la mayoría de los
        emisores SÍ taggean el trimestre discreto (~80-100 días) para las
        partidas de resultados en sus 10-Q — pero el 10-K anual normalmente NO
        taggea un "Q4 discreto" por separado, solo la cifra ANUAL (~350-380
        días). Para no perder sistemáticamente uno de cada cuatro trimestres,
        el Q4 se DERIVA como anual - acumulado de 9 meses (~250-300 días) del
        mismo año fiscal, cuando ambos hechos están disponibles. Las partidas
        del estado de flujos de caja a veces solo se taggean acumuladas
        (YTD) en trimestres intermedios (Q2/Q3); esos casos quedan más
        dispersos — limitación conocida, no se implementa una des-acumulación
        completa de Q2/Q3.

        Devuelve (valores, fechas_de_publicación) indexadas por fin de período.
        Ante restatements O migraciones de tag que solapan un mismo período, se
        conserva el hecho con la fecha de publicación ('filed') más temprana —
        la versión que el mercado conocía en su momento, no la corregida después.
        """
        raw_rows = []
        for taxonomy in taxonomies:
            taxonomy_facts = facts.get("facts", {}).get(taxonomy, {})
            for tag in tags:
                concept = taxonomy_facts.get(tag)
                if not concept:
                    continue
                units = concept.get("units", {})
                unit_facts = units.get("USD") or units.get("shares") or units.get("USD/shares") or []
                for fact in unit_facts:
                    if "end" not in fact or "val" not in fact or "filed" not in fact:
                        continue
                    end = pd.Timestamp(fact["end"])
                    filed = pd.Timestamp(fact["filed"])
                    if is_duration:
                        if "start" not in fact:
                            continue
                        start = pd.Timestamp(fact["start"])
                        raw_rows.append({"start": start, "end": end, "val": fact["val"], "filed": filed})
                    else:
                        raw_rows.append({"end": end, "val": fact["val"], "filed": filed})

        if not raw_rows:
            return pd.Series(dtype=float), pd.Series(dtype="datetime64[ns]")

        raw_df = pd.DataFrame(raw_rows)

        if not is_duration:
            df = raw_df.sort_values("filed").drop_duplicates(subset="end", keep="first")
            df = df.set_index("end").sort_index()
            return df["val"], df["filed"]

        duration_days = (raw_df["end"] - raw_df["start"]).dt.days
        lo_q, hi_q = DataEngine._QUARTER_DURATION_DAYS
        lo_9m, hi_9m = DataEngine._NINE_MONTH_DURATION_DAYS
        lo_fy, hi_fy = DataEngine._ANNUAL_DURATION_DAYS

        quarterly = raw_df[duration_days.between(lo_q, hi_q)]
        nine_month = raw_df[duration_days.between(lo_9m, hi_9m)]
        annual = raw_df[duration_days.between(lo_fy, hi_fy)]

        derived_rows = []
        for _, fy_row in annual.iterrows():
            candidates = nine_month[nine_month["start"] == fy_row["start"]]
            if candidates.empty:
                continue
            # El checkpoint YTD más cercano al cierre de año fiscal.
            closest_idx = (fy_row["end"] - candidates["end"]).idxmin()
            ytd_row = candidates.loc[closest_idx]
            derived_rows.append({
                "end": fy_row["end"],
                "val": fy_row["val"] - ytd_row["val"],
                "filed": max(fy_row["filed"], ytd_row["filed"]),
            })

        combined = quarterly[["end", "val", "filed"]]
        if derived_rows:
            combined = pd.concat([combined, pd.DataFrame(derived_rows)], ignore_index=True)

        if combined.empty:
            return pd.Series(dtype=float), pd.Series(dtype="datetime64[ns]")

        df = combined.sort_values("filed").drop_duplicates(subset="end", keep="first")
        df = df.set_index("end").sort_index()
        return df["val"], df["filed"]

    def _fetch_fundamental_data_edgar(self, ticker: str) -> pd.DataFrame:
        """
        Descarga companyfacts de SEC EDGAR (gratis, sin API key) y lo normaliza
        al mismo esquema plano que espera FundamentalFactorEngine, pero con
        muchísimo más histórico (típicamente 40-80 trimestres vs. los ~5 de
        yfinance) y con dos garantías que yfinance no ofrece:

        - Point-in-time: el DataFrame resultante se indexa por la fecha de
          PUBLICACIÓN del trimestre (el 'filed' más tardío entre los conceptos
          que lo componen), no por el fin de período contable. Un balance de
          Q4 no era público hasta el 10-K de febrero/marzo siguiente; indexarlo
          por fin de trimestre haría que FundamentalFactorEngine lo tratara
          como disponible meses antes de que existiera — lookahead bias.
        - Restatements: ante varios valores publicados para el mismo período
          (una cifra corregida en un filing posterior), se conserva el primero
          publicado — lo que el mercado sabía en su momento.
        """
        cik = self._resolve_cik(ticker)
        if cik is None:
            logger.warning(f"No se encontró CIK de SEC EDGAR para el ticker {ticker}.")
            return pd.DataFrame()

        facts = self._fetch_companyfacts(cik)

        concept_values: Dict[str, pd.Series] = {}
        concept_filed: Dict[str, pd.Series] = {}

        for name, tags in _EDGAR_DURATION_CONCEPTS.items():
            concept_values[name], concept_filed[name] = self._extract_concept_quarterly_series(
                facts, tags, is_duration=True)

        for name, tags in _EDGAR_INSTANT_CONCEPTS.items():
            if name == "shares_outstanding":
                # Muchos emisores solo taggean las acciones en circulación en la
                # portada del filing (taxonomía 'dei'), no en us-gaap.
                combined_tags = tags + _EDGAR_SHARES_OUTSTANDING_DEI_TAGS
                taxonomies = ("us-gaap", "dei")
            else:
                combined_tags, taxonomies = tags, ("us-gaap",)
            concept_values[name], concept_filed[name] = self._extract_concept_quarterly_series(
                facts, combined_tags, is_duration=False, taxonomies=taxonomies)

        if all(series.empty for series in concept_values.values()):
            logger.warning(f"SEC EDGAR (CIK {cik}) no devolvió ningún concepto US-GAAP reconocido para {ticker}.")
            return pd.DataFrame()

        values_df = pd.DataFrame(concept_values)
        filed_df = pd.DataFrame(concept_filed).reindex(values_df.index)

        # Point-in-time: el trimestre completo (todas sus columnas) recién se
        # conoce cuando se publicó el ÚLTIMO de sus componentes disponibles.
        point_in_time = filed_df.max(axis=1)
        values_df = values_df[point_in_time.notna()]
        point_in_time = point_in_time[point_in_time.notna()]

        values_df = values_df.set_index(point_in_time).sort_index()
        values_df = values_df[~values_df.index.duplicated(keep="first")]

        result = pd.DataFrame(index=values_df.index)
        result["net_income"] = values_df["net_income"]
        result["ebit"] = values_df["ebit"]
        result["total_revenue"] = values_df["total_revenue"]
        result["ebitda"] = values_df["ebit"] + values_df["depreciation_amortization"]
        result["interest_expense"] = values_df["interest_expense"].abs()

        # Misma lógica defensiva que _fetch_fundamental_data_yfinance: la tasa
        # de impuestos es un ratio calculado, validado antes de usarse.
        with np.errstate(divide="ignore", invalid="ignore"):
            effective_tax_rate = values_df["tax_expense"] / values_df["pretax_income"]
        effective_tax_rate = effective_tax_rate.replace([np.inf, -np.inf], np.nan)
        implausible = effective_tax_rate.notna() & ~effective_tax_rate.between(0.0, 0.60)
        if implausible.any():
            logger.warning(f"{ticker}: tasa efectiva de impuestos (SEC EDGAR) fuera de [0, 0.60] en "
                            f"{int(implausible.sum())} trimestre(s); se usará el default 0.21 para esos casos.")
            effective_tax_rate = effective_tax_rate.where(~implausible, np.nan)
        result["tax_rate"] = effective_tax_rate.fillna(0.21)

        result["total_equity"] = values_df["total_equity"]
        # Nunca Total Liabilities como fallback de deuda (ver _fetch_fundamental_data_yfinance):
        # deuda financiera real = deuda de largo plazo + porción corriente.
        result["total_debt"] = values_df["long_term_debt"].add(values_df["current_debt"], fill_value=0.0)
        result["cash_and_equiv"] = values_df["cash_and_equiv"]
        result["shares_outstanding"] = values_df["shares_outstanding"]
        result["free_cash_flow"] = values_df["operating_cash_flow"] - values_df["capex"]
        result["total_dividends"] = values_df["total_dividends"].abs()

        # Conceptos como 'shares_outstanding' (taxonomía 'dei') aparecen en la
        # portada de MUCHOS más filings que los que traen estados financieros
        # completos (8-Ks, proxies, etc.), generando filas "huecas" en el índice
        # que no aportan ninguna señal fundamental real. Sin este filtro, una de
        # esas filas podría terminar siendo la última del histórico y dejar el
        # score de Calidad/Valoración huérfano de datos recientes reales.
        core_columns = ["net_income", "ebit", "total_revenue", "total_equity"]
        has_substance = result[core_columns].notna().any(axis=1)
        result = result[has_substance]

        result = self._validate_fundamentals(result, ticker=ticker)
        return result.sort_index()

    def calculate_forward_returns(self, horizon_months: int = 6) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Calcula retornos a futuro y genera la variable objetivo cross-sectional (Y).
        Evita lookahead bias al no usar estos datos en las fases de cálculo de factores.
        
        Formula del retorno compuesto continuo: R = ln(P_t_plus_h / P_t)
        """
        if self.market_data.empty or self.benchmark_data.empty:
            raise ValueError("Datos de mercado no inicializados. Ejecuta fetch_market_data primero.")
            
        # Horizonte en SESIONES. Fase 3.6: se deriva de
        # trading_days_per_year (252/12 = 21 en acciones, 365/12 = 30,4 en
        # cripto) en vez del literal 21, que daba "6 meses" = 126 sesiones
        # también para un activo que cotiza los 365 días — o sea 4 meses de
        # calendario, no 6.
        trading_days = int(round(horizon_months * self.trading_days_per_year / 12))
        
        # Retornos logarítmicos futuros de los activos
        asset_future_returns = np.log(self.market_data.shift(-trading_days) / self.market_data)
        # Retornos logarítmicos futuros del benchmark
        benchmark_future_returns = np.log(self.benchmark_data.shift(-trading_days) / self.benchmark_data)
        
        # Variable binaria objetivo: 1 si supera al benchmark, 0 si no
        binary_target = pd.DataFrame(index=self.market_data.index)
        for ticker in self.tickers:
            binary_target[ticker] = (asset_future_returns[ticker] > benchmark_future_returns[self.benchmark]).astype(int)
            # Asignar NaN a las últimas filas donde no podemos conocer el futuro real
            binary_target.loc[self.market_data.index[-trading_days]:, ticker] = np.nan
            
        return asset_future_returns, binary_target