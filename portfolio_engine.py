import datetime
import json
import logging
import os
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd

from config import PipelineConfig
from data_engine import DataEngine, last_closed_session
from technical_engine import TechnicalFactorEngine
from time_series_features import TimeSeriesFeatureExtractor
from fundamental_engine import FundamentalFactorEngine, MIN_QUARTERS_FOR_RELIABILITY
from entry_context_engine import EntryContextEngine, percentil_historico

# FASE 3.3 — este módulo ya NO importa MachineLearningEngine. El modo cartera
# dejó de usar la probabilidad ML para cualquier cosa: la señal que la
# consumía (MANTENER/VIGILAR/VENDER) se eliminó porque la Fase 2 midió su
# poder predictivo y es cero. ml_engine.py y sus tests se conservan intactos;
# lo que desapareció es su uso en la ruta de decisión.

# Orden de preferencia del múltiplo con el que se reporta el percentil de
# valoración de una posición. Se usa el PRIMERO que tenga dato, y se dice cuál
# fue (`valuation_multiple_used`): no se promedian los percentiles de varios
# múltiplos, porque un promedio de percentiles vuelve a ser un número
# compuesto que esconde de qué salió — exactamente lo que la Fase 3 elimina.
# El orden va del múltiplo más usado al más tolerante a denominadores
# degenerados: el PER es NaN en una empresa en pérdidas (ver _positivo_o_nan),
# el EV/EBITDA en una con EBITDA negativo, y P/S casi nunca lo es.
_VALUATION_MULTIPLE_PREFERENCE = (
    ("per", "PER"),
    ("ev_to_ebitda", "EV/EB"),
    ("price_to_sales", "P/S"),
    ("price_to_book", "P/B"),
)

# Configuración del Logger institucional (mismo patrón que data_engine.py)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("QuantSystem.PortfolioEngine")

# Mapeo de las columnas REALES del export Flex Query de IBKR (nombres exactos
# de la cabecera) a snake_case. 'AssetClass' se mapea porque se usa para
# filtrar, pero no forma parte de las columnas de salida.
_COLUMN_MAP = {
    "CurrencyPrimary": "currency",
    "AssetClass": "asset_class",
    "Symbol": "symbol",
    "Description": "description",
    "Quantity": "quantity",
    "PositionValue": "position_value",
    "CostBasisMoney": "cost_basis_money",
    "PercentOfNAV": "percent_of_nav",
    "FifoPnlUnrealized": "unrealized_pnl",
    "OpenDateTime": "open_date",
}

# Único campo estrictamente obligatorio: sin símbolo no hay posición que
# identificar. Todo lo demás se tolera ausente (se rellena con NaN).
_REQUIRED_COLUMN = "Symbol"

_NUMERIC_COLUMNS = ["quantity", "position_value", "cost_basis_money", "percent_of_nav", "unrealized_pnl"]

_OUTPUT_COLUMNS = [
    "symbol", "description", "quantity", "position_value", "cost_basis_money",
    "cost_basis_price", "percent_of_nav", "unrealized_pnl", "open_date", "currency",
]


def _read_raw(path: str) -> pd.DataFrame:
    """
    El export de IBKR Flex Query viene separado por TABULADORES pese a
    llamarse .csv. Probamos '\\t' primero (formato real confirmado); si el
    resultado da una sola columna (el separador probado no era el correcto),
    reintentamos con ','.
    """
    df = pd.read_csv(path, sep="\t", dtype=str)
    if df.shape[1] <= 1:
        df = pd.read_csv(path, sep=",", dtype=str)
    return df


def load_positions(path: str) -> pd.DataFrame:
    """
    Lee el export de posiciones abiertas de IBKR (Flex Query) y lo normaliza
    a un esquema plano en snake_case: symbol, description, quantity,
    position_value, cost_basis_money, cost_basis_price, percent_of_nav,
    unrealized_pnl, open_date, currency.

    Dos particularidades confirmadas del export real que este parser maneja
    explícitamente:
    - No existe columna CostBasisPrice: el precio medio se DERIVA como
      cost_basis_money / quantity, protegido contra Quantity 0/NaN.
    - OpenDateTime viene completamente vacía en el export real (aunque la
      columna exista) — se parsea tolerantemente (yyyyMMdd o
      yyyyMMdd;HHmmss) con errors='coerce', dejando NaT sin loguear ruido
      si la columna entera está vacía.

    Solo 'Symbol' es estrictamente obligatorio (ValueError si falta). Toda
    otra columna esperada que falte se rellena con NaN y se loguea qué se
    pierde con eso.
    """
    raw = _read_raw(path)

    if _REQUIRED_COLUMN not in raw.columns:
        raise ValueError(
            f"El export de posiciones no tiene la columna obligatoria '{_REQUIRED_COLUMN}' "
            f"(columnas encontradas: {list(raw.columns)})."
        )

    missing_source_columns = [c for c in _COLUMN_MAP if c != _REQUIRED_COLUMN and c not in raw.columns]
    if missing_source_columns:
        lost_fields = sorted(_COLUMN_MAP[c] for c in missing_source_columns)
        logger.warning(
            f"Columnas ausentes en el export de posiciones: {missing_source_columns} -> "
            f"quedan en NaN {lost_fields}. Sin Quantity/CostBasisMoney no hay precio medio "
            f"ni P&L derivable, solo símbolo y veredicto."
        )

    df = pd.DataFrame(index=raw.index)
    for source_col, target_col in _COLUMN_MAP.items():
        df[target_col] = raw[source_col] if source_col in raw.columns else np.nan

    # --- Filtrado por AssetClass == 'STK' ---
    if "AssetClass" in raw.columns:
        is_stock = df["asset_class"].astype(str).str.strip() == "STK"
        n_dropped = int((~is_stock).sum())
        if n_dropped:
            dropped_reasons = df.loc[~is_stock, "asset_class"].value_counts().to_dict()
            logger.info(
                f"Se descartan {n_dropped} fila(s) con AssetClass != 'STK' "
                f"(desglose por clase: {dropped_reasons})."
            )
        df = df[is_stock].copy()

    for col in _NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- cost_basis_price derivado, protegido contra división por cero ---
    if "Quantity" in raw.columns:
        bad_quantity = df["quantity"].isna() | (df["quantity"] == 0)
        n_bad_quantity = int(bad_quantity.sum())
        if n_bad_quantity:
            logger.warning(
                f"{n_bad_quantity} posición(es) con Quantity 0 o ausente: cost_basis_price "
                f"queda en NaN para esas filas (no se divide por cero)."
            )
    safe_quantity = df["quantity"].replace(0, np.nan)
    df["cost_basis_price"] = df["cost_basis_money"] / safe_quantity

    # --- open_date: parseo tolerante, sin loguear ruido si la columna está vacía ---
    raw_open_date = df["open_date"].astype(str).str.strip()
    has_value = raw_open_date.notna() & raw_open_date.ne("") & raw_open_date.str.lower().ne("nan")
    date_only = raw_open_date.str.split(";").str[0]
    parsed_open_date = pd.to_datetime(date_only, format="%Y%m%d", errors="coerce")

    if has_value.any():
        n_unparsed = int((has_value & parsed_open_date.isna()).sum())
        if n_unparsed:
            logger.warning(
                f"{n_unparsed} valor(es) de OpenDateTime no se pudieron interpretar (formato "
                f"esperado yyyyMMdd o yyyyMMdd;HHmmss); se dejan como NaT."
            )
    df["open_date"] = parsed_open_date

    df["symbol"] = df["symbol"].astype(str).str.strip()

    return df[_OUTPUT_COLUMNS].reset_index(drop=True)


# =============================================================================
#                    ANÁLISIS POR POSICIÓN (MODO CARTERA)
# =============================================================================
# Versión REDUCIDA de main_pipeline.run_production_system: mismos motores de
# ingesta, factores técnicos, volatilidad condicional y fundamentales, pero sin
# Monte Carlo, sin abanicos cuantílicos y sin gráficos, que son el grueso del
# tiempo de corrida y no aportan nada al contexto de una posición ya abierta.
# Pensado para correr en segundos por ticker, no minutos, porque una cartera
# típica tiene varias posiciones a evaluar en una sola sesión.
#
# FASE 3.3 — YA NO ENTRENA NINGÚN MODELO. Antes esta función existía para
# producir `current_ml_prob`; ahora produce contexto: percentil de valoración,
# percentil de volatilidad condicional, beta, drawdown de entrada típico y
# tendencia del margen neto TTM. Quitar el entrenamiento del ML la abarata
# mucho más de lo que la encarece añadir los fundamentales.

def analyze_position(ticker: str, config: PipelineConfig) -> Dict[str, Any]:
    """
    Pipeline reducido de CONTEXTO para UNA posición de cartera. Devuelve:
      - precio_actual: último cierre ajustado disponible.
      - valuation_percentile / valuation_multiple_used / valuation_n_quarters /
        valuation_fiable: percentil point-in-time del múltiplo de valoración
        (ver _VALUATION_MULTIPLE_PREFERENCE) dentro del histórico de la propia
        empresa, con el n que lo sostiene.
      - volatility_ann, max_drawdown, beta: perfil de riesgo estadístico
        (beta = último valor de la beta ROLLING vs. el benchmark).
      - garch_vol_esperada: volatilidad diaria esperada sobre el horizonte.
      - vol_percentile: percentil de la volatilidad condicional GARCH de hoy
        dentro de su propio histórico — es lo que permite hablar de "cambio de
        régimen" sin inventar un umbral absoluto de volatilidad.
      - drawdown_horizonte / drawdown_mediano: distribución del drawdown de
        entrada a config.horizon_months (EntryContextEngine), con su n
        independiente y su flag de fiabilidad.
      - net_margin_ttm / margen_trimestres_cayendo: nivel y tendencia del
        margen neto TTM, para el tercer disparador de "REVISAR".

    Trade-off de rendimiento — config.portfolio_ts_refit_every (63 días,
    trimestral) en vez de config.ts_refit_every (21 días, mensual): el
    walk-forward de GARCH (reajuste MLE periódico) es el paso más caro que
    queda en esta función. Reajustar cada trimestre en vez de cada mes reduce
    sustancialmente las veces que se reoptimiza el modelo por ticker; se pierde
    algo de fidelidad intra-trimestre en la volatilidad condicional, pero el
    percentil que se lee de ella opera a resolución de meses, no de días.

    NO se calcula el target forward ni ninguna tendencia ARIMA: los dos existían
    solo como insumo del clasificador ML, que ya no se entrena acá.
    """
    engine = DataEngine(tickers=[ticker], benchmark=config.benchmark, cache_dir=config.cache_dir,
                        trading_days_per_year=config.trading_days_per_year)
    # end_date = última sesión CERRADA, no today() — mismo motivo que en
    # run_production_system (ver data_engine.last_closed_session): con today()
    # la última barra es un precio VIVO y todo lo que se derive de ella cambia
    # entre corridas del mismo día.
    ultima_sesion = last_closed_session()
    end_date = ultima_sesion.strftime("%Y-%m-%d")
    start_date = (ultima_sesion - datetime.timedelta(days=config.lookback_years * 365)).strftime("%Y-%m-%d")

    market_prices = engine.fetch_market_data(start_date=start_date, end_date=end_date)

    tech_engine = TechnicalFactorEngine(market_data=market_prices, benchmark_data=engine.benchmark_data,
                                        trading_days_per_year=config.trading_days_per_year)
    ts_extractor = TimeSeriesFeatureExtractor(log_returns=tech_engine.log_returns)
    garch_vol = ts_extractor.extract_garch_volatility(ticker=ticker, refit_every=config.portfolio_ts_refit_every)
    garch_vol_esperada = ts_extractor.extract_garch_forecast(ticker=ticker, horizon_days=config.horizon_days)
    vol_ctx = percentil_historico(garch_vol)

    stat_risk = tech_engine.compute_statistical_risk()
    beta_series = stat_risk['beta'][ticker].dropna()
    beta = float(beta_series.iloc[-1]) if not beta_series.empty else float("nan")
    volatility_ann = float(stat_risk['volatility_ann'][ticker]) if ticker in stat_risk['volatility_ann'] else float("nan")
    max_drawdown = float(stat_risk['max_drawdown'][ticker]) if ticker in stat_risk['max_drawdown'] else float("nan")

    precio_serie = market_prices[ticker].dropna()
    precio_actual = float(precio_serie.iloc[-1])

    # --- Valoración point-in-time -----------------------------------------
    fundamental_data = engine.fetch_fundamental_data(ticker=ticker, source=config.fundamentals_source)
    fund_engine = FundamentalFactorEngine(
        fundamental_data=fundamental_data,
        min_invested_capital_fraction=config.roic_min_invested_capital_fraction,
    )
    valuation_df = fund_engine.compute_valuation_scores(current_prices=precio_serie)
    valuation_ctx = EntryContextEngine.valuation_context(
        valuation_df, n_quarters_reliable=MIN_QUARTERS_FOR_RELIABILITY,
    )
    valuation_percentile = float("nan")
    valuation_multiple_used = None
    for clave, etiqueta in _VALUATION_MULTIPLE_PREFERENCE:
        datos = (valuation_ctx.get("multiplos") or {}).get(clave)
        if datos and datos.get("n_obs"):
            pct = datos.get("percentil")
            if pct is not None and not pd.isna(pct):
                valuation_percentile = float(pct)
                valuation_multiple_used = etiqueta
                break

    net_margin = fund_engine.net_margin_ttm()
    net_margin_ttm = float(net_margin.iloc[-1]) if not net_margin.empty else float("nan")
    # Fase 3.6: solo se cuentan trimestres consecutivos de caída sobre
    # ventanas TTM que cubren de verdad 12 meses. Una "caída" entre una
    # ventana de 12 meses y otra de 18 no es un deterioro del negocio, es un
    # cambio de la ventana — y disparar "REVISAR" por eso sería avisar de un
    # hueco de EDGAR como si fuera un hecho de la empresa.
    ventana_ok = fund_engine.net_margin_ttm_window_valid()
    margen_trimestres_cayendo = contar_trimestres_cayendo(
        net_margin, ventana_valida=ventana_ok,
    )

    # --- Drawdown de entrada al horizonte configurado ----------------------
    # Un solo horizonte (config.horizon_months) y no el abanico de cinco: acá
    # se necesita una cifra de referencia por posición para la tabla, y el
    # abanico completo vive en la ficha de entrada del modo [1].
    entry_engine = EntryContextEngine(price_series=precio_serie,
                                       horizontes_meses=(config.horizon_months,),
                                       dias_por_mes=config.trading_days_per_month)
    drawdown_horizonte = entry_engine.drawdown_distribution().get(config.horizon_months, {})

    return {
        "ticker": ticker,
        "precio_actual": precio_actual,
        "valuation_percentile": valuation_percentile,
        "valuation_multiple_used": valuation_multiple_used,
        "valuation_n_quarters": valuation_ctx.get("n_quarters", 0),
        "valuation_fiable": valuation_ctx.get("fiable", False),
        "volatility_ann": volatility_ann,
        "max_drawdown": max_drawdown,
        "beta": beta,
        "garch_vol_esperada": garch_vol_esperada,
        "vol_percentile": vol_ctx["percentil"],
        "vol_condicional_actual": vol_ctx["actual"],
        "drawdown_horizonte": drawdown_horizonte,
        "drawdown_mediano": drawdown_horizonte.get("p50", float("nan")),
        "net_margin_ttm": net_margin_ttm,
        "margen_trimestres_cayendo": margen_trimestres_cayendo,
    }


def contar_trimestres_cayendo(serie: pd.Series,
                               ventana_valida: Optional[pd.Series] = None) -> Optional[int]:
    """
    Cuántos trimestres CONSECUTIVOS lleva cayendo una serie contable, contando
    hacia atrás desde el último dato.

    `ventana_valida` (Fase 3.6) es una máscara booleana opcional: los
    trimestres marcados False se descartan ANTES de contar. Se usa para no
    contar como "caída del margen" un escalón que en realidad viene de
    comparar una ventana TTM de 12 meses con otra de 18 (ver
    FundamentalFactorEngine.window_coverage). Si al filtrar quedan menos de
    dos observaciones, el resultado es None: no evaluable.

    Devuelve None —no 0— cuando no hay al menos dos observaciones con las que
    comparar: "no se puede saber" y "no está cayendo" son estados distintos, y
    confundirlos es lo que hace que un dato ausente se lea como una medición
    tranquilizadora (mismo criterio que `reliable=False` o `relative_drop=None`
    en el resto del sistema).

    Un valor plano (igual al anterior) corta la cuenta: cae quien baja, no
    quien se queda quieto.
    """
    limpia = serie.dropna() if isinstance(serie, pd.Series) else pd.Series(dtype=float)
    if ventana_valida is not None and not limpia.empty:
        mascara = ventana_valida.reindex(limpia.index).fillna(False).astype(bool)
        limpia = limpia[mascara]
    if len(limpia) < 2:
        return None
    valores = limpia.to_numpy(dtype=float)
    cuenta = 0
    for i in range(len(valores) - 1, 0, -1):
        if valores[i] < valores[i - 1]:
            cuenta += 1
        else:
            break
    return int(cuenta)


# =============================================================================
#                 REGISTROS HISTÓRICOS POR TICKER (JSON en cache/)
# =============================================================================
# Dos ficheros con el mismo formato: {ticker: [{"date": "YYYY-MM-DD", ...}, ...]}.
#
#   - ml_prob_history.json (config.ml_prob_history_path): histórico de la
#     probabilidad ML. FASE 3.3: YA NO SE ESCRIBE — nada llama a
#     record_ml_prob. El fichero NO se borra porque es histórico (describe qué
#     afirmaba el sistema en cada fecha) y get_earliest_ml_prob sigue pudiendo
#     leerlo; los dos se conservan con sus tests.
#   - context_history.json (config.context_history_path): histórico de
#     CONTEXTO (percentil de valoración, percentil de volatilidad). Es el que
#     alimenta ahora los disparadores de "REVISAR": responde "¿cambió algo
#     material desde que empecé a mirar esta posición?" sin recurrir a ninguna
#     predicción.


def _load_json_history(path: str) -> Dict[str, list]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"No se pudo leer el histórico en {path} ({e}); se empieza de cero.")
        return {}


def _load_ml_prob_history(path: str) -> Dict[str, list]:
    """Alias histórico de _load_json_history (los dos ficheros comparten formato)."""
    return _load_json_history(path)


def _persist_json_atomic(path: str, contenido: Any) -> None:
    """
    Escribe `contenido` como JSON en `path` de forma ATÓMICA (.tmp + os.replace).

    Antes se escribía directamente sobre el fichero definitivo, así que una
    interrupción a mitad de volcado —Ctrl+C, corte de luz— dejaba un JSON
    truncado y se perdía TODO el histórico acumulado, que es justamente el dato
    que no se puede recuperar porque describe el pasado. os.replace es atómico
    en Windows y POSIX.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp_path = f"{path}.tmp"
    renombrado = False
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(contenido, f, indent=2)
        os.replace(tmp_path, path)
        renombrado = True
    finally:
        # Si el volcado no llegó a completarse, no dejar el .tmp truncado
        # tirado: la siguiente corrida no lo leería (solo se lee `path`), pero
        # ensucia el cache_dir y confunde al depurar.
        #
        # `finally` y no `except Exception`: la interrupción que motiva toda
        # esta escritura atómica es un Ctrl+C, y KeyboardInterrupt hereda de
        # BaseException, así que un `except Exception` no lo captura — el
        # propio test de esta sub-fase lo detectó.
        if not renombrado and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def _upsert_por_fecha(entries: list, date: Any, valores: Dict[str, Any],
                       claves_requeridas: Sequence[str] = ()) -> list:
    """
    Normaliza una lista de entradas a UNA por fecha (la última gana) y la
    devuelve ordenada por fecha.

    Upsert y no append: `entries.append(...)` sin deduplicar hacía que cada
    corrida acumulara una entrada más, y el fichero real acabó con duplicados
    de la misma fecha. Eso no es solo ruido — los lectores recorren el
    histórico para identificar la entrada más ANTIGUA, y duplicar entradas
    distorsiona ese registro. El orden explícito hace falta porque el orden de
    inserción no garantiza nada si alguna vez se registra una fecha pasada.

    `claves_requeridas` descarta las entradas ya presentes a las que les falte
    alguno de esos campos (además de las que no tengan fecha, que se descartan
    siempre). Se usa para el histórico de probabilidades ML, cuyo lector hace
    `entrada["ml_prob"]` y se rompería con una entrada a medias. El histórico de
    CONTEXTO no lo usa a propósito: sus claves son aditivas —una corrida futura
    puede registrar una métrica que las viejas no traen— y descartar una entrada
    por eso tiraría historia real.
    """
    por_fecha: Dict[str, Dict[str, Any]] = {}
    for entry in entries or []:
        fecha = entry.get("date")
        if fecha is None:
            continue
        if any(entry.get(clave) is None for clave in claves_requeridas):
            continue
        por_fecha[str(fecha)] = {k: v for k, v in entry.items() if k != "date"}
    por_fecha[str(date)] = dict(valores)
    return [{"date": f, **por_fecha[f]} for f in sorted(por_fecha)]


def record_context(ticker: str, date: Any, metrics: Dict[str, Any], path: str) -> None:
    """
    Registra los percentiles de contexto de hoy para el ticker (upsert por
    fecha, escritura atómica) en config.context_history_path.

    Es lo que hace evaluable, en la corrida siguiente, si algo cambió de forma
    material. En la PRIMERA corrida de una posición no hay nada con lo que
    comparar y los disparadores relativos quedan NO EVALUABLES — el estado
    esperado, no un fallo.
    """
    history = _load_json_history(path)
    # NaN fuera: un percentil que no se pudo calcular no debe quedar
    # registrado como si fuera un valor, o la corrida siguiente lo compararía.
    limpio = {k: (float(v) if v is not None and not pd.isna(v) else None)
              for k, v in metrics.items()}
    history[ticker] = _upsert_por_fecha(history.get(ticker, []), date, limpio)
    _persist_json_atomic(path, history)


def get_earliest_context(ticker: str, path: str) -> Optional[Dict[str, Any]]:
    """
    Devuelve la entrada de contexto MÁS ANTIGUA registrada para el ticker, o
    None si no hay ningún histórico previo — nunca un valor por defecto.
    """
    history = _load_json_history(path)
    entries = history.get(ticker, [])
    if not entries:
        return None
    return min(entries, key=lambda e: e["date"])


def record_ml_prob(ticker: str, date, prob: float, path: str) -> None:
    """
    Registra {date, ml_prob} en el histórico del ticker y persiste el JSON
    completo en `path` (los llamadores usan config.ml_prob_history_path).
    `date` acepta cualquier valor date-like (se guarda como str(date),
    típicamente ISO yyyy-mm-dd).

    UPSERT por (ticker, fecha): una sola entrada por ticker y día, y la última
    corrida del día gana. Antes hacía `entries.append(...)` sin deduplicar, así
    que cada corrida acumulaba una entrada más y el fichero real acabó con
    duplicados de la misma fecha para los cuatro tickers de la cartera. Eso no
    es solo ruido: `get_earliest_ml_prob` recorre el histórico para responder
    "¿se deterioró la tesis?", y duplicar entradas distorsiona ese registro.

    Las entradas se mantienen ORDENADAS por fecha. `get_earliest_ml_prob`
    depende de poder identificar la más antigua, y el orden de inserción no lo
    garantiza si alguna vez se registra una fecha pasada.

    ESCRITURA ATÓMICA (.tmp + os.replace): antes se escribía directamente sobre
    el fichero definitivo, así que una interrupción a mitad de volcado
    —Ctrl+C, corte de luz— dejaba un JSON truncado y se perdía TODO el
    histórico acumulado, que es justamente el dato que no se puede recuperar
    porque describe el pasado. os.replace es atómico en Windows y POSIX.
    """
    history = _load_json_history(path)
    # Normalizar TODA la lista a una entrada por fecha (la última gana), no
    # solo la de hoy: el fichero real ya venía con duplicados de las corridas
    # anteriores al arreglo, y así se auto-repara en cuanto ese ticker vuelve a
    # registrarse, sin necesitar un script de migración aparte.
    history[ticker] = _upsert_por_fecha(history.get(ticker, []), date, {"ml_prob": float(prob)},
                                        claves_requeridas=("ml_prob",))
    _persist_json_atomic(path, history)


def get_earliest_ml_prob(ticker: str, path: str) -> Optional[float]:
    """
    Devuelve la probabilidad ML MÁS ANTIGUA registrada para el ticker, o
    None si no hay ningún histórico previo — NUNCA un valor por defecto ni
    False. Mismo patrón que `reliable=False` en FundamentalFactorEngine: la
    ausencia de historia es un estado explícito ("no evaluable"), no un
    error ni un "0% de caída".

    FASE 3.3 — este fichero YA NO SE ALIMENTA (nada llama a record_ml_prob),
    así que esta función solo sirve para LEER el histórico acumulado hasta
    entonces. Se conserva porque ese histórico es la única constancia de qué
    probabilidad afirmaba el sistema en cada fecha, y borrarlo sería tirar la
    evidencia con la que algún día se podrá juzgar aquella capa. El registro
    que sí se alimenta ahora es el de contexto (ver get_earliest_context).
    """
    history = _load_ml_prob_history(path)
    entries = history.get(ticker, [])
    if not entries:
        return None
    earliest = min(entries, key=lambda e: e["date"])
    return float(earliest["ml_prob"])


# =============================================================================
#                        MARCA "REVISAR" (FASE 3.3)
# =============================================================================
# `generate_hold_sell_signal` SE ELIMINÓ. Combinaba dos umbrales sobre
# `current_ml_prob` para emitir MANTENER / VIGILAR / VENDER, y toda esa señal
# descansaba en que la probabilidad ML dijera algo sobre el futuro. La Fase 2
# lo midió con la vara del propio sistema (ventanas independientes, no fechas
# solapadas) y no dice nada: Information Coefficient medio -0,0093, signo
# repartido 5-5, y los dos grupos del event study no estimables al horizonte de
# producción en ninguno de los 10 tickers. Una orden de vender sostenida en eso
# es peor que ninguna orden.
#
# Lo que la sustituye NO es otra etiqueta de decisión. "REVISAR" significa
# "míralo tú": se marca cuando algo cambió de forma MATERIAL frente a lo que
# quedó registrado la primera vez que se analizó la posición, y la explicación
# dice qué cambió y cuánto. No dice qué hacer.


def generate_review_flags(*, valuation_percentile: Optional[float],
                           valuation_percentile_earliest: Optional[float],
                           vol_percentile: Optional[float],
                           vol_percentile_earliest: Optional[float],
                           margen_trimestres_cayendo: Optional[int],
                           config: PipelineConfig) -> Dict[str, Any]:
    """
    Tres disparadores INDEPENDIENTES de "REVISAR", cada uno con su umbral en
    config.py (nunca hardcodeado):

    1. VALORACIÓN (config.review_threshold_valuation_percentile_jump): el
       percentil point-in-time del múltiplo se movió más que el umbral desde el
       primer registro. En VALOR ABSOLUTO: que se haya vuelto mucho más barata
       también es un cambio material que merece una mirada — el sistema no
       decide en qué dirección es "malo".
    2. VOLATILIDAD (config.review_threshold_vol_percentile_jump): lo mismo con
       el percentil de la volatilidad condicional GARCH. Es la
       operacionalización de "cambió de régimen": un umbral absoluto de
       volatilidad ("más del 40% anual") marcaría siempre a los mismos activos
       por ser lo que son, no por haber cambiado.
    3. MARGEN (config.review_margin_declining_quarters): el margen neto TTM
       lleva N trimestres consecutivos cayendo. Este NO necesita histórico
       propio del registro —sale de los datos contables—, así que es el único
       evaluable en la primera corrida de una posición.

    Cada disparador devuelve None cuando NO ES EVALUABLE (sin registro previo,
    o sin percentil calculable), nunca False: ausencia de histórico no es
    evidencia de que nada haya cambiado. `revisar` es True solo si algún
    disparador dio True — un None nunca marca.
    """
    def _salto(actual: Optional[float], anterior: Optional[float], umbral: float) -> Optional[bool]:
        if actual is None or anterior is None:
            return None
        if pd.isna(actual) or pd.isna(anterior):
            return None
        return bool(abs(float(actual) - float(anterior)) > umbral)

    valoracion = _salto(valuation_percentile, valuation_percentile_earliest,
                        config.review_threshold_valuation_percentile_jump)
    volatilidad = _salto(vol_percentile, vol_percentile_earliest,
                         config.review_threshold_vol_percentile_jump)

    if margen_trimestres_cayendo is None:
        margen = None
    else:
        margen = bool(margen_trimestres_cayendo >= config.review_margin_declining_quarters)

    motivos = {"valoracion": valoracion, "volatilidad": volatilidad, "margen": margen}
    return {"revisar": any(v is True for v in motivos.values()), "motivos_revisar": motivos}


# =============================================================================
#          EVALUACIÓN COMPLETA DE UNA POSICIÓN (CONTEXTO + MARCA "REVISAR")
# =============================================================================

def evaluate_position(position: Dict[str, Any], config: PipelineConfig) -> Dict[str, Any]:
    """
    Une analyze_position + el histórico de contexto + los disparadores de
    "REVISAR" para UNA fila de posición (la que produce load_positions, como
    dict o pd.Series) y devuelve un único resultado.

    FASE 3.3 — NO EMITE NINGÚN VEREDICTO. Antes devolvía
    MANTENER/VIGILAR/VENDER (o "SIN SEÑAL") a partir de la probabilidad ML;
    ahora devuelve contexto y, como mucho, una marca "REVISAR" que significa
    "míralo tú". No hay etiqueta de decisión, ni sustituta suya con otro
    nombre: cualquier resumen categórico de varias dimensiones en una palabra
    reintroduce lo que esta fase elimina.

    CRÍTICO — cost_basis_price (el precio medio de entrada) NO PARTICIPA EN
    NADA. Se devuelve, junto con la cantidad y el P&L no realizado, en claves
    separadas y marcadas como contexto informativo, nunca mezclado con lo que
    se mide.

    Por qué: el precio de entrada es un coste hundido (sunk cost). La pregunta
    que gestiona el riesgo de la posición no es "¿gano o pierdo frente a lo que
    pagué?" sino "¿qué tengo delante hoy?". Meter cost_basis_price en la
    lectura introduciría disposition effect (aferrarse a una posición perdedora
    esperando 'recuperar' el precio de entrada, o vender una ganadora
    demasiado pronto para 'asegurar' la ganancia) y anclaje al precio de
    compra — dos sesgos de comportamiento bien documentados, no una ventaja
    informativa real. Dai, Zhang & Zhu (2010, "Optimal Trend Following Trading
    Rules under a Three-State Regime Switching Model") derivan la regla de
    salida ÓPTIMA para este problema (mantener o liquidar bajo un régimen de
    mercado no observable) y muestran que depende del estado inferido, nunca
    del nivel de precio al que se entró.
    """
    ticker = position["symbol"]
    history_path = config.context_history_path

    # El histórico se lee ANTES de registrar la corrida de hoy: si se leyera
    # después, la primera corrida de un ticker se compararía consigo misma
    # (salto 0 en vez de "no evaluable").
    earliest = get_earliest_context(ticker, path=history_path) or {}

    analysis = analyze_position(ticker, config)

    flags = generate_review_flags(
        valuation_percentile=analysis["valuation_percentile"],
        valuation_percentile_earliest=earliest.get("valuation_percentile"),
        vol_percentile=analysis["vol_percentile"],
        vol_percentile_earliest=earliest.get("vol_percentile"),
        margen_trimestres_cayendo=analysis["margen_trimestres_cayendo"],
        config=config,
    )

    record_context(
        ticker,
        date=datetime.date.today(),
        metrics={
            "valuation_percentile": analysis["valuation_percentile"],
            "vol_percentile": analysis["vol_percentile"],
        },
        path=history_path,
    )

    cost_basis_price = position.get("cost_basis_price")
    has_cost_basis = cost_basis_price is not None and not pd.isna(cost_basis_price) and cost_basis_price != 0
    unrealized_pnl_pct_vs_cost = (
        (analysis["precio_actual"] / cost_basis_price) - 1 if has_cost_basis else None
    )

    return {
        **analysis,
        "revisar": flags["revisar"],
        "motivos_revisar": flags["motivos_revisar"],
        "has_history": bool(earliest),
        "valuation_percentile_earliest": earliest.get("valuation_percentile"),
        "vol_percentile_earliest": earliest.get("vol_percentile"),
        # --- Contexto informativo, NUNCA usado en nada de lo de arriba ---
        "cost_basis_price": cost_basis_price,
        "quantity": position.get("quantity"),
        "unrealized_pnl": position.get("unrealized_pnl"),
        "unrealized_pnl_pct_vs_cost": unrealized_pnl_pct_vs_cost,
    }
