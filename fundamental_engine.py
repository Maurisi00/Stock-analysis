import pandas as pd
import numpy as np
import logging
from typing import Any, Dict

logger = logging.getLogger("QuantSystem.FundamentalEngine")

# Mínimo de trimestres históricos necesarios para que un percentil (rank pct=True)
# tenga algún significado estadístico. Por debajo de esto, la ficha se marca
# como no fiable aunque sus valores sean correctos: el dato contable vale, el
# PERCENTIL sobre 5 observaciones no. Con SEC EDGAR como fuente (40-80
# trimestres típicos) 12 es alcanzable; con yfinance (~5) marcará no fiable,
# que es el comportamiento correcto.
MIN_QUARTERS_FOR_RELIABILITY = 12

# FASE 3.3 — QUALITY_ABSOLUTE_THRESHOLDS y _absolute_level_score SE ELIMINARON.
# Eran el parche que mezclaba el percentil histórico de la propia empresa con un
# nivel absoluto de la literatura, y existían SOLO para alimentar `score_quality`
# (a su vez insumo del score multifactorial). Sin score no hay nada que anclar:
# la Calidad pasa a ficha DESCRIPTIVA (ver `describe_quality`), donde el valor y
# su percentil se imprimen por separado y el lector los interpreta — que es
# justo lo que el promedio de los dos impedía hacer.


# UMBRAL DE MATERIALIDAD del capital invertido para el ROIC (Fase 3.6). Es el
# DEFAULT; el valor operativo vive en
# config.PipelineConfig.roic_min_invested_capital_fraction, que es donde está
# la justificación completa del 0.10 con las cifras que la sostienen.
#
# En una frase: el ROIC es asintótico en cero porque su denominador es una
# RESTA de números grandes (deuda + equity - caja), así que cuando el resultado
# es una fracción diminuta de sus propios inputs el ratio deja de ser "alto" y
# pasa a ser NO ESTIMABLE. La Fase 1.5 solo cubrió el caso <= 0.
MIN_INVESTED_CAPITAL_FRACTION = 0.10

# --- Cobertura real de las ventanas TTM / YoY (Fase 3.6) -------------------
# `.rolling(4)` y `.pct_change(4)` cuentan FILAS, no trimestres. Con los huecos
# de EDGAR (trimestres que no se pudieron derivar y se descartan), una ventana
# "de 12 meses" puede abarcar bastante más tiempo real sin avisar de nada.
#
# MEDIDO sobre 7 tickers y 282 trimestres reales (2026-09): el 69% de las
# ventanas TTM y el 78% de las YoY NO cubren 12 meses, y la cobertura MEDIANA
# de una ventana TTM es de 455 días — o sea que el "TTM" típico de este sistema
# es en realidad un acumulado de 15 meses. La separación entre filas
# consecutivas tiene mediana 92 días (correcta) pero percentil 75 de 182 días:
# uno de cada cuatro pares de filas consecutivas está a dos trimestres o más.
# El peor caso encontrado es un hueco de 1.099 días en AMZN.
#
# No se arregla reindexando a trimestres fiscales (el plan lo deja como
# opcional y es una reescritura del extractor de EDGAR). Lo que NO es opcional
# es dejar de ocultarlo: se mide la cobertura real de cada ventana y se
# publica junto al valor, para que el lector sepa que el "ROE TTM" que está
# leyendo puede ser de 15 meses.
DIAS_ANIO_CONTABLE = 365
# Un trimestre de calendario. Se suma al lapso entre la primera y la última
# FILA de una ventana TTM: 4 filas separadas ~91 días abarcan 273 días de
# lapso, pero cubren 4 trimestres de actividad, o sea 273 + 91 = 365.
DIAS_POR_TRIMESTRE_CALENDARIO = 365.25 / 4
# Desvío tolerado respecto de los 365 días esperados, en días. ~45 es el valor
# que propone el plan: media estación. Por debajo de eso el desvío es el ruido
# normal de las fechas de publicación (un 10-K se presenta más tarde que un
# 10-Q); por encima, la ventana está sumando trimestres que no corresponden.
TOLERANCIA_DIAS_VENTANA = 45


def cobertura_ventana_ttm(index: pd.Index) -> pd.Series:
    """
    Días de calendario que CUBRE de verdad la ventana TTM de 4 filas que
    termina en cada fila.

    Se calcula como el lapso entre la fila t-3 y la t más un trimestre: las 4
    filas de una ventana correcta están separadas ~91 días cada una (273 de
    lapso total) y entre todas cubren 4 trimestres de actividad, 365 días. Si
    el resultado se va muy por encima, la ventana está sumando trimestres
    salteados y su "TTM" abarca más de un año.

    Las 3 primeras filas salen NaN: es el mismo burn-in que `_ttm`.
    """
    if index is None or len(index) == 0:
        return pd.Series(dtype=float)
    fechas = pd.Series(pd.to_datetime(pd.Index(index)), index=index)
    lapso = (fechas - fechas.shift(3)).dt.days
    return lapso + DIAS_POR_TRIMESTRE_CALENDARIO


def cobertura_ventana_yoy(index: pd.Index) -> pd.Series:
    """
    Días de calendario entre cada fila y la de 4 filas antes, que es lo que
    `.pct_change(4)` trata como "el mismo trimestre del año anterior".

    Acá no hay que sumar ningún trimestre: se comparan DOS instantes, no se
    acumula un período, así que el valor esperado es directamente 365.
    """
    if index is None or len(index) == 0:
        return pd.Series(dtype=float)
    fechas = pd.Series(pd.to_datetime(pd.Index(index)), index=index)
    return (fechas - fechas.shift(4)).dt.days.astype(float)


def ventana_cubre_un_anio(cobertura: pd.Series,
                           tolerancia: float = TOLERANCIA_DIAS_VENTANA) -> pd.Series:
    """
    True donde la ventana cubre 365 +/- `tolerancia` días; False donde no.

    Donde la cobertura es NaN (burn-in) devuelve False: no es que la ventana
    sea mala, es que no hay ventana — y el valor de esa fila también es NaN,
    así que nunca se publica un número marcado como fiable sin serlo.
    """
    if cobertura.empty:
        return pd.Series(dtype=bool)
    return (cobertura - DIAS_ANIO_CONTABLE).abs() <= tolerancia


def _positivo_o_nan(serie: pd.Series) -> pd.Series:
    """
    Anula (NaN) los valores <= 0 de un denominador de múltiplo o de ratio.

    Motivo (AUDITORIA_2026-09.md A5/A6): el score de valoración que existía
    hasta la Fase 3.3 puntuaba con `1.0 - rank(pct)`, o sea "múltiplo más bajo
    = más barato = mejor". Un denominador negativo produce un múltiplo
    NEGATIVO, que es el más bajo de toda la serie, así que el percentil salía
    mínimo y el score ~1.0: **una empresa en pérdidas aparecía como la
    valoración más atractiva de su propia historia**. La regla sigue siendo
    igual de necesaria sin score: `EntryContextEngine.valuation_context`
    reporta el percentil crudo, y un PER negativo colado ahí saldría como el
    múltiplo más BARATO del histórico de la empresa. Lo mismo con EV/EBITDA si el EBITDA es negativo, con P/B si el
    equity es negativo, y con el ROIC si el capital invertido es <= 0 (caso
    net-cash, casi toda la tecnológica: el denominador tiende a cero o cambia
    de signo y el ratio explota).

    Un múltiplo con denominador negativo no es "barato": no está definido. La
    regla correcta es NaN — que el motor ya sabe propagar y reportar como no
    evaluable — nunca un número que ordena mal.

    Acepta también un ESCALAR: cuando una columna no existe en los datos
    contables, `df.get(col, np.nan)` / `df.get(col, 0.0)` devuelve un float y
    no una Series, así que `invested_capital` puede degradarse a escalar si
    faltan todas sus componentes (pasa con fundamentales vacíos o parciales,
    p. ej. la fixture de crecimiento YoY que solo trae total_revenue).
    """
    if isinstance(serie, pd.Series):
        return serie.where(serie > 0)
    # Escalar (o NaN): mismo criterio, sin pandas de por medio.
    if serie is None:
        return np.nan
    valor = float(serie)
    return valor if valor > 0 else np.nan


def _capital_invertido_material(invertido, capital_bruto, min_fraccion: float):
    """
    Anula (NaN) el capital invertido que no es MATERIAL respecto del capital
    bruto del que sale, además del que ya era <= 0.

    EL PROBLEMA. `capital_invertido = deuda + equity - caja` es una resta de
    números grandes. Su error relativo respecto del error de la caja se
    amplifica por `caja / capital_invertido`: en una empresa que tiene casi
    todo su balance en caja, una diferencia del 1% en la cifra de caja —una
    reclasificación normal entre "efectivo" y "inversiones a corto", o una
    diferencia de timing a cierre de trimestre— se convierte en un error
    enorme del denominador, y por tanto del ROIC. Con el capital invertido al
    1% del bruto, ese 1% de error en la caja es un 99% de error en el ROIC: el
    número no es alto, es ruido.

    LA REGLA. Si `capital_invertido < min_fraccion * (deuda + equity)`, el
    ROIC es NaN. Con el default de 0.10 la amplificación queda acotada en ~9x
    (un 1% de error en la caja -> ~9% en el ROIC). Medido sobre 8 tickers
    reales y 182 trimestres con ROIC definido, no anula NI UNA observación: el
    mínimo real es del 22,75%. Dispara solo en el régimen patológico.

    POR QUÉ NaN Y NO UN TECHO. Recortar el ROIC a, digamos, 3.0 lo dejaría
    leyéndose como una medición ("ROIC del 300%"), y no lo es. El invariante
    del repo es que un número que no se puede medir se declara ausente, no se
    sustituye por el valor más plausible.

    Acepta escalares por el mismo motivo que `_positivo_o_nan` (ver su
    docstring): con fundamentales parciales, `df.get(col, 0.0)` degrada la
    expresión a float.
    """
    invertido = _positivo_o_nan(invertido)

    if not isinstance(invertido, pd.Series):
        # Escalar: si no hay un capital bruto positivo con el que comparar, no
        # hay materialidad que juzgar y se deja lo que ya dijo _positivo_o_nan.
        try:
            bruto = float(capital_bruto)
        except (TypeError, ValueError):
            return invertido
        if not np.isfinite(bruto) or bruto <= 0 or invertido != invertido:
            return invertido
        return invertido if invertido >= min_fraccion * bruto else np.nan

    bruto = capital_bruto if isinstance(capital_bruto, pd.Series) else \
        pd.Series(capital_bruto, index=invertido.index, dtype=float)
    # Solo se juzga la materialidad donde el capital bruto es positivo y
    # conocido; donde no lo es, el propio invertido ya salió NaN (no puede ser
    # positivo con un bruto <= 0 y una caja >= 0).
    juzgable = bruto.notna() & (bruto > 0)
    inmaterial = juzgable & (invertido < min_fraccion * bruto)
    n_anulados = int((inmaterial & invertido.notna()).sum())
    if n_anulados:
        logger.info(
            f"ROIC no estimable en {n_anulados} trimestre(s): el capital invertido "
            f"(deuda + equity - caja) queda por debajo del {min_fraccion:.0%} del capital "
            f"bruto, así que su error relativo lo domina — ver "
            f"config.roic_min_invested_capital_fraction."
        )
    return invertido.where(~inmaterial)


def _ttm(serie: pd.Series) -> pd.Series:
    """
    Suma móvil de 4 trimestres (Trailing Twelve Months) de una partida de
    FLUJO (resultados, flujo de caja). Las primeras 3 filas salen NaN, que es
    el burn-in correcto: no hay 4 trimestres todavía.

    Por qué hace falta: los umbrales de QUALITY_ABSOLUTE_THRESHOLDS ("ROE > 15%
    = excelente") son ANUALES, pero las partidas de EDGAR/yfinance son
    trimestrales. Dividir un beneficio trimestral por un equity (que es un
    stock, ya anual por naturaleza) da un ROE ~4x más chico que el anual: una
    empresa con ROE anual del 15% producía ~3,75% y _absolute_level_score le
    daba 0,25 en vez de 1,0. Todo el factor Calidad estaba sesgado a la baja
    para TODOS los tickers (AUDITORIA_2026-09.md A4).

    Regla general al usar esto: acumular SOLO los flujos. Los stocks de balance
    (equity, deuda, capital invertido, caja) NO se acumulan — se toma el valor
    del trimestre, porque ya representan un saldo a una fecha.

    DEUDA CONOCIDA (AUDITORIA_2026-09.md, sección MEDIO — fuera del alcance de
    la Fase 1): `.rolling(4)` cuenta FILAS, no trimestres. Con la fuente EDGAR
    el índice puede tener huecos (trimestres que no se pudieron derivar), así
    que un "TTM" puede abarcar 5-7 trimestres calendario sin avisar. Arreglarlo
    requiere validar la ventana por fechas y devolver NaN si no cubre ~365
    días; se documenta acá para no repetir el aviso en cada llamada. La rama de
    Valoración (compute_valuation_scores) usa `.rolling(4).sum()` inline y
    arrastra la misma deuda.
    """
    return serie.rolling(4).sum()


class FundamentalFactorEngine:
    """
    Engine encargado del procesamiento, normalización y cálculo de factores de 
    Calidad (Quality) y Valoración (Value) basados en percentiles históricos.
    """
    def __init__(self, fundamental_data: pd.DataFrame,
                 min_invested_capital_fraction: float = MIN_INVESTED_CAPITAL_FRACTION,
                 tolerancia_dias_ventana: float = TOLERANCIA_DIAS_VENTANA):
        """
        Se espera un DataFrame estructurado de tal forma que las métricas
        contables estén limpias y ordenadas por fechas.

        :param min_invested_capital_fraction: umbral de MATERIALIDAD del
            capital invertido para el ROIC (Fase 3.6, ver
            `_capital_invertido_material` y
            config.PipelineConfig.roic_min_invested_capital_fraction). Se pasa
            por constructor en vez de leerse de un global para que sea un
            parámetro operativo enhebrado desde config, como el resto.
        :param tolerancia_dias_ventana: desvío tolerado, en días, entre lo que
            una ventana TTM/YoY cubre de VERDAD y los 365 días que su nombre
            promete (Fase 3.6, ver `window_coverage`).
        """
        self.raw_fundamentals = fundamental_data
        self.min_invested_capital_fraction = float(min_invested_capital_fraction)
        self.tolerancia_dias_ventana = float(tolerancia_dias_ventana)

    def compute_quality_scores(self) -> pd.DataFrame:
        """
        Calcula las métricas de calidad y estabilidad financiera.
        """
        df = self.raw_fundamentals.copy()
        quality_df = pd.DataFrame(index=df.index)
        
        # 1. Rentabilidades (Profitability)
        # Numeradores de FLUJO en TTM (ver _ttm): los umbrales absolutos contra
        # los que se comparan (QUALITY_ABSOLUTE_THRESHOLDS) son ANUALES, así
        # que un numerador trimestral los subestimaba ~4x. Los denominadores de
        # BALANCE (equity, capital invertido) son stocks y NO se acumulan.
        ttm_net_income = _ttm(df.get('net_income', pd.Series(dtype=float)))
        ttm_ebit = _ttm(df.get('ebit', pd.Series(dtype=float)))
        ttm_ebitda = _ttm(df.get('ebitda', pd.Series(dtype=float)))
        ttm_revenue = _ttm(df.get('total_revenue', pd.Series(dtype=float)))

        quality_df['roe'] = ttm_net_income / df.get('total_equity', np.nan)

        # ROIC = NOPAT TTM / Invested Capital (stock, sin acumular).
        # La tax_rate es una TASA (no un flujo): se aplica al EBIT ya acumulado,
        # no se suma. Se toma la del trimestre más reciente de cada fila.
        nopat = ttm_ebit * (1 - df.get('tax_rate', 0.25))
        invested_capital = df.get('total_debt', 0.0) + df.get('total_equity', np.nan) - df.get('cash_and_equiv', 0.0)
        # Capital BRUTO: el mismo denominador antes de restar la caja. Es la
        # escala contra la que se juzga si lo que sobrevive a la resta es
        # material (Fase 3.6).
        gross_capital = df.get('total_debt', 0.0) + df.get('total_equity', np.nan)
        # DOS reglas sobre el mismo denominador:
        #  (1) Capital invertido <= 0 -> ROIC NaN (Fase 1.5). En una empresa
        #      net-cash (AAPL histórico, casi toda la tech) el denominador
        #      cambia de signo y el ROIC sale negativo sin que eso signifique
        #      nada sobre su rentabilidad.
        #  (2) Capital invertido < min_invested_capital_fraction del bruto ->
        #      ROIC NaN (Fase 3.6). La regla (1) sola dejaba pasar el caso
        #      ASINTÓTICO —positivo pero diminuto—, que producía un ROIC del
        #      18.300%: igual de inútil que el infinito que (1) eliminó.
        quality_df['roic'] = nopat / _capital_invertido_material(
            invested_capital, gross_capital, self.min_invested_capital_fraction,
        )

        # Márgenes: numerador Y denominador son flujos, así que ambos van en
        # TTM. Además de dejar el margen en escala anual, esto lo estabiliza
        # frente a la estacionalidad de un trimestre suelto.
        quality_df['operating_margin'] = ttm_ebit / ttm_revenue
        quality_df['net_margin'] = ttm_net_income / ttm_revenue

        # 2. Apalancamiento y Solvencia (Solvency)
        # debt_to_equity: stock / stock, no se acumula nada.
        quality_df['debt_to_equity'] = df.get('total_debt', np.nan) / df.get('total_equity', np.nan)
        # debt_to_ebitda: stock / FLUJO -> el EBITDA va en TTM. Con EBITDA
        # trimestral el ratio salía ~4x inflado ("4 años de deuda" leídos como
        # 16). Es el mismo defecto de unidades que A4 en estas mismas líneas.
        # Desde la Fase 3.3 SÍ se imprime, en la ficha de salud financiera.
        quality_df['debt_to_ebitda'] = df.get('total_debt', np.nan) / ttm_ebitda
        # interest_coverage: flujo / flujo del MISMO período — consistente en
        # trimestral, pero se pasa a TTM por coherencia con el resto y porque
        # la cobertura de intereses se cita siempre en base anual.
        quality_df['interest_coverage'] = ttm_ebit / _ttm(df.get('interest_expense', pd.Series(dtype=float)))
        
        # 3. Crecimiento ESTRUCTURAL (Growth YoY): comparar contra el MISMO
        # trimestre del año anterior (pct_change(4) sobre datos trimestrales),
        # no contra el trimestre inmediato anterior (QoQ) — un pct_change()
        # simple queda contaminado por estacionalidad (ej. Q4 vs. Q3 de
        # retail/consumo no es "crecimiento", es el ciclo normal del negocio).
        quality_df['revenue_growth'] = df.get('total_revenue', pd.Series(dtype=float)).pct_change(4, fill_method=None)
        quality_df['earnings_growth'] = df.get('net_income', pd.Series(dtype=float)).pct_change(4, fill_method=None)
        quality_df['fcf_growth'] = df.get('free_cash_flow', pd.Series(dtype=float)).pct_change(4, fill_method=None)
        
        # Reemplazar infinitos por NaN producidos por divisiones entre cero contables
        quality_df = quality_df.replace([np.inf, -np.inf], np.nan)
        return quality_df

    def compute_valuation_scores(self, current_prices: pd.Series) -> pd.DataFrame:
        """
        Calcula múltiplos de valoración alineando, para CADA fecha de reporte
        contable, el precio de mercado VIGENTE en ese momento (merge_asof hacia
        atrás sobre la serie completa de precios) — no un único precio "de hoy"
        aplicado a toda la serie histórica.

        Aplicar el precio actual a todo el histórico congela el market cap: el
        múltiplo pasa a depender solo del denominador contable, y como equity
        y beneficios normalmente CRECEN con el tiempo, el trimestre más
        reciente sale SIEMPRE como el "más barato" de su historia (percentil
        más bajo de PER/EV-EBITDA) sin importar cuál sea el precio real hoy —
        el score de valoración queda ciego al precio.
        """
        df = self.raw_fundamentals.copy().sort_index()
        valuation_df = pd.DataFrame(index=df.index)

        prices = current_prices.dropna().sort_index()
        if df.empty or prices.empty:
            for col in ('per', 'price_to_book', 'ev_to_ebitda', 'price_to_sales', 'dividend_yield'):
                valuation_df[col] = np.nan
            return valuation_df

        report_dates = pd.DataFrame({"date": df.index})
        price_lookup = prices.rename("price").rename_axis("date").reset_index()
        merged = pd.merge_asof(report_dates, price_lookup, on="date", direction="backward")
        # merge_asof preserva el orden de filas del lado izquierdo (report_dates,
        # ya ordenado como df.index), así que .values se alinea 1:1 con df.index.
        price_at_report_date = pd.Series(merged["price"].values, index=df.index)

        shares_outstanding = df.get('shares_outstanding', np.nan)
        market_cap = price_at_report_date * shares_outstanding
        enterprise_value = market_cap + df.get('total_debt', 0.0) - df.get('cash_and_equiv', 0.0)

        # Beneficios TTM (suma móvil de los últimos 4 trimestres): comparar un
        # market cap (una foto instantánea) contra un solo trimestre de
        # resultados infla los múltiplos ~4x frente al estándar TTM del mercado.
        ttm_net_income = df.get('net_income', pd.Series(dtype=float)).rolling(4).sum()
        ttm_ebitda = df.get('ebitda', pd.Series(dtype=float)).rolling(4).sum()
        ttm_revenue = df.get('total_revenue', pd.Series(dtype=float)).rolling(4).sum()
        ttm_dividends = df.get('total_dividends', pd.Series(dtype=float)).rolling(4).sum()

        # Denominadores <= 0 -> múltiplo NaN, NUNCA un número (ver
        # _positivo_o_nan): con `1.0 - rank(pct)`, un PER negativo por pérdidas
        # TTM salía como el percentil más bajo y por tanto como la valoración
        # MÁS atractiva de la historia de la empresa. Aplica a beneficio TTM
        # (PER), equity (P/B) y EBITDA TTM (EV/EBITDA).
        valuation_df['per'] = market_cap / _positivo_o_nan(ttm_net_income)
        valuation_df['price_to_book'] = market_cap / _positivo_o_nan(df.get('total_equity', pd.Series(dtype=float)))
        valuation_df['ev_to_ebitda'] = enterprise_value / _positivo_o_nan(ttm_ebitda)
        # Revenue negativo no existe en la práctica, pero cero sí (empresa
        # pre-ingresos) y dividiría por cero.
        valuation_df['price_to_sales'] = market_cap / _positivo_o_nan(ttm_revenue)
        # dividend_yield lleva el market cap en el DENOMINADOR: un market cap
        # <= 0 no tiene sentido (implicaría acciones o precio negativos).
        valuation_df['dividend_yield'] = ttm_dividends / _positivo_o_nan(market_cap)

        valuation_df = valuation_df.replace([np.inf, -np.inf], np.nan)
        return valuation_df
    
    # ------------------------------------------------------------------
    # FICHA DESCRIPTIVA (Fase 3.3) — reemplaza a score_quality/score_valuation
    # ------------------------------------------------------------------
    # `score_quality` y `score_valuation` SE ELIMINARON. Los dos comprimían
    # varias dimensiones en un número 0-1 que después entraba en el score
    # multifactorial, y `score_valuation` además invertía el percentil
    # (`1.0 - rank`) para que "más fuera mejor". Ninguna de las dos cosas es
    # verificable: "Valoración: 0,72" no dice qué se midió ni sobre cuántos
    # trimestres. Lo que queda es lo que sí se puede comprobar — el valor de
    # hoy, su percentil dentro del histórico de la propia empresa, y el n que
    # sostiene ese percentil. La lectura de la valoración vive en
    # `EntryContextEngine.valuation_context`, que consume
    # `compute_valuation_scores` sin puntuarlo.

    # Métricas de la ficha de SALUD FINANCIERA, en el orden en que se
    # imprimen. La tupla es (clave, etiqueta, unidad): la unidad viaja con la
    # métrica porque un margen se lee en % y una cobertura de intereses en
    # veces, y equivocarlo es el tipo de error de unidades que la Fase 1 tuvo
    # que arreglar dos veces.
    # (clave, etiqueta, unidad, ventana). `ventana` dice de qué tipo de
    # ventana depende el valor, para poder publicar su cobertura REAL al lado
    # (Fase 3.6): "ttm" si acumula 4 filas, "yoy" si compara con 4 filas atrás,
    # "stock" si es un saldo de balance a una fecha y no depende de ventana
    # ninguna.
    DESCRIPTIVE_METRICS = (
        ("roe", "ROE (TTM)", "pct", "ttm"),
        ("roic", "ROIC (TTM)", "pct", "ttm"),
        ("operating_margin", "Margen operativo (TTM)", "pct", "ttm"),
        ("net_margin", "Margen neto (TTM)", "pct", "ttm"),
        ("debt_to_equity", "Deuda / Equity", "ratio", "stock"),
        ("debt_to_ebitda", "Deuda / EBITDA (TTM)", "ratio", "ttm"),
        ("interest_coverage", "Cobertura de intereses", "ratio", "ttm"),
        ("revenue_growth", "Crecimiento ingresos YoY", "pct", "yoy"),
        ("earnings_growth", "Crecimiento beneficio YoY", "pct", "yoy"),
        ("fcf_growth", "Crecimiento FCF YoY", "pct", "yoy"),
    )

    # Columnas que deciden si la ficha tiene datos suficientes para que sus
    # percentiles signifiquen algo. Mismas cuatro que usaba `score_quality`
    # para su `n_quarters`, para que el criterio de fiabilidad no cambie de
    # vara al cambiar de presentación.
    RELIABILITY_COLUMNS = ("roe", "roic", "net_margin", "debt_to_equity")

    def window_coverage(self) -> pd.DataFrame:
        """
        Cobertura REAL, en días, de las ventanas TTM y YoY de cada fila, con su
        flag de validez contra los 365 días esperados (Fase 3.6).

        Columnas: dias_ttm, ttm_valida, dias_yoy, yoy_valida.

        Es la respuesta al residuo "TTM y YoY cuentan filas, no trimestres".
        No lo arregla —eso exigiría reindexar a trimestres fiscales, que el
        plan deja como opcional— pero lo hace VISIBLE, que es lo que no era
        opcional: sin esto, un "ROE TTM" calculado sobre 15 meses de actividad
        se publica exactamente igual que uno calculado sobre 12.
        """
        idx = self.raw_fundamentals.index
        dias_ttm = cobertura_ventana_ttm(idx)
        dias_yoy = cobertura_ventana_yoy(idx)
        return pd.DataFrame({
            "dias_ttm": dias_ttm,
            "ttm_valida": ventana_cubre_un_anio(dias_ttm, self.tolerancia_dias_ventana),
            "dias_yoy": dias_yoy,
            "yoy_valida": ventana_cubre_un_anio(dias_yoy, self.tolerancia_dias_ventana),
        }, index=idx)

    def describe_quality(self) -> Dict[str, Any]:
        """
        FICHA DESCRIPTIVA de salud financiera: por métrica, el valor del último
        trimestre disponible, su PERCENTIL dentro del histórico de la propia
        empresa, cuántas observaciones sostienen ese percentil, y la variación
        frente al mismo trimestre del año anterior (4 filas atrás).

        NO PUNTÚA NADA y no devuelve ninguna etiqueta. El percentil se reporta
        en su orientación NATURAL (percentil alto = valor alto), sin invertirlo
        para las métricas donde "menos es mejor" como Deuda/Equity: invertir el
        signo de unas y no de otras es lo que obligaba a leer una tabla con dos
        convenciones a la vez. Un D/E en su percentil 0,95 significa "la deuda
        está en su nivel más alto del histórico", y eso se lee sin ayuda.

        `reliable` es False cuando hay menos de MIN_QUARTERS_FOR_RELIABILITY
        trimestres: los valores siguen siendo válidos (son el dato contable),
        pero los PERCENTILES no significan gran cosa sobre 5 observaciones, y
        el informe tiene que decirlo en vez de imprimirlos igual.

        La variación YoY (`delta_4t`) es la "tendencia" del bloque: se compara
        contra el MISMO trimestre del año anterior, no contra el trimestre
        inmediato anterior, por la misma razón que `compute_quality_scores` usa
        `pct_change(4)` — un QoQ está contaminado por estacionalidad.
        """
        q_df = self.compute_quality_scores()
        cols = [c for c in self.RELIABILITY_COLUMNS if c in q_df.columns]
        n_quarters = int(q_df[cols].dropna(how="all").shape[0]) if (not q_df.empty and cols) else 0

        if q_df.empty or n_quarters == 0:
            logger.warning("Sin datos fundamentales de calidad disponibles; la ficha descriptiva "
                           "queda vacía y marcada como no fiable.")
            return {"metricas": {}, "n_quarters": 0, "reliable": False}

        # Cobertura REAL de las ventanas TTM/YoY (Fase 3.6): cada métrica
        # publica, junto a su valor, cuántos días abarca de verdad la ventana
        # de la que sale. Sin esto un "ROE TTM" de 15 meses se imprime igual
        # que uno de 12.
        cobertura = self.window_coverage()

        metricas: Dict[str, Any] = {}
        for clave, etiqueta, unidad, ventana in self.DESCRIPTIVE_METRICS:
            if clave not in q_df.columns:
                continue
            col_completa = q_df[clave]
            serie = col_completa.dropna()
            if serie.empty:
                metricas[clave] = {"etiqueta": etiqueta, "unidad": unidad, "ventana": ventana,
                                   "actual": float("nan"), "percentil": float("nan"), "n_obs": 0,
                                   "mediana_historica": float("nan"), "delta_4t": float("nan"),
                                   "ventana_dias": float("nan"), "ventana_valida": None,
                                   "n_ventanas_validas": 0}
                continue

            actual = float(serie.iloc[-1])
            # Percentil del valor de hoy dentro de su propio histórico, mismo
            # cálculo que EntryContextEngine.valuation_context para que las dos
            # fichas del informe hablen la misma lengua.
            percentil = float((serie <= actual).mean())
            # delta_4t sobre la serie COMPLETA (no la limpia): 4 filas atrás
            # tiene que ser el trimestre de hace un año, y saltarse los NaN
            # movería la referencia sin avisar.
            pos_actual = col_completa.index.get_loc(serie.index[-1])
            delta = float("nan")
            if isinstance(pos_actual, int) and pos_actual >= 4:
                anterior = col_completa.iloc[pos_actual - 4]
                if pd.notna(anterior):
                    delta = float(actual - anterior)

            # Cobertura de la ventana del ÚLTIMO valor (el que se imprime) y
            # cuántas de las observaciones que sostienen el percentil salen de
            # una ventana que sí cubre un año. Un "stock" de balance no depende
            # de ventana: es un saldo a una fecha.
            if ventana == "stock":
                ventana_dias, ventana_valida, n_validas = float("nan"), None, int(len(serie))
            else:
                col_dias = cobertura[f"dias_{ventana}"]
                col_ok = cobertura[f"{ventana}_valida"]
                fecha_actual = serie.index[-1]
                ventana_dias = float(col_dias.get(fecha_actual, float("nan")))
                valida_actual = col_ok.get(fecha_actual, False)
                ventana_valida = bool(valida_actual) if pd.notna(valida_actual) else None
                n_validas = int(col_ok.reindex(serie.index).fillna(False).sum())

            metricas[clave] = {
                "etiqueta": etiqueta,
                "unidad": unidad,
                "ventana": ventana,
                "actual": actual,
                "percentil": percentil,
                "n_obs": int(len(serie)),
                "mediana_historica": float(serie.median()),
                "delta_4t": delta,
                "ventana_dias": ventana_dias,
                "ventana_valida": ventana_valida,
                "n_ventanas_validas": n_validas,
            }

        reliable = n_quarters >= MIN_QUARTERS_FOR_RELIABILITY
        if not reliable:
            logger.warning(f"Solo {n_quarters} trimestres de calidad disponibles (minimo "
                           f"{MIN_QUARTERS_FOR_RELIABILITY}); los percentiles de la ficha "
                           f"descriptiva no son interpretables.")

        # Resumen agregado de la cobertura de ventanas, para que el informe
        # pueda decir de una vez "36 de 48 ventanas TTM no cubren 12 meses"
        # en vez de obligar a mirar métrica por métrica.
        resumen_ventanas: Dict[str, Any] = {}
        for tipo in ("ttm", "yoy"):
            dias = cobertura[f"dias_{tipo}"].dropna()
            ok = cobertura[f"{tipo}_valida"].reindex(dias.index).fillna(False)
            resumen_ventanas[tipo] = {
                "n_obs": int(len(dias)),
                "n_validas": int(ok.sum()),
                "n_invalidas": int((~ok.astype(bool)).sum()),
                "mediana_dias": float(dias.median()) if not dias.empty else float("nan"),
                "peor_dias": float(dias.iloc[(dias - DIAS_ANIO_CONTABLE).abs().argmax()])
                              if not dias.empty else float("nan"),
                "esperado_dias": DIAS_ANIO_CONTABLE,
                "tolerancia_dias": self.tolerancia_dias_ventana,
            }
        if resumen_ventanas["ttm"]["n_invalidas"]:
            logger.warning(
                f"{resumen_ventanas['ttm']['n_invalidas']} de "
                f"{resumen_ventanas['ttm']['n_obs']} ventanas TTM no cubren "
                f"{DIAS_ANIO_CONTABLE} +/- {self.tolerancia_dias_ventana:.0f} días "
                f"(mediana real: {resumen_ventanas['ttm']['mediana_dias']:.0f} días): con huecos "
                f"en los datos contables, `.rolling(4)` acumula trimestres salteados."
            )

        return {"metricas": metricas, "n_quarters": n_quarters, "reliable": reliable,
                "ventanas": resumen_ventanas}

    def net_margin_ttm_window_valid(self) -> pd.Series:
        """
        Máscara booleana de si la ventana TTM de cada trimestre del margen
        neto cubre de verdad 12 meses.

        La consume `portfolio_engine` antes de contar trimestres consecutivos
        de caída: una "caída del margen TTM" entre dos ventanas que abarcan 12
        y 18 meses respectivamente no es una caída del negocio, es un cambio
        de la ventana.
        """
        cobertura = self.window_coverage()
        serie = self.net_margin_ttm()
        if serie.empty or cobertura.empty:
            return pd.Series(dtype=bool)
        return cobertura["ttm_valida"].reindex(serie.index).fillna(False).astype(bool)

    def net_margin_ttm(self) -> pd.Series:
        """
        Serie del margen neto TTM por trimestre, sin puntuar.

        La consume `portfolio_engine` para el tercer disparador de "REVISAR"
        (margen neto TTM cayendo N trimestres seguidos). Se expone como método
        propio en vez de que el llamador se cuele en `compute_quality_scores` y
        elija una columna a mano.
        """
        q_df = self.compute_quality_scores()
        if q_df.empty or "net_margin" not in q_df.columns:
            return pd.Series(dtype=float)
        return q_df["net_margin"].dropna()
