"""
Motor de CONTEXTO DE ENTRADA (PLAN_REFACTOR.md 3.1).

Responde la única pregunta que una sola acción sí permite contestar:
**"¿a qué me apunto si compro hoy?"** — no "¿va a subir?".

Es la diferencia entre una afirmación PREDICTIVA (que la Fase 2 demostró que
este sistema no puede sostener: IC medio -0,0093 con el signo repartido 5-5 en
el único horizonte con potencia estadística) y una afirmación DESCRIPTIVA sobre
la distribución histórica de experiencias de entrada, que no requiere que el
modelo prediga nada.

QUÉ CALCULA
  (a) Distribución del DRAWDOWN DE ENTRADA: para cada fecha histórica t y cada
      horizonte H, cuánto llegó a caer el precio por debajo de lo que se pagó.
      Es la métrica más útil del sistema nuevo: determina si aguantas la
      posición o vendes en pánico, y ninguna herramienta retail la da.
  (b) Distribución del TIEMPO HASTA RECUPERACIÓN: desde t, cuántos días hasta
      volver al precio de entrada.
  (d) Contexto de VALORACIÓN point-in-time: el múltiplo de hoy frente al
      histórico de la propia empresa. No es un factor predictivo reetiquetado
      como contexto: es la misma medición respondiendo la pregunta legítima
      ("¿estoy pagando caro respecto a como ha cotizado esta empresa?") en vez
      de la ilegítima ("¿va a subir?").

(c) — ANÁLOGOS CONDICIONALES: DESCARTADO POR DECISIÓN DEL PROPIETARIO (2026-09).
El plan original contemplaba repetir (a) y (b) restringiendo a fechas cuyo
estado se pareciera al de hoy (percentil de volatilidad condicional, caída
desde máximos). NO se implementa, y no debe implementarse sin una decisión
nueva: con 12 años y ventanas de 126 días, el número de análogos
INDEPENDIENTES cae casi siempre por debajo del mínimo fiable, y es exactamente
la puerta por la que el sobreajuste volvería a entrar — justo después de haber
echado la capa ML por esa misma razón. Las distribuciones de (a) y (b) son
INCONDICIONALES, sobre todo el histórico disponible.

PRINCIPIO TRANSVERSAL: TODO NÚMERO VIAJA CON SU TAMAÑO MUESTRAL.
Cada distribución devuelve `n_obs`, `n_independiente` (= n_obs / horizonte, la
misma corrección por solapamiento que `run_holding_period_backtest`) y un flag
`fiable`. Dos fechas separadas por una semana comparten casi toda su ventana de
6 meses: no son dos observaciones. Sin esta corrección, "el drawdown mediano a
24 meses es del -18%" suena a dato y descansa en media docena de ventanas
reales. Es el criterio de aceptación de la Fase 3: ningún número sin su muestra
al lado.
"""
import logging
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger("QuantSystem.EntryContextEngine")

# Abanico de horizontes del informe nuevo, en MESES. Sustituye al horizonte
# único de 126 días sobre el que estaba anclado todo el sistema anterior: la
# estrategia real del propietario no tiene plazo de mantenimiento fijo (vende
# de forma discrecional), así que un solo horizonte era una suposición, no un
# dato.
HORIZONTES_MESES = (1, 3, 6, 12, 24)

# Sesiones por mes. Es el DEFAULT de acciones (252/12); la Fase 3.6 lo
# convirtió en un parámetro que viene de `config.trading_days_per_month`
# (21 en acciones, 30,42 en cripto), porque con el literal 21 un "horizonte de
# 24 meses" son 504 sesiones tanto para una acción —24 meses de calendario—
# como para BTC, donde son 16 meses. El default se mantiene para que llamar a
# este motor sin config siga dando exactamente lo de antes.
# DIAS_POR_ANIO se eliminó: era un literal que nadie leía, y el único sitio que
# necesita las sesiones por AÑO es config.trading_days_per_year.
DIAS_POR_MES = 21

# Percentiles que se reportan de cada distribución. No se reporta el mínimo ni
# el máximo: son el peor y el mejor caso de UNA realización histórica, y se
# leerían como cotas cuando son solo el extremo de la muestra que tocó.
PERCENTILES = (5, 25, 50, 75, 95)

# Por debajo de estas observaciones INDEPENDIENTES, la distribución se marca
# como no fiable. Mismo número y mismo criterio que
# MIN_INDEPENDENT_SIGNALS_FOR_CONCLUSIVE en scoring_backtest_engine: no tiene
# sentido exigir 10 ventanas independientes al event study y conformarse con 3
# acá.
MIN_OBS_INDEPENDIENTES = 10


def percentil_historico(serie: pd.Series) -> Dict[str, Any]:
    """
    Valor de HOY de una serie, su percentil dentro de su propio histórico, y el
    n que sostiene ese percentil.

    Es el patrón de presentación de toda la Fase 3: en vez de comprimir una
    dimensión en un score, se dice cuál es el valor, dónde cae respecto de su
    propia historia, y sobre cuántas observaciones se calculó eso. El percentil
    va en su orientación NATURAL (alto = valor alto), sin invertirse para las
    métricas donde "menos es mejor": invertir unas y no otras obliga a leer una
    misma tabla con dos convenciones.

    Se usa desde el informe (percentil de la volatilidad condicional) y desde
    el modo cartera (mismo percentil, para detectar un cambio de régimen frente
    al primer registro de la posición), así que vive acá y no duplicado en los
    dos sitios.

    `n_obs` es el número CRUDO de observaciones, sin corrección por
    solapamiento: a diferencia del drawdown o del tiempo de recuperación, un
    percentil no mide ventanas hacia adelante, así que no hay solapamiento que
    corregir — cada día aporta su propio valor de volatilidad condicional.
    """
    limpia = serie.dropna() if isinstance(serie, pd.Series) else pd.Series(dtype=float)
    if limpia.empty:
        return {"actual": float("nan"), "percentil": float("nan"), "n_obs": 0,
                "mediana_historica": float("nan")}
    actual = float(limpia.iloc[-1])
    return {
        "actual": actual,
        "percentil": float((limpia <= actual).mean()),
        "n_obs": int(len(limpia)),
        "mediana_historica": float(limpia.median()),
    }


def horizontes_en_dias(horizontes_meses: Sequence[int] = HORIZONTES_MESES,
                        dias_por_mes: float = DIAS_POR_MES) -> Dict[int, int]:
    """
    Mapea meses -> SESIONES, con la convención del resto del sistema.

    `dias_por_mes` viene de `config.trading_days_per_month` (Fase 3.6): 21 para
    acciones, 30,42 para un activo que cotiza los 365 días. Se redondea al
    entero porque un horizonte se mide en filas del panel de precios.
    """
    return {m: int(round(m * dias_por_mes)) for m in horizontes_meses}


def _describir(valores: pd.Series, horizonte_dias: int, unidad: str) -> Dict[str, Any]:
    """
    Resume una distribución con sus percentiles Y su tamaño muestral efectivo.

    `n_independiente` divide por el horizonte porque las ventanas de fechas
    consecutivas se solapan casi por completo: 500 fechas con ventanas de 126
    días contienen ~4 experiencias independientes, no 500.
    """
    limpio = valores.dropna()
    n = int(len(limpio))
    n_indep = n / horizonte_dias if horizonte_dias > 0 else float(n)

    base: Dict[str, Any] = {
        "n_obs": n,
        "n_independiente": float(n_indep),
        "fiable": bool(n_indep >= MIN_OBS_INDEPENDIENTES),
        "unidad": unidad,
    }
    if n == 0:
        base.update({f"p{p}": float("nan") for p in PERCENTILES})
        base["media"] = float("nan")
        return base

    cuantiles = limpio.quantile([p / 100 for p in PERCENTILES])
    base.update({f"p{p}": float(cuantiles.loc[p / 100]) for p in PERCENTILES})
    base["media"] = float(limpio.mean())
    return base


#: Sentinela para "no ocurrió dentro de la ventana examinada". No se usa NaN
#: porque estos arrays son de enteros y NaN forzaría a float, y no se usa un
#: número grande arbitrario porque contaminaría cualquier percentil que lo
#: incluyera por error.
NO_OCURRE = -1


def _offsets_hundimiento_y_recuperacion(precios: np.ndarray, max_dias: int) -> Dict[str, np.ndarray]:
    """
    Para cada fecha de compra t, calcula dos desplazamientos (en días desde t),
    mirando como máximo `max_dias` hacia adelante:

      - `hundimiento[t]`: primer j > t con P[j] < P[t]. Es decir, cuándo la
        posición se pone en pérdidas por primera vez. NO_OCURRE si nunca pasa
        dentro de la ventana.
      - `recuperacion[t]`: primer j > hundimiento[t] con P[j] >= P[t]. Es
        decir, cuándo se vuelve al precio pagado DESPUÉS de haberse hundido.
        NO_OCURRE si no vuelve dentro de la ventana (o si nunca se hundió).

    POR QUÉ NO ES SIMPLEMENTE "primer j > t con P[j] >= P[t]". Esa es la
    lectura literal de "tiempo hasta recuperación", y no mide lo que dice: si
    el precio sube el día siguiente a la compra, devuelve 1 día INCLUSO SI
    después se hunde un 30% y tarda dos años en volver. En un activo alcista da
    1 día en casi todas las fechas y la mediana resultante no informa de nada.
    Lo detectaron los propios tests de esta sub-fase.

    Lo que interesa, y lo que se lee junto a la distribución de drawdown, es:
    "si compro y la posición se pone en pérdidas, cuánto tardo en volver a
    estar en paz". Eso requiere localizar primero el hundimiento y buscar la
    recuperación DESPUÉS de él.

    Se calcula UNA vez con el horizonte más largo y los horizontes más cortos
    se derivan comparando estos desplazamientos con sus propios límites, en vez
    de repetir el barrido cinco veces.
    """
    n = len(precios)
    hundimiento = np.full(n, NO_OCURRE, dtype=np.int64)
    recuperacion = np.full(n, NO_OCURRE, dtype=np.int64)

    for t in range(n - 1):
        fin = min(n, t + 1 + max_dias)
        ventana = precios[t + 1:fin]
        if ventana.size == 0:
            continue

        p_entrada = precios[t]
        bajo_agua = ventana < p_entrada
        if not bajo_agua.any():
            continue

        # +1 porque `ventana` arranca en t+1.
        off_hund = int(bajo_agua.argmax()) + 1
        hundimiento[t] = off_hund

        resto = ventana[off_hund - 1:]
        de_vuelta = resto >= p_entrada
        if de_vuelta.any():
            recuperacion[t] = off_hund + int(de_vuelta.argmax())

    return {"hundimiento": hundimiento, "recuperacion": recuperacion}


class EntryContextEngine:
    """
    Contexto de entrada de un activo: qué experiencias históricas produjo
    comprarlo en una fecha cualquiera y aguantar H días.

    NO predice. Describe la distribución de lo que pasó, que es una afirmación
    verificable sobre el pasado y no una apuesta sobre el futuro.
    """

    def __init__(self, price_series: pd.Series,
                 horizontes_meses: Sequence[int] = HORIZONTES_MESES,
                 dias_por_mes: float = DIAS_POR_MES):
        self.precios = price_series.dropna().astype(float)
        self.horizontes_meses = list(horizontes_meses)
        # dias_por_mes (Fase 3.6): sesiones por mes del activo, para que "6
        # meses" signifique seis meses de CALENDARIO tanto en una acción como
        # en cripto. Ver config.PipelineConfig.trading_days_per_month.
        self.dias_por_mes = dias_por_mes
        self.horizontes_dias = horizontes_en_dias(self.horizontes_meses, dias_por_mes)

        if self.precios.empty:
            logger.warning("Serie de precios vacía: el contexto de entrada no será evaluable.")

        # Se calculan una vez con el horizonte más largo; los más cortos se
        # derivan comparando los desplazamientos con su propio límite.
        max_dias = max(self.horizontes_dias.values()) if self.horizontes_dias else 0
        if self.precios.empty or max_dias == 0:
            vacio = np.array([], dtype=np.int64)
            self._offsets = {"hundimiento": vacio, "recuperacion": vacio}
        else:
            self._offsets = _offsets_hundimiento_y_recuperacion(
                self.precios.to_numpy(), max_dias,
            )

    # ------------------------------------------------------------------
    # (a) Distribución del drawdown de entrada
    # ------------------------------------------------------------------

    def drawdown_distribution(self) -> Dict[int, Dict[str, Any]]:
        """
        Para cada horizonte H: distribución de `min(P[t..t+H]) / P[t] - 1`.

        El mínimo INCLUYE la fecha de compra t, así que el drawdown está
        acotado en 0 por arriba: si el precio nunca baja del precio pagado, la
        respuesta correcta es "nunca estuviste en pérdidas" (0%), no un número
        positivo. Excluir t daría un "drawdown" positivo en las entradas que
        solo subieron, que sería una ganancia disfrazada de caída.

        Solo se usan las fechas con la ventana COMPLETA observada (se descartan
        las últimas H): de las demás no se sabe cuánto habrían caído, y
        rellenarlas con lo poco que se ve sesgaría la distribución hacia
        drawdowns pequeños justo en el tramo más reciente.
        """
        resultado: Dict[int, Dict[str, Any]] = {}
        for meses, dias in self.horizontes_dias.items():
            if len(self.precios) <= dias:
                resultado[meses] = _describir(pd.Series(dtype=float), dias, "retorno")
                resultado[meses]["horizonte_dias"] = dias
                continue

            # rolling(dias+1).min() en la fila t+dias abarca [t, t+dias];
            # shift(-dias) lo trae a la fila t.
            min_ventana = self.precios.rolling(dias + 1).min().shift(-dias)
            drawdown = (min_ventana / self.precios) - 1.0

            resumen = _describir(drawdown, dias, "retorno")
            resumen["horizonte_dias"] = dias
            resultado[meses] = resumen
        return resultado

    # ------------------------------------------------------------------
    # (b) Distribución del tiempo hasta recuperación
    # ------------------------------------------------------------------

    def recovery_time_distribution(self) -> Dict[int, Dict[str, Any]]:
        """
        Para cada horizonte H: cuántos días desde la compra hasta volver al
        precio pagado DESPUÉS de haberse puesto en pérdidas, y qué porcentaje
        de las entradas que se hunden no vuelve dentro de H.

        LA DISTRIBUCIÓN ESTÁ RESTRINGIDA A LAS ENTRADAS QUE SE HUNDEN, y eso es
        deliberado: en una entrada que nunca cotizó por debajo de lo que se
        pagó no hay nada que recuperar, y meterla en la muestra con un 0
        arrastraría la mediana a 0 en cualquier activo alcista. Se reporta
        aparte cuántas entradas nunca se hundieron (`pct_nunca_bajo_agua`),
        que es información por sí misma.

        ESTO NO ES UN FILTRO DE SIMILITUD DE LOS QUE 3.1c DESCARTA. Ahí el
        problema era condicionar en el ESTADO DE HOY (volatilidad parecida,
        caída desde máximos parecida): eso selecciona un puñado de fechas por
        parecerse al presente y reintroduce sobreajuste. Acá se condiciona en
        que la métrica esté DEFINIDA —no se puede medir "cuánto tardó en
        recuperarse" en una entrada que nunca perdió nada—, es una condición
        sobre el propio suceso y no mira el presente en absoluto. La
        distribución sigue siendo INCONDICIONAL respecto al estado actual: usa
        todas las fechas del histórico que cumplen la condición.

        Las entradas que se hunden y NO vuelven dentro de H se excluyen de los
        percentiles (no se les asigna un número grande inventado, que
        contaminaría la mediana) y se reportan en `pct_no_recupera`, que es la
        cifra honesta y a menudo la más importante del bloque.
        """
        resultado: Dict[int, Dict[str, Any]] = {}
        n = len(self.precios)

        for meses, dias in self.horizontes_dias.items():
            if n <= dias:
                vacio = _describir(pd.Series(dtype=float), dias, "dias")
                resultado[meses] = {
                    "horizonte_dias": dias,
                    "recuperacion": vacio,
                    "pct_no_recupera": float("nan"),
                    "pct_nunca_bajo_agua": float("nan"),
                    "n_bajo_agua": 0,
                    "n_evaluables": 0,
                }
                continue

            # Mismo tramo evaluable que el drawdown (fechas con la ventana
            # completa observada), para que ambos bloques hablen de lo mismo.
            n_evaluables = n - dias
            idx = self.precios.index[:n_evaluables]

            off_hund = self._offsets["hundimiento"][:n_evaluables]
            off_rec = self._offsets["recuperacion"][:n_evaluables]

            # Se hundió DENTRO de este horizonte.
            bajo_agua = (off_hund != NO_OCURRE) & (off_hund <= dias)
            # Y volvió al precio de entrada, también dentro del horizonte.
            recupero = bajo_agua & (off_rec != NO_OCURRE) & (off_rec <= dias)

            dias_recuperacion = pd.Series(
                np.where(recupero, off_rec, np.nan), index=idx, dtype=float,
            )

            n_bajo_agua = int(bajo_agua.sum())
            resumen = _describir(dias_recuperacion.dropna(), dias, "dias")

            resultado[meses] = {
                "horizonte_dias": dias,
                "recuperacion": resumen,
                # De las que se hundieron, cuántas no volvieron dentro de H.
                "pct_no_recupera": (float((bajo_agua & ~recupero).sum() / n_bajo_agua)
                                     if n_bajo_agua else float("nan")),
                # Cuántas entradas nunca llegaron a estar en pérdidas.
                "pct_nunca_bajo_agua": float((~bajo_agua).mean()) if n_evaluables else float("nan"),
                "n_bajo_agua": n_bajo_agua,
                "n_evaluables": int(n_evaluables),
            }
        return resultado

    # ------------------------------------------------------------------
    # (d) Contexto de valoración point-in-time
    # ------------------------------------------------------------------

    @staticmethod
    def valuation_context(valuation_df: pd.DataFrame,
                           n_quarters_reliable: Optional[int] = None) -> Dict[str, Any]:
        """
        Reetiqueta los múltiplos de `FundamentalFactorEngine.compute_valuation_scores`
        como CONTEXTO DE ENTRADA en vez de factor predictivo.

        Para cada múltiplo devuelve su valor de hoy, su PERCENTIL dentro del
        histórico point-in-time de la propia empresa, y cuántos trimestres
        sostienen ese percentil. Deliberadamente NO produce ningún score
        compuesto ni ninguna etiqueta ("caro"/"barato"): un percentil con su n
        al lado es una afirmación verificable; "Valoración: 72/100" no lo es.

        El percentil se reporta en su orientación NATURAL (percentil alto =
        múltiplo alto = caro), sin el `1.0 - rank` que usaba `score_valuation`
        para convertirlo en "más es mejor". Esa inversión tenía sentido dentro
        de un score donde todo debía apuntar en la misma dirección; leído como
        contexto, invertirlo solo confunde.

        No modifica `fundamental_engine`: lo consume. Los múltiplos siguen
        calculándose exactamente igual, incluida la alineación point-in-time
        del precio por fecha de reporte y la anulación de denominadores <= 0.
        """
        multiplos = ("per", "price_to_book", "ev_to_ebitda", "price_to_sales", "dividend_yield")
        contexto: Dict[str, Any] = {}

        if valuation_df is None or valuation_df.empty:
            return {"multiplos": {}, "n_quarters": 0, "fiable": False}

        for col in multiplos:
            if col not in valuation_df.columns:
                continue
            serie = valuation_df[col].dropna()
            if serie.empty:
                contexto[col] = {"actual": float("nan"), "percentil": float("nan"),
                                  "n_obs": 0, "mediana_historica": float("nan")}
                continue

            actual = float(serie.iloc[-1])
            # Percentil del valor de hoy dentro de su propio histórico.
            percentil = float((serie <= actual).mean())
            contexto[col] = {
                "actual": actual,
                "percentil": percentil,
                "n_obs": int(len(serie)),
                "mediana_historica": float(serie.median()),
            }

        n_quarters = int(valuation_df.dropna(how="all").shape[0])
        fiable = (n_quarters_reliable is not None and n_quarters >= n_quarters_reliable)

        return {"multiplos": contexto, "n_quarters": n_quarters, "fiable": bool(fiable)}

    # ------------------------------------------------------------------

    def build_full_context(self) -> Dict[str, Any]:
        """
        Empaqueta (a) y (b) para todos los horizontes en una sola estructura,
        que es lo que consumirá el informe nuevo (Fase 3.4).
        """
        return {
            "horizontes_meses": list(self.horizontes_meses),
            "horizontes_dias": dict(self.horizontes_dias),
            "n_sesiones": int(len(self.precios)),
            "drawdown": self.drawdown_distribution(),
            "recuperacion": self.recovery_time_distribution(),
        }
