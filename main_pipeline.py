import datetime
import sys
import pandas as pd
import numpy as np
import yfinance as yf
from data_engine import DataEngine, last_closed_session
from technical_engine import TechnicalFactorEngine
from time_series_features import TimeSeriesFeatureExtractor
from risk_simulation_engine import RiskSimulationEngine
from fundamental_engine import FundamentalFactorEngine, MIN_QUARTERS_FOR_RELIABILITY
from quantile_forecast_engine import QuantileFanForecaster
from entry_context_engine import (
    EntryContextEngine,
    MIN_OBS_INDEPENDIENTES,
    horizontes_en_dias,
    percentil_historico,
)
from visualization_engine import ReportVisualizer
from portfolio_engine import load_positions, evaluate_position
from prediction_log import registrar_prediccion
from config import PipelineConfig, parse_args, MIN_LOOKBACK_YEARS_FOR_INTERPRETABLE_BACKTEST
from typing import Any, Dict, List, Optional

# python main_pipeline.py


def _forzar_salida_utf8() -> None:
    """
    Fuerza UTF-8 en stdout/stderr.

    En Windows, si la salida NO es una consola —redirigida a un fichero
    (`python main_pipeline.py > informe.txt`) o canalizada a otro comando—
    Python usa cp1252, que no sabe codificar los caracteres que el informe
    imprime a lo largo de todo su recorrido (█ del banner, ⚠ de las
    advertencias, ─ de los separadores). El resultado era un
    UnicodeEncodeError que abortaba la corrida ENTERA justo al empezar a
    imprimir el informe — es decir, después de haber pagado los
    walk-forwards de GARCH/ARIMA, el Monte Carlo y los abanicos cuantílicos.
    Todo el cálculo, tirado por la codificación de la terminal.

    Peor: el manejador de errores del bucle del menú (ver run_menu_loop)
    también imprime ⚠, así que cuando el fallo ERA de codificación el propio
    manejador volvía a fallar y el traceback se escapaba igual, anulando la
    degradación que la Fase 1.7 introdujo. Por eso esto se arregla en la
    raíz y no envolviendo prints en try/except.

    Se llama al importar el módulo. Los `except` cubren los casos en los que
    stdout no es un stream reconfigurable (pytest lo sustituye por su propio
    objeto de captura, por ejemplo), donde además no hace falta.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass


_forzar_salida_utf8()


# Cuantiles normales de las bandas 80/90/95%. Extraído a constante en la Fase
# 3.2 para que la versión de horizonte único y la de abanico no puedan
# divergir en los z que usan.
_Z_SCORES_BANDAS = {"80": 1.282, "90": 1.645, "95": 1.960}

# Ancho de la primera columna de las tablas del informe (la etiqueta de fila).
_ANCHO_ETIQUETA = 12


def lognormal_price_fan(current_price: float, mu_anual_log: float, vol_diaria: float,
                         horizons_days: Dict[int, int],
                         trading_days_per_year: int = 252) -> Dict[str, Any]:
    """
    ABANICO lognormal: las bandas 80/90/95% para VARIOS horizontes a la vez
    (PLAN_REFACTOR.md 3.2).

    Generaliza `lognormal_price_bands`, que recibe la deriva y la sigma ya
    escaladas a un horizonte concreto. Acá se reciben en su escala NATURAL
    —deriva log ANUALIZADA y volatilidad DIARIA, que es como las producen
    `RiskSimulationEngine.mu` y `extract_garch_forecast`— y el escalado a cada
    horizonte se hace acá dentro, una sola vez y de una sola forma:

        drift_h = mu_anual * (dias_h / trading_days_per_year)
        sigma_h = vol_diaria * sqrt(dias_h)

    Hacerlo así evita el error que la Fase 1.1 tuvo que arreglar: cuando cada
    ruta escala por su cuenta, dos proyecciones del mismo activo acaban con mu
    y sigma distintas y el informe las presenta como comparables.

    OJO CON LA RAÍZ DEL TIEMPO A HORIZONTES LARGOS. `sigma_h = vol_diaria *
    sqrt(dias_h)` asume retornos iid, y a 504 días eso es una suposición fuerte:
    ignora la reversión a la media de la volatilidad que el propio GJR-GARCH
    modela. La banda de 24 meses saldrá más ancha de lo que el modelo de
    volatilidad implicaría si se proyectara correctamente. Se mantiene la
    fórmula simple porque es la convención estándar y es transparente, pero el
    informe debe presentar los horizontes largos como lo que son: una
    extrapolación, no una medición. `extract_garch_forecast(horizon_days=...)`
    sí respeta la reversión y es la fuente correcta de `vol_diaria` por
    horizonte si se quiere refinar.

    ADITIVO: `lognormal_price_bands` sigue existiendo y funcionando igual.
    """
    resultado: Dict[int, Dict[str, Any]] = {}
    for meses, dias in horizons_days.items():
        drift_h = mu_anual_log * (dias / trading_days_per_year)
        sigma_h = vol_diaria * np.sqrt(dias)
        bandas = lognormal_price_bands(current_price, drift_h, sigma_h)
        resultado[meses] = {
            **bandas,
            "horizonte_dias": dias,
            "central": current_price * np.exp(drift_h),
            "drift_log": float(drift_h),
            "sigma": float(sigma_h),
        }
    return {
        "bandas": resultado,
        "precio_actual": float(current_price),
        "mu_anual_log": float(mu_anual_log),
        "vol_diaria": float(vol_diaria),
    }


def lognormal_price_bands(current_price: float, drift_log: float, sigma: float) -> dict:
    """
    Intervalos de confianza 80/90/95% LOGNORMALES: precio = current_price *
    exp(drift_log ± z*sigma). drift_log y sigma deben estar en la MISMA escala
    (log-retorno acumulado sobre el horizonte de proyección).

    Los precios no pueden ser negativos, así que un modelo bien especificado
    no necesita recortar el resultado con max(0, ...) — ese recorte era la
    señal de que unas bandas ARITMÉTICAS simétricas (current_price * (1 +
    drift ± z*sigma)) estaban mal especificadas para una variable lognormal.

    No incluye un 99%: estimar un cuantil tan extremo con las ~500
    observaciones solapadas típicas del histórico disponible equivale a ~2.5
    observaciones EFECTIVAS en la cola (mismo problema de solapamiento que
    documenta run_holding_period_backtest.n_independent_signals) — no es una
    banda estimable, solo un número que aparenta precisión. El 80% (nuevo)
    sí es estimable y más útil para decidir una compra a horizonte.
    """
    z_scores = _Z_SCORES_BANDAS
    return {
        label: (current_price * np.exp(drift_log - z * sigma), current_price * np.exp(drift_log + z * sigma))
        for label, z in z_scores.items()
    }


# =============================================================================
#          FORMATEO DEL INFORME (funciones puras, para poder testearlas)
# =============================================================================

def _fmt(valor: Any, unidad: str = "pct", decimales: int = 1) -> str:
    """
    Formatea un número del informe, o 'n/d' si no hay dato.

    'n/d' y nunca un 0 ni un guion suelto: un dato ausente tiene que leerse
    como ausente, que es el invariante "un número que falta no es una
    medición" aplicado a la capa de presentación.
    """
    if valor is None:
        return "n/d"
    try:
        v = float(valor)
    except (TypeError, ValueError):
        return str(valor)
    if not np.isfinite(v):
        return "n/d"
    if unidad == "pct":
        return f"{v:.{decimales}%}"
    if unidad == "pct_signo":
        return f"{v:+.{decimales}%}"
    if unidad == "usd":
        return f"${v:,.2f}"
    if unidad == "ratio":
        return f"{v:,.2f}x"
    if unidad == "dias":
        return f"{v:,.0f}d"
    return f"{v:,.{decimales}f}"


def _etiqueta_horizonte(meses: int) -> str:
    """'1 mes' / '3 meses'. Existe para que la concordancia se escriba una vez
    y no aparezca un '1 meses' en la mitad de las tablas y no en la otra."""
    return f"{meses} mes" if int(meses) == 1 else f"{meses} meses"


def _fmt_percentil(valor: Any) -> str:
    """Percentil 0-1 como 'p78'; 'n/d' si no hay dato."""
    if valor is None:
        return "n/d"
    try:
        v = float(valor)
    except (TypeError, ValueError):
        return "n/d"
    if not np.isfinite(v):
        return "n/d"
    return f"p{v*100:.0f}"


def _fiable_txt(fiable: Optional[bool]) -> str:
    """
    'sí' / 'NO' / 'n/d'. `None` es un tercer estado real (no evaluable), no un
    False disfrazado — ver el invariante de "no evaluable" del repo.
    """
    if fiable is None:
        return "n/d"
    return "sí" if fiable else "NO"


def _tabla_drawdown(drawdown_por_horizonte: Dict[int, Dict[str, Any]]) -> List[str]:
    """
    Tabla del drawdown de entrada por horizonte, con el tamaño muestral y el
    flag de fiabilidad EN LA MISMA FILA que los percentiles.

    Va en la misma fila a propósito: es el criterio de aceptación de la Fase 3
    ("ningún número sin su muestra al lado"), y una nota al pie se lee después
    de haberse creído el número.
    """
    lineas = [
        f"    {'Horizonte':<{_ANCHO_ETIQUETA}}{'P5':>9}{'P25':>9}{'Mediana':>9}"
        f"{'P75':>9}{'P95':>9}{'n':>8}{'n indep.':>10}{'fiable':>8}"
    ]
    for meses in sorted(drawdown_por_horizonte):
        d = drawdown_por_horizonte[meses]
        etiqueta = _etiqueta_horizonte(meses)
        lineas.append(
            f"    {etiqueta:<{_ANCHO_ETIQUETA}}"
            f"{_fmt(d.get('p5')):>9}{_fmt(d.get('p25')):>9}{_fmt(d.get('p50')):>9}"
            f"{_fmt(d.get('p75')):>9}{_fmt(d.get('p95')):>9}"
            f"{d.get('n_obs', 0):>8}{d.get('n_independiente', float('nan')):>10.1f}"
            f"{_fiable_txt(d.get('fiable')):>8}"
        )
    return lineas


def _tabla_recuperacion(recuperacion_por_horizonte: Dict[int, Dict[str, Any]]) -> List[str]:
    """
    Tabla del tiempo hasta recuperación. El '% no recupera' es, con frecuencia,
    la cifra más importante del bloque: una mediana de 40 días no dice nada si
    un tercio de las entradas no volvió al precio pagado dentro del horizonte.
    """
    lineas = [
        f"    {'Horizonte':<{_ANCHO_ETIQUETA}}{'Mediana':>9}{'P95':>9}{'No recup.':>11}"
        f"{'Nunca baja':>12}{'n':>8}{'n indep.':>10}{'fiable':>8}"
    ]
    for meses in sorted(recuperacion_por_horizonte):
        r = recuperacion_por_horizonte[meses]
        rec = r.get("recuperacion", {})
        etiqueta = _etiqueta_horizonte(meses)
        lineas.append(
            f"    {etiqueta:<{_ANCHO_ETIQUETA}}"
            f"{_fmt(rec.get('p50'), 'dias'):>9}{_fmt(rec.get('p95'), 'dias'):>9}"
            f"{_fmt(r.get('pct_no_recupera'), 'pct', 0):>11}"
            f"{_fmt(r.get('pct_nunca_bajo_agua'), 'pct', 0):>12}"
            f"{rec.get('n_obs', 0):>8}{rec.get('n_independiente', float('nan')):>10.1f}"
            f"{_fiable_txt(rec.get('fiable')):>8}"
        )
    return lineas


def _tabla_abanico(bandas_por_horizonte: Dict[int, Dict[str, Any]]) -> List[str]:
    """
    Tabla de bandas de precio en abanico. Sirve para las dos rutas (lognormal
    clásica y cuantílica) porque ambas devuelven el mismo diccionario de
    etiquetas '80'/'90'/'95' -> (inferior, superior) más un 'central'.
    """
    lineas = [
        f"    {'Horizonte':<{_ANCHO_ETIQUETA}}{'Central':>12}"
        f"{'80%':>26}{'90%':>26}{'95%':>26}"
    ]
    for meses in sorted(bandas_por_horizonte):
        b = bandas_por_horizonte[meses]
        etiqueta = _etiqueta_horizonte(meses)
        # Las bandas cuantílicas usan las claves 'p80'/'p90'/'p95' y las
        # lognormales '80'/'90'/'95'; se aceptan ambas para no duplicar tabla.
        celdas = []
        for label in ("80", "90", "95"):
            banda = b.get(label) or b.get(f"p{label}")
            if banda is None:
                celdas.append(f"{'n/d':>26}")
                continue
            lo, hi = banda
            celdas.append(f"{('[ ' + _fmt(lo, 'usd') + ' - ' + _fmt(hi, 'usd') + ' ]'):>26}")
        lineas.append(f"    {etiqueta:<{_ANCHO_ETIQUETA}}{_fmt(b.get('central'), 'usd'):>12}"
                      + "".join(celdas))
    return lineas


def _ventana_txt(datos: Dict[str, Any]) -> str:
    """
    Cobertura REAL de la ventana de la que sale el valor de hoy (Fase 3.6).

    `.rolling(4)` y `.pct_change(4)` cuentan FILAS, no trimestres, así que con
    huecos en los datos contables una ventana "de 12 meses" puede abarcar
    bastante más. Se imprime el número de días, no un sí/no: "455d" dice de
    inmediato que ese "TTM" es en realidad de 15 meses, y un booleano no.
    Un '!' marca que se sale de la tolerancia; '—' es una métrica de balance,
    que no depende de ventana ninguna.
    """
    if datos.get("ventana") == "stock":
        return "—"
    dias = datos.get("ventana_dias")
    if dias is None or not np.isfinite(dias):
        return "n/d"
    marca = "" if datos.get("ventana_valida") else "!"
    return f"{dias:.0f}d{marca}"


def _tabla_salud_financiera(salud: Dict[str, Any]) -> List[str]:
    """
    Ficha descriptiva de salud financiera: valor, percentil histórico propio,
    mediana histórica, variación YoY y n. SIN puntuar y sin etiquetas.

    Se imprime el valor Y el percentil por separado, no su promedio: promediar
    ambos era lo que hacía `score_quality`, y el resultado no era interpretable
    (un 0,62 podía ser "excelente en absoluto y en su peor momento histórico" o
    "mediocre y en su mejor momento").
    """
    metricas = salud.get("metricas", {})
    if not metricas:
        return ["    Sin datos contables suficientes para la ficha descriptiva."]

    lineas = [
        f"    {'Métrica':<26}{'Actual':>12}{'Percentil':>11}{'Mediana hist.':>15}"
        f"{'Δ vs. 1 año':>14}{'n':>6}{'Ventana':>10}"
    ]
    for datos in metricas.values():
        unidad = datos.get("unidad", "pct")
        delta_unidad = "pct_signo" if unidad == "pct" else "num"
        delta = datos.get("delta_4t")
        delta_txt = _fmt(delta, delta_unidad)
        if delta_unidad == "num" and delta_txt != "n/d":
            delta_txt = f"{float(delta):+,.2f}"
        lineas.append(
            f"    {datos.get('etiqueta', '?'):<26}"
            f"{_fmt(datos.get('actual'), unidad):>12}"
            f"{_fmt_percentil(datos.get('percentil')):>11}"
            f"{_fmt(datos.get('mediana_historica'), unidad):>15}"
            f"{delta_txt:>14}"
            f"{datos.get('n_obs', 0):>6}"
            f"{_ventana_txt(datos):>10}"
        )
    return lineas


def _tabla_valoracion(valuation_ctx: Dict[str, Any]) -> List[str]:
    """
    Múltiplos point-in-time con su percentil dentro del histórico de la propia
    empresa. Percentil ALTO = múltiplo alto = caro, sin la inversión
    `1.0 - rank` que usaba el score: leído como contexto, invertirlo solo
    confunde.
    """
    etiquetas = {
        "per": "PER (TTM)",
        "ev_to_ebitda": "EV / EBITDA (TTM)",
        "price_to_book": "Precio / Valor contable",
        "price_to_sales": "Precio / Ventas (TTM)",
        "dividend_yield": "Rentabilidad por dividendo",
    }
    multiplos = valuation_ctx.get("multiplos", {})
    if not multiplos:
        return ["    Sin múltiplos de valoración disponibles."]

    lineas = [f"    {'Múltiplo':<28}{'Actual':>12}{'Percentil':>11}{'Mediana hist.':>15}{'n':>6}"]
    for clave, etiqueta in etiquetas.items():
        if clave not in multiplos:
            continue
        d = multiplos[clave]
        unidad = "pct" if clave == "dividend_yield" else "ratio"
        lineas.append(
            f"    {etiqueta:<28}{_fmt(d.get('actual'), unidad):>12}"
            f"{_fmt_percentil(d.get('percentil')):>11}"
            f"{_fmt(d.get('mediana_historica'), unidad):>15}{d.get('n_obs', 0):>6}"
        )
    return lineas


def _recolectar_advertencias(*, config: PipelineConfig, engine: DataEngine,
                             historico_efectivo_corto: bool,
                             valuation_ctx: Dict[str, Any], salud: Dict[str, Any],
                             contexto: Dict[str, Any],
                             fan_cuantilico: Dict[str, Any],
                             validacion_cuantilica: Dict[int, Dict[str, Any]],
                             horizontes_dias: Dict[int, int]) -> List[str]:
    """
    Junta TODAS las advertencias activas en una sola lista, que el informe
    imprime en su propio bloque final.

    Un bloque explícito y no una nota suelta al lado de cada número: el informe
    nuevo no tiene un veredicto que resuma nada, así que lo que el lector
    necesita saber es exactamente qué partes de lo que acaba de leer no se
    sostienen y por qué. Cada entrada nombra la causa concreta y su cifra, no
    "puede haber problemas con los datos".
    """
    advertencias: List[str] = []

    if config.lookback_years < MIN_LOOKBACK_YEARS_FOR_INTERPRETABLE_BACKTEST:
        advertencias.append(
            f"lookback_years={config.lookback_years} por debajo del mínimo "
            f"{MIN_LOOKBACK_YEARS_FOR_INTERPRETABLE_BACKTEST}: todas las distribuciones de "
            f"'a qué te apuntas' se calculan sobre menos histórico del recomendado."
        )

    if historico_efectivo_corto:
        advertencias.append(
            f"histórico EFECTIVO de solo ~{engine.effective_history_years:.1f} años (desde "
            f"{engine.effective_history_start.date()}), por debajo del mínimo "
            f"{MIN_LOOKBACK_YEARS_FOR_INTERPRETABLE_BACKTEST}: se pidieron {config.lookback_years} "
            f"años pero el activo no cotiza desde entonces."
        )

    if not valuation_ctx.get("fiable"):
        advertencias.append(
            f"múltiplos de valoración sobre {valuation_ctx.get('n_quarters', 0)} trimestres "
            f"(mínimo {MIN_QUARTERS_FOR_RELIABILITY}): los VALORES son el dato contable y valen, "
            f"pero sus PERCENTILES no son interpretables con tan pocas observaciones."
        )

    # Ventanas TTM/YoY que no cubren 12 meses (Fase 3.6). Se advierte cuando
    # afecta al valor de HOY —que es el que se imprime— y, aparte, cuando
    # afecta a la mayoría del histórico, porque entonces los percentiles se
    # calculan comparando ventanas de duraciones distintas entre sí.
    ventanas = salud.get("ventanas") or {}
    for tipo, nombre in (("ttm", "TTM"), ("yoy", "YoY")):
        v = ventanas.get(tipo) or {}
        n_obs = v.get("n_obs", 0)
        n_mal = v.get("n_invalidas", 0)
        if n_obs and n_mal / n_obs > 0.5:
            advertencias.append(
                f"{n_mal} de {n_obs} ventanas {nombre} NO cubren 12 meses reales (mediana "
                f"{v.get('mediana_dias', float('nan')):.0f} días, peor caso "
                f"{v.get('peor_dias', float('nan')):.0f}): `.rolling(4)`/`.pct_change(4)` cuentan "
                f"FILAS y los datos de EDGAR tienen huecos, así que el percentil de esas métricas "
                f"compara ventanas de duración distinta entre sí. La columna VENTANA de la ficha "
                f"dice la cobertura real de cada valor de hoy."
            )

    metricas_hoy_mal = [d.get("etiqueta", k) for k, d in (salud.get("metricas") or {}).items()
                        if d.get("ventana") != "stock" and d.get("ventana_valida") is False]
    if metricas_hoy_mal:
        advertencias.append(
            f"el valor de HOY de {', '.join(metricas_hoy_mal)} sale de una ventana que no cubre "
            f"12 meses (ver la columna VENTANA): no es comparable con el mismo dato de otra fecha "
            f"ni con el umbral anual de la literatura."
        )

    if not salud.get("reliable"):
        advertencias.append(
            f"ficha de salud financiera sobre {salud.get('n_quarters', 0)} trimestres "
            f"(mínimo {MIN_QUARTERS_FOR_RELIABILITY}): mismos reparos que arriba para sus "
            f"percentiles."
        )

    # Distribuciones con muestra independiente insuficiente: se nombran los
    # horizontes concretos, porque en la práctica los cortos son fiables y los
    # largos no, y decir "algunas distribuciones" obliga a volver a la tabla.
    dd_no_fiables = [m for m, d in contexto.get("drawdown", {}).items() if not d.get("fiable")]
    if dd_no_fiables:
        advertencias.append(
            f"drawdown de entrada NO fiable a {', '.join(f'{m}m' for m in sorted(dd_no_fiables))}: "
            f"menos de {MIN_OBS_INDEPENDIENTES} ventanas independientes (las fechas consecutivas "
            f"comparten casi toda su ventana, así que n/horizonte es la muestra real)."
        )
    rec_no_fiables = [m for m, r in contexto.get("recuperacion", {}).items()
                      if not r.get("recuperacion", {}).get("fiable")]
    if rec_no_fiables:
        advertencias.append(
            f"tiempo hasta recuperación NO fiable a "
            f"{', '.join(f'{m}m' for m in sorted(rec_no_fiables))}: misma corrección por "
            f"solapamiento, y además solo cuenta las entradas que llegaron a estar en pérdidas."
        )

    for meses, motivo in (fan_cuantilico.get("omitidos") or {}).items():
        advertencias.append(f"banda cuantílica de {_etiqueta_horizonte(meses)} OMITIDA — {motivo}.")

    for meses, val in sorted(validacion_cuantilica.items()):
        for label in ("80", "90", "95"):
            cov = (val.get("coverage") or {}).get(label)
            if cov is None or not np.isfinite(cov):
                continue
            nominal = int(label) / 100
            if abs(cov - nominal) > 0.15:
                advertencias.append(
                    f"banda cuantílica {label}% a {_etiqueta_horizonte(meses)} MAL CALIBRADA fuera de muestra: "
                    f"cubrió {cov:.1%} de los casos frente al {nominal:.0%} nominal "
                    f"({val.get('n_test_obs', 0)} observaciones de test)."
                )

    # Horizontes largos del abanico lognormal: la advertencia es estructural (la
    # fórmula, no los datos), así que se emite siempre que se impriman.
    largos = [m for m in horizontes_dias if m >= 12]
    if largos:
        advertencias.append(
            f"las bandas lognormales a {', '.join(f'{m}m' for m in sorted(largos))} escalan la "
            f"volatilidad por sqrt(t), lo que asume retornos iid e ignora la reversión a la media "
            f"que el propio GJR-GARCH modela: son una EXTRAPOLACIÓN, no una medición."
        )

    return advertencias


_ticker_validation_cache: dict = {}


def prompt_main_menu() -> int:
    """
    Imprime el menú principal y devuelve la opción elegida (0/1/2). Ante
    cualquier entrada que no sea 0/1/2, vuelve a preguntar sin salirse.
    """
    while True:
        print("\n" + "="*60)
        print("   HERRAMIENTA DE CONTEXTO DE ENTRADA")
        print("="*60)
        print("  [1] Analizar una acción (ficha de entrada completa)")
        print("  [2] Analizar mi cartera actual (contexto por posición)")
        print("  [0] Salir")
        opcion = input("\nElegí una opción: ").strip()
        if opcion in ("0", "1", "2"):
            return int(opcion)
        print(f" ⚠ Opción '{opcion}' no válida — ingresá 0, 1 o 2.")


def prompt_ticker() -> Optional[str]:
    """
    Pide un ticker por consola, lo normaliza (.strip().upper()) y lo valida
    contra yfinance (~5 días de precios) antes de devolverlo. Reintenta en
    bucle si el ticker no existe o no trae datos. Devuelve None si el
    usuario escribe 0 o deja la entrada vacía (volver al menú).

    _ticker_validation_cache evita repetir la descarga de validación si el
    mismo ticker se valida dos veces en la misma sesión.
    """
    while True:
        raw = input("\nIngresá el ticker a analizar (0 para volver al menú): ").strip().upper()
        if raw in ("", "0"):
            return None

        if raw in _ticker_validation_cache:
            is_valid = _ticker_validation_cache[raw]
        else:
            end_date = datetime.date.today()
            start_date = end_date - datetime.timedelta(days=5)
            try:
                data = yf.download(raw, start=start_date.strftime("%Y-%m-%d"),
                                    end=(end_date + datetime.timedelta(days=1)).strftime("%Y-%m-%d"),
                                    progress=False)
            except Exception:
                data = None
            is_valid = data is not None and not data.empty
            _ticker_validation_cache[raw] = is_valid

        if is_valid:
            return raw
        print(f" Ticker '{raw}' no encontrado o sin datos de mercado — probá con otro.")


def run_production_system(ticker: str, config: PipelineConfig = None):
    """
    FICHA DE ENTRADA de un ticker (PLAN_REFACTOR.md 3.4).

    Responde *"¿a qué me apunto si compro hoy a este precio?"* — no "¿va a
    subir?". El informe DESCRIBE; no dictamina.

    QUÉ SALIÓ DE ACÁ EN LA FASE 3.3, y por qué no reponerlo. Se eliminaron el
    Score Multifactorial 0-100, el veredicto COMPRAR/MANTENER/EVITAR, la
    probabilidad ML y su calibración, los dos backtests como secciones del
    informe y `ValidationEngine` entero. No es una regresión ni un descuido:
    la Fase 2 midió el poder predictivo de esa capa con la vara del propio
    sistema y no lo hay (Information Coefficient medio -0,0093 con el signo
    repartido 5-5 en el único horizonte con potencia estadística, y los dos
    grupos del event study no estimables en ninguno de los 10 tickers al
    horizonte de producción). Está autorizado explícitamente por el
    propietario como excepción a su regla de "solo añadir al output", con
    fecha y motivo anotados en `Ajustes y Arreglos.txt`.

    TAMPOCO SE SUSTITUYE POR OTRA ETIQUETA. Ni "atractivo/neutro/caro", ni
    semáforos, ni estrellas: cualquier resumen categórico de varias
    dimensiones en una palabra reintroduce por la puerta de atrás lo mismo que
    esta fase elimina — un número que no se puede auditar porque esconde de
    qué medición salió y con cuántas observaciones.

    El código del ML se conserva (ml_engine.py, validation_engine.py,
    scripts/diagnostico_multihorizonte.py) con todos sus tests: solo sale de
    la ruta de decisión. Si algún día se construye el panel transversal de la
    Fase 6, esa infraestructura —walk-forward, purga, embargo, IC,
    Newey-West— se reutiliza entera.
    """
    config = config or PipelineConfig()

    print("\n" + "="*60)
    print("   HERRAMIENTA DE CONTEXTO DE ENTRADA")
    print("="*60 + "\n")

    ticker_objetivo = ticker
    # Horizonte "de producción" que comparten el Monte Carlo y la volatilidad
    # esperada del GARCH. Ya NO es el ancla del sistema: las distribuciones de
    # entrada y las bandas de precio se reportan en abanico sobre los cinco
    # horizontes de entry_context_engine.HORIZONTES_MESES, precisamente porque
    # la estrategia real no tiene plazo de mantenimiento fijo.
    dias_horizonte = config.horizon_days
    horizontes_dias = horizontes_en_dias(dias_por_mes=config.trading_days_per_month)

    # 1. INGESTA Y PROCESAMIENTO DE DATOS
    engine = DataEngine(tickers=[ticker_objetivo], benchmark=config.benchmark, cache_dir=config.cache_dir,
                        trading_days_per_year=config.trading_days_per_year)
    # end_date = última sesión CERRADA, no today(): corriendo en horario de
    # mercado, today() mete como última barra un precio VIVO que cambia entre
    # corridas del mismo día (ver data_engine.last_closed_session). start_date
    # se ancla a ese mismo end_date para que la ventana completa quede fija.
    ultima_sesion = last_closed_session()
    end_date = ultima_sesion.strftime("%Y-%m-%d")
    start_date = (ultima_sesion - datetime.timedelta(days=config.lookback_years*365)).strftime("%Y-%m-%d")

    market_prices = engine.fetch_market_data(start_date=start_date, end_date=end_date)

    # Histórico EFECTIVO vs. solicitado. Al quitar el bfill() (Fase 1.6), un
    # ticker que cotiza desde hace poco ya no recibe su precio de IPO replicado
    # hacia atrás, así que el panel dice la verdad sobre su tamaño.
    historico_efectivo_corto = (
        engine.effective_history_years is not None
        and engine.effective_history_years < MIN_LOOKBACK_YEARS_FOR_INTERPRETABLE_BACKTEST
    )

    # Descarga datos contables trimestrales (SEC EDGAR por defecto — 40-80 trimestres
    # point-in-time; yfinance como fallback automático o explícito, ver config.fundamentals_source)
    fundamental_data = engine.fetch_fundamental_data(ticker=ticker_objetivo, source=config.fundamentals_source)

    # 2. FACTORES TÉCNICOS Y SERIES TEMPORALES
    tech_engine = TechnicalFactorEngine(market_data=market_prices, benchmark_data=engine.benchmark_data,
                                        trading_days_per_year=config.trading_days_per_year)
    momentum_factors = tech_engine.compute_momentum_factors()
    trend_signals = tech_engine.compute_trend_signals()

    ts_extractor = TimeSeriesFeatureExtractor(log_returns=tech_engine.log_returns)
    garch_vol = ts_extractor.extract_garch_volatility(ticker=ticker_objetivo, refit_every=config.ts_refit_every)
    arima_trend = ts_extractor.extract_arima_trend(ticker=ticker_objetivo, refit_every=config.ts_refit_every)

    # 3. MATRIZ DE CARACTERÍSTICAS
    # Sigue siendo la misma matriz que antes alimentaba al clasificador ML: la
    # consume el abanico cuantílico (QuantileFanForecaster), que predice la
    # DISTRIBUCIÓN del retorno forward por horizonte en vez de una etiqueta
    # binaria "bate al benchmark". Los features son walk-forward (GARCH/ARIMA
    # con ventana expansiva), así que no hay lookahead en ninguna fila.
    features_df = pd.DataFrame(index=market_prices.index)
    features_df['mom_3m'] = momentum_factors['mom_3m'][ticker_objetivo]
    features_df['mom_6m'] = momentum_factors['mom_6m'][ticker_objetivo]
    features_df['mom_12m'] = momentum_factors['mom_12m'][ticker_objetivo]
    features_df['rsi'] = trend_signals[f'{ticker_objetivo}_rsi']
    features_df['ma_crossover'] = trend_signals[f'{ticker_objetivo}_crossover']
    features_df['garch_volatility'] = garch_vol
    features_df['arima_trend_pred'] = arima_trend
    latest_features = features_df.tail(1)

    # 4. PERFIL DE RIESGO
    # Volatilidad diaria ESPERADA sobre el horizonte (promedio de las varianzas
    # proyectadas por el GARCH desde hoy), NO la volatilidad condicional
    # instantánea de hoy (garch_vol.iloc[-1]): esa última congela el estado de
    # vol de un único día, ignorando la reversión a la media que el propio
    # modelo estima.
    #
    # ESTA MISMA sigma alimenta el Monte Carlo (acá) y el abanico lognormal
    # (más abajo). Se calcula UNA vez y ambas rutas consumen el mismo valor:
    # antes el Monte Carlo recibía garch_vol.iloc[-1] y la proyección clásica
    # el forecast, así que el informe imprimía dos proyecciones lognormales del
    # mismo activo con sigma (y deriva) distintas presentándolas como
    # comparables (Fase 1.1).
    garch_vol_esperada = ts_extractor.extract_garch_forecast(ticker=ticker_objetivo, horizon_days=dias_horizonte)
    risk_engine = RiskSimulationEngine(
        price_series=market_prices[ticker_objetivo],
        log_returns=tech_engine.log_returns[ticker_objetivo],
        predicted_volatility=garch_vol_esperada,
        drift_mode=config.drift_mode,
        risk_free_rate=config.risk_free_rate,
        random_state=config.random_state,
        trading_days_per_year=config.trading_days_per_year,
    )
    mc_results = risk_engine.run_monte_carlo(horizon_days=dias_horizonte, n_simulations=config.monte_carlo_simulations,
                                              target_return=config.monte_carlo_target_return)
    risk_metrics = risk_engine.calculate_risk_metrics(risk_free_rate=config.risk_free_rate)

    stat_risk = tech_engine.compute_statistical_risk()
    # stat_risk['beta'] es una serie ROLLING (252 días) por fecha, no un
    # escalar de muestra completa — tomamos el valor MÁS RECIENTE, que refleja
    # el perfil de riesgo actual del activo, no el promedio de todo su histórico.
    beta_series = stat_risk['beta'][ticker_objetivo].dropna()
    beta = float(beta_series.iloc[-1]) if not beta_series.empty else float("nan")
    benchmark_vol_ann = float(tech_engine.bench_returns.std().iloc[0]
                              * np.sqrt(config.trading_days_per_year))
    vol_relativa = (risk_metrics['volatility_ann'] / benchmark_vol_ann
                    if benchmark_vol_ann else float("nan"))
    # Volatilidad condicional de hoy Y su percentil dentro de su propio
    # histórico: el nivel solo no dice si el activo está en un momento
    # tranquilo o agitado PARA ÉL, que es lo que importa al entrar hoy.
    vol_condicional = percentil_historico(garch_vol)

    # 5. VALORACIÓN Y SALUD FINANCIERA (descriptivas, sin puntuar)
    fund_engine = FundamentalFactorEngine(
        fundamental_data=fundamental_data,
        min_invested_capital_fraction=config.roic_min_invested_capital_fraction,
    )
    valuation_df = fund_engine.compute_valuation_scores(current_prices=market_prices[ticker_objetivo])
    valuation_ctx = EntryContextEngine.valuation_context(
        valuation_df, n_quarters_reliable=MIN_QUARTERS_FOR_RELIABILITY,
    )
    salud = fund_engine.describe_quality()

    # 6. CONTEXTO DE ENTRADA — el bloque central del informe
    entry_engine = EntryContextEngine(price_series=market_prices[ticker_objetivo],
                                       dias_por_mes=config.trading_days_per_month)
    contexto = entry_engine.build_full_context()

    precio_actual = float(market_prices[ticker_objetivo].dropna().iloc[-1])
    # "Últimos 12 meses" = un año de SESIONES del activo (Fase 3.6): 252 en
    # una acción, 365 en cripto. Con el literal 252 el rango de un activo que
    # cotiza todos los días abarcaba 8 meses y se rotulaba como 12.
    precios_recientes = market_prices[ticker_objetivo].dropna().tail(config.trading_days_per_year)
    precio_max = float(precios_recientes.max())
    precio_min = float(precios_recientes.min())
    precio_medio = float(precios_recientes.mean())

    # 7. ABANICOS DE BANDAS DE PRECIO (1/3/6/12/24 meses)
    # (a) Clásica lognormal. La deriva se toma de risk_engine.mu (media de
    # log-retornos anualizada YA ajustada según config.drift_mode) en vez de
    # recalcularla, para que respetar drift_mode sea automático; el escalado
    # por horizonte vive entero dentro de lognormal_price_fan.
    fan_lognormal = lognormal_price_fan(
        current_price=precio_actual,
        mu_anual_log=float(risk_engine.mu),
        vol_diaria=float(garch_vol_esperada),
        horizons_days=horizontes_dias,
        trading_days_per_year=config.trading_days_per_year,
    )
    # (b) Cuantílica: un QuantileForecastEngine por horizonte, con el embargo
    # de cada split dimensionado a su propio target. Se valida fuera de muestra
    # ANTES de publicar (pinball loss + cobertura empírica por horizonte): las
    # bandas no se presentan a ciegas.
    qf_fan = QuantileFanForecaster(
        feature_matrix=features_df,
        price_series=market_prices[ticker_objetivo],
        horizons_days=horizontes_dias,
        train_size=config.ml_train_size,
        random_state=config.random_state,
    )
    validacion_cuantilica = qf_fan.evaluate_out_of_sample()
    qf_fan.refit_on_full_data()
    fan_cuantilico = qf_fan.predict_price_fan(latest_features, precio_actual)

    # 8. ADVERTENCIAS
    advertencias = _recolectar_advertencias(
        config=config, engine=engine, historico_efectivo_corto=historico_efectivo_corto,
        valuation_ctx=valuation_ctx, salud=salud, contexto=contexto,
        fan_cuantilico=fan_cuantilico, validacion_cuantilica=validacion_cuantilica,
        horizontes_dias=horizontes_dias,
    )

    # =========================================================================
    #                     INFORME: FICHA DE ENTRADA
    # =========================================================================
    print("\n" + "█"*80)
    print(f"        FICHA DE ENTRADA — {ticker_objetivo} — {ultima_sesion}")
    print(f"        ¿A qué me apunto si compro hoy a ${precio_actual:,.2f}?")
    print("█"*80)

    # --- PRECIO Y VALORACIÓN (solo con --verbose) --------------------------
    if config.verbose:
        print("\n PRECIO Y VALORACIÓN")
        print("─"*80)
        print(f"  Precio actual (cierre de {ultima_sesion}):  ${precio_actual:,.2f}")
        print(f"  Rango de los últimos 12 meses:  mín ${precio_min:,.2f}  |  "
              f"medio ${precio_medio:,.2f}  |  máx ${precio_max:,.2f}")
        print(f"  Histórico efectivo utilizado: "
              f"{_fmt(engine.effective_history_years, 'num', 1)} años"
              f"{'' if engine.effective_history_start is None else f' (desde {engine.effective_history_start.date()})'}")
        print(f"  Múltiplos point-in-time frente al histórico de la propia empresa "
              f"({valuation_ctx.get('n_quarters', 0)} trimestres, fiable: "
              f"{_fiable_txt(valuation_ctx.get('fiable'))}):")
        for linea in _tabla_valoracion(valuation_ctx):
            print(linea)
        print("  Percentil alto = múltiplo alto = caro respecto de su propia historia. No hay")
        print("  puntuación ni etiqueta: un percentil con su n al lado se puede comprobar; un")
        print("  'Valoración: 72/100' no dice qué se midió ni sobre cuántas observaciones.")

    # --- PERFIL DE RIESGO (solo con --verbose) ------------------------------
    if config.verbose:
        print("\n PERFIL DE RIESGO")
        print("─"*80)
        print(f"  Volatilidad condicional GARCH (hoy):   {_fmt(vol_condicional['actual'], 'pct', 2)} diaria"
              f"   |   percentil de su propio histórico: {_fmt_percentil(vol_condicional['percentil'])}"
              f"  (n={vol_condicional['n_obs']} días)")
        print(f"  Volatilidad esperada a {config.horizon_months} meses (GARCH):    "
              f"{_fmt(garch_vol_esperada, 'pct', 2)} diaria")
        print(f"  Volatilidad anualizada (histórica):    {_fmt(risk_metrics['volatility_ann'])}"
              f"   |   benchmark {config.benchmark}: {_fmt(benchmark_vol_ann)}"
              f"   |   relativa: {_fmt(vol_relativa, 'ratio')}")
        print(f"  Beta rolling {config.trading_days_per_year}d (1 año) vs. {config.benchmark}:"
              f"{'':<{max(0, 12 - len(str(config.trading_days_per_year)))}}{_fmt(beta, 'num', 2)}")
        print(f"  Máximo drawdown histórico:             {_fmt(risk_metrics['max_drawdown'])}")
        print(f"  VaR diario 95%:                        {_fmt(risk_metrics['var_daily_95_pct'])}"
              f"   (${risk_metrics['var_daily_95_monetary']:,.2f})")
        print(f"  Expected Shortfall diario:             {_fmt(risk_metrics['expected_shortfall_daily'])}")
        print(f"  Sharpe ratio (histórico):              {_fmt(risk_metrics['sharpe_ratio'], 'num', 2)}")
        print(f"  Sortino ratio (histórico):             {_fmt(risk_metrics['sortino_ratio'], 'num', 2)}")
        print(f"  [Monte Carlo — {config.monte_carlo_simulations:,} simulaciones a {config.horizon_months} meses, "
              f"deriva log anual {risk_engine.mu:+.4f} ({config.drift_mode}), "
              f"vol. diaria {garch_vol_esperada:.4%}]")
        print(f"    Precio medio simulado:               ${mc_results['expected_mean_price']:,.2f}")
        print(f"    Probabilidad de acabar por debajo del precio de hoy: {mc_results['prob_loss']:.1%}")
        print(f"    Probabilidad de alcanzar +{config.monte_carlo_target_return:.0%}: "
              f"{mc_results['prob_target_reached']:.1%}")
        print(f"    Intervalo de confianza 95%:          [ ${mc_results['confidence_interval_95'][0]:,.2f}  -  "
              f"${mc_results['confidence_interval_95'][1]:,.2f} ]")
        print("    Es una SIMULACIÓN bajo un movimiento browniano geométrico con esa deriva y esa")
        print("    volatilidad, no una medición de lo que pasó: su probabilidad vale lo que valgan")
        print("    esos dos supuestos.")

    # --- A QUÉ TE APUNTAS (siempre; es el bloque central) -------------------
    print("\n A QUÉ TE APUNTAS SI COMPRAS HOY")
    print("─"*80)
    print(" Distribución de lo que históricamente le pasó a quien compró este activo en una")
    print(" fecha CUALQUIERA y aguantó H. Son afirmaciones sobre el pasado, verificables y sin")
    print(" condicionar al estado de hoy — no una predicción. 'n indep.' = n/horizonte: dos")
    print(" fechas separadas por una semana comparten casi toda su ventana (la de 6 meses, por")
    print(" ejemplo, se solapa en más del 95%), así que no son dos observaciones.")
    print(f"\n  1) DRAWDOWN DE ENTRADA — cuánto llegó a caer por debajo de lo que pagaste")
    print(f"     (min(P[t..t+H])/P[t] - 1; acotado en 0: si nunca bajó, nunca estuviste en pérdidas)")
    for linea in _tabla_drawdown(contexto["drawdown"]):
        print(linea)
    print(f"\n  2) TIEMPO HASTA RECUPERACIÓN — de las entradas que SÍ se pusieron en pérdidas,")
    print(f"     cuántos días hasta volver al precio pagado")
    for linea in _tabla_recuperacion(contexto["recuperacion"]):
        print(linea)
    print("     'No recup.' = de las que se hundieron, cuántas no habían vuelto al precio de")
    print("     entrada al terminar el horizonte (se excluyen de la mediana en vez de meterles")
    print("     un número grande inventado). 'Nunca baja' = entradas que jamás cotizaron por")
    print("     debajo de lo pagado, y por tanto no tenían nada que recuperar.")
    print(f"\n  3) BANDAS DE PRECIO EN ABANICO")
    print(f"     (a) Clásica lognormal: precio · exp(deriva ± z·sigma), con deriva log anual "
          f"{risk_engine.mu:+.4f} ({config.drift_mode})")
    print(f"         y volatilidad diaria GARCH {garch_vol_esperada:.4%} escalada por sqrt(t)")
    for linea in _tabla_abanico(fan_lognormal["bandas"]):
        print(linea)
    print(f"     (b) Cuantílica (una regresión por cuantil y por horizonte), con su cobertura")
    print(f"         empírica fuera de muestra al lado — la banda que no se validó no se publica")
    for linea in _tabla_abanico(fan_cuantilico["bandas"]):
        print(linea)
    if fan_cuantilico["bandas"]:
        print(f"    {'Horizonte':<{_ANCHO_ETIQUETA}}{'Cobertura 80%':>16}{'Cobertura 90%':>16}"
              f"{'Cobertura 95%':>16}{'n test':>9}")
        for meses in sorted(fan_cuantilico["bandas"]):
            val = validacion_cuantilica.get(meses, {})
            cov = val.get("coverage", {})
            etiqueta = _etiqueta_horizonte(meses)
            print(f"    {etiqueta:<{_ANCHO_ETIQUETA}}"
                  f"{(_fmt(cov.get('80')) + ' /80%'):>16}{(_fmt(cov.get('90')) + ' /90%'):>16}"
                  f"{(_fmt(cov.get('95')) + ' /95%'):>16}{val.get('n_test_obs', 0):>9}")
    for meses, motivo in (fan_cuantilico.get("omitidos") or {}).items():
        print(f"     ⚠ {_etiqueta_horizonte(meses)}: omitido — {motivo}")

    # --- SALUD FINANCIERA (solo con --verbose) ------------------------------
    if config.verbose:
        print("\n SALUD FINANCIERA (descriptivo, sin puntuar)")
        print("─"*80)
        print(f"  {salud.get('n_quarters', 0)} trimestres point-in-time (fiable: "
              f"{_fiable_txt(salud.get('reliable'))}). Flujos en TTM; los stocks de balance "
              f"no se acumulan.")
        for linea in _tabla_salud_financiera(salud):
            print(linea)
        print("  Percentil en su orientación natural (alto = valor alto), también donde 'menos es")
        print("  mejor': un D/E en p95 significa 'la deuda está en su nivel más alto del")
        print("  histórico'. 'Δ vs. 1 año' compara con el MISMO trimestre del año anterior, no")
        print("  con el trimestre anterior, que estaría contaminado por estacionalidad.")
        v_ttm = (salud.get("ventanas") or {}).get("ttm", {})
        v_yoy = (salud.get("ventanas") or {}).get("yoy", {})
        print(f"  VENTANA = días que cubre DE VERDAD la ventana de la que sale el valor de hoy;")
        print(f"  se esperan {v_ttm.get('esperado_dias', 365)} ± {v_ttm.get('tolerancia_dias', 45):.0f}, "
              f"y '!' marca que se sale. '—' = métrica de balance, sin ventana.")
        print(f"  Sobre TODO el histórico: {v_ttm.get('n_invalidas', 0)} de {v_ttm.get('n_obs', 0)} "
              f"ventanas TTM y {v_yoy.get('n_invalidas', 0)} de {v_yoy.get('n_obs', 0)} YoY no cubren 12")
        print(f"  meses (mediana real TTM: {_fmt(v_ttm.get('mediana_dias'), 'num', 0)} días). Es un hueco "
              f"de los datos contables de EDGAR, no un")
        print(f"  error de cálculo: `.rolling(4)` y `.pct_change(4)` cuentan FILAS, no trimestres.")

    # --- ADVERTENCIAS (siempre) ---------------------------------------------
    # Se imprimen en los dos modos de verbosidad a propósito: el informe nuevo
    # no tiene ningún veredicto que resuma cuánto fiarse de lo de arriba, así
    # que la lista de lo que NO se sostiene es parte del resultado, no un
    # apéndice que un flag pueda esconder.
    print("\n ADVERTENCIAS")
    print("─"*80)
    if advertencias:
        for a in advertencias:
            print(f"  ⚠ {a}")
    else:
        print("  Ninguna: todas las distribuciones de arriba tienen muestra independiente")
        print("  suficiente y los fundamentales cubren el mínimo de trimestres.")

    # REGISTRO DE PREDICCIONES (Fase 1.12, esquema aditivo — ver prediction_log).
    # Se escribe SIEMPRE y con los valores ya calculados para el informe; nada
    # se recomputa para esto. Las claves de la Fase 3 son NUEVAS: las entradas
    # viejas seguirán teniendo `score`/`recomendacion`/`ml` y las nuevas no, y
    # ninguna clave existente cambia de significado — si se renombrara o
    # reutilizara una, se invalidaría el histórico acumulado, que es
    # exactamente lo que este fichero existe para evitar.
    registrar_prediccion(
        path=config.prediction_log_path,
        ticker=ticker_objetivo,
        config=config,
        payload={
            "precio_actual": precio_actual,
            "modo": "ficha_entrada",
            # --- claves nuevas de la Fase 3.3/3.4 -------------------------
            "valoracion_contexto": valuation_ctx,
            "salud_financiera": salud,
            "vol_condicional": vol_condicional,
            "drawdown_entrada": contexto["drawdown"],
            "tiempo_recuperacion": contexto["recuperacion"],
            "abanico_lognormal": {
                str(meses): {k: (list(v) if isinstance(v, tuple) else v) for k, v in b.items()}
                for meses, b in fan_lognormal["bandas"].items()
            },
            "abanico_cuantilico": {
                str(meses): {k: (list(v) if isinstance(v, tuple) else v) for k, v in b.items()}
                for meses, b in fan_cuantilico["bandas"].items()
            },
            "abanico_cuantilico_omitidos": {str(k): v for k, v in
                                             (fan_cuantilico.get("omitidos") or {}).items()},
            "abanico_cuantilico_validacion": {str(meses): v for meses, v in
                                               validacion_cuantilica.items()},
            "advertencias": advertencias,
            # --- claves con el MISMO significado que antes de la Fase 3 ---
            # (se mantienen tal cual para que el histórico siga comparable)
            "fiabilidad": {
                "valuation": valuation_ctx.get("fiable"),
                "valuation_n_quarters": valuation_ctx.get("n_quarters"),
                "quality": salud.get("reliable"),
                "quality_n_quarters": salud.get("n_quarters"),
            },
            "proyeccion_inputs": {
                "drift_log_horizonte": (float(risk_engine.mu)
                                        * (dias_horizonte / config.trading_days_per_year)),
                "sigma_horizonte": float(garch_vol_esperada) * float(np.sqrt(dias_horizonte)),
                "garch_vol_diaria_esperada": garch_vol_esperada,
                "drift_mode": config.drift_mode,
            },
            "monte_carlo": {
                "prob_loss": mc_results["prob_loss"],
                "prob_target_reached": mc_results["prob_target_reached"],
                "expected_mean_price": mc_results["expected_mean_price"],
            },
            "riesgo": {
                "volatility_ann": risk_metrics["volatility_ann"],
                "sharpe_ratio": risk_metrics["sharpe_ratio"],
                "sortino_ratio": risk_metrics["sortino_ratio"],
                "max_drawdown": risk_metrics["max_drawdown"],
                "beta": beta,
                "benchmark_volatility_ann": benchmark_vol_ann,
            },
            "historico_efectivo_anios": engine.effective_history_years,
            "historico_efectivo_corto": historico_efectivo_corto,
        },
    )

    # GRÁFICOS. Un fallo al guardar (ej. permisos de disco) no debe tumbar un
    # informe que ya terminó de calcularse.
    #
    # La CURVA DE EQUITY desapareció con la Fase 3.3: era el dibujo de
    # `run_strategy_backtest` (rebalanceo diario según la probabilidad ML), o
    # sea de la señal que salió de la ruta de decisión. Mantenerla obligaría a
    # entrenar el ML en cada corrida solo para pintarla. `plot_equity_curve`
    # sigue existiendo en ReportVisualizer para quien le pase un df_bt.
    print("\n GRÁFICOS GENERADOS:")
    try:
        visualizer = ReportVisualizer(ticker=ticker_objetivo, output_dir=config.output_dir)
        chart_paths = visualizer.generate_all(
            final_prices=mc_results['final_prices'],
            current_price=precio_actual,
            garch_vol=garch_vol,
        )
        print(f"  * Distribución Monte Carlo: {chart_paths['monte_carlo']}")
        print(f"  * Volatilidad GARCH:        {chart_paths['garch_volatility']}")
    except Exception as e:
        print(f"  ⚠ No se pudieron generar los gráficos: {e}")
    print("█"*80 + "\n")


# =============================================================================
#                    ANÁLISIS DE CARTERA (MENÚ OPCIÓN [2])
# =============================================================================
# FASE 3.3 — CONTEXTO, SIN VEREDICTO. La tabla ya no lleva columna de veredicto
# (MANTENER/VIGILAR/VENDER) ni probabilidad ML: la señal que las producía salió
# de la ruta de decisión por la misma razón que en el modo [1]. Lo que queda es
# contexto por posición, y una marca "REVISAR" que significa "míralo tú", no
# una orden de vender.
#
# Tabla: Ticker | Cant. | P.Medio | P.Actual | P&L | Val. | Vol.ann | Vol. | Beta | DD med | Revisar
#   Val.    = percentil point-in-time de valoración (alto = caro para su propia historia)
#   Vol.    = percentil de la volatilidad condicional GARCH de hoy en su propio histórico
#   DD med  = drawdown de entrada MEDIANO al horizonte de config.horizon_months (bloque 3.1)
_PORTFOLIO_COL_WIDTHS = {
    "ticker": 8, "quantity": 8, "cost_basis": 10, "current_price": 10,
    "pnl": 22, "valuation_pct": 7, "vol_ann": 9, "vol_pct": 7, "beta": 7,
    "drawdown": 9, "review": 10,
}


def _portfolio_table_header() -> str:
    w = _PORTFOLIO_COL_WIDTHS
    return (f"{'Ticker':<{w['ticker']}}{'Cant.':>{w['quantity']}}{'P.Medio':>{w['cost_basis']}}"
            f"{'P.Actual':>{w['current_price']}}{'P&L':>{w['pnl']}}{'Val.':>{w['valuation_pct']}}"
            f"{'Vol.ann':>{w['vol_ann']}}{'Vol.':>{w['vol_pct']}}{'Beta':>{w['beta']}}"
            f"{'DD med':>{w['drawdown']}}{'Revisar':>{w['review']}}")


def _format_optional(value: Any, fmt: str) -> str:
    """'n/d' si value es None/NaN, si no lo formatea con fmt (ej. '.2f', '+.2%')."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "n/d"
    return format(value, fmt)


def _format_portfolio_row(position: Dict[str, Any], analysis: Optional[Dict[str, Any]],
                           error: Optional[str]) -> str:
    w = _PORTFOLIO_COL_WIDTHS
    ticker = str(position.get("symbol", "?"))
    quantity_str = _format_optional(position.get("quantity"), ".2f")

    cost_basis_price = position.get("cost_basis_price")
    cost_basis_str = _format_optional(cost_basis_price, ".2f")

    cost_basis_money = position.get("cost_basis_money")
    unrealized_pnl = position.get("unrealized_pnl")
    if (cost_basis_money is not None and not pd.isna(cost_basis_money) and cost_basis_money != 0
            and unrealized_pnl is not None and not pd.isna(unrealized_pnl)):
        pnl_pct = unrealized_pnl / cost_basis_money
        pnl_str = f"{pnl_pct:+.2%} ({unrealized_pnl:+,.2f})"
    else:
        pnl_str = "n/d"

    if error is not None:
        current_price_str = "n/d"
        valuation_str = "n/d"
        vol_ann_str = "n/d"
        vol_pct_str = "n/d"
        beta_str = "n/d"
        drawdown_str = "n/d"
        review_str = "ERROR"
    else:
        current_price_str = _format_optional(analysis["precio_actual"], ".2f")
        valuation_str = _fmt_percentil(analysis["valuation_percentile"])
        vol_ann_str = _fmt(analysis["volatility_ann"], "pct", 0)
        vol_pct_str = _fmt_percentil(analysis["vol_percentile"])
        beta_str = _format_optional(analysis["beta"], ".2f")
        drawdown_str = _fmt(analysis["drawdown_mediano"], "pct", 0)
        # "REVISAR" o vacío. Vacío y no "OK": el sistema no está afirmando que
        # la posición esté bien, solo que ninguno de los tres disparadores
        # detectó un cambio material frente al primer registro.
        review_str = "REVISAR" if analysis["revisar"] else ""

    return (f"{ticker:<{w['ticker']}}{quantity_str:>{w['quantity']}}{cost_basis_str:>{w['cost_basis']}}"
            f"{current_price_str:>{w['current_price']}}{pnl_str:>{w['pnl']}}"
            f"{valuation_str:>{w['valuation_pct']}}{vol_ann_str:>{w['vol_ann']}}"
            f"{vol_pct_str:>{w['vol_pct']}}{beta_str:>{w['beta']}}"
            f"{drawdown_str:>{w['drawdown']}}{review_str:>{w['review']}}")


def _portfolio_review_explanation(position: Dict[str, Any], analysis: Dict[str, Any],
                                   config: PipelineConfig) -> str:
    """
    Explica una marca REVISAR nombrando el disparador concreto y sus números.

    "Revisar" significa "míralo tú", no una orden: por eso la explicación dice
    QUÉ cambió y cuánto, y no qué hacer al respecto.
    """
    ticker = position.get("symbol", "?")
    razones: List[str] = []
    motivos = analysis.get("motivos_revisar", {})

    if motivos.get("valoracion"):
        razones.append(
            f"el percentil de valoración pasó de {_fmt_percentil(analysis['valuation_percentile_earliest'])} "
            f"a {_fmt_percentil(analysis['valuation_percentile'])} desde el primer registro "
            f"(umbral: {config.review_threshold_valuation_percentile_jump:.0%} de salto)"
        )
    if motivos.get("volatilidad"):
        razones.append(
            f"el percentil de volatilidad condicional pasó de "
            f"{_fmt_percentil(analysis['vol_percentile_earliest'])} a "
            f"{_fmt_percentil(analysis['vol_percentile'])} — cambio de régimen "
            f"(umbral: {config.review_threshold_vol_percentile_jump:.0%})"
        )
    if motivos.get("margen"):
        razones.append(
            f"el margen neto TTM lleva {analysis['margen_trimestres_cayendo']} trimestres "
            f"consecutivos cayendo (umbral: {config.review_margin_declining_quarters}); "
            f"último: {_fmt(analysis['net_margin_ttm'])}"
        )
    return f"  - {ticker}: " + "; ".join(razones)


def run_portfolio_analysis(config: PipelineConfig) -> None:
    """
    Opción [2] del menú: carga las posiciones abiertas (load_positions) y
    muestra CONTEXTO por posición (portfolio_engine.evaluate_position).

    FASE 3.3 — SIN VEREDICTO. Ya no emite MANTENER/VIGILAR/VENDER ni usa la
    probabilidad ML para nada. Por posición muestra el precio actual, el precio
    medio de entrada y el P&L (contexto, NUNCA criterio — se mantiene el
    invariante de que el precio de entrada no entra en ninguna valoración), el
    percentil point-in-time de valoración, la volatilidad y su percentil, la
    beta, y el drawdown de entrada mediano al horizonte configurado. Marca
    "REVISAR" —no "vender"— cuando algo cambió de forma material frente a lo
    que quedó registrado la primera vez.

    Cada posición se procesa secuencialmente, mostrando progreso ("Analizando
    2/4: JD...") — no hay forma honesta de paralelizar sin ocultarle al
    usuario cuánto va a tardar cada ticker. Si el análisis de una posición
    falla (ej. un ADR con histórico corto, un ticker deslistado), se marca
    ERROR en su fila y se continúa con el resto.
    """
    print("\n" + "="*60)
    print("   CARTERA: CONTEXTO POR POSICIÓN")
    print("="*60 + "\n")

    try:
        positions = load_positions(config.portfolio_csv_path)
    except (ValueError, pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as e:
        # Un CSV presente pero ilegible (export a medias, separador inesperado,
        # cabecera sin las columnas mínimas, codificación distinta de cp1252)
        # lanzaba una excepción que solo el except de FileNotFoundError cubría,
        # así que MATABA el bucle del menú. El fichero existe: no toca repetir
        # las instrucciones de la Flex Query, sino decir qué no se pudo parsear.
        print(f" ⚠ El archivo de posiciones existe pero no se pudo leer: {config.portfolio_csv_path}")
        print(f"   {type(e).__name__}: {e}")
        print(" Comprobá que sea el CSV/TSV que exporta la Flex Query de IBKR sin editar, que")
        print(" tenga la columna 'Symbol' y que la descarga no se haya cortado a medias.\n")
        return
    except FileNotFoundError:
        print(f" ⚠ No se encontró el archivo de posiciones en: {config.portfolio_csv_path}\n")
        print(" Para generarlo desde Interactive Brokers:")
        print("   1. IBKR > Performance & Reports > Flex Queries")
        print("   2. Crear (o abrir) una Flex Query con una sección 'Posiciones abiertas'")
        print("      (Open Positions) que incluya al menos las columnas: CurrencyPrimary,")
        print("      AssetClass, Symbol, Description, Quantity, PositionValue, CostBasisMoney,")
        print("      PercentOfNAV, FifoPnlUnrealized, OpenDateTime.")
        print("   3. Formato de salida: CSV.")
        print(f"   4. Ejecutar la query y guardar el resultado en: {config.portfolio_csv_path}")
        print(f"      (o apuntar --portfolio-csv-path a donde lo hayas guardado)\n")
        return

    if positions.empty:
        print(" El archivo de posiciones no tiene ninguna posición de tipo STK — nada que analizar.\n")
        return

    n = len(positions)
    entries: List[Dict[str, Any]] = []
    for i, (_, position_row) in enumerate(positions.iterrows(), start=1):
        position = position_row.to_dict()
        ticker = position["symbol"]
        print(f" Analizando {i}/{n}: {ticker}...")
        try:
            analysis = evaluate_position(position, config)
            entries.append({"position": position, "analysis": analysis, "error": None})
            # Registro de predicciones también en modo [2] (Fase 1.12). El
            # payload es más chico que el del modo [1] porque analyze_position
            # es una versión reducida del pipeline (sin Monte Carlo ni
            # abanicos cuantílicos); el esquema aditivo del log tolera esa
            # diferencia por diseño. 'modo' distingue el origen.
            registrar_prediccion(
                path=config.prediction_log_path,
                ticker=ticker,
                config=config,
                payload={
                    "modo": "cartera",
                    "precio_actual": analysis["precio_actual"],
                    "revisar": analysis["revisar"],
                    "motivos_revisar": analysis["motivos_revisar"],
                    "valoracion_contexto": {
                        "percentil": analysis["valuation_percentile"],
                        "percentil_primer_registro": analysis["valuation_percentile_earliest"],
                        "n_quarters": analysis["valuation_n_quarters"],
                        "fiable": analysis["valuation_fiable"],
                    },
                    "riesgo": {
                        "volatility_ann": analysis["volatility_ann"],
                        "max_drawdown": analysis["max_drawdown"],
                        "beta": analysis["beta"],
                        "garch_vol_esperada": analysis["garch_vol_esperada"],
                    },
                    "vol_condicional": {
                        "percentil": analysis["vol_percentile"],
                        "percentil_primer_registro": analysis["vol_percentile_earliest"],
                    },
                    "drawdown_entrada": analysis["drawdown_horizonte"],
                    "salud_financiera": {
                        "net_margin_ttm": analysis["net_margin_ttm"],
                        "trimestres_cayendo": analysis["margen_trimestres_cayendo"],
                    },
                    # Contexto de la posición, NO entra en ninguna valoración
                    # (ver evaluate_position): se registra para poder
                    # reconstruir después qué se tenía en cada fecha.
                    "posicion": {
                        "cost_basis_price": analysis["cost_basis_price"],
                        "quantity": analysis["quantity"],
                        "unrealized_pnl": analysis["unrealized_pnl"],
                    },
                },
            )
        except Exception as e:
            print(f"   ⚠ Error analizando {ticker}: {e}")
            entries.append({"position": position, "analysis": None, "error": str(e)})

    header = _portfolio_table_header()
    print("\n" + header)
    print("-" * len(header))
    for entry in entries:
        print(_format_portfolio_row(entry["position"], entry["analysis"], entry["error"]))
    print()

    flagged = [e for e in entries if e["analysis"] is not None and e["analysis"]["revisar"]]
    if flagged:
        print(" POSICIONES A REVISAR (qué cambió y con qué números):")
        for e in flagged:
            print(_portfolio_review_explanation(e["position"], e["analysis"], config))
        print()

    sin_historico = [e for e in entries if e["analysis"] is not None
                     and not e["analysis"]["has_history"]]
    if sin_historico:
        # La explicación va UNA vez y la lista de tickers en una línea: es la
        # misma razón para todas, y repetirla por posición hacía que el bloque
        # tapara la tabla en la primera corrida, que es justo cuando aplica a
        # todas las posiciones a la vez.
        simbolos = ", ".join(str(e["position"].get("symbol", "?")) for e in sin_historico)
        print(f" SIN HISTÓRICO CON EL QUE COMPARAR (primera corrida de estas posiciones): {simbolos}")
        print("  Los disparadores de valoración y de volatilidad quedan NO EVALUABLES hasta que")
        print("  exista un registro anterior con el que comparar — no son un 'no ha cambiado'. El")
        print("  del margen neto TTM sí se evalúa, porque sale de los datos contables y no")
        print("  necesita histórico propio.")
        print()

    errored = [e for e in entries if e["error"] is not None]
    if errored:
        print(" ERRORES DURANTE EL ANÁLISIS (se excluyen de la tabla; el resto de la cartera")
        print(" se analizó igual):")
        for e in errored:
            print(f"  - {e['position'].get('symbol', '?')}: {e['error']}")
        print()

    print("-" * len(header))
    print(" NOTA: esta tabla es CONTEXTO, no un veredicto. No dice comprar, mantener ni vender:")
    print(" desde la Fase 3.3 el sistema no emite ninguna etiqueta de decisión, porque la que")
    print(" emitía descansaba en una señal cuyo poder predictivo se midió y es cero.")
    print(" REVISAR significa 'míralo tú': se marca cuando el percentil de valoración o el de")
    print(" volatilidad se movieron más que el umbral desde el primer registro de esa posición,")
    print(" o cuando el margen neto TTM cae varios trimestres seguidos. No es una orden.")
    print(" P.Medio es el precio de entrada: se muestra como CONTEXTO y NO participa en nada")
    print(" (es un coste hundido; la pregunta relevante es '¿compraría esto hoy?', no '¿gano o")
    print(" pierdo frente a lo que pagué?'). P&L viene del propio CSV, no se recalcula.")
    print(" Val. y Vol. son PERCENTILES dentro del histórico de cada activo (alto = caro / vol.")
    print(" alta para su propia historia). DD med = drawdown de entrada MEDIANO al horizonte")
    print(f" configurado ({config.horizon_months} meses), medido sobre todo el histórico.")
    print(" Las implicaciones fiscales de vender (tributación de plusvalías, regla de recompra")
    print(" a 2 meses en España) NO están consideradas acá — conviene consultarlas por separado.")
    print("="*60 + "\n")


def run_menu_loop(config: PipelineConfig) -> None:
    """
    Bucle principal del menú interactivo: repite hasta que el usuario elige
    salir, reutilizando el mismo PipelineConfig en cada análisis de la sesión.

    Está extraído del bloque `if __name__ == "__main__":` para que la
    degradación ante fallos de datos (Fase 1.7) se pueda probar por
    COMPORTAMIENTO y no inspeccionando el código fuente con regex.
    """
    while True:
        opcion = prompt_main_menu()
        if opcion == 0:
            print("\nHasta luego.\n")
            break
        elif opcion == 1:
            ticker_elegido = prompt_ticker()
            if ticker_elegido is None:
                continue
            # El modo [1] tiene que degradar como ya lo hace el modo [2]:
            # data_engine hace `raise e` ante un fallo de descarga y esto no
            # estaba capturado en ningún sitio, así que sin red (o con yfinance
            # caído, o con un ticker que dejó de devolver datos entre la
            # validación y el análisis) el programa escupía un traceback y se
            # SALÍA, perdiendo la sesión entera. Un fallo de datos es una
            # condición esperada de un pipeline que vive de una API externa,
            # no un bug del que haya que abortar.
            try:
                run_production_system(ticker_elegido, config)
            except KeyboardInterrupt:
                # Ctrl+C es una decisión del usuario, no un error: se respeta
                # y se vuelve al menú sin tratarlo como fallo.
                print("\n\n ⚠ Análisis interrumpido por el usuario — se vuelve al menú.\n")
            except Exception as e:
                print(f"\n ⚠ No se pudo completar el análisis de {ticker_elegido}: "
                      f"{type(e).__name__}: {e}")
                print(" Causas típicas: sin conexión, yfinance no responde, el ticker no tiene")
                print(" suficiente histórico, o SEC EDGAR rechazó la petición. El sistema vuelve")
                print(" al menú; se puede reintentar o probar con otro ticker.\n")
        elif opcion == 2:
            # run_portfolio_analysis ya captura por ticker, pero load_positions
            # puede fallar por causas que no son FileNotFoundError (ver el
            # try/except de esa función) y el resto del informe también podría
            # romperse; la misma red de seguridad aplica.
            try:
                run_portfolio_analysis(config)
            except KeyboardInterrupt:
                print("\n\n ⚠ Análisis de cartera interrumpido por el usuario — se vuelve al menú.\n")
            except Exception as e:
                print(f"\n ⚠ No se pudo completar el análisis de cartera: {type(e).__name__}: {e}\n")


if __name__ == "__main__":
    run_menu_loop(parse_args())

# python main_pipeline.py --horizon-months 6
