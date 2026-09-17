import argparse
import os
from dataclasses import dataclass, fields
from typing import Dict, Optional, Sequence

# Por debajo de este lookback, la ventana out-of-sample resultante es
# demasiado chica para que el backtest / la selección de modelo de
# MachineLearningEngine sean interpretables (ver PipelineConfig.lookback_years).
MIN_LOOKBACK_YEARS_FOR_INTERPRETABLE_BACKTEST = 8


@dataclass(frozen=True)
class PipelineConfig:
    """
    Punto único de configuración operativa del pipeline. Evita constantes
    hardcodeadas y duplicadas (horizonte, umbrales, tasa libre de riesgo,
    costes, semilla aleatoria) repartidas entre varios engines. El ticker
    a analizar YA NO vive acá — se pide interactivamente por consola (ver
    main_pipeline.prompt_ticker) porque es una elección de cada corrida, no
    un parámetro operativo del pipeline.

    No incluye constantes de metodología "de forma" (ventanas de
    momentum/RSI) — esas son decisiones cuantitativas fijas, no parámetros
    operativos de una corrida.

    FASE 3.3 — los pesos del score multifactorial (weight_*) YA NO EXISTEN.
    Vivían acá como excepción deliberada, esperando recalibrarse
    empíricamente; la Fase 2 midió el poder predictivo de la capa que
    ponderaban y no lo hay (IC medio -0,0093, `grupos_suficientes` False en
    los 10 tickers al horizonte de producción), así que el score entero sale
    del informe y sus pesos con él. Lo mismo con `recommendation_bands_path`:
    sin score no hay nada que cortar en bandas.
    """
    benchmark: str = "^GSPC"
    # Con 4 años (1008 filas): tras el calentamiento de mom_12m (252) y la cola
    # sin target resuelto (horizon_days, ~126), quedan ~630 filas usables; el
    # 20% de test (config.ml_train_size) son ~126 filas — con un target que
    # mira 126 días hacia adelante, eso es ~1 observación independiente, y
    # elegir el mejor de 3 modelos sobre eso es selección sobre ruido. Por
    # debajo de 8 años el backtest y la selección de modelo NO son
    # interpretables (ver MachineLearningEngine.generate_walkforward_probabilities).
    lookback_years: int = 12
    horizon_months: int = 6
    buy_threshold: float = 0.50
    risk_free_rate: float = 0.04
    transaction_cost_bps: float = 5.0
    slippage_bps: float = 5.0
    monte_carlo_simulations: int = 5000
    monte_carlo_target_return: float = 0.10
    ml_train_size: float = 0.8
    # Tamaño de bloque (en filas) del walk-forward de MachineLearningEngine:
    # tras la ventana de entrenamiento inicial (ml_train_size), reentrena cada
    # ml_block_days filas y predice solo el bloque siguiente, purgando
    # embargo_days en cada reajuste. Reemplaza el único split 80/20 tanto para
    # seleccionar el modelo ganador (Balanced-Accuracy agregada sobre todos
    # los bloques) como para las probabilidades históricas del backtest.
    ml_block_days: int = 63
    random_state: int = 42
    output_dir: str = "output"
    cache_dir: str = "cache"
    # 'edgar': SEC EDGAR companyfacts, point-in-time e indexado por fecha de
    # publicación (40-80 trimestres típicos). 'yfinance': ~5 trimestres sin
    # restatements ni point-in-time; también se usa automáticamente como
    # fallback si EDGAR falla (ver DataEngine.fetch_fundamental_data).
    fundamentals_source: str = "edgar"
    # Cada cuántos días hábiles se reajusta (MLE completo) el walk-forward de
    # GJR-GARCH/ARIMA en TimeSeriesFeatureExtractor; entre reajustes se propaga
    # con los mismos parámetros. Un valor más chico es más fiel a un despliegue
    # real (recalibra más seguido) pero más lento — ver la advertencia de log
    # si el walk-forward completo tarda más de ~60s.
    ts_refit_every: int = 21
    # Mismo reajuste walk-forward que ts_refit_every, pero usado por
    # portfolio_engine.analyze_position (modo cartera) en vez de ts_refit_every:
    # el walk-forward GARCH/ARIMA es, con diferencia, el paso más caro de todo
    # el pipeline, y analizar cada posición de una cartera debe tomar segundos,
    # no minutos. Reajustar cada trimestre (63 días hábiles) en vez de cada mes
    # (21) recorta sustancialmente el tiempo por ticker; se acepta perder algo
    # de fidelidad en la volatilidad/tendencia estimada porque la señal de
    # mantener/vender que consume ese resultado ya opera a resolución mensual
    # (horizon_months), no diaria — un reajuste trimestral no le resta
    # resolución práctica a esa decisión.
    portfolio_ts_refit_every: int = 63
    # Cómo estima RiskSimulationEngine la deriva anualizada del Monte Carlo:
    # 'historical' (media histórica, ruidosa con pocos años de datos),
    # 'zero' (sin tendencia), o 'shrunk' (default: la histórica encogida 50%
    # hacia risk_free_rate — ver RiskSimulationEngine.__init__).
    drift_mode: str = "shrunk"
    # --- Fase 3.6: calendario de negociación --------------------------------
    # Días de negociación por año del activo. 252 para acciones (5 sesiones
    # por semana menos festivos); 365 para cripto, que cotiza todos los días.
    #
    # Era un LITERAL repetido por todo el repo, y ese literal es incorrecto
    # para BTC: anualizar su volatilidad con sqrt(252) cuando genera 365
    # retornos al año la SUBESTIMA en un ~20% (sqrt(365/252) = 1,20), y un
    # "horizonte de 6 meses" de 126 sesiones son 6 meses de calendario en una
    # acción pero solo 4 en cripto. Ver "Ajustes y Arreglos.txt", primer punto
    # del fichero, abierto desde el principio del proyecto.
    #
    # QUÉ SE ENHEBRA CON ESTO (todo lo que es ANUALIZACIÓN o CALENDARIO):
    #   - horizon_days y trading_days_per_month (abajo).
    #   - DataEngine: el histórico efectivo en años, y el horizonte de los
    #     retornos forward.
    #   - RiskSimulationEngine: la deriva anualizada, la sigma anualizada, el
    #     dt del Monte Carlo, y la vol/vol a la baja de Sharpe y Sortino.
    #   - TechnicalFactorEngine: la volatilidad anualizada y la ventana de la
    #     beta rolling (que es una ventana de UN AÑO, no de 252 días).
    #   - BacktestAndScoringEngine: el factor de anualización de los retornos
    #     y del Sharpe del backtest.
    #   - ValidationEngine: la conversión Sharpe anual <-> diario del DSR.
    #   - EntryContextEngine y el abanico de bandas: los horizontes en días.
    #   - main_pipeline: el rango de precios "de los últimos 12 meses".
    #
    # QUÉ NO SE ENHEBRA, A PROPÓSITO: las ventanas de FORMA de la metodología
    # (los 21/252 días de las ventanas de momentum académico 12-1, el burn-in
    # de 252 observaciones del GARCH/ARIMA, el window_size de sus reajustes).
    # Esas definen la FORMA de un factor, no una conversión de unidades: el
    # momentum "12-1" es una definición de la literatura con esas ventanas
    # concretas, y cambiarlas no es adaptar una unidad sino redefinir el
    # factor. Con trading_days_per_year=365 esas ventanas siguen contando
    # sesiones, así que un "momentum 12-1" de cripto abarcaría 8 meses de
    # calendario: es una limitación REAL y conocida, documentada acá en vez de
    # medio arreglada. La detección automática de cripto también queda fuera.
    trading_days_per_year: int = 252
    # True (default): la FICHA DE ENTRADA completa (precio y valoración,
    # perfil de riesgo, "a qué te apuntas", salud financiera, advertencias).
    # False: solo el bloque central "A QUÉ TE APUNTAS" y las ADVERTENCIAS —
    # ver el informe en main_pipeline.run_production_system. En los dos casos
    # se calcula lo mismo; el flag solo controla cuánto se imprime.
    verbose: bool = True
    # Ruta al export de posiciones abiertas de IBKR (Flex Query) que consume
    # portfolio_engine.load_positions — ver ese módulo para el formato esperado.
    portfolio_csv_path: str = "portfolio/posiciones.csv"
    # --- Modo cartera [2]: umbrales de "REVISAR" (Fase 3.3) ---------------
    # El modo cartera dejó de emitir MANTENER/VIGILAR/VENDER y de usar
    # current_ml_prob para nada (los umbrales hold_threshold_* desaparecieron
    # con esa señal). Ahora solo marca "REVISAR" —que significa "míralo tú",
    # no una orden— cuando algo cambió de forma MATERIAL respecto de lo que
    # quedó registrado la primera vez que se analizó esa posición.
    #
    # Salto de PERCENTIL de valoración (en puntos de percentil, 0-1) desde el
    # primer registro del ticker. 0.40 es deliberadamente grande: un percentil
    # calculado sobre 40-80 trimestres se mueve solo con que entre un trimestre
    # nuevo, así que un umbral chico marcaría ruido contable como si fuera un
    # cambio de la tesis.
    review_threshold_valuation_percentile_jump: float = 0.40
    # Mismo criterio para el percentil de la volatilidad condicional GARCH:
    # es la operacionalización de "la volatilidad cambió de régimen".
    review_threshold_vol_percentile_jump: float = 0.40
    # Trimestres CONSECUTIVOS de caída del margen neto TTM que disparan
    # "REVISAR". Dos es el mínimo que distingue una tendencia de un trimestre
    # malo suelto; con uno, cualquier estacionalidad residual marcaría.
    review_margin_declining_quarters: int = 2

    # --- Fase 3.6: umbral de MATERIALIDAD del capital invertido (ROIC) ------
    # El ROIC es asintótico en cero porque su denominador es una RESTA de
    # números grandes: capital_invertido = deuda + equity - caja. Cuando el
    # resultado es una fracción diminuta de sus propios inputs, su error
    # relativo se amplifica por caja/capital_invertido, así que el ratio deja
    # de ser "muy alto" y pasa a ser NO ESTIMABLE. La Fase 1.5 solo cubrió el
    # caso <= 0 -> NaN; el caso asintótico (positivo pero diminuto) seguía
    # produciendo un ROIC del 18.300%.
    #
    # Si capital_invertido < esta fracción de (deuda + equity), el ROIC es NaN.
    #
    # POR QUÉ 0.10 (decisión del propietario, 2026-09-10):
    #   - La amplificación del error es caja/invertido = (1-frac)/frac. Exigir
    #     que un error del 1% en la caja —una diferencia normal de timing o de
    #     clasificación a cierre de trimestre— no supere un 10% de error en el
    #     ROIC obliga a frac >= 1/11 = 9,1%, que se redondea a 10%.
    #     Amplificación por tramo: frac 20% -> 4%, frac 10% -> 9%,
    #     frac 5% -> 19%, frac 1% -> 99% (ruido puro).
    #   - Ataca el MECANISMO (la cancelación catastrófica) y no un proxy suyo.
    #     La alternativa que contemplaba el plan (capital invertido < 1% de los
    #     ingresos TTM) anularía de paso a las empresas asset-light, cuya baja
    #     intensidad de capital es un hecho económico REAL y no un artefacto
    #     aritmético.
    #   - Es GRATIS en datos reales: medido sobre 8 tickers (AAPL, MSFT, META,
    #     AMZN, NVDA, JNJ, KO, XOM) y 182 trimestres con ROIC definido, el
    #     mínimo de invertido/(deuda+equity) es del 22,75% (NVDA, 2020-05) y el
    #     ROIC máximo es 0,905. El umbral no anula NI UNA observación real:
    #     dispara solo en el régimen patológico, que es lo que un umbral de
    #     materialidad debe hacer.
    #   - Se anula (NaN) en vez de recortar a un techo: un valor recortado se
    #     lee como una medición, y el invariante del repo es que un número que
    #     no se puede medir se declara ausente.
    # LÍMITE CONOCIDO: sobre el fixture sintético `fundamentals_net_cash`, que
    # cruza el cero a propósito, el umbral anula los trimestres con ROIC del
    # 3.190% y del 18.290% pero deja pasar uno del 1.750%. Ese fixture está
    # construido con un NOPAT/capital que ninguna empresa real tiene, así que
    # el residuo es de la fixture y no del umbral.
    roic_min_invested_capital_fraction: float = 0.10

    @property
    def trading_days_per_month(self) -> float:
        """
        Días de negociación por mes, derivados de trading_days_per_year.

        Con el default de acciones (252) sale exactamente 21, que es el valor
        que estaba hardcodeado por todo el repo; con 365 (cripto) sale 30,42.
        Se deriva en vez de ser un segundo campo para que no puedan quedar
        desalineados entre sí.
        """
        return self.trading_days_per_year / 12


    @property
    def horizon_days(self) -> int:
        """Horizonte en SESIONES. 6 meses -> 126 con acciones, 182 con cripto."""
        return int(round(self.horizon_months * self.trading_days_per_month))

    @property
    def ml_prob_history_path(self) -> str:
        """
        Ruta del histórico de probabilidades ML por ticker.

        FASE 3.3: YA NO SE ESCRIBE. El modo cartera dejó de usar la
        probabilidad ML, así que nada añade entradas nuevas; el fichero
        existente NO se borra porque es histórico —describe qué afirmaba el
        sistema en cada fecha— y `portfolio_engine.get_earliest_ml_prob`
        sigue pudiendo leerlo. El histórico que sí se alimenta ahora es
        `context_history_path`.
        """
        return os.path.join(self.cache_dir, "ml_prob_history.json")

    @property
    def context_history_path(self) -> str:
        """
        Ruta del histórico de CONTEXTO por ticker del modo cartera (Fase 3.3):
        {ticker: [{date, valuation_percentile, vol_percentile}, ...]}.

        Es lo que permite responder "¿cambió algo material desde que empecé a
        mirar esta posición?" sin recurrir a ninguna predicción: se comparan
        los percentiles de hoy contra los del primer registro. Reemplaza el
        papel que jugaba ml_prob_history_path, que dejó de escribirse.
        """
        return os.path.join(self.cache_dir, "context_history.json")

    @property
    def prediction_log_path(self) -> str:
        """
        Ruta del REGISTRO DE PREDICCIONES (JSON Lines, append-only): una línea
        por corrida con todo lo que el sistema afirmó ese día.

        Es la única forma de saber algún día si el sistema sirve. Todo lo demás
        que produce el pipeline se puede recalcular; esto no, porque describe
        qué se predijo ANTES de conocer el resultado. Cuanto antes empiece a
        acumular, antes sirve — ver prediction_log.registrar_prediccion y
        scripts/revisar_predicciones.py.
        """
        return os.path.join(self.cache_dir, "prediction_log.jsonl")


def parse_args(argv: Optional[Sequence[str]] = None) -> PipelineConfig:
    defaults = PipelineConfig()
    parser = argparse.ArgumentParser(description="Sistema Cuantitativo de Modelado Multifactorial")

    for f in fields(defaults):
        flag = "--" + f.name.replace("_", "-")
        if f.name == "fundamentals_source":
            parser.add_argument(flag, type=str, choices=["edgar", "yfinance"], default=getattr(defaults, f.name))
        elif f.name == "drift_mode":
            parser.add_argument(flag, type=str, choices=["historical", "zero", "shrunk"], default=getattr(defaults, f.name))
        elif f.type is bool:
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=getattr(defaults, f.name))
        else:
            parser.add_argument(flag, type=f.type if f.type in (int, float, str) else str,
                                 default=getattr(defaults, f.name))

    args = parser.parse_args(argv)
    return PipelineConfig(**vars(args))
