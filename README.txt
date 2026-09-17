=================================================================
       MANUAL DE REFERENCIA: FICHA DE ENTRADA
       Qué significa cada número del informe, y cómo leerlo
=================================================================

QUÉ RESPONDE ESTE SISTEMA, Y QUÉ NO
-----------------------------------------------------------------
Responde UNA pregunta: **"¿a qué me apunto si compro hoy a este precio?"**

NO responde "¿va a subir?". Esa diferencia no es un matiz de redacción: es la
única razón por la que los números de este informe se pueden comprobar. Todo
lo que aparece abajo es una afirmación DESCRIPTIVA sobre lo que ya pasó (con
el tamaño de muestra impreso al lado) o una PROYECCIÓN con sus supuestos a la
vista. No hay ninguna afirmación predictiva sin validar.

QUÉ SE ELIMINÓ EN SEPTIEMBRE DE 2026, Y POR QUÉ
  Hasta la Fase 3 del refactor este informe imprimía un Score Multifactorial
  de 0 a 100, un veredicto COMPRAR / MANTENER / EVITAR, una probabilidad de
  Machine Learning de "batir al benchmark", dos backtests y un veredicto de
  robustez ALTA/MEDIA/BAJA. Nada de eso existe ya.

  El motivo no es de estilo. Se midió el poder predictivo de esa capa con la
  vara del propio sistema (ventanas INDEPENDIENTES, no fechas solapadas) sobre
  10 tickers y 5 horizontes: de 49 celdas evaluables, solo 10 tenían un
  Information Coefficient con muestra suficiente — las diez en el horizonte de
  1 mes — y ahí salió -0,009 de media con el signo repartido 5 a 5. Al
  horizonte de producción (6 meses) NINGUNO de los 10 tickers tenía los dos
  grupos del event study estimables: el "exceso señal-control" que el informe
  imprimía cada corrida como su validación principal no tenía tamaño muestral
  para existir.

  Un promedio ponderado de cinco factores cuyo componente predictivo mide cero
  no deja de medir cero por promediarse: solo se vuelve más difícil de
  auditar, porque una etiqueta ("COMPRA") esconde de qué medición salió y con
  cuántas observaciones. Por eso tampoco se ha sustituido por otra etiqueta
  —ni "atractivo/neutro/caro", ni semáforos, ni estrellas—: cualquier resumen
  categórico de varias dimensiones en una palabra reintroduce exactamente lo
  que se eliminó.

  EL INFORME DESCRIBE; NO DICTAMINA. La decisión es del lector.

ESTRUCTURA DEL INFORME
-----------------------------------------------------------------
  FICHA DE ENTRADA — {TICKER} — {fecha}

    1. PRECIO Y VALORACIÓN
    2. PERFIL DE RIESGO
    3. A QUÉ TE APUNTAS SI COMPRAS HOY      <-- el bloque central
    4. SALUD FINANCIERA (descriptivo, sin puntuar)
    5. ADVERTENCIAS

  --verbose      (default) los cinco bloques.
  --no-verbose   solo el bloque 3 (A QUÉ TE APUNTAS) y el 5 (ADVERTENCIAS).

  Las ADVERTENCIAS se imprimen en los dos modos a propósito: sin un veredicto
  que diga cuánto fiarse de lo que se acaba de leer, la lista de lo que NO se
  sostiene es parte del resultado, no un apéndice que un flag pueda esconder.

  El flag controla cuánto se IMPRIME, nunca qué se calcula.

  Ejemplo: python main_pipeline.py --no-verbose
  (el ticker no es un flag: se elige en el menú interactivo que aparece al
  arrancar, opción [1])

  Este manual documenta la opción [1] del menú. La opción [2] (cartera) es un
  informe distinto y más chico — ver la Sección 6 al final.

CÓMO LEER CUALQUIER NÚMERO DE ESTE INFORME (leer esto primero)
-----------------------------------------------------------------
Tres convenciones se repiten en todas las secciones. Entenderlas una vez
ahorra malinterpretar la mitad del informe.

* "n" Y "n indep." NO SON LO MISMO, y la que importa es la segunda.
  Muchas métricas se calculan sobre ventanas que MIRAN HACIA ADELANTE (el
  drawdown a 6 meses de cada fecha histórica, por ejemplo). Dos fechas
  separadas por una semana comparten casi toda su ventana de 6 meses: no son
  dos observaciones independientes. La corrección es n / horizonte, y el
  informe la imprime al lado del n crudo:
      2.888 fechas con ventana de 126 días  ->  ~22,9 observaciones reales
  Un número sostenido por 4 ventanas independientes es ruido con aspecto de
  dato, por espectacular que parezca. Por eso está impreso.

* "fiable" ES UN FLAG DE TAMAÑO MUESTRAL, no de calidad del activo.
    sí   --> la distribución descansa en >= 10 ventanas independientes.
    NO   --> descansa en menos: el número está, pero no es evidencia.
    n/d  --> no se pudo evaluar (no es lo mismo que "NO").
  Esos tres estados aparecen en todo el sistema y siempre significan lo mismo.
  "n/d" nunca es un 0 disfrazado: un dato que falta se imprime como ausente.

* LOS PERCENTILES SON CONTRA SU PROPIA HISTORIA, y en orientación NATURAL.
  Un "p85" significa "el valor de hoy es más alto que el 85% de los valores
  que este mismo activo tuvo en su propio histórico". No es una comparación
  con otras empresas ni con un sector (eso sería un percentil transversal, que
  este sistema no calcula).
  Orientación natural = percentil alto significa VALOR alto, siempre, también
  donde "menos es mejor": un PER en p93 está caro, y una Deuda/Equity en p95
  significa "la deuda está en su nivel más alto del histórico". No hay
  métricas invertidas, para que toda la tabla se lea con una sola convención.

-----------------------------------------------------------------
1. PRECIO Y VALORACIÓN
-----------------------------------------------------------------
¿Qué mide?: A qué precio se entra hoy, y si ese precio es caro o barato
            RESPECTO DE COMO HA COTIZADO ESTA MISMA EMPRESA. La pregunta
            "¿estoy pagando caro frente a su propia historia?" es
            contestable; "¿va a subir?" no lo es.

* Precio actual (cierre de {fecha}):
  - Mide: el último cierre ajustado de la última sesión bursátil CERRADA. No
    es el precio en vivo: usar el precio vivo haría que dos corridas del mismo
    día dieran informes distintos (ver la sección de reproducibilidad).

* Rango de los últimos 12 meses (mín / medio / máx):
  - Mide: el recorrido del precio en el último año de sesiones. Contexto
    inmediato de si el precio de hoy está en la parte alta o baja de su año.

* Histórico efectivo utilizado:
  - Mide: los años de datos que hay DE VERDAD, no los que se pidieron. Un
    ticker que salió a bolsa hace 5 años analizado con una ventana de 12 años
    tiene 5 años de historia, y todo lo que dependa de la muestra queda
    debilitado en consecuencia.
  - Regla Práctica: por debajo de 8 años aparece una advertencia explícita en
    la Sección 5.

* Múltiplos point-in-time (PER, EV/EBITDA, P/B, P/S, rentabilidad por
  dividendo), con su percentil:
  - Mide: cada múltiplo de HOY y su percentil dentro del histórico de la
    propia empresa, con el número de trimestres que sostiene ese percentil.
  - "Point-in-time" quiere decir que cada trimestre histórico se calculó con
    el precio que REGÍA en su momento, no con el precio de hoy aplicado hacia
    atrás. Sin eso, el market cap queda congelado y el múltiplo se mueve solo
    con el denominador contable (que normalmente crece), así que el trimestre
    más reciente saldría SIEMPRE como el más barato de la historia,
    independientemente del precio real.
  - Interpretación:
      p0-p25    --> barato respecto de su propia historia
      p25-p75   --> en su rango habitual
      p75-p100  --> caro respecto de su propia historia
  - Regla Práctica: un percentil alto no es una señal de venta ni uno bajo una
    de compra. Una empresa puede estar cara porque ha mejorado su negocio, y
    barata porque se ha deteriorado. Esto mide lo que se paga, no lo que se
    recibe — la Sección 4 (SALUD FINANCIERA) es la que dice cómo va el negocio.
  - "fiable: NO" --> hay menos de 12 trimestres. Los VALORES siguen siendo el
    dato contable y valen; los PERCENTILES no significan nada sobre 5
    observaciones.
  - "n/d" en todos los múltiplos --> no se pudo calcular el market cap porque
    faltan las acciones en circulación en los datos de SEC EDGAR (le pasa a
    algunos tickers, META entre ellos). Es una carencia de la extracción de
    datos, no del precio: el bloque sale vacío y advertido en vez de
    inventarse un número.
  - Un múltiplo con denominador <= 0 (beneficio negativo, EBITDA negativo,
    equity negativo) sale NaN, nunca un número negativo: un PER de -8 no es
    "baratísimo", es que no está definido.

-----------------------------------------------------------------
2. PERFIL DE RIESGO
-----------------------------------------------------------------
¿Qué mide?: El riesgo del ACTIVO en sí (no de una estrategia de trading):
            cuánto se mueve, si se mueve más o menos que de costumbre, cuánto
            depende del mercado, y qué dice una simulación sobre el rango de
            resultados posibles desde el precio actual.

* Volatilidad condicional GARCH (hoy) + su percentil histórico:
  - Mide: la volatilidad diaria que el modelo GJR-GARCH estima para HOY, y
    dónde cae ese valor en el propio histórico de volatilidad del activo.
  - Por qué van los dos juntos: el nivel solo no dice si el activo está en un
    momento tranquilo o agitado PARA ÉL. Un 2,4% diario es alto en términos
    absolutos y puede ser perfectamente normal para ese activo (p50) o un
    máximo histórico (p97). Lo segundo es lo que cambia una decisión de entrada.
  - Regla Práctica: entrar con la volatilidad en p85+ significa comprar en un
    momento de estrés del activo; el bloque 3 (drawdown de entrada) dice qué
    ha pasado históricamente en general, no condicionado a este estado — ver
    la nota sobre análogos condicionales al final de la Sección 3.

* Volatilidad esperada al horizonte (GARCH):
  - Mide: la volatilidad diaria MEDIA que el modelo proyecta para el horizonte
    configurado, no la de hoy congelada. El GARCH modela reversión a la media:
    tras un pico de volatilidad, la esperada del horizonte es más baja que la
    de hoy, y tras una racha calma más alta.
  - Es el número que alimenta el Monte Carlo Y el abanico lognormal de la
    Sección 3, los dos, para que no puedan partir de supuestos distintos.

* Volatilidad anualizada (histórica) / benchmark / relativa:
  - Mide: la desviación estándar de los retornos diarios anualizada, la del
    benchmark, y el cociente entre ambas.
  - Rangos:
      > 40% anual  --> propio de crecimiento o especulación alta
      20-40%       --> rango habitual de una acción grande
      < 20%        --> defensiva / madura
      relativa > 2,0x --> se mueve el doble que el mercado

* Beta rolling (1 año) vs. benchmark:
  - Mide: cuánto se mueve el activo por cada 1% que se mueve el benchmark,
    calculado sobre el ÚLTIMO año de sesiones, no sobre todo el histórico. Un
    activo cambia de perfil de riesgo con el tiempo (una empresa que madura de
    crecimiento a value), y una beta de muestra completa no lo capta.
  - Rangos:
      Beta > 1,3  --> ALTA sensibilidad al mercado (amplifica subidas y bajadas)
      Beta ~ 1,0  --> se mueve en línea con el mercado
      Beta < 0,7  --> BAJA sensibilidad (más defensivo)

* Máximo drawdown histórico:
  - Mide: la peor caída del precio desde un máximo previo, sobre todo el
    histórico disponible. Es un hecho del activo, no una proyección.
  - Rangos:
      0 a -15%    --> excelente
      -15 a -25%  --> moderado
      -25 a -50%  --> alto
      más allá    --> el activo ya ha perdido la mitad de su valor al menos
                      una vez; conviene saberlo antes de entrar
  - No confundir con el DRAWDOWN DE ENTRADA de la Sección 3: este es el peor
    caso de TODA la historia desde un máximo; aquel es lo que le pasó a quien
    compró en una fecha cualquiera, que es la pregunta relevante al entrar.

* VaR diario 95%:
  - Mide: la pérdida diaria que históricamente solo se superó el 5% de las
    veces, en % y en euros/dólares sobre el precio actual.
  - Regla Práctica: es una pérdida de "mal día normal", no el peor escenario.

* Expected Shortfall diario:
  - Mide: la pérdida diaria PROMEDIO dentro de ese 5% peor. A diferencia del
    VaR, sí dice qué tan mala es la cola, no solo dónde empieza.
  - Regla Práctica: cuanto más se aleje del VaR, más "gorda" es la cola.

* Sharpe / Sortino ratio (histórico, del activo):
  - Mide: el retorno del ACTIVO por encima de la tasa libre de riesgo,
    ajustado por su volatilidad total (Sharpe) o solo por la volatilidad a la
    baja (Sortino, más indulgente con la volatilidad al alza).
  - Rangos:
      >= 2,0    --> excelente
      1,0-2,0   --> bueno
      0,5-1,0   --> mediocre
      < 0,5     --> pobre (el activo compensa mal su propio riesgo)
  - Se calculan con la media histórica REAL de los retornos, sin encoger.

* Monte Carlo (N simulaciones al horizonte):
  - Mide: miles de trayectorias de precio simuladas con un movimiento
    browniano geométrico, partiendo de la deriva y la volatilidad que se
    imprimen en la misma línea del encabezado. Resumido en:
      Precio medio simulado, probabilidad de acabar por debajo del precio de
      hoy, probabilidad de alcanzar el objetivo (+10% por defecto), e
      intervalo de confianza del 95%.
  - CÓMO LEERLO, Y ESTO ES LO IMPORTANTE: es una SIMULACIÓN bajo dos
    supuestos, no una medición de lo que pasó. Su probabilidad vale
    exactamente lo que valgan esos dos supuestos, y por eso están impresos al
    lado (deriva log anual y volatilidad diaria). Cambiar --drift-mode a
    'zero' cambia todas estas probabilidades.
  - La deriva por defecto ('shrunk') es la media histórica encogida un 50%
    hacia la tasa libre de riesgo. Motivo: la media histórica de unos pocos
    años tiene un error estándar enorme (~15% anual para una acción típica),
    así que usarla cruda es proyectar ruido. Encogerla es un seguro contra eso.

-----------------------------------------------------------------
3. A QUÉ TE APUNTAS SI COMPRAS HOY  (el bloque central)
-----------------------------------------------------------------
¿Qué mide?: La distribución de lo que históricamente le pasó a quien compró
            este activo en una fecha CUALQUIERA y aguantó H, para cinco
            horizontes: 1, 3, 6, 12 y 24 meses.

            Son afirmaciones sobre el pasado, verificables, y sin condicionar
            al estado de hoy. No son una predicción. Es el bloque que más
            debería influir en la decisión, y el único que se imprime con
            --no-verbose.

POR QUÉ CINCO HORIZONTES Y NO UNO. El sistema anterior proyectaba a 6 meses
fijos porque era el ancla con que se había construido, no porque los datos
dijeran que ahí funciona. Si la estrategia real es vender de forma
discrecional, un solo horizonte es una suposición; un abanico responde la
pregunta de verdad y además deja ver cómo empeora todo al alargar el plazo.

* 3.1 DRAWDOWN DE ENTRADA  (la métrica más útil del informe)
  - Mide: para cada fecha histórica t y cada horizonte H,
        min(P[t..t+H]) / P[t] - 1
    o sea CUÁNTO LLEGÓ A CAER EL PRECIO POR DEBAJO DE LO QUE SE PAGÓ, en el
    peor momento de la ventana. Se reportan los percentiles P5/P25/mediana/
    P75/P95 de esa distribución.
  - Por qué importa más que un retorno esperado: es lo que determina si se
    aguanta la posición o se vende en pánico. Un activo puede acabar en verde
    a 12 meses después de haber estado un 40% en rojo por el camino, y esas
    dos experiencias no son la misma inversión.
  - Está ACOTADO EN 0 por arriba: la ventana incluye la fecha de compra, así
    que si el precio nunca bajó del precio pagado la respuesta correcta es
    "nunca estuviste en pérdidas" (0%), no un número positivo.
  - Cómo leer cada columna:
      Mediana --> la experiencia típica: la mitad de las entradas históricas
                  vieron una caída peor que esta, y la mitad mejor.
      P25     --> una entrada de cada cuatro vio algo peor que esto.
      P5      --> el 5% peor de las entradas. NO es el peor caso posible: es
                  el peor caso de la muestra que tocó. Un escenario peor no
                  está descartado, simplemente no ocurrió en este histórico.
      P95     --> normalmente 0,0%: son las entradas que nunca estuvieron
                  en rojo.
  - Regla Práctica: la pregunta que hay que hacerse es si se aguantaría el P25
    sin vender. Si la respuesta es no, el tamaño de la posición es demasiado
    grande, y eso se sabe ANTES de entrar, no durante la caída.
  - Solo se usan las fechas con la ventana COMPLETA observada (se descartan
    las últimas H): de las demás no se sabe cuánto habrían caído, y
    rellenarlas sesgaría la distribución hacia drawdowns pequeños justo en el
    tramo más reciente.

* 3.2 TIEMPO HASTA RECUPERACIÓN
  - Mide: de las entradas que SÍ se pusieron en pérdidas, cuántos días
    pasaron hasta volver al precio pagado. Se reportan la mediana, el P95, el
    porcentaje que no recupera dentro del horizonte, y el porcentaje que nunca
    estuvo bajo agua.
  - "No recup." --> de las que se hundieron, cuántas NO habían vuelto al
    precio de entrada al terminar el horizonte. Suele ser la cifra más
    importante del bloque: una mediana de 7 días no dice nada si un 28% no
    recuperó nunca dentro del plazo.
    Esas entradas se EXCLUYEN de la mediana en vez de asignarles un número
    grande inventado, que contaminaría el percentil.
  - "Nunca baja" --> entradas que jamás cotizaron por debajo de lo pagado, y
    que por tanto no tenían nada que recuperar. Se reportan aparte porque son
    información por sí mismas.
  - POR QUÉ SOLO CUENTA LAS QUE SE HUNDIERON: la lectura literal ("primer día
    en que el precio vuelve a estar por encima del de compra") devuelve 1 día
    para cualquier entrada cuyo precio suba al día siguiente, AUNQUE después
    se hunda un 30% y tarde dos años en volver. En un activo alcista eso daría
    una mediana de 1 día que no informa de nada. Lo que se mide es: "si
    compro y la posición se pone en rojo, cuánto tardo en volver a estar en
    paz".
  - Regla Práctica: la mediana suele ser corta (una semana) y el P95 largo
    (meses). El P95 es el que hay que poder tolerar.

* 3.3 BANDAS DE PRECIO EN ABANICO
  Dos rutas independientes para el mismo horizonte, a propósito: si coinciden,
  la conclusión es más robusta; si no, la discrepancia misma es información.

  (a) CLÁSICA LOGNORMAL:  precio · exp(deriva ± z·sigma)
    - Mide: el rango de precios bajo el supuesto de que los retornos son
      lognormales, con la deriva y la volatilidad GARCH que se imprimen en la
      cabecera del bloque (las MISMAS que usa el Monte Carlo de la Sección 2).
    - Bandas: 80% (zona operativa estrecha), 90% (estándar), 95% (límite de
      riesgo). No hay banda del 99%: estimar un cuantil tan extremo con el
      histórico disponible equivale a ~2,5 observaciones efectivas en la cola
      — sería un número que aparenta precisión sin ser estimable.
    - LIMITACIÓN DECLARADA: la volatilidad se escala por sqrt(t), lo que asume
      retornos independientes e ignora la reversión a la media que el propio
      GARCH modela. A 12 y 24 meses eso es una suposición fuerte y la banda
      sale más ancha de lo que el modelo de volatilidad implicaría. Los
      horizontes largos son una EXTRAPOLACIÓN, no una medición, y el informe
      lo advierte en la Sección 5.

  (b) CUANTÍLICA (una regresión por cuantil y por horizonte):
    - Mide: los percentiles del retorno futuro estimados directamente con
      Gradient Boosting, sin asumir normalidad ni simetría. Captura sesgo y
      colas gordas reales.
    - Se corrige el "cruce de cuantiles" antes de convertir a precio: los 7
      modelos son independientes y nada les impide predecir un p05 por encima
      del p50, así que se reordenan (rearrangement de Chernozhukov,
      Fernández-Val y Galichon, 2010). En AAPL, un 32,6% de las filas de test
      tenían cuantiles cruzados: no es un caso raro.
    - COBERTURA EMPÍRICA (la tabla que va debajo): de todas las veces que se
      proyectó una banda del 90% sobre datos que el modelo NUNCA vio, qué
      porcentaje contuvo el precio real. Es la validación honesta de estas
      bandas, y va impresa al lado de cada horizonte con su número de
      observaciones de test.
      Interpretación:
        cobertura ~ nominal (±15 pts)  --> la banda es utilizable
        cobertura << nominal           --> la banda es DEMASIADO ESTRECHA: el
                                           riesgo real es mayor que el que
                                           dibuja. El informe lo advierte.
      Regla Práctica: en la práctica, las bandas de 1 mes y de 24 meses salen
      mal calibradas con frecuencia (cobertura del 30-60% donde prometen
      80-95%), y las de 3-12 meses razonablemente. Una banda mal calibrada NO
      debe usarse como referencia: está impresa con su advertencia justamente
      para que se pueda descartar con criterio.
    - "OMITIDO" en un horizonte --> el histórico no alcanza para entrenar ese
      target (mira H días hacia adelante y se pierde esa cola). Se dice el
      motivo en vez de presentar cuatro horizontes como si fueran cinco.

* NOTA — VERSIÓN CONDICIONADA AL ESTADO DE HOY: NO EXISTE, POR DECISIÓN.
  Sería natural querer estas mismas distribuciones restringidas a las fechas
  históricas cuyo estado se pareciera al de hoy (volatilidad parecida, caída
  desde máximos parecida). No se implementa: con 12 años y ventanas de 126
  días, el número de análogos INDEPENDIENTES cae casi siempre por debajo del
  mínimo fiable, y es exactamente la puerta por la que el sobreajuste volvería
  a entrar — justo después de haber eliminado la capa ML por esa misma razón.
  Las distribuciones son INCONDICIONALES sobre todo el histórico disponible.

-----------------------------------------------------------------
4. SALUD FINANCIERA (descriptivo, sin puntuar)
-----------------------------------------------------------------
¿Qué mide?: Cómo va el negocio, en crudo. Por métrica: el valor del último
            trimestre disponible, su percentil dentro del histórico de la
            propia empresa, la mediana histórica, la variación frente al MISMO
            trimestre del año anterior, el n, y la cobertura real de la
            ventana de la que sale el valor.

            NO PUNTÚA NADA. Antes estas métricas se promediaban con un score
            de "nivel absoluto" en un único número de 0 a 1, y el resultado no
            era interpretable: un 0,62 podía ser "excelente en absoluto y en su
            peor momento histórico" o "mediocre y en su mejor momento". Ahora
            el valor y el percentil se imprimen por separado.

* Rentabilidad — ROE y ROIC (TTM):
  - Miden: beneficio neto sobre fondos propios (ROE) y beneficio operativo
    después de impuestos sobre capital invertido (ROIC).
  - Rangos orientativos (referencias generales, no sectoriales):
      ROE  > 15%  --> excelente;  5-15% normal;  < 5% pobre
      ROIC > 12%  --> excelente;  5-12% normal;  < 5% pobre
  - El ROIC puede salir "n/d" por dos reglas distintas, las dos deliberadas:
      capital invertido <= 0     --> no está definido (empresa net-cash: la
                                     caja supera deuda + fondos propios)
      capital invertido < 10% de (deuda + fondos propios) --> NO ESTIMABLE.
    La segunda merece explicación: el capital invertido es una RESTA de
    números grandes (deuda + equity - caja), así que cuando lo que sobrevive a
    la resta es una fracción diminuta de sus propios inputs, el error relativo
    del denominador lo domina. Con el capital invertido al 1% del bruto, una
    diferencia del 1% en la cifra de caja —una reclasificación normal a cierre
    de trimestre— es un 99% de error en el ROIC. Sin esta regla el sistema
    llegaba a imprimir un ROIC del 18.300%. Umbral configurable con
    --roic-min-invested-capital-fraction; el 10% acota el error a ~9x y no
    anula ninguna observación real medida sobre 8 tickers y 182 trimestres.

* Márgenes — operativo y neto (TTM):
  - Miden: beneficio operativo y neto sobre ingresos, ambos en TTM.
  - Rangos orientativos:
      Margen neto > 15%  --> excelente;  5-15% normal;  < 5% ajustado
  - Van en TTM (numerador y denominador) para que no los contamine la
    estacionalidad de un trimestre suelto.

* Apalancamiento — Deuda/Equity, Deuda/EBITDA, cobertura de intereses:
  - Rangos orientativos:
      Deuda/Equity   < 1,0 saludable;  1,0-2,0 vigilar;  > 3,0 alto
      Deuda/EBITDA   < 3,0 saludable;  3,0-4,0 vigilar;  > 5,0 alto
      Cobertura int.  > 5x cómoda;  2-5x ajustada;  < 2x preocupante
  - Un Deuda/Equity NEGATIVO no es un error: significa fondos propios
    negativos, un estado contable real y relevante (empresas que han
    recomprado acciones financiándose con deuda). Se imprime tal cual en vez
    de anularse.

* Crecimiento YoY — ingresos, beneficio, flujo de caja libre:
  - Miden: la variación frente al MISMO trimestre del año anterior, no frente
    al trimestre inmediato anterior (que estaría contaminado por
    estacionalidad: el Q4 de una empresa de consumo no es "crecimiento", es el
    ciclo normal del negocio).

* "Δ vs. 1 año":
  - Mide: cuánto ha cambiado la métrica respecto del mismo trimestre del año
    anterior. En una métrica que ya es un porcentaje (un margen), la
    diferencia está en PUNTOS porcentuales. Es la "tendencia" del bloque.

* "VENTANA"  (columna nueva, y conviene entenderla)
  - Mide: cuántos días de calendario cubre DE VERDAD la ventana de la que sale
    el valor de hoy. Se esperan 365 ± 45, y un "!" marca que se sale.
    "—" es una métrica de balance (un saldo a una fecha), que no depende de
    ventana ninguna.
  - POR QUÉ EXISTE: los cálculos "TTM" y "YoY" usan `.rolling(4)` y
    `.pct_change(4)`, que cuentan FILAS, no trimestres. Los datos de SEC EDGAR
    tienen huecos (trimestres que no se pudieron derivar y se descartan), así
    que una ventana "de 12 meses" puede abarcar bastante más tiempo real.
  - LA MAGNITUD, MEDIDA: sobre 7 tickers y 282 trimestres reales, el 69% de
    las ventanas TTM y el 78% de las YoY NO cubren 12 meses, y la cobertura
    MEDIANA de una ventana TTM es de 455 días. Es decir: el "TTM" típico de
    este sistema es en realidad un acumulado de ~15 meses. El peor hueco
    encontrado es de 1.099 días (AMZN). El único ticker sano de la muestra es
    META (mediana 365).
  - Regla Práctica: si el valor de hoy lleva "!", ese ROE "TTM" no es
    comparable ni con el mismo dato de otra fecha ni con el umbral anual de la
    literatura de la lista de rangos de arriba. Y si la nota al pie dice que
    la mayoría del histórico está fuera de tolerancia, el PERCENTIL de esa
    métrica está comparando ventanas de duraciones distintas entre sí.
  - No está arreglado, está MEDIDO Y PUBLICADO. Arreglarlo del todo exigiría
    reindexar la extracción de EDGAR a trimestres fiscales; ocultarlo no era
    una opción.

-----------------------------------------------------------------
5. ADVERTENCIAS
-----------------------------------------------------------------
¿Qué mide?: La lista de lo que NO se sostiene de todo lo que se acaba de leer,
            con la causa y la cifra concreta de cada caso.

Sin un veredicto que resuma cuánto fiarse del informe, esta lista ES parte del
resultado. Se imprime en los dos modos de verbosidad. Cada entrada nombra su
causa, nunca dice "puede haber problemas con los datos".

Las advertencias posibles:
  * lookback_years por debajo del mínimo (8 años).
  * histórico EFECTIVO corto: se pidieron N años pero el activo no cotiza
    desde entonces.
  * múltiplos de valoración sobre menos de 12 trimestres.
  * ficha de salud financiera sobre menos de 12 trimestres.
  * drawdown de entrada NO fiable a los horizontes X (menos de 10 ventanas
    independientes). Es normal verla en 24 meses: con 12 años de datos, 24
    meses dan ~5 ventanas independientes. Es el diseño funcionando.
  * tiempo hasta recuperación NO fiable a los horizontes X.
  * banda cuantílica OMITIDA en un horizonte, con el motivo.
  * banda cuantílica MAL CALIBRADA fuera de muestra, con su cobertura real
    frente a la nominal y el n de test.
  * ventanas TTM/YoY que no cubren 12 meses reales (ver Sección 4).
  * el valor de HOY de una métrica sale de una ventana que no cubre 12 meses.
  * las bandas lognormales a 12 y 24 meses son una extrapolación (sqrt(t)).

Si no hay ninguna, lo dice explícitamente.

-----------------------------------------------------------------
6. OPCIÓN [2] DEL MENÚ — CARTERA: CONTEXTO POR POSICIÓN
-----------------------------------------------------------------
¿Qué mide?: Una tabla de contexto por cada posición abierta del export de
            IBKR (portfolio/posiciones.csv, una Flex Query de "Open
            Positions"). ES CONTEXTO, NO UN VEREDICTO: no dice comprar,
            mantener ni vender.

Columnas:
  Ticker | Cant. | P.Medio | P.Actual | P&L | Val. | Vol.ann | Vol. | Beta |
  DD med | Revisar

  * P.Medio (precio medio de entrada) y P&L: CONTEXTO, y no participan en
    NADA de lo que se mide. El precio de entrada es un coste hundido: la
    pregunta que gestiona el riesgo de una posición no es "¿gano o pierdo
    frente a lo que pagué?" sino "¿qué tengo delante hoy?". Meterlo en la
    lectura introduciría efecto de disposición (aferrarse a una perdedora
    esperando "recuperar" el precio de entrada, o vender una ganadora
    demasiado pronto) y anclaje al precio de compra — dos sesgos documentados,
    no una ventaja informativa. El P&L se toma del propio CSV, no se recalcula.
  * Val. --> percentil point-in-time del múltiplo de valoración (alto = caro
    para su propia historia). Usa el PRIMER múltiplo con dato del orden
    PER -> EV/EBITDA -> P/S -> P/B. No promedia varios a propósito: un
    promedio de percentiles vuelve a ser un número compuesto que esconde de
    qué salió.
  * Vol.ann --> volatilidad anualizada del activo.
  * Vol. --> percentil de la volatilidad condicional de hoy en su propio
    histórico (alto = el activo está agitado para lo que es habitual en él).
  * Beta --> beta rolling de 1 año vs. el benchmark.
  * DD med --> drawdown de entrada MEDIANO al horizonte configurado, medido
    sobre todo el histórico (la métrica de la Sección 3.1, resumida en un
    número por posición).
  * Revisar --> "REVISAR" o vacío.

QUÉ SIGNIFICA "REVISAR": significa **"míralo tú"**, no una orden de vender.
Se marca cuando algo cambió de forma MATERIAL frente a lo que quedó registrado
la PRIMERA vez que se analizó esa posición, con tres disparadores
independientes y sus umbrales en config.py:

  * el percentil de valoración se movió más de 40 puntos
    (--review-threshold-valuation-percentile-jump). En valor ABSOLUTO: que se
    haya vuelto mucho más barata también es un cambio material que merece una
    mirada. El sistema no decide en qué dirección es "malo".
  * el percentil de volatilidad se movió más de 40 puntos
    (--review-threshold-vol-percentile-jump). Es la operacionalización de
    "cambió de régimen". Un umbral absoluto de volatilidad marcaría siempre a
    los mismos activos por ser lo que son, no por haber cambiado.
  * el margen neto TTM lleva 2 trimestres consecutivos cayendo
    (--review-margin-declining-quarters). Este NO necesita histórico previo,
    así que es el único evaluable en la primera corrida — y solo cuenta
    trimestres cuya ventana TTM cubre de verdad 12 meses, para no confundir un
    hueco de EDGAR con un deterioro del negocio.

  Cada disparador vale "no evaluable" (y NUNCA "no ha cambiado") cuando no hay
  registro previo o el percentil no se pudo calcular. Un "no evaluable" nunca
  marca REVISAR: el sistema no se alarma por lo que no pudo medir.

  Vacío en la columna Revisar NO significa "está bien": significa que ninguno
  de los tres disparadores detectó un cambio material.

PRIMERA CORRIDA DE UNA POSICIÓN: aparece un bloque "SIN HISTÓRICO CON EL QUE
COMPARAR" listando los tickers cuyos disparadores relativos quedan no
evaluables. Es el estado esperado, no un fallo: el histórico de contexto
(cache/context_history.json) se empieza a poblar con la primera corrida.

LO QUE NO ESTÁ CONSIDERADO: las implicaciones fiscales de vender (tributación
de plusvalías, regla de recompra a 2 meses en España). Conviene consultarlas
por separado.

-----------------------------------------------------------------
7. GRÁFICOS (carpeta output/)
-----------------------------------------------------------------
Dos PNG por corrida:
  * {TICKER}_monte_carlo.png     --> histograma de los precios finales
    simulados, con el precio actual y la mediana simulada marcados.
  * {TICKER}_garch_volatility.png --> la serie de volatilidad condicional
    GARCH, útil para ver visualmente si el percentil de volatilidad de hoy
    (Sección 2) es un pico o parte de un régimen sostenido.

Un fallo al guardar gráficos no tumba el informe: se avisa y se sigue.

LA CURVA DE EQUITY DESAPARECIÓ: dibujaba un backtest de rebalanceo diario
según la probabilidad ML, o sea la señal que se eliminó de la ruta de decisión.
Mantenerla habría obligado a entrenar el modelo en cada corrida solo para
pintar el gráfico.

=================================================================
  REGISTRO DE PREDICCIONES (cache/prediction_log.jsonl)
=================================================================

QUÉ ES:
  Un fichero al que cada corrida — opción [1] y opción [2], con --verbose y
  con --no-verbose — añade UNA línea con todo lo que el sistema afirmó ese
  día: precio, contexto de valoración, salud financiera, percentil de
  volatilidad, distribuciones de drawdown y recuperación, los dos abanicos de
  bandas con su validación, las advertencias activas, y los parámetros con los
  que se calculó todo eso.

POR QUÉ ES LO MÁS IMPORTANTE QUE ESCRIBE EL SISTEMA:
  Todo lo demás que produce el pipeline (el informe, los gráficos, la caché de
  precios) se puede volver a calcular. Esto no, porque su valor está
  precisamente en haberse escrito ANTES de conocer el resultado. Es la única
  forma de responder algún día si este sistema sirve, y solo funciona si se
  acumula con el tiempo.

CÓMO CONSULTARLO:
    venv\Scripts\python.exe scripts/revisar_predicciones.py
    venv\Scripts\python.exe scripts/revisar_predicciones.py --ticker META
    venv\Scripts\python.exe scripts/revisar_predicciones.py --min-dias 90
    venv\Scripts\python.exe scripts/revisar_predicciones.py --sin-red

  Muestra, por predicción: cuántos días lleva, el retorno realizado desde
  entonces, y si el precio de hoy cayó DENTRO de la banda del 90% que se había
  proyectado — que es una afirmación de CALIBRACIÓN, y es la que se puede
  seguir comprobando ahora que el sistema no afirma ninguna dirección.

  Las entradas anteriores a septiembre de 2026 llevan además Score,
  recomendación y probabilidad ML, y el script las sigue puntuando: son la
  única evidencia que quedará de si aquella capa servía. Las nuevas no llevan
  nada de eso, y su ausencia no es un dato que falte — es que el sistema dejó
  de afirmar una dirección.

FORMATO:
  JSON Lines: una línea JSON autocontenida por corrida. Si una línea se
  corrompe (una corrida interrumpida a mitad de escritura), se descarta sola y
  no arrastra al resto del fichero.
  El esquema es ADITIVO: las claves son opcionales y varían entre entradas de
  distintas épocas del sistema. Cualquier herramienta que lo lea debe tolerar
  claves ausentes en vez de asumir un formato fijo — así las predicciones
  antiguas siguen siendo válidas cuando el sistema cambia, que es justo para lo
  que existe este fichero.

=================================================================
  REPRODUCIBILIDAD, HISTÓRICO Y CALENDARIO
=================================================================

DOS CORRIDAS DEL MISMO DÍA DAN EL MISMO INFORME:
  Un informe que no se puede reproducir no se puede auditar. Cuatro cosas lo
  garantizan:
  * Los precios se descargan hasta la última sesión bursátil CERRADA, no hasta
    "hoy". Corriendo en horario de mercado, la última barra sería un precio
    VIVO que cambia entre una corrida y la siguiente.
  * El panel de precios se guarda en caché (cache/prices_*.pkl) por
    combinación de tickers y fechas, así que dos corridas del mismo día leen
    exactamente los mismos datos.
  * Los modelos se ejecutan en un solo hilo: con varios, el orden de las sumas
    en punto flotante varía y el modelo ajustado difiere en los últimos
    decimales.
  * La lista de tickers se deduplica preservando el orden, no con un `set`
    (cuyo orden de iteración cambia entre PROCESOS).
  Verificado en datos reales: dos corridas de META en procesos distintos
  producen informes byte a byte idénticos.

HISTÓRICO SOLICITADO vs. EFECTIVO:
  Los precios NO se rellenan hacia atrás. Un ticker que salió a bolsa en 2020
  analizado con una ventana de 12 años tiene 5 años de datos, no 12, y el
  panel se recorta al primer precio real en vez de replicar el precio de salida
  a bolsa hacia el pasado (lo que fabricaría años de retornos exactamente
  cero: volatilidad artificialmente baja, momentum cero, beta cero).
  El informe imprime el histórico efectivo y avisa cuando queda por debajo de
  8 años.

CALENDARIO DE NEGOCIACIÓN (--trading-days-per-year):
  252 por defecto (acciones: 5 sesiones por semana menos festivos). Para un
  activo que cotiza los 365 días del año, como BTC, hay que pasarle 365:

      python main_pipeline.py --trading-days-per-year 365

  Sin eso, la volatilidad anualizada sale SUBESTIMADA en un factor
  sqrt(365/252) = 1,20 —y con ella el Sharpe, el Sortino, la sigma del Monte
  Carlo y todas las bandas de precio— y un "horizonte de 6 meses" son 126
  sesiones, o sea cuatro meses de calendario en vez de seis.

  QUÉ AJUSTA ESTE FLAG: la anualización de la volatilidad, el Sharpe, el
  Sortino, la deriva y el dt del Monte Carlo, la ventana de la beta rolling
  (que es una ventana de un año), los cinco horizontes del abanico, el
  histórico efectivo en años, y el horizonte de los retornos forward.

  QUÉ NO AJUSTA, A PROPÓSITO: las ventanas de FORMA de la metodología — los
  21/252 días del momentum académico "12-1" y el burn-in de 252 observaciones
  del GARCH/ARIMA. Esas definen un factor de la literatura con esas ventanas
  concretas; cambiarlas no sería adaptar el calendario sino redefinir el
  factor. Es una limitación conocida: con 365, el "momentum 12-1" de un activo
  de cripto sigue contando 252 sesiones, o sea 8 meses de calendario.
  La detección automática de cripto queda fuera: el flag es explícito.

=================================================================
  PARÁMETROS DE LÍNEA DE COMANDOS
=================================================================
Todos los parámetros operativos viven en config.py y se pueden pasar por CLI.
`python main_pipeline.py --help` da la lista completa. Los que más cambian lo
que se imprime:

  --horizon-months 6            horizonte del Monte Carlo y de la volatilidad
                                esperada (el abanico usa siempre 1/3/6/12/24)
  --lookback-years 12           años de histórico solicitados
  --trading-days-per-year 252   calendario del activo (365 para cripto)
  --drift-mode shrunk           historical | zero | shrunk — cambia el Monte
                                Carlo y el abanico lognormal
  --monte-carlo-simulations 5000
  --monte-carlo-target-return 0.10   el objetivo cuya probabilidad se reporta
  --risk-free-rate 0.04         entra en Sharpe, Sortino y la deriva encogida
  --fundamentals-source edgar   edgar | yfinance
  --ts-refit-every 21           cada cuántos días se reajusta el GARCH/ARIMA
  --no-verbose                  solo el bloque central y las advertencias
  --roic-min-invested-capital-fraction 0.10   umbral de materialidad del ROIC
  --review-threshold-valuation-percentile-jump 0.40   umbrales de "REVISAR"
  --review-threshold-vol-percentile-jump 0.40         del modo cartera
  --review-margin-declining-quarters 2

=================================================================
