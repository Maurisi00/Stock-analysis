# Plan de refactor — de "recomendador" a "herramienta de entrada"

Documento de especificación para ejecutar con Claude Code. Complementa a
`AUDITORIA_2026-09.md` (diagnóstico); este fichero dice **qué hacer**.

**Contexto de la decisión.** La estrategia real del propietario es: *"¿es buena idea
comprar ahora?"*, sin horizonte de mantenimiento fijo — vende de forma discrecional.
Eso invalida el ancla de 126 días sobre la que está construido el 80% del sistema
(target del ML, bandas cuantílicas, Monte Carlo, event study). El refactor reorienta el
sistema hacia la única pregunta que **sí** es contestable con una sola acción:
*"¿a qué me estoy apuntando si compro hoy a este precio?"*.

---

## Reglas transversales (aplican a TODAS las fases)

1. **Entorno Windows con venv en el repo.** Ejecutar los tests con
   `venv\Scripts\python.exe -m pytest`. No hay linter configurado — no añadir uno.
2. **Una fase = una rama = un commit final.** No mezclar fases. `pytest` en verde antes
   de cerrar cada fase.
3. **Varios tests de caracterización van a fallar A PROPÓSITO.** Son tests que fijan el
   comportamiento *actual*, y varias correcciones cambian ese comportamiento. **No
   debilitar la aserción para que pase.** Procedimiento obligatorio: verificar a mano que
   el valor nuevo es el correcto, actualizar el valor esperado, y dejar un comentario en
   el test explicando qué cambió y por qué. Si no puedes justificar el valor nuevo, el
   cambio está mal, no el test.
4. **Cada corrección lleva su test.** Un test que falle antes del arreglo y pase después.
5. **Español** en docstrings, logs y salida de consola, como el resto del repo.
6. **No refactorizar de más.** Nada fuera del alcance de la fase, aunque se vea feo.
7. `config.py` sigue siendo la única fuente de parámetros operativos. No hardcodear.

---

# FASE 1 — Correcciones de corrección y reproducibilidad

Objetivo: que el sistema deje de contradecirse a sí mismo y produzca el mismo resultado
dos veces seguidas. **No cambia el diseño, solo arregla lo que está mal.**

### 1.1 — Sigma incoherente entre Monte Carlo y Opción A `[CRÍTICO]`

`main_pipeline.py:171` pasa `garch_vol.iloc[-1]` (volatilidad condicional instantánea de
hoy, congelada sobre todo el horizonte) al `RiskSimulationEngine`, mientras la Opción A
usa `extract_garch_forecast` (línea 348-349). Además las derivas difieren: Opción A usa
la media histórica pura (línea 337), el Monte Carlo usa `drift_mode='shrunk'`.

- Pasar al Monte Carlo la **misma** sigma que usa la Opción A (`extract_garch_forecast`).
- Unificar la deriva: ambas rutas deben respetar `config.drift_mode`.
- **Test:** un test de integración que verifique que la sigma y la mu que recibe
  `RiskSimulationEngine` son las mismas que consume `lognormal_price_bands`. Hoy no
  existe ningún test que mire esto, y por eso el bug sobrevivió a toda la suite.

### 1.2 — Snapshot de datos congelado y determinismo `[CRÍTICO]`

Evidencia del fallo: `cache/ml_prob_history.json` tiene JD con `0.66208637` y
`0.66246547` **la misma fecha**, con `random_state=42`.

- `main_pipeline.py:120` y `portfolio_engine.py:193`: `end_date` debe ser la **última
  sesión bursátil cerrada**, no `today`. Correr intradía usa un precio vivo como última
  barra.
- Cachear el panel de precios en `config.cache_dir` con clave `(ticker, start, end)`, de
  forma que dos corridas del mismo día lean el mismo fichero.
- `ml_engine.py:65-67`: fijar `n_jobs=1` en RandomForest y `nthread=1`/`n_jobs=1` en
  XGBoost. La suma de gradientes depende del reparto de hilos.
- `data_engine.py:104`: sustituir `list(set(tickers))` por deduplicación que **preserve
  el orden** (`dict.fromkeys`).
- **Test obligatorio:** `test_reproducibilidad.py` — construir el pipeline dos veces con
  los mismos datos mockeados y afirmar igualdad **exacta** del score, de `current_ml_prob`
  y de las bandas. Este es el test que valida toda la sub-fase.

### 1.3 — NaN no puede imprimirse como "EVITAR" `[ALTO]`

`scoring_backtest_engine.py:130`: `np.clip(nan,0,100)` = `nan`; en
`generate_recommendation` (143-150) toda comparación con NaN es False → cae al `else` →
`"EVITAR (Avoid / Short)"`.

- Guarda explícita al principio de `generate_recommendation`: si el score es NaN, devolver
  `"NO EVALUABLE"`. Nunca una recomendación.
- Que `main_pipeline` imprima el motivo (qué factor salió NaN), no solo la etiqueta.
- **Test:** pasar NaN y afirmar `"NO EVALUABLE"`.

### 1.4 — ROE y ROIC trimestrales contra umbrales anuales `[ALTO]`

`fundamental_engine.py:68` calcula ROE con `net_income` **trimestral**;
`QUALITY_ABSOLUTE_THRESHOLDS` (línea 24) ancla "excelente" en 0,15 **anual**. Una empresa
con ROE anual del 15% puntúa 0,25 en vez de 1,0. **Todo el factor Calidad está sesgado a
la baja para todos los tickers.**

- Usar TTM (`.rolling(4).sum()`) en el numerador de ROE, ROIC, margen neto y margen
  operativo, igual que ya se hace en la rama de Valoración (líneas 133-136).
- El denominador de balance (equity, capital invertido) es un **stock**, no un flujo: NO
  se acumula, se toma el valor del trimestre.
- **Test:** empresa sintética con ROE anual conocido del 15% → `_absolute_level_score`
  debe dar ≈ 1,0.

### 1.5 — Denominadores negativos o nulos `[ALTO]`

- `fundamental_engine.py:233-235`: `1.0 - rank(pct)` hace que un PER negativo (empresa en
  pérdidas) salga como la valoración **más atractiva** de su historia. Regla: si el
  denominador del múltiplo (beneficio TTM, EBITDA, equity) es ≤ 0, el múltiplo es **NaN**,
  nunca un número.
- `fundamental_engine.py:72`: `invested_capital = total_debt + total_equity - cash` puede
  ser ≤ 0 en empresas net-cash (casi toda la tech) → ROIC explota. Si es ≤ 0 → NaN.
- `data_engine.py:23`: `_validate_fundamentals` **anula el equity negativo**, que es un
  estado real y relevante (BA, MCD, SBUX, HD). Permitir equity negativo; que sean los
  ratios los que devuelvan NaN cuando no tengan sentido.
- **Tests:** un fixture "empresa en pérdidas" y otro "empresa net-cash". No existen hoy.

### 1.6 — `bfill()` en precios introduce lookahead `[ALTO]`

`data_engine.py:145`: `df_close.ffill().bfill()`. Un ticker que salió a bolsa en 2020 con
`lookback_years=12` recibe su precio de IPO replicado hacia atrás hasta 2013 → retornos
exactamente 0, vol artificialmente baja, momentum 0, beta 0.

- Eliminar el `bfill()`. Recortar la serie al **primer precio real** del ticker.
- Registrar en el log el histórico efectivo obtenido vs. el solicitado.
- Si el histórico efectivo < `MIN_LOOKBACK_YEARS_FOR_INTERPRETABLE_BACKTEST`, avisarlo
  de forma prominente en el informe.

### 1.7 — Fallos de red no deben matar el menú `[ALTO]`

`data_engine.py:156` hace `raise e`, no capturado en `run_production_system` ni en el
bucle del menú (`main_pipeline.py:782-793`) → traceback y salida del programa.

- Capturar los fallos de datos en el bucle del menú, imprimir un mensaje claro y
  **volver al menú**. El modo `[2]` ya lo hace bien; replicar ese patrón.
- Proteger también `main_pipeline.py:707` frente a `ValueError` y `pd.errors.ParserError`
  del parser de IBKR, no solo `FileNotFoundError`.

### 1.8 — Historial de probabilidades: upsert y escritura atómica `[ALTO]`

`portfolio_engine.py:272-278`: `entries.append(...)` sin deduplicar → el fichero tiene
entradas duplicadas con la misma fecha. Y la escritura no es atómica: una interrupción
corrompe todo el historial.

- Upsert por `(ticker, fecha)`: una entrada por ticker y día, la última gana.
- Escritura atómica: escribir a `.tmp` y `os.replace()`.

### 1.9 — Sharpe reportado a la mitad `[ALTO]`

`risk_simulation_engine.py:117-118`: `excess_return = self.mu - rf` con `self.mu` ya
encogido al 50% hacia `rf` (líneas 46-47). El álgebra da exactamente
`0,5·(μ_hist − rf)/σ`, pero se imprime como "Sharpe Ratio (histórico)".

- Calcular el Sharpe con la media histórica **sin encoger**.
- La deriva encogida sigue usándose para el Monte Carlo (ahí es correcta); son dos usos
  distintos y hay que separarlos explícitamente.

### 1.10 — Unidades del Deflated Sharpe Ratio `[CRÍTICO]`

`validation_engine.py:304` usa `√(T-1)` con `T` en observaciones **diarias**, pero
`sr_hat` viene anualizado (`scoring_backtest_engine.py:281`). El numerador queda inflado
~√252 ≈ 16× → **DSR = 1.0000 siempre**.

- Usar el Sharpe en la **misma frecuencia** que T (diario con T diario).
- `n_trials` (línea 296) usa `len(sharpe_values)` = 3. Documentar en el docstring que es
  un **suelo**, no el número real de especificaciones probadas (que son decenas), y que
  por tanto el DSR resultante es **optimista por construcción**.
- **Test:** que el DSR no salga saturado en 1.0 con valores realistas.

### 1.11 — Signo de "Δ Prob" invertido `[ALTO]`

`portfolio_engine.py:338`: `relative_drop = earliest - current`, impreso con formato
`"+.2%"` (`main_pipeline.py:649`) bajo la cabecera "Δ Prob". **Un deterioro se muestra
como número positivo.** Renombrar la columna a `Caída` o invertir el signo.

### 1.12 — Registro de predicciones `[EMPEZAR YA]`

Es la única forma de saber algún día si el sistema sirve, y cuesta poco. **Cuanto antes
empiece a acumular, antes sirve.**

- Nuevo `cache/prediction_log.jsonl` (una línea JSON por corrida, append-only).
- Cada entrada: `timestamp`, `ticker`, `precio_actual`, todos los factores, el score, la
  recomendación, `current_ml_prob`, las bandas de precio, el veredicto de robustez y la
  versión de `config` usada.
- Escribir siempre, tanto en modo `[1]` como en modo `[2]`.
- Añadir un script `scripts/revisar_predicciones.py` que lea el log, descargue los precios
  actuales y muestre qué ha pasado desde cada predicción.

### Criterio de aceptación de la Fase 1

- `pytest` en verde, con los tests nuevos de 1.1, 1.2, 1.3, 1.4, 1.5 y 1.10.
- Correr `python main_pipeline.py` dos veces seguidas sobre el mismo ticker produce un
  informe **idéntico**.
- El informe ya no imprime dos sigmas distintas para el mismo horizonte.

---

# FASE 2 — Diagnóstico multi-horizonte (puerta de decisión)

**Esta fase no cambia el sistema: genera la evidencia que justifica la Fase 3.** No
borrar nada del score hasta ver estos resultados.

### 2.1 — Script de diagnóstico

Nuevo `scripts/diagnostico_multihorizonte.py`. Para una lista de tickers (usar al menos
AAPL, MSFT, META, AMZN, NVDA, UBER, JD, JNJ, KO, XOM — mezcla deliberada de growth,
value, defensivo y energía) y para horizontes de **1, 3, 6, 12 y 24 meses**:

- Ejecutar `run_holding_period_backtest` con cada horizonte.
- Reportar por ticker y horizonte: hit rate del grupo señal, hit rate del control,
  `excess_difference`, `n_signals` y `n_independent_signals`.
- Calcular el **Information Coefficient**: correlación de Spearman entre
  `current_ml_prob` en cada fecha y el retorno forward a ese horizonte.
- Añadir un **error estándar Newey-West** con lag = `horizon_days` sobre el exceso
  señal-control, y reportar el intervalo de confianza al 95%. Hoy
  `scoring_backtest_engine.py:378` reporta una diferencia de medias cruda, sin ninguna
  medida de incertidumbre.
- Guardar todo en `output/diagnostico_multihorizonte.csv` y una tabla resumen por consola.

### 2.2 — Cómo leer el resultado

La pregunta que responde: **¿la señal ML aguanta en todos los horizontes o solo en 126
días?** Si el exceso solo es positivo en un horizonte y cambia de signo en los demás, es
ruido y el 20% de peso del ML no está justificado. Si el IC es indistinguible de cero en
todos los horizontes (que es lo que espero, ver `AUDITORIA_2026-09.md` §3 A1-A3), la
Fase 3 está justificada empíricamente y no por opinión.

**No saltarse esta fase.** Borrar el score porque un auditor lo dijo es peor práctica que
borrarlo porque los datos lo dicen — y si resulta que la señal sí aguanta, hay que saberlo.

### 2.3 — Añadir al análisis de sensibilidad las variantes gratuitas

`validation_engine.py:440`: hoy se paga la única variante cara (un segundo walk-forward
GARCH completo) y se omiten las que cuestan cero y más informan. Añadir:
`factor_weights`, las bandas de recomendación, `buy_threshold`, `drift_mode` y
**`random_state`** (esta última revela cuánto del score es puro ruido de semilla).

---

# FASE 3 — Reorientar el sistema a la pregunta real

Esta es la fase que sube la fiabilidad práctica, y es la más invasiva en
`main_pipeline.py`. **No empezar sin los resultados de la Fase 2.**

**La Fase 2 ya se ejecutó y la justifica por los datos** (ver `Ajustes y Arreglos.txt`,
bloque FASE 2, y `output/diagnostico_multihorizonte.csv`): IC medio de -0,0093 con el
signo repartido 5-5 en el único horizonte con potencia estadística, y `grupos_suficientes`
= False en los 10 tickers al horizonte de producción de 6 meses.

### Decisiones del propietario para esta fase (ya tomadas, no volver a preguntar)

1. **Autorización explícita para ELIMINAR del output.** El propietario tiene una regla
   permanente en `Ajustes y Arreglos.txt` (*"NO QUITE NADA DEL OUTPUT (SOLO AÑADIR) A NO
   SER QUE LO PIDA YO EXPLÍCITAMENTE"*). **Esta fase es la excepción explícita a esa
   regla**, autorizada el 2026-09. Se elimina del informe el Score 0-100, el veredicto
   COMPRAR/MANTENER/EVITAR y la probabilidad ML. **Obligatorio:** dejarlo escrito en
   `Ajustes y Arreglos.txt` con la justificación y la fecha, para que ningún agente futuro
   lo interprete como una regresión y lo reponga. La regla general sigue vigente para todo
   lo demás.
2. **El código del ML se conserva**, solo sale de la ruta de decisión (ver 3.3).
3. **Modo cartera: contexto sin veredicto** (ver 3.3).
4. **Analogías condicionales: descartadas** (ver 3.1).
5. **Precios objetivo de analistas: fuera de esta fase**, pasan a Fase 5.

### 3.1 — Nuevo motor: `entry_context_engine.py`

Responde *"¿a qué me apunto si compro hoy?"*. Todo sin horizonte fijo — se reporta en
abanico para 1, 3, 6, 12 y 24 meses.

**a) Distribución del drawdown de entrada.** Para cada fecha histórica `t` (excluyendo la
cola sin datos) y cada horizonte `H`:
`drawdown(t,H) = min(P[t..t+H]) / P[t] - 1`.
Reportar percentiles 5/25/50/75/95 por horizonte. **Esta es la métrica más útil del
sistema nuevo:** es lo que determina si aguantas la posición o vendes en pánico, y
ninguna herramienta retail te la da.

**b) Distribución del tiempo hasta recuperación.** Desde `t`, primer día en que
`P >= P[t]`, con tope en el horizonte máximo. Reportar percentiles y el % de casos que no
recuperan dentro del tope.

**c) DESCARTADO — versión condicional al estado actual.** El plan original contemplaba
repetir (a) y (b) restringiendo a fechas históricas cuyo estado se pareciera al de hoy
(percentil de volatilidad, caída desde máximos). **NO se implementa.** Con 12 años y
ventanas de 126 días, el número de análogos independientes queda casi siempre por debajo
del mínimo para ser fiable, y es exactamente el punto por donde el sobreajuste volvería a
entrar por la puerta de atrás — justo después de haber echado por esa misma razón la capa
ML. Las distribuciones de (a) y (b) son **incondicionales**, sobre todo el histórico
disponible. No añadir "análogos" ni filtros de similitud en ninguna forma sin una decisión
nueva del propietario.

**d) Valoración point-in-time frente a su propio histórico.** Ya existe en
`fundamental_engine`; aquí solo se **reetiqueta**: pasa de ser un factor predictivo a ser
contexto de entrada. La pregunta *"¿estoy pagando caro respecto a como ha cotizado esta
empresa?"* es legítima; *"¿va a subir?"* no lo es.

### 3.2 — Bandas de precio en abanico

`quantile_forecast_engine` y `lognormal_price_bands` pasan a producir bandas para los 5
horizontes en vez de uno solo. Mantener el rearrangement de Chernozhukov, y **aplicarlo
también en `evaluate_out_of_sample`** (`quantile_forecast_engine.py:63-75`): hoy se valida
un objeto distinto del que se publica.

### 3.3 — Eliminar la capa no validada

La Fase 2 confirmó que no hay señal. Autorizado explícitamente (ver decisiones arriba).

- Eliminar del informe `calculate_multifactor_score`, `generate_recommendation` y todo el
  score 0-100, junto con `calibrate_recommendation_bands`/`load_recommendation_bands` y el
  fichero de bandas: sin score no hay nada que cortar en bandas.
- Eliminar el veredicto `COMPRAR / MANTENER / EVITAR`. **No sustituirlo por otra etiqueta
  categórica** — ni "atractivo/neutro/caro", ni semáforos, ni estrellas. Cualquier
  etiqueta compuesta reintroduce por la puerta de atrás exactamente lo que esta fase
  elimina. El informe describe; no dictamina.
- Sacar `MachineLearningEngine` de la ruta de decisión: fuera del informe la probabilidad
  ML, fuera `evaluate_probability_calibration`, fuera `run_holding_period_backtest` y
  `run_strategy_backtest` como secciones del informe, y fuera `ValidationEngine` entero
  (sus tres chequeos validan un modelo que ya no decide nada).
  **Conservar los ficheros y todos sus tests**, y conservar
  `scripts/diagnostico_multihorizonte.py`: si algún día se construye el panel transversal
  (Fase 6), esa infraestructura —walk-forward, purga, embargo, IC, Newey-West— se
  reutiliza entera y es lo mejor construido del repo.
- Calidad y crecimiento pasan a **ficha descriptiva sin puntuar**: se imprimen los valores
  (ROE, ROIC, márgenes, apalancamiento, crecimiento YoY) y su percentil histórico propio,
  pero desaparece `score_quality` y su mezcla percentil/nivel absoluto — ese parche existía
  solo para alimentar el score.
- **Al eliminar `ValidationEngine` desaparecen también los dos chequeos que la Fase 1 dejó
  documentados como estructuralmente inalcanzables** (régimen-dependencia y concluyencia
  estadística, ver los pendientes de la Fase 1 en `Ajustes y Arreglos.txt`). No hay que
  "arreglarlos": el problema de diseño que los hacía inertes desaparece con lo que
  validaban.
- **Modo cartera `[2]`: contexto, sin veredicto.** Deja de emitir MANTENER/VIGILAR/VENDER
  y de usar `current_ml_prob` para nada. Por posición muestra: precio actual, precio medio
  de entrada y P&L (contexto, nunca criterio — se mantiene el invariante de que el precio
  de entrada no entra en ninguna valoración), percentil point-in-time de valoración,
  volatilidad y su percentil, beta, y el drawdown típico del bloque 3.1.
  Marca **"revisar"** —no "vender"— cuando algo cambie de forma material frente a lo que
  registró el histórico: la valoración salta más de 40 puntos de percentil, la volatilidad
  cambia de régimen, o el margen neto TTM cae dos trimestres seguidos. Umbrales en
  `config.py`, nunca hardcodeados. **"Revisar" significa "míralo tú", no una orden.**
  `cache/ml_prob_history.json` deja de escribirse; **no borrar el fichero existente**, es
  histórico.
- `prediction_log.jsonl` **sigue escribiéndose en cada corrida**, con las claves nuevas de
  esta fase. Respetar el esquema versionado y aditivo documentado en `CLAUDE.md` punto 13:
  añadir claves nuevas, **nunca renombrar ni reutilizar una existente con otro
  significado**, o se invalida el histórico acumulado — que es justo lo que ese fichero
  existe para evitar. Las entradas viejas seguirán teniendo `score`/`recomendacion` y las
  nuevas no: `scripts/revisar_predicciones.py` debe tolerar ambas formas y dejar de tratar
  la ausencia de recomendación como un fallo.

### 3.4 — Nuevo informe

Estructura, sin ningún número compuesto:

```
FICHA DE ENTRADA — {TICKER} — {fecha}

PRECIO Y VALORACIÓN
  Precio actual, y percentil point-in-time de PER / EV-EBITDA / P-B
  frente a su propio histórico (n trimestres, fiable sí/no)

PERFIL DE RIESGO
  Volatilidad condicional y su percentil histórico
  Beta rolling 252d · Volatilidad relativa al benchmark
  Máximo drawdown histórico

A QUÉ TE APUNTAS  ← el bloque central
  Drawdown esperado por horizonte (P5/P25/mediana/P75/P95)
  Tiempo hasta recuperación (mediana y P95, % que no recupera)
  Bandas de precio 80/90/95% en abanico 1/3/6/12/24 meses
  [Versión condicionada al estado actual, con n_analogos visible]

SALUD FINANCIERA (descriptivo, sin puntuar)
  Márgenes, ROE/ROIC TTM, apalancamiento, crecimiento YoY, tendencia

ADVERTENCIAS
  Fundamentales no fiables / histórico corto / analogías insuficientes / etc.
```

`config.verbose=False` sigue imprimiendo solo el bloque "A QUÉ TE APUNTAS".

### 3.5 — Deuda documental (requisito explícito del propietario)

- **Reescribir `CLAUDE.md`.** Hoy describe en detalle un sistema que esta fase destruye
  (el score, el Resumen Ejecutivo, las bandas de recomendación, el peso del ML). Un
  `CLAUDE.md` desactualizado es peor que no tenerlo: hace que cualquier agente futuro
  trabaje sobre una descripción falsa.
- **Crear `README.txt`.** No existe, pese a que `CLAUDE.md` lo declara requisito
  permanente. Debe documentar qué significa cada métrica impresa y sus rangos de
  interpretación.
- **Crear `Ajustes y Arreglos.txt`.** Tampoco existe, y también se cita como requisito.

### 3.6 — Residuos de la Fase 1 que esta fase debe cerrar

Están anotados como pendientes en `Ajustes y Arreglos.txt` y ahora sí toca resolverlos,
porque en el informe nuevo los fundamentales se imprimen **crudos** en vez de digeridos en
un score, así que un valor absurdo ya no se diluye: se ve.

- **ROIC de 18.300%.** La regla actual (`<= 0 → NaN`) no cubre el caso asintótico de un
  capital invertido positivo pero diminuto. Hace falta un **umbral de materialidad**
  (p. ej. capital invertido < 1% del activo total, o < 1% de los ingresos TTM → NaN).
  Es una decisión de modelado nueva: proponer el umbral, justificarlo en el docstring,
  llevarlo a `config.py` y fijarlo en un test.
- **"TTM" y "YoY" cuentan filas, no trimestres.** Con huecos en EDGAR una ventana "de 12
  meses" puede abarcar 5-7 trimestres reales. Como mínimo, **detectarlo y marcar el valor
  como no fiable** cuando el lapso real entre la primera y la última fila de la ventana se
  aleje más de ~45 días de los 365 esperados. Arreglarlo del todo (reindexar a trimestres
  fiscales) es opcional; ocultarlo no lo es.
- **BTC y los 252 días/año.** Sigue sin resolver y es el primer punto del fichero. En el
  informe nuevo afecta a la anualización de volatilidad y a los horizontes en días. Basta
  con un campo en `config.py` (`trading_days_per_year`, 252 por defecto) enhebrado por
  todas partes en vez del literal repetido. La detección automática de cripto queda fuera.

### Criterio de aceptación de la Fase 3

- El informe no contiene ningún número que no se pueda justificar con una muestra cuyo
  tamaño se imprime al lado.
- No queda ninguna afirmación predictiva sin validar, ni ninguna etiqueta categórica que
  resuma varias dimensiones en una palabra.
- Correr el pipeline dos veces sigue dando informes idénticos (el criterio de la Fase 1 no
  se rompe).
- `CLAUDE.md` describe el sistema que existe de verdad, y `README.txt` documenta cada
  métrica nueva con su rango de interpretación.
- La excepción a la regla de "solo añadir" queda anotada en `Ajustes y Arreglos.txt` con
  fecha y motivo.

---

# FASE 5 — Precios objetivo externos (pendiente del propietario)

Punto abierto en `Ajustes y Arreglos.txt`: *"dar intervalo de precios a 52 semanas sobre un
activo de al menos 3 fuentes distintas (y citarlas) — alto, medio y bajo"*. Encaja con el
enfoque nuevo (es **contexto externo**, no predicción propia), pero se deja fuera de la
Fase 3 deliberadamente: añade una dependencia de red nueva y mezclarla con la cirugía del
informe hace imposible saber qué rompió qué. Requisito cuando se haga: **citar la fuente y
la fecha de cada objetivo**, y presentarlos como lo que son — la opinión agregada de
terceros, no una estimación del sistema.

---

# FASE 6 — OPCIONAL: comparables sectoriales

Solo si más adelante se quiere recuperar un score puntuable.

25-40 nombres del mismo sector, percentiles **transversales** por fecha en vez de contra
el propio histórico. Eso hace innecesarios los parches de "percentil + nivel absoluto" de
`score_quality` y `score_momentum`, que existen precisamente porque el percentil propio no
significaba nada.

**Ser honesto sobre el límite:** 30 acciones del mismo sector correlacionan 0,5-0,7 entre
sí, así que el N efectivo sube mucho menos que el nominal. Esto da **comparabilidad**, no
tanto tamaño muestral. Aun así permite calcular un IC con barras de error reales, que es
más de lo que hay hoy. El salto de verdad sería un panel de 400-500 nombres
(`AUDITORIA_2026-09.md` §5).

**Condición de entrada, no negociable:** esta fase solo se abre si el IC medido sobre el
panel supera cero de forma estable, con la misma vara que tumbó al ML en la Fase 2
(ventanas independientes suficientes, signo estable entre horizontes, Newey-West). El
diagnóstico de la Fase 2 ya existe y sirve tal cual para juzgarlo. **Si el panel no pasa
ese examen, no se reintroduce ningún score** — el trabajo no se "aprovecha" publicando un
número que no lo aprobó.
