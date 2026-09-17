# Auditoría del sistema cuantitativo — septiembre 2026

**Alcance:** revisión completa de los 13 módulos, los 15 ficheros de test y la evidencia
de tus corridas reales guardadas en `output/` y `cache/`.

**Limitación importante y honesta:** el entorno de ejecución de esta sesión no arrancó
(`HYPERVISOR_VIRT_DISABLED`), así que **no he podido ejecutar `pytest` ni el pipeline**.
Todo lo que sigue es análisis estático línea a línea, más la evidencia empírica de las
corridas que tú ya habías hecho (los tres PNG de META, el `ml_prob_history.json` de
AMZN/JD/NVDA/UBER y los `companyfacts` cacheados de AAPL/MSFT/META). Donde afirmo algo
cito el fichero y la línea para que puedas verificarlo tú mismo.

---

## 1. Veredicto

| Dimensión | Nota |
|---|---|
| **Ingeniería y arquitectura** | **8 / 10** |
| **Rigor de proceso (anti-lookahead, purga, walk-forward, tests)** | **7 / 10** |
| **Fiabilidad predictiva demostrada** | **2 / 10** |
| **Uso real como herramienta de decisión de compra** | **3 / 10** |
| **GLOBAL** | **4 / 10** |

El 4 no es un suspenso al trabajo: es la nota de un sistema **muy bien construido para
responder una pregunta que, tal y como está planteada, no tiene respuesta estadística
posible**. Has hecho bien casi todo lo que un quant junior suele hacer mal (purga y
embargo dimensionados al horizonte del target, walk-forward de GARCH/ARIMA en vez de un
ajuste único, fundamentales point-in-time indexados por fecha de publicación, escalado
por bloque, semilla explícita, tests de caracterización). Eso está por encima del 90% de
los proyectos de este tipo que circulan.

El problema no es el código. Es que **has montado una infraestructura de validación
excelente sobre ~17 observaciones independientes**.

---

## 2. La evidencia empírica de tus propias corridas

Antes de la lista de bugs, esto es lo que dicen tus ficheros:

**2.1 — `output/META_equity_curve.png`.** La ventana out-of-sample va de enero 2024 a
febrero 2026: **2 años**, no 12. Es el último 20% del histórico (`ml_engine.py:97`,
`train_size=0.8`). Y en esa ventana:

| Serie | Retorno acumulado |
|---|---|
| Comprar y mantener META | **+83%** |
| Benchmark (S&P 500) | +47% |
| Estrategia bruta | +52% |
| **Estrategia neta** | **+36%** |

La estrategia **pierde contra comprar y mantener por 47 puntos**, y neta de costes
**pierde también contra el índice**. Sé que `run_strategy_backtest` es "referencia
interna, no ejecutable" — pero es la única serie de rentabilidad realizada que produce
el sistema, y va en contra de la señal.

**2.2 — `output/META_monte_carlo.png`.** Precio actual $589,85; mediana simulada a 6
meses $617,93 (+4,8%); distribución de $200 a $1.800. El intervalo útil es de
aproximadamente ±60%. Una banda de ese ancho no cambia ninguna decisión: es compatible
con "cómpralo" y con "no lo toques" a la vez.

**2.3 — `cache/ml_prob_history.json`.** Dos cosas graves:

- Entradas **duplicadas** con la misma fecha para los cuatro tickers → `record_ml_prob`
  no hace upsert (`portfolio_engine.py:272-278`).
- Para JD, la misma fecha da **0.66208637** y **0.66246547**. Difieren en el cuarto
  decimal *con `random_state=42`*. **El sistema no es reproducible.** Un informe que no
  se puede reproducir no se puede auditar, y no se puede confiar en él.

**2.4 — No existe `cache/recommendation_bands.json`.** Nunca has ejecutado la
calibración de bandas, y `calibrate_recommendation_bands` **no tiene ningún llamador en
producción** (solo tests). Es decir: siempre corres con los cortes fijos 80/60/40 sobre
una distribución de score de media ~51 y desviación ~13. **"COMPRA FUERTE" tiene una
probabilidad de ~0,8% de salir jamás.** La mitad del diseño de scoring está inerte.

**2.5 — No existen `README.txt` ni `Ajustes y Arreglos.txt`**, que tu propio `CLAUDE.md`
declara requisito permanente. Hay decenas de métricas impresas sin documentación de rangos.

---

## 3. Bugs y defectos, por severidad

### CRÍTICO

**C1 — El Monte Carlo usa una sigma distinta de la Opción A.**
`main_pipeline.py:171` pasa `garch_vol.iloc[-1]` (volatilidad condicional **instantánea
de hoy**, congelada sobre 126 días) al `RiskSimulationEngine`, mientras la Opción A usa
`extract_garch_forecast` (línea 348-349). Es exactamente el antipatrón que tu `CLAUDE.md`
declara corregido: sigue vivo en el Monte Carlo. Y las derivas también difieren
(histórica pura en A, encogida en el MC). **El informe imprime dos proyecciones
lognormales del mismo activo con μ distinta y σ distinta y las presenta como comparables.**

**C2 — El Deflated Sharpe Ratio está saturado en 1.0 por mezcla de unidades.**
`validation_engine.py:304` usa `√(T-1)` con `T` en observaciones **diarias**, pero
`sr_hat` viene anualizado (`scoring_backtest_engine.py:281`). El numerador queda inflado
~√252 ≈ 16×. Resultado: DSR = 1.0000 siempre que el ganador supere el umbral. El chequeo
más sofisticado del sistema es un sello de goma.

**C3 — `regime_dependent` y `statistically_conclusive` no pueden activarse nunca.**
`_assess_regime_dependence` exige `n_signals ≥ 1260` dentro de un sub-período que como
mucho tiene ~130 señales. Y en el flujo principal, `n_independent_signals ≤ 3,2 < 10`
siempre. **Dos de los tres chequeos de robustez son inertes**, y `generate_robustness_verdict`
trata "no evaluable" como "no rojo" → **el veredicto por defecto es ALTA confianza**.
El sistema se autoproclama fiable precisamente porque no puede medirse.

**C4 — Datos no congelados → resultados irreproducibles.**
`main_pipeline.py:120`: `end_date = today`, sin caché de precios. Si corres intradía, la
última barra es un precio vivo. Con modelos de árbol eso a veces no cambia nada (AMZN,
UBER idénticos) y a veces voltea una hoja (JD, Δ≈4e-4). Súmale que XGBoost no fija
`n_jobs`/`nthread` (`ml_engine.py:67`) y tienes no determinismo por reparto de hilos.

**C5 — ~17 observaciones independientes.**
12 años ≈ 3.020 días. Menos ~252 de burn-in y ~126 de target sin resolver → ~2.640 filas.
Train ≈ 2.110 / 126 = **~17 observaciones independientes**; test ≈ 530 / 126 = **~4,2**.
Entrenar XGBoost (100 árboles, profundidad 4, 7 features) con eso no es defendible: el
error estándar de la Balanced Accuracy resultante es de ±0,12. Es la misma corrección
`n / horizon_days` que sí aplicas en `run_holding_period_backtest` — pero **no la aplicas
en ninguna parte de `ml_engine.py`**.

### ALTO

**A1 — El target es relativo al benchmark, pero ninguna feature describe el benchmark.**
`main_pipeline.py:141-147`: las 7 features son `mom_3m`, `mom_6m`, `mom_12m`, `rsi`,
`ma_crossover`, `garch_volatility`, `arima_trend_pred`. El target es
`(retorno_activo > retorno_benchmark)`. **Le estás pidiendo al modelo que prediga una
resta dándole solo un lado.** Es un error de especificación de manual, y por sí solo
explica que el modelo no pueda superar el azar.

**A2 — Las 7 features son ~1,5 dimensiones reales.** `mom_6m = mom_3m + ln(P[t-63]/P[t-126])`
por construcción (correlaciones 0,6-0,85); `ma_crossover` es casi la función signo de
`mom_12m`; `arima_trend_pred` es el valor ajustado de un ARIMA(1,0,1) sobre retornos
**diarios** (magnitud ~1e-4: la media incondicional más ruido, contenido informativo
sobre 126 días ≈ nulo); `garch_volatility` es un **nivel** no estacionario que los
árboles usarán como proxy temporal y sobre el que no pueden extrapolar.

**A3 — Target desbalanceado sin corregir.** `data_engine.py:592`. Para un valor que ha
batido al S&P (el caso típico del que analizas), la clase 1 sale al 60-70%. Ningún
clasificador lleva `class_weight`/`scale_pos_weight` (`ml_engine.py:65-67`) → todos
tienden a la clase mayoritaria → Balanced Accuracy ≈ 0,50 → empate → el desempate
`if balanced_acc > best` da el ganador **por orden del diccionario**: siempre Regresión
Logística. La "selección entre 3 modelos" es vacía, y el DSR la penaliza como si hubiera
habido búsqueda real.

**A4 — ROE y ROIC trimestrales contra umbrales anuales.**
`fundamental_engine.py:68` calcula `roe = net_income_trimestral / total_equity`;
`QUALITY_ABSOLUTE_THRESHOLDS` (línea 24) ancla "excelente" en 0,15 **anual**. Una empresa
con ROE anual del 15% produce ~3,75% trimestral → `_absolute_level_score` = **0,25**, no
1,0. **El factor Calidad está sistemáticamente sesgado a la baja para todos los tickers.**
La rama de Valoración sí usa TTM (línea 133-136); la de Calidad no. Los fixtures lo
enmascaran porque su ROE trimestral cae casualmente cerca del umbral anual.

**A5 — Múltiplos negativos puntúan como "baratísimo".** `fundamental_engine.py:233-235`:
`1.0 - rank(pct)`. Un trimestre TTM con pérdidas da PER negativo → percentil mínimo →
`per_score ≈ 1,0`. Igual con EV/EBITDA negativo y con P/B si el equity es negativo.
**Una empresa en pérdidas sale como la valoración más atractiva de su historia.**

**A6 — Capital invertido puede ser negativo.** `fundamental_engine.py:72`:
`total_debt + total_equity - cash`. Para una empresa net-cash (AAPL histórico, casi toda
la tech) el denominador tiende a cero o cambia de signo → ROIC explota.

**A7 — `bfill()` sobre precios introduce lookahead.** `data_engine.py:145`:
`df_close.ffill().bfill()`. Un ticker que salió a bolsa en 2020 con `lookback_years=12`
recibe su precio de IPO replicado hacia atrás hasta 2013 → retornos exactamente 0 → vol
artificialmente baja, momentum 0, beta 0, y filas de entrenamiento falsas que el pipeline
no distingue de datos reales. Es el punto donde más contaminación entra y contradice tu
propio invariante declarado.

**A8 — NaN en el score se imprime silenciosamente como "EVITAR".**
`scoring_backtest_engine.py:130`: `np.clip(nan,0,100) = nan`; en `generate_recommendation`
(143-150) todas las comparaciones con NaN son False → cae al `else` → **"EVITAR (Avoid /
Short)"**. `score_momentum` devuelve NaN con serie vacía (`technical_engine.py:85-86`).
Sin ninguna guarda y sin ningún test.

**A9 — El event study no tiene test de significancia.**
`run_holding_period_backtest` (`scoring_backtest_engine.py:378`) reporta una diferencia
de medias cruda: sin error estándar, sin t-stat, sin p-valor, sin intervalo de confianza.
El `std` de la línea 367 es la dispersión de ventanas solapadas, no un SE.
`n_independent_signals` es una heurística sensata pero **no entra en ningún cálculo**,
solo en un booleano. Lo correcto sería Newey-West con lag ≈ `horizon_days` o bootstrap
por bloques estacionario.

**A10 — El Sharpe reportado es sistemáticamente la mitad del real.**
`risk_simulation_engine.py:117-118`: `excess_return = self.mu - rf` con `self.mu` **ya
encogido** al 50% hacia `rf` (líneas 46-47). El álgebra da exactamente
`0,5·(μ_hist − rf)/σ`. Se imprime como "Sharpe Ratio (histórico)"
(`main_pipeline.py:470`) y se interpreta contra bandas estándar.

**A11 — Columna "Δ Prob" con el signo invertido.** `portfolio_engine.py:338`:
`relative_drop = earliest - current`, impresa con formato `"+.2%"` (`main_pipeline.py:649`)
bajo la cabecera "Δ Prob". **Un deterioro aparece como número positivo.** Un lector lo
interpreta como mejora.

**A12 — `prob_loss` es una fórmula cerrada disfrazada de simulación.** Con μ y σ
constantes y Z iid normal, `log(S_T/S_0) ~ N(μT, σ²T)` exactamente, así que
`prob_loss = Φ(−μ√T/σ)`. Las 5.000 trayectorias solo añaden ruido de ±0,7 pp a un número
analítico. Peor: ajustas un GJR-GARCH-t (colas gruesas, clustering, apalancamiento) y
luego **tiras toda esa especificación** para simular choques normales iid con un σ escalar.

**A13 — Sin red o con yfinance caído, el modo [1] crashea.** `data_engine.py:156`
`raise e`, no capturado en `run_production_system` ni en el bucle del menú
(`main_pipeline.py:782-793`) → traceback y salida. El modo [2] sí degrada bien.

**A14 — `VIGILAR` es inalcanzable para parte de tu cartera.** Con
`hold_threshold_absolute=0.45` y `hold_threshold_relative_drop=0.20`, VIGILAR requiere
`earliest > 0.65`. UBER quedó registrado en 0.549 → cualquier caída de 20 puntos rompe
antes el umbral absoluto → **siempre VENDER, nunca VIGILAR**.

### MEDIO (resumen)

- `pct_change(4)` y `rolling(4).sum()` cuentan **filas**, no trimestres. Con EDGAR el
  índice tiene huecos → "YoY" y "TTM" pueden abarcar 5-7 trimestres reales sin aviso
  (`fundamental_engine.py:88-90, 133-136`).
- La rearrangement de Chernozhukov se aplica solo en `predict_price_quantiles`, no en
  `evaluate_out_of_sample` → **validas un objeto distinto del que publicas**
  (`quantile_forecast_engine.py:63-75` vs `115-117`).
- Fuga residual de 1 día en cada refit GARCH/ARIMA: el `sample` del MLE incluye `r_t`
  (`time_series_features.py:119-128`, `266-276`). Pequeña, pero es exactamente la clase
  de contaminación que el módulo dice eliminar.
- `_validate_fundamentals` **anula** el equity negativo (`data_engine.py:23`), que es un
  estado real y relevante (BA, MCD, SBUX, HD) → ROE, D/E y P/B a NaN justo donde importan.
- El fallback `dei` de acciones en circulación es prácticamente inerte: su `end` es la
  fecha de portada, que casi nunca coincide con un fin de período → esas filas caen por
  `has_substance` (`data_engine.py:504-512, 565`).
- ADRs, emisores extranjeros (20-F, taxonomía `ifrs-full`, reporte semestral) y
  financieras caen todos al fallback yfinance → `reliable=False` → el 40% del peso se
  redistribuye silenciosamente en buena parte del universo no-US.
- `expected_mean_price` es la **media** de una lognormal (sesgada al alza ~+3%) mientras
  la Opción B reporta **medianas**: tres centralidades distintas en un mismo informe.
- VaR/ES son **diarios históricos**, no de la simulación, en un informe a 6 meses. El
  número útil (ES sobre `final_prices` a 126 días) está disponible y nunca se calcula.
- El modo cartera hace un `extract_garch_forecast` completo por ticker cuyo resultado
  **nunca se imprime** (`portfolio_engine.py:227-231`): un MLE regalado por posición.
- El parser IBKR no fija `encoding` (cp1252 en Windows), no mapea `BRK B` → `BRK-B`, no
  agrega lotes duplicados del mismo símbolo, y solo `FileNotFoundError` está protegido en
  `main_pipeline.py:707` — un `ValueError` mata el bucle del menú.
- `run_sensitivity_analysis` paga la única variante cara (segundo walk-forward GARCH
  completo) y **omite las gratuitas y más influyentes**: los pesos del score, las bandas,
  `buy_threshold`, `horizon_months`, `drift_mode` y el `random_state` (que revelaría
  cuánto del score es puro ruido de semilla).

### Sobre los tests

Son tests de caracterización honestos, pero:

- **Todo se prueba sobre GBM sintético sin señal** (`conftest.py:48`), donde lo correcto
  es Balanced Accuracy ≈ 0,50. **Ningún test distingue un motor funcional de uno roto.**
  No hay ni un solo test de "señal plantada".
- El smoke test ejecuta el pipeline de verdad, pero sus únicas aserciones son "existe la
  línea SCORE", "0 ≤ score ≤ 100" y "existe VEREDICTO". Los tests `growing` y
  `deteriorating` afirman **exactamente lo mismo**.
- `test_main_pipeline.py` prueba `lognormal_price_bands` como función pura (5 tests) pero
  **nada verifica qué sigma le pasa `main_pipeline`** → C1 sobrevive al suite entero.
- No hay **ningún test de reproducibilidad** (correr dos veces → mismo output), que es
  justo el defecto que se observa en tu `ml_prob_history.json`.
- Las fixtures usan `refit_every=252`, así que el camino de propagación (`fix()`/`append`)
  se ejercita ~11 veces en vez de las ~143 de producción.
- `test_portfolio_cli.py:165-176` verifica una rama con `inspect.getsource` + regex:
  comprueba texto fuente, no comportamiento.
- Varios tests reimplementan la fórmula que auditan (`scoring_backtest_engine`, línea 318)
  → **blindan el bug en lugar de detectarlo**.
- Contaminación de fixture de sesión (`conftest.py:209`, `scope="session"`): dos tests
  mutan el mismo objeto `MachineLearningEngine` → dependencia del orden.

---

## 4. El diagnóstico de fondo

Todos los problemas estadísticos de arriba (17 observaciones independientes, chequeos de
robustez inalcanzables, percentiles que solo significan algo dentro de un ticker, bandas
de recomendación imposibles, umbrales absolutos que hay que inventar a mano) tienen **una
sola raíz**:

> **El sistema está construido sobre N = 1 activo.**

Con una sola acción tienes una única realización de un proceso estocástico. Puedes
extraerle 3.000 filas, pero solo contiene ~17 trozos de información independiente sobre
la pregunta "¿qué pasa en 6 meses?". No hay técnica —ni walk-forward, ni purga, ni DSR—
que fabrique información que no está en los datos. La infraestructura de validación que
has construido es correcta y está, literalmente, midiendo ruido con mucha precisión.

Y hay un problema conceptual encima: **el momentum, la calidad y el valor son anomalías
transversales, no de serie temporal.** Jegadeesh-Titman (1993) no dice "compra una acción
cuando su momentum esté alto respecto a su propio pasado"; dice "compra el decil superior
de momentum **del universo** y vende el inferior". Fama-French y Asness-Frazzini-Pedersen
(Quality Minus Junk) son igual: comparan empresas **entre sí**, no consigo mismas.
Rankear una acción contra su propio histórico no es la anomalía documentada, y no hay
razón teórica para esperar que funcione. De hecho, has tenido que parchearlo dos veces
(la mezcla percentil + nivel absoluto en `score_momentum` y `score_quality`) precisamente
porque el percentil propio no significaba nada. Esos parches son el síntoma, no la cura.

---

## 5. La propuesta: pasar de 1 acción a un panel transversal

**El cambio:** en vez de `1 ticker × 12 años`, entrenar sobre
`400-500 tickers × 12 años` (S&P 500 o Russell 1000), y cambiar el target de
"¿bate al benchmark?" a **"¿en qué decil del retorno a 6 meses de la sección transversal
caerá esta acción?"**.

### Por qué esto funciona — siete razones concretas

**1. Resuelve el problema de tamaño muestral, que es el único que no se puede resolver de
otra forma.** 500 acciones × ~24 ventanas de 6 meses no solapadas = **~12.000 observaciones
independientes**, frente a 17. De golpe:
- XGBoost pasa a ser defendible en vez de decorativo.
- `MIN_INDEPENDENT_SIGNALS_FOR_CONCLUSIVE = 10` se supera por tres órdenes de magnitud →
  `statistically_conclusive` **empieza a poder ser True** y tu backtest deja de ser
  automáticamente inconcluyente (C3).
- El DSR, una vez corregidas las unidades, se convierte en un test con potencia real.
- La diferencia señal vs. control admite un error estándar de verdad (A9).

**2. Arregla A1 gratis.** Si el target es el rango transversal, el benchmark está
implícito en la comparación: ya no necesitas darle al modelo un lado de la resta que no
tiene. La especificación pasa a ser coherente.

**3. Elimina los parches de percentil vs. nivel absoluto.** Los percentiles pasan a ser
transversales (¿ROE de 18% frente a las otras 499 empresas de este mes?), que es lo que
la literatura mide. Ya no hay que inventar `QUALITY_ABSOLUTE_THRESHOLDS`, y desaparece de
paso A4 (el desajuste trimestral/anual deja de importar porque todas las empresas se
miden con la misma vara) y buena parte de A5 y A6 (los outliers se recortan por winsorizado
transversal, que es el estándar en factor investing y es trivial de implementar).

**4. Hace el score verificable por primera vez.** Hoy no existe forma de saber si el
número 0-100 significa algo: no hay ni un test que relacione score con retorno futuro.
Con un panel puedes calcular el **Information Coefficient** — la correlación de Spearman
entre el score de cada mes y el retorno a 6 meses de la sección transversal. Es un solo
número, se calcula en tres líneas, y responde la pregunta que hoy el sistema evita:
*¿este score predice algo?* (Referencia: un IC mensual estable de 0,02-0,05 es lo que
consiguen fondos cuantitativos reales. Si el tuyo sale en 0,00, lo sabrás en una tarde.)

**5. Calibra los pesos en vez de justificarlos narrativamente.** Hoy el 25/25/15/20/15
solo está respaldado por comentarios. Con un panel puedes calcular el IC **de cada factor
por separado** y ponderar por IC/volatilidad del IC. Eso convierte la decisión más
arbitraria del sistema en una decisión medida. Y de paso resuelve A11 (los pesos
nominales no son los efectivos): podrás medir la descomposición de varianza real del score.

**6. Las bandas de recomendación se calibran solas.** Hoy `calibrate_recommendation_bands`
está escrito pero nunca se llama, y sin él "COMPRA FUERTE" es inalcanzable. Con una
sección transversal, "decil superior de hoy" es la banda, se recalcula cada mes por
construcción y es estable por definición. El fichero persistido deja de hacer falta.

**7. Convierte el backtest en algo que puedes ejecutar de verdad.** Un long/flat sobre una
sola acción rebalanceado a diario (que tú mismo declaras no ejecutable) se sustituye por
una cartera de deciles con rebalanceo semestral: exactamente lo que tu horizonte de 6
meses describe, con costes de transacción realistas y diversificación que reduce la
varianza del estimador otro orden de magnitud.

### Lo que NO cambia

La capa de presentación. El informe por ticker, el Resumen Ejecutivo, las bandas de
precio, el Monte Carlo, el modo cartera: todo eso se mantiene igual. Solo cambia de dónde
sale el score. Sigues escribiendo `[1] Analizar una acción`; lo que hay detrás es un
modelo entrenado sobre 500, no sobre esa.

### Coste realista

- **Datos de precios:** `yfinance.download(lista_de_500, ...)` es una sola llamada. Minutos.
- **Fundamentales:** aquí está el cuello de botella. EDGAR a ≤10 req/s son ~500 ficheros
  `companyfacts` (varios MB cada uno): una tarde de descarga, cacheada a disco, refrescada
  trimestralmente. Es perfectamente viable pero hay que planificarlo.
- **GARCH/ARIMA:** aquí hay que ser realista — 500 walk-forwards de GJR-GARCH son
  inviables en un portátil (uno solo ya supera tu `WALK_FORWARD_SLOW_WARNING_SECONDS`).
  **Recomendación: sácalos del modelo transversal.** Ya has visto (A2) que
  `arima_trend_pred` no aporta nada y que `garch_volatility` es un nivel no estacionario
  que los árboles no pueden usar. Consérvalos donde sí valen: en el *informe* del ticker
  concreto, para las bandas de precio y el riesgo. El modelo transversal se entrena con
  factores baratos y vectorizables (momentum, reversión, calidad, valor, vol realizada,
  beta, tamaño).
- **Esfuerzo:** el 70% del código que necesitas ya existe. Lo que falta es un
  `universe_engine.py` que descargue y alinee el panel, y refactorizar los engines para
  aceptar un `MultiIndex (fecha, ticker)` en vez de una serie. Es una semana de trabajo,
  no un rediseño.

---

## 6. La alternativa honesta, si no quieres construir el panel

Si el panel es demasiado, la otra opción **buena** no es dejar el sistema como está: es
**degradarlo conscientemente de recomendador a panel de contexto y riesgo**.

Elimina el score 0-100 y el veredicto COMPRAR/MANTENER/EVITAR, y quédate con lo que el
sistema mide bien y de forma auditable:

- volatilidad realizada y condicional, drawdown máximo, beta rolling;
- bandas de precio a 6 meses (con la sigma corregida y una sola convención);
- múltiplos actuales frente al histórico point-in-time de la propia empresa — que para
  *contexto* sí es una pregunta legítima ("¿está caro respecto a como ha cotizado?"),
  aunque no lo sea para *predicción*;
- calidad y crecimiento como ficha descriptiva, sin puntuar.

Eso es un producto honesto, útil, y que no promete lo que no puede cumplir. Un sistema
que dice "META cotiza a 1,3 desviaciones por encima de su PER mediano de 10 años y su
volatilidad condicional está en el percentil 80" vale más que uno que dice "Score 58/100 —
MANTENER" cuando ese 58 no ha sido validado nunca contra un retorno.

---

## 7. Plan de acción por prioridad

### P0 — Correcciones baratas que arreglan cosas que hoy están mal (1-2 días)

1. **C1**: pasar `garch_vol_esperada` (no `garch_vol.iloc[-1]`) al Monte Carlo y unificar
   la deriva con `drift_mode` en la Opción A. Dos líneas.
2. **C4**: `end_date` = última sesión **cerrada**; cachear el panel de precios por fecha;
   fijar `n_jobs=1` / `nthread=1` en los tres clasificadores. Añadir un test de
   reproducibilidad (correr dos veces → mismo output).
3. **A8**: guarda explícita — si `final_score` es NaN, imprimir "NO EVALUABLE", nunca
   "EVITAR".
4. **A4**: anualizar ROE/ROIC (TTM, como ya haces en Valoración) antes de compararlos con
   `QUALITY_ABSOLUTE_THRESHOLDS`.
5. **A5**: si el denominador de un múltiplo es ≤ 0, el score es NaN, no 1,0.
6. **A7**: recortar el histórico al primer precio real en vez de `bfill()`.
7. **A11**: invertir el signo de "Δ Prob" o renombrar la columna a "Caída".
8. **2.3**: upsert por `(ticker, date)` en `record_ml_prob` con escritura atómica
   (`tmp` + `os.replace`).

### P1 — Empezar a medir en vez de asumir (1 semana)

9. **C2**: corregir las unidades del DSR (Sharpe diario con T diario) y subir `n_trials`
   al número de especificaciones realmente probadas (son decenas, no 3).
10. **A9**: error estándar Newey-West (lag = `horizon_days`) o bootstrap por bloques
    sobre el exceso señal-control, y reportar el intervalo de confianza.
11. Añadir al análisis de sensibilidad las variantes **gratuitas**: pesos, bandas,
    `buy_threshold`, `random_state`. Cuestan cero y son las más informativas.
12. **Registro de predicciones.** Extiende `ml_prob_history.json` a un log completo:
    fecha, ticker, score, recomendación, precio, probabilidad, bandas. En 6 meses tendrás
    la única evidencia que de verdad importa: si el sistema acertó. Empieza hoy; no hay
    forma más barata de ganar (o perder) confianza en él.
13. Un test de **señal plantada**: fixture donde una feature predice el target por
    construcción, y afirmar que el pipeline la encuentra. Hoy ningún test distingue un
    motor que funciona de uno roto.

### P2 — El cambio estructural (2-4 semanas)

14. `universe_engine.py`: panel de 400-500 tickers, precios + fundamentales, cacheado.
15. Refactorizar `technical_engine` y `fundamental_engine` a `MultiIndex (fecha, ticker)`
    con winsorizado y z-score transversales por fecha.
16. Target = decil de retorno a 6 meses en la sección transversal.
17. **Calcular el IC del score y de cada factor.** Este es el hito: es el momento en que
    el sistema deja de ser un ejercicio de ingeniería y pasa a ser una herramienta con
    poder predictivo medido — o el momento en que descubres que no lo tiene, que también
    es información valiosa y barata de obtener.
18. Backtest de cartera de deciles con rebalanceo semestral y costes.

---

## 8. Resumen en una frase

Has construido con mucho cuidado el andamiaje correcto alrededor de una pregunta que,
con una sola acción, no admite respuesta estadística. La corrección no es añadir más
modelos: es **ampliar el universo de 1 a 500**, porque es lo único que convierte tus 17
observaciones en 12.000 y hace que toda la maquinaria de validación que ya has escrito
—que es buena— empiece a medir algo real.
