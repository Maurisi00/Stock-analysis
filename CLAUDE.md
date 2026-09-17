# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A quantitative equity **entry-context** tool: it pulls market and fundamental data for a stock and
describes *"what am I signing up for if I buy today at this price?"* — the historical distribution of
entry experiences (drawdown you'd have sat through, time to recover), the point-in-time valuation and
risk profile, a raw financial-health card, and price bands fanned across 1/3/6/12/24 months. There is no
web app, API, or database — it's a script-driven research pipeline run from the console.

**It describes; it does not dictate.** There is no score, no buy/hold/avoid verdict, no ML probability
and no categorical label of any kind. That layer existed until 2026-09 and was removed with the owner's
explicit authorization after Phase 2 measured its predictive power and found none — see
**The Phase 3 removal** below before "restoring" anything.

## Commands

Setup (no lockfile beyond `requirements.txt`; a `venv/` already exists in the repo root):
```
pip install -r requirements.txt
```

Run the full pipeline:
```
python main_pipeline.py
```
This drops into an interactive menu (`prompt_main_menu`) — `[1]` prompts for a ticker (`prompt_ticker`,
validated against yfinance before continuing) and prints the FICHA DE ENTRADA; `[2]` runs
`run_portfolio_analysis`, a per-position context table over the positions in `config.portfolio_csv_path`
(default `portfolio/posiciones.csv`, an IBKR Flex Query export — see point 9 in Architecture); `[0]` exits.
The ticker is a per-run choice made at the prompt, not a CLI flag or a `PipelineConfig` field.

All other operational parameters are CLI flags backed by `config.py::PipelineConfig` — run
`python main_pipeline.py --help` for the full list, e.g.:
```
python main_pipeline.py --horizon-months 6 --drift-mode zero
python main_pipeline.py --trading-days-per-year 365     # crypto: 365 sessions/yr, not 252
python main_pipeline.py --no-verbose
```
`--verbose`/`--no-verbose` (`config.verbose`, default `True`) controls report length, not what gets
computed — everything is always calculated either way. `--no-verbose` prints only the central
"A QUÉ TE APUNTAS" block **and the ADVERTENCIAS block**; the latter is a deliberate deviation from
PLAN_REFACTOR.md 3.4 ("solo el bloque A QUÉ TE APUNTAS"), because the report has no verdict telling the
reader how much to trust it, so a warnings list that vanishes with a verbosity flag would be exactly the
buried-caveat failure this phase exists to prevent.

`config.py` is the single source of truth for operational values — don't hardcode a new operational
parameter in an engine file; add it to `PipelineConfig` and thread it through instead. Methodology
constants that describe a fixed *shape* (momentum/RSI windows, GARCH burn-in) are intentionally *not* in
`PipelineConfig` and stay hardcoded in the engines that own them — see
`PipelineConfig.trading_days_per_year`'s docstring for where that line is drawn and why.

Inspect the prediction registry — what the system claimed vs. what happened (see point 10 in Architecture):
```
venv\Scripts\python.exe scripts/revisar_predicciones.py
venv\Scripts\python.exe scripts/revisar_predicciones.py --ticker META --min-dias 90
venv\Scripts\python.exe scripts/revisar_predicciones.py --sin-red
```

Multi-horizon diagnostic of the (now-retired) ML signal — the Phase 2 decision gate, kept as the yardstick
for judging any future signal (see point 11 in Architecture):
```
venv\Scripts\python.exe scripts/diagnostico_multihorizonte.py
venv\Scripts\python.exe scripts/diagnostico_multihorizonte.py --tickers AAPL,MSFT --horizontes 6,12
```
Takes ~6 minutes for the default 10 tickers x 5 horizons; writes `output/diagnostico_multihorizonte.csv`.

Run the test suite (characterization tests + a mocked full-pipeline smoke test — see `tests/`):
```
pytest
```
Run a single engine's tests, or a single test, the usual pytest way:
```
pytest tests/test_entry_context_engine.py
pytest tests/test_pipeline_smoke.py::test_el_informe_no_reintroduce_ningun_score_ni_veredicto
```
`tests/conftest.py` provides seeded synthetic fixtures (a ~3000-business-day GBM price series + correlated
benchmark, and 20-quarter "growing company" / "deteriorating company" fundamentals, plus
"empresa en pérdidas" / "net-cash" / "equity negativo" for the degenerate-denominator cases) so the suite
never touches the network.
Some tests deliberately spawn **real subprocesses** (`tests/test_reproducibilidad.py`,
`tests/test_degradacion_fallos_datos.py`) because the bugs they guard — per-process hash-order variation,
and a cp1252 stdout — are invisible inside a single interpreter or under pytest's stdout capture; don't
"optimize" those into in-process calls. These are characterization tests: they lock in the pipeline's
*current* behavior (shapes, index alignment, value ranges, determinism) to catch regressions during
refactors — they are not a validation that the underlying methodology is correct. There is no linter or
build step configured in this repo — don't assume `ruff`/etc. exist unless you add it yourself.

## The Phase 3 removal — read this before "restoring" anything

In 2026-09 the following were **deleted from the report and from the decision path**, with the owner's
explicit authorization as a documented exception to their standing "only add to the output, never remove"
rule (recorded with date and justification in `Ajustes y Arreglos.txt`):

- the 0-100 Score Multifactorial (`calculate_multifactor_score`) and its weights;
- the COMPRA FUERTE / COMPRA / MANTENER / EVITAR verdict (`generate_recommendation`,
  `calibrate_recommendation_bands`, `load_recommendation_bands`, the persisted bands file);
- `calculate_risk_factor` (the 0-1 composite risk factor that only fed the score);
- `FundamentalFactorEngine.score_quality` / `score_valuation`, `_absolute_level_score` and
  `QUALITY_ABSOLUTE_THRESHOLDS` (the percentile-plus-absolute-level blend existed only to feed the score);
- the ML probability and `evaluate_probability_calibration` as report content;
- both backtests as report sections;
- `ValidationEngine`'s `run_sensitivity_analysis` (all `sensitivity_*`) and
  `generate_robustness_verdict`;
- mode `[2]`'s MANTENER/VIGILAR/VENDER verdict, `generate_hold_sell_signal`, the `hold_threshold_*`
  config fields, and the Prob.ML / Bal.Acc / Caída columns;
- the equity-curve chart (it plotted the daily-rebalanced ML strategy).

**Why, with the numbers.** Phase 2 measured that layer's predictive power with the system's own yardstick
(INDEPENDENT windows, not overlapping dates) across 10 tickers x 5 horizons. Of 49 evaluable cells only 10
had an Information Coefficient resting on enough independent windows — all ten at the 1-month horizon —
and there the IC came out at mean **-0.0093**, median -0.002, with the sign split 5-5. At the production
horizon (6 months) **not one** of the 10 tickers had both event-study groups estimable, i.e. the
`excess_difference` the report printed every run as its primary validation had no sample size to exist.
Mean Balanced-Accuracy was 0.5157 (19 of 49 below 0.50). A weighted average of five factors whose
predictive component measures zero does not stop measuring zero by being averaged — it just becomes harder
to audit, because a label ("COMPRA") hides which measurement it came from and on how many observations.
Full record in `Ajustes y Arreglos.txt` (FASE 2 and FASE 3.3/3.4 blocks) and
`output/diagnostico_multihorizonte.csv`.

**It was NOT replaced by another label**, and must not be: no "attractive/neutral/expensive", no traffic
lights, no stars, no renamed score. Any categorical summary of several dimensions into one word
reintroduces exactly what was removed. Three guards fail if it comes back:
`tests/test_pipeline_smoke.py::test_el_informe_no_reintroduce_ningun_score_ni_veredicto`,
`tests/test_fundamental_engine.py::test_describe_quality_no_produce_ningun_score_compuesto`, and
`tests/test_portfolio_cli.py` (which asserts the removed columns don't reappear).

**The ML code itself is kept**, out of the decision path: `ml_engine.py`, `validation_engine.py`'s two
surviving checks, and `scripts/diagnostico_multihorizonte.py`, with all their tests intact. What's worth
keeping there is the infrastructure — walk-forward, purge, embargo, IC, Newey-West standard errors — which
is the best-built part of the repo, is the yardstick that retired the ML in the first place, and is the
entry condition for a hypothetical Phase 6 cross-sectional panel. If a score is ever reintroduced, it gets
measured with that machinery *first*.

## Architecture

`main_pipeline.py` is the orchestrator (`run_production_system(ticker: str, config: PipelineConfig)`,
called from the interactive menu in `run_menu_loop`). It wires together independent engine classes, each
in its own file, executed in this order:

1. **`data_engine.DataEngine`** — downloads adjusted close prices via `yfinance` for the target ticker(s)
   + benchmark (`^GSPC`) and normalizes yfinance's inconsistent column structures. Three reproducibility
   guarantees live here (see Key Invariants): the ticker list is deduplicated with `dict.fromkeys`
   (order-preserving — `list(set(...))` iterates differently in each *process* under `PYTHONHASHSEED`
   randomization); callers pass `end_date=last_closed_session()` (module-level helper: last weekday whose
   16:00 ET close has passed — holidays deliberately unmodelled, since asking yfinance for a holiday just
   returns the prior session) rather than `date.today()`, which would make the last bar a *live* price;
   and the normalized panel is cached to `config.cache_dir` keyed on `(tickers, start, end)` as a pickle
   written atomically (`_price_cache_path`/`_persist_price_cache`) — pickle and not CSV so float
   round-trip is exact. NaN handling is **`ffill` only, never `bfill`**;
   `_recortar_al_primer_precio_real` then trims the panel to the first date where every column has a
   price and records `effective_history_years`/`effective_history_start`, which `main_pipeline.py` surfaces
   as a warning when it falls below `MIN_LOOKBACK_YEARS_FOR_INTERPRETABLE_BACKTEST`.
   `calculate_forward_returns` still exists (the diagnostic script and `ml_engine`'s tests consume it) and
   sizes its horizon from `trading_days_per_year`, not a hardcoded 21.
   Also pulls quarterly fundamentals and normalizes them into the flat schema `fundamental_engine.py`
   expects (`fetch_fundamental_data(ticker, source=config.fundamentals_source)`), from one of two sources:
   - **`'edgar'` (default)** — `_fetch_fundamental_data_edgar`: SEC EDGAR's free `companyfacts` API
     (ticker→CIK resolved from `company_tickers.json`, both cached on disk under `config.cache_dir`).
     Typically yields 40-80 clean quarters vs. yfinance's ~5, which is what makes the historical-percentile
     figures below meaningful at all. Merges every alternative US-GAAP tag per concept (companies migrate
     tags over time, e.g. `Revenues` → `RevenueFromContractWithCustomerExcludingAssessedTax` after
     adopting ASC 606), derives the fiscal Q4 quarter as annual-minus-9-months when a 10-K doesn't tag a
     discrete Q4 duration (most don't), and drops "orphan" rows where only a cover-page concept has data.
     See Key Invariants for the point-in-time indexing and restatement handling — both load-bearing for
     this being safe in a no-lookahead pipeline.
     **Known gap:** `shares_outstanding` is only looked up under `CommonStockSharesOutstanding` (plus the
     `dei` cover-page fallback), and some issuers never tag it — META among them, which only reports
     `WeightedAverageNumberOfDilutedSharesOutstanding`. Without a share count there's no market cap, so
     *every* valuation multiple comes back NaN and the whole PRECIO Y VALORACIÓN block prints `n/d` with a
     warning. Adding the diluted-weighted-average tag is a modeling decision (it changes the entire
     historical multiple series), so it's deliberately open rather than silently patched.
   - **`'yfinance'`** (`_fetch_fundamental_data_yfinance`) — the original ~5-quarter source, with its own
     tax-rate/debt fallback hardening. Used automatically if EDGAR raises or returns no usable data for
     the ticker (e.g. no CIK match), or if requested explicitly.
   Either path runs through `_validate_fundamentals` as a final backstop: implausible or infinite values
   per column get nulled (NaN) and logged rather than silently contaminating downstream figures.

2. **`technical_engine.TechnicalFactorEngine`** — academic "X-1" momentum factors (`compute_momentum_factors`:
   3/6/12-month *lookback*, each one excluding the most recent month). The `mom_Xm` names follow that
   lookback-minus-1-month academic convention, **not** the actual price window measured: `mom_3m` =
   `ln(P[t-21]/P[t-63])` spans only 42 trading days between t-63 and t-21, `mom_6m` spans 105 days, and
   `mom_12m` spans 231 days — the textbook 12-1 momentum. MA50/MA200 trend crossover, Wilder RSI, and
   statistical risk (annualized vol, max drawdown, and a **rolling** beta vs. benchmark —
   `compute_statistical_risk(beta_window=None)`: `Cov(asset, bench) / Var(bench)` over a rolling window
   defaulting to `trading_days_per_year`, i.e. one *year* of sessions rather than a hardcoded 252,
   algebraically equivalent to a single-regressor OLS slope on that window but far cheaper than refitting
   `sm.OLS` per window. Returns a `pd.DataFrame` (one column per ticker, NaN for the burn-in rows), not a
   per-ticker scalar like `volatility_ann`/`max_drawdown` — a full-sample OLS beta freezes a single number
   over an asset's entire history even as its risk profile changes; `main_pipeline.py` takes the *latest*
   non-NaN value as "today's" beta). The momentum factors now feed only the quantile fan (point 6), not any
   score.

3. **`time_series_features.TimeSeriesFeatureExtractor`** — `extract_garch_volatility` (GJR-GARCH(1,1),
   `o=1`, `dist='t'`) and `extract_arima_trend` (ARIMA(1,0,1)) both use an **expanding walk-forward
   window**, not a single fit over the whole series: `refit_every` days (`config.ts_refit_every`, default
   21) a fresh MLE fit runs on data up to that date; between refits, the same fitted parameters are
   propagated forward (`arch_model.fix()` for GARCH, `ARIMAResults.append(refit=False)` for ARIMA) rather
   than re-estimated. A single whole-series fit would estimate each date's parameters using *future* data,
   and those per-date features feed the quantile fan — the exact lookahead bias the rest of the pipeline
   avoids everywhere else. Both require 252 observations before their first estimate (NaN before that);
   with fewer than 252 total observations, or if the walk-forward raises, both fall back to a simple
   rolling stat (std / mean(5)) — don't remove this without replacing it. GJR (asymmetric leverage term)
   and Student-t (fat tails) replace a plain symmetric-normal GARCH(1,1): equities react to a drop with
   more vol than an equal-sized rally, and daily-return tails are fatter than normal — a mismatched spec
   here distorts VaR and the Monte Carlo directly. `extract_garch_forecast(ticker, horizon_days)` is
   separate and *not* walk-forward: it fits once on 100% of history (a legitimate one-shot "today's
   forecast", not a training feature) and averages `res.forecast(horizon=...)`'s projected variances
   before taking the square root, so the horizon's expected volatility reflects the model's own
   mean-reversion. Logs a warning if the walk-forward takes longer than
   `WALK_FORWARD_SLOW_WARNING_SECONDS` (60s) — real, not hypothetical. Internally works off a plain
   `RangeIndex`, not the original `DatetimeIndex`, for the ARIMA calls — real market data's index has no
   inferable `freq` (irregular holiday gaps), and `ARIMAResults.append()` requires one; without this it
   silently fell back to the crude rolling-mean fallback on every real ticker.
   Its 252-observation burn-in and `window_size` are **not** threaded through
   `trading_days_per_year` on purpose — see that field's docstring.

4. **`fundamental_engine.FundamentalFactorEngine`** — turns raw quarterly fundamentals into **descriptive**
   figures, no scores. `compute_quality_scores` produces ROE, ROIC, margins, leverage and YoY growth
   (growth columns use `.pct_change(4)`, i.e. YoY same-quarter comparison, not QoQ, which would be
   contaminated by seasonality); `compute_valuation_scores` produces PER, P/B, EV/EBITDA, P/S and dividend
   yield. **Flow numerators go through `_ttm()`** (ROE, ROIC, both margins, debt/EBITDA, interest
   coverage): the literature thresholds these get read against are *annual*, so dividing a single quarter's
   earnings by a balance-sheet *stock* understated the ratio ~4x. Balance-sheet denominators (equity,
   invested capital) are stocks and are **not** accumulated. Every multiple/ratio denominator passes
   through **`_positivo_o_nan()`**: `<= 0` yields NaN, never a number — a loss-making company's negative
   PER would otherwise land in the lowest percentile of its own history and read as the *cheapest* it has
   ever been. The ROIC has a second, independent guard, **`_capital_invertido_material()`** (Phase 3.6):
   `invested_capital = debt + equity − cash` is a *difference of large numbers*, so when what survives the
   subtraction is a tiny fraction of its own inputs, the denominator's relative error dominates and the
   ratio is not "high" but **unestimable** — at 1% of gross capital, a 1% error in the cash figure is a 99%
   error in the ROIC, and the engine used to print 18,300%. If
   `invested_capital < config.roic_min_invested_capital_fraction * (debt + equity)` the ROIC is NaN. The
   0.10 default is derived in `PipelineConfig`'s docstring (it caps the error amplification at ~9x) and
   costs **zero** real observations: measured across 8 tickers and 182 quarters with a defined ROIC, the
   minimum real fraction is 22.75%.
   `compute_valuation_scores` takes the *full* market-price series (not a single scalar) and, for every
   reporting date, aligns it to the price actually in effect then via
   `pd.merge_asof(..., direction='backward')` — applying today's price uniformly across history would
   freeze market cap and make every multiple move only with the growing accounting denominator, so the most
   recent quarter would *always* rank as the historical cheapest regardless of the real price (verified
   empirically: PER 208x and PER 4x both scored identically). PER, EV/EBITDA and P/S use trailing-twelve-
   month figures rather than a single quarter's.
   `describe_quality()` is the report's card: per metric, today's value, its percentile within the
   company's own history in **natural orientation** (high = high value, also where "less is better" — no
   inverted metrics, so the whole table reads with one convention), the historical median, the YoY delta,
   `n_obs`, and the window coverage below. `MIN_QUARTERS_FOR_RELIABILITY` is 12 — reachable with EDGAR,
   while yfinance's ~5 quarters correctly reports `reliable: False`, which means *the percentiles* aren't
   interpretable (the values are still the accounting data).
   **`window_coverage()`** (Phase 3.6) closes the "TTM counts rows, not quarters" residual:
   `.rolling(4)`/`.pct_change(4)` count *rows*, and with EDGAR's gaps a "12-month" window can span much
   more real time. It reports, per row, how many calendar days the TTM and YoY windows actually cover
   (`cobertura_ventana_ttm` = span between rows t-3 and t plus one quarter; `cobertura_ventana_yoy` = span
   between t-4 and t) and whether that is within `TOLERANCIA_DIAS_VENTANA` (45) of the expected 365.
   **Measured magnitude: 69% of TTM windows and 78% of YoY windows do not cover 12 months, and the median
   TTM window covers 455 days** — the typical "TTM" in this system is a ~15-month accumulation. The report
   publishes the per-value coverage in a `VENTANA` column and the aggregate in ADVERTENCIAS. It is *not*
   fixed (that needs reindexing the EDGAR extractor to fiscal quarters, explicitly optional per the plan);
   hiding it was not an option. `net_margin_ttm_window_valid()` exposes the same mask so
   `portfolio_engine` doesn't count a window-length change as a margin decline.

5. **`entry_context_engine.EntryContextEngine`** — the central engine of the new report. It answers
   *"what am I signing up for if I buy today?"* with **descriptive** claims about the historical
   distribution of entry experiences, not predictive ones.
   (a) `drawdown_distribution()`: for each date t and horizon H, `min(P[t..t+H])/P[t] - 1`. The window
   **includes** t so the drawdown is capped at 0 — if the price never dips below what you paid, the honest
   answer is "you were never underwater", not a positive number. Only dates with the full window observed
   are used.
   (b) `recovery_time_distribution()`: days from purchase until the price returns to the entry price
   **after having dipped below it**. NOT the literal reading of the plan ("first j > t with P[j] >= P[t]"):
   that returns 1 day whenever the price rises the day after purchase, *even if it then crashes 30% and
   takes two years to come back*, so on a rising asset the median is 1 and says nothing. The distribution
   is restricted to entries that actually went underwater — there is nothing to recover from otherwise —
   and `pct_nunca_bajo_agua` reports how many those were. Entries that dip and don't return within H are
   excluded from the percentiles (no invented large value contaminating the median) and reported as
   `pct_no_recupera`, often the most important figure in the block.
   (d) `valuation_context()`: consumes `compute_valuation_scores` and re-presents each multiple as today's
   value plus its percentile within the company's own point-in-time history, with `n_obs`. Deliberately
   produces **no composite score and no label** — a percentile with its n beside it is verifiable;
   "Valuation: 72/100" is not. The percentile keeps its natural orientation (high = expensive).
   Also exports **`percentil_historico(serie)`**, the value/percentile/n presentation pattern the whole
   phase is built on, used by both menu modes for the conditional-volatility percentile.
   **(c) conditional analogues are DISCARDED** by owner decision (2026-09) and must not be added back
   without a new one: with 12 years and 126-day windows the count of *independent* analogues falls below
   the reliable minimum, and it is precisely the door through which overfitting would return right after
   the ML layer was removed for that same reason. `tests/test_entry_context_engine.py` has an explicit
   guard that fails if any public name suggests a similarity filter. Every distribution carries `n_obs`,
   `n_independiente` (= n / horizon, the overlap correction) and a `fiable` flag against
   `MIN_OBS_INDEPENDIENTES` (10) — on 12 years the 24-month horizon comes out with ~5 independent windows
   and is correctly flagged unreliable. `HORIZONTES_MESES` is `(1, 3, 6, 12, 24)`; `horizontes_en_dias`
   and the constructor take `dias_por_mes` from `config.trading_days_per_month` so "6 months" means six
   *calendar* months on any asset.

6. **`quantile_forecast_engine`** — `QuantileForecastEngine` fits one
   `GradientBoostingRegressor(loss='quantile')` per quantile (2.5%, 5%, 10%, 50%, 90%, 95%, 97.5%) of the
   forward return distribution, exposed as 80/90/95% confidence bands (`CONFIDENCE_BANDS`). No 99% band:
   estimating a quantile that extreme off ~500 overlapping observations amounts to ~2.5 *effective*
   observations in the tail — a number that looks precise without being estimable. Two-phase pattern:
   `evaluate_out_of_sample()` fits on a purged/embargoed train split and reports pinball loss + empirical
   interval coverage vs. nominal, so the bands aren't presented blind; `refit_on_full_data()` then refits
   on 100% of history for "today's" prediction via `predict_price_quantiles`, which also fixes **quantile
   crossing**: the 7 regressors are independent models with no ordering constraint, so nothing stops the
   p05 model predicting above the p50. Both paths apply the Chernozhukov, Fernández-Val & Galichon (2010)
   rearrangement — sort the quantile *levels* ascending, sort the *predicted values* ascending, zip them —
   which guarantees monotonicity by construction without retraining. Applying it inside
   `evaluate_out_of_sample` too matters because otherwise a *different object* was being validated than the
   one published: **32.6% of AAPL's test rows (172 of 528) had crossed quantiles**.
   `rearrange_quantiles`/`rearrange_quantile_matrix` are pure functions so both call sites share one
   implementation; the matrix version sorts **per row** (per observation) — sorting along the time axis
   would mix one date's p05 with another's p95.
   **`QuantileFanForecaster`** holds one engine per horizon, because the forward return at 21 days is a
   different learning problem than at 504, with each split's embargo sized to its own target. Horizons a
   short history cannot support are skipped **with a stated reason** (`skipped`) rather than silently
   absent.

7. **`risk_simulation_engine.RiskSimulationEngine`** — runs a vectorized Monte Carlo (Geometric Brownian
   Motion) seeded with GARCH-derived annualized volatility, and computes VaR, Expected Shortfall, Sharpe,
   Sortino, and max drawdown. It carries **two drifts for two different jobs**, and they must not be
   confused: `self.mu` (adjusted per `drift_mode`) is for *projecting* — `run_monte_carlo` and the
   lognormal fan, where shrinking a noisy historical mean toward `rf` genuinely reduces estimator variance
   — while `self.historical_mu` (the raw annualized mean of log returns) is what `calculate_risk_metrics`
   uses for Sharpe and Sortino, which *describe* what the asset already did. Using `self.mu` there made the
   algebra come out as exactly `0.5*(mu_hist - rf)/sigma` under the default `'shrunk'`, i.e. **half the
   real Sharpe**, printed as "Sharpe Ratio (histórico)" and read against the standard literature bands.
   `self.mu` is the annualized mean of *log* returns, so `run_monte_carlo`'s GBM step is
   `exp(mu*dt + sigma*sqrt(dt)*Z)` — **no** `-0.5*sigma²` convexity term. That correction converts an
   *arithmetic* drift into a log-scale one via Itô's lemma; `self.mu` is already log-scale, so subtracting
   it again silently shifted the whole simulated distribution downward (verified empirically). `drift_mode`
   (`config.drift_mode`, default `'shrunk'`) controls how `self.mu` is estimated: `'historical'` (raw
   annualized mean — huge standard error, ~15%/yr for a typical stock), `'zero'`, or `'shrunk'`
   (historical mean shrunk `SHRINKAGE_TOWARD_RISK_FREE` (50%) toward `risk_free_rate`, a simple Bayesian
   hedge against that noise). `random_state` (`config.random_state`) seeds a `np.random.default_rng`
   instance stored as `self.rng` — the constructor used to draw from numpy's *global* unseeded state, so
   `prob_loss` silently changed between identical back-to-back runs. All five annualizations
   (`historical_mu`, `sigma`, the GBM `dt`, and the total/downside vol of Sharpe and Sortino) come from
   `trading_days_per_year`, not a repeated 252 literal.

8. **`main_pipeline.lognormal_price_fan` / `lognormal_price_bands`** (module-level functions) — the
   classical projection's 80/90/95% intervals: `current_price * exp(drift_log ± z*sigma)` (no 99% band,
   same effective-sample-size reasoning as `CONFIDENCE_BANDS`). `lognormal_price_fan` takes drift and vol
   in their *natural* scales (annualized log drift, daily vol) and does the per-horizon scaling in one
   place, so the two routes cannot drift apart. **Its two inputs are shared with the Monte Carlo, by
   construction:** `extract_garch_forecast` is called exactly once in `run_production_system`, before
   `RiskSimulationEngine` is built, and the same daily `garch_vol_esperada` feeds both; the drift is taken
   from `risk_engine.mu` rather than recomputed, so honouring `config.drift_mode` is automatic. Before
   this, the Monte Carlo got `garch_vol.iloc[-1]` (today's *instantaneous* conditional vol, frozen across
   the horizon) while the classical route used the horizon *forecast*, and it used the raw historical mean
   while the Monte Carlo used the shrunk one: the report printed **two lognormal projections of the same
   asset with different mu AND different sigma and presented them as comparable**. The report prints the
   shared drift/sigma pair so it can be audited at a glance, and
   `tests/test_coherencia_proyecciones.py` spies on the arguments both routes actually receive during a
   full run — no test looked at that, which is why the bug survived all 196 of them.
   Prices are lognormal, not normal, so the bands must be built by exponentiating an additive log-space
   interval, not by adding/subtracting a percentage and then clamping negative results with `max(0, ...)`.
   The `sqrt(t)` scaling in the fan assumes iid returns, which at 504 days ignores the vol mean-reversion
   the GJR-GARCH itself models — documented in the docstring; long horizons must be presented as
   extrapolation, not measurement, and the report warns accordingly.

9. **`portfolio_engine.py`** — a separate, smaller pipeline for menu option `[2]` ("Analizar mi cartera
   actual"), independent of the option-`[1]` flow. `load_positions(path)` parses an IBKR Flex Query
   "Open Positions" export — **TSV despite the `.csv` extension** (separator auto-detected: tries `'\t'`
   first, falls back to `','` if that yields a single column), filtered to `AssetClass == 'STK'`. Two
   confirmed real-export quirks it handles explicitly: there's no `CostBasisPrice` column (average entry
   price is derived as `CostBasisMoney / Quantity`, guarded against `Quantity` 0/NaN), and `OpenDateTime`
   comes back completely empty (parsed tolerantly — `yyyyMMdd` or `yyyyMMdd;HHmmss`, `errors='coerce'`, no
   noisy logging when the whole column is blank). Only `Symbol` is strictly required; any other expected
   column missing from the header is filled with NaN and logged.
   `analyze_position(ticker, config)` is a **reduced** version of the option-`[1]` pipeline: prices,
   statistical risk, the GARCH walk-forward (for the conditional-vol percentile) and forecast, the
   point-in-time valuation context, the TTM net margin and its trend, and the entry drawdown at
   `config.horizon_months`. It **trains no model at all** — it used to exist to produce `current_ml_prob`,
   and removing that made it much cheaper than adding the fundamentals made it expensive. It uses
   `config.portfolio_ts_refit_every` (default 63, quarterly) instead of `config.ts_refit_every` (default
   21, monthly) because that walk-forward is the most expensive step left, and a whole portfolio needs to
   run in seconds per ticker.
   `generate_review_flags` produces a **"REVISAR"** marker — meaning *"look at it yourself"*, never an
   order — from three independent triggers, thresholds in `config.py`: the valuation percentile jumped
   more than `review_threshold_valuation_percentile_jump` (0.40) from the first record **in absolute
   value** (getting much cheaper is a material change too; the system doesn't decide which direction is
   "bad"); the conditional-volatility percentile jumped more than
   `review_threshold_vol_percentile_jump` (0.40) — the operationalization of "changed regime", since an
   absolute vol threshold would always flag the same assets for being what they are rather than for having
   changed; or the TTM net margin fell `review_margin_declining_quarters` (2) consecutive quarters, the
   only trigger evaluable on a position's first run because it needs no recorded history. Each trigger is
   `None` (not evaluable) and **never `False`** when there's no prior record or the percentile isn't
   computable, and a `None` never marks.
   `record_context`/`get_earliest_context` persist per-ticker context history to
   `config.context_history_path` (`cache/context_history.json`) with the same upsert-by-date and atomic
   `.tmp`+`os.replace` write as the legacy ML-probability history. **`cache/ml_prob_history.json` is no
   longer written** (nothing calls `record_ml_prob`); the existing file is *not* deleted because it is
   history, and `get_earliest_ml_prob` can still read it — both functions and their tests are kept.
   `evaluate_position(position, config)` ties it together for one `load_positions` row and reads history
   *before* recording today's values (so a ticker's first run correctly sees no history instead of
   comparing itself to itself). **Critically, `cost_basis_price` never enters anything measured** — it's
   the sunk entry price, and using it would introduce disposition effect and anchoring (see Dai, Zhang &
   Zhu 2010 on optimal exit rules under regime switching: the optimal rule depends on the inferred state,
   never on entry price). `cost_basis_price`/`quantity`/`unrealized_pnl` are returned as separate
   context-only keys.
   `main_pipeline.run_portfolio_analysis(config)` drives the flow: loads positions (printing IBKR Flex
   Query setup instructions instead of a raw traceback if the CSV is missing), analyzes each position
   sequentially with a `"Analizando i/n: TICKER..."` progress line (catching per-ticker exceptions into an
   `ERROR` row so one bad ticker doesn't take down the rest), prints the fixed-width context table
   (Ticker | Cant. | P.Medio | P.Actual | P&L | Val. | Vol.ann | Vol. | Beta | DD med | Revisar — `P&L`
   uses the CSV's own `unrealized_pnl`, never recomputed), an explanation naming which trigger fired with
   concrete numbers for every REVISAR, a "SIN HISTÓRICO CON EL QUE COMPARAR" block on a position's first
   run, and a closing note reiterating that this is context and not a verdict, that entry price
   participates in nothing, and that tax implications of selling (capital gains, Spain's 2-month
   wash-sale-equivalent rule) are out of scope.

10. **`prediction_log.py`** — append-only prediction registry at `config.prediction_log_path`
   (`cache/prediction_log.jsonl`), written by **both** menu modes on **every** run. `registrar_prediccion`
   is called before the `config.verbose` early-return precisely so `--no-verbose` still records; the values
   are the ones already computed for the report, nothing is recalculated for it. This is the only output
   the pipeline produces that **cannot be regenerated**: its value is that it was written *before* the
   outcome was known.
   Two properties are load-bearing and easy to break. (a) **It never raises** — serialization included,
   which is why `_serializar` sits *inside* the `try`: a logging failure must not take down a report that
   is already computed and printed. NaN/Inf are stored as `null` so the file stays strict JSON, and
   numpy/pandas scalars are converted losslessly. (b) **The schema is versioned and additive** — every key
   is optional. Phase 3 stopped emitting `score`, `recomendacion`, `factores`, `ml` and `robustez` and
   added `valoracion_contexto`, `salud_financiera`, `vol_condicional`, `drawdown_entrada`,
   `tiempo_recuperacion`, `abanico_lognormal`, `abanico_cuantilico` (+ its `omitidos` and `validacion`)
   and `advertencias`; keys that survived (`precio_actual`, `fiabilidad`, `proyeccion_inputs`,
   `monte_carlo`, `riesgo`, `historico_efectivo_*`) kept their exact meaning. **Never rename or reuse a
   key with a different meaning** — add a new one, or the accumulated history is invalidated, which is
   exactly what this file exists to prevent. JSON Lines rather than one JSON document so a truncated line
   doesn't take the rest of the file with it (`leer_predicciones` drops unreadable lines with a warning).
   `scripts/revisar_predicciones.py` reads it back, downloads current prices (via `last_closed_session`,
   same rule as the pipeline) and reports realized return and whether today's price landed inside the
   projected 90% band — reading that band from the old single-horizon keys *or* the new fan, so a log
   spanning both eras works. It still scores the old COMPRA/EVITAR entries (they're the only evidence that
   layer will ever leave) and treats a missing recommendation as "the system no longer asserts a
   direction", not as a failure.

11. **`ml_engine.py`, `validation_engine.py`, `scoring_backtest_engine.py`,
   `scripts/diagnostico_multihorizonte.py`** — **kept, but out of the decision path.** Nothing in the
   report comes from here. See **The Phase 3 removal** above for why they're kept and what was deleted
   from them.
   `MachineLearningEngine` trains Logistic Regression, Random Forest and XGBoost and picks a winner by
   Balanced-Accuracy (not F1 — a test slice can end up single-class if the regime shifts, which degenerates
   F1/ROC-AUC to a meaningless tie), with selection and the historical probability series both
   **walk-forward** (`_walkforward_predict`: expanding window, purging `embargo_days` at each step, pooled
   metric across blocks) rather than a single 80/20 split, whose test slice can amount to ~1 independent
   observation once you account for the forward-looking target's own horizon.
   `BacktestAndScoringEngine` keeps the two backtests: `run_holding_period_backtest` (the event study —
   signal-date forward returns vs. a control group over the same horizon, reporting hit rate, percentiles,
   the signal-vs-control excess, and `n_independent_signals = n_signals / horizon_days` next to every
   metric, gating `statistically_conclusive` on `MIN_INDEPENDENT_SIGNALS_FOR_CONCLUSIVE` (10)) and
   `run_strategy_backtest` (daily-rebalanced long/flat with per-trade costs, which nobody executes —
   internal input to the Deflated Sharpe only). Both annualize from `trading_days_per_year`.
   `ValidationEngine` keeps `run_subperiod_validation` (retrains on each of `N_SUBPERIODS` (4) independent
   chronological blocks and reruns the event study per block; `_assess_regime_dependence` flags a hit-rate
   swing over `HIT_RATE_REGIME_DEPENDENT_THRESHOLD` (20 points), and fewer than 2 evaluable blocks yields
   `regime_dependent=None` — not evaluable, not "stable") and `compute_deflated_sharpe_ratio` (Bailey &
   López de Prado 2014: trying 3 classifiers and keeping the winner inflates its Sharpe by multiple-testing
   selection, so the Probabilistic Sharpe Ratio of the *selected* model is evaluated against the expected
   Sharpe of the best of 3 under the null, not against 0). **Everything in the DSR is computed in DAILY
   frequency** — the PSR requires the Sharpe and `T` to share a frequency, and the conversion must be
   applied to the **whole** Sharpe vector, not just `sr_hat`, since `sigma_sr` and `expected_max_sharpe`
   derive from it; de-annualizing only the winner collapses the DSR to ~0 always, the mirror of the
   original bug that pinned it at 1.0000 (verified on the same inputs: 1.0000 → 0.0000002 → 0.9474). The
   de-annualization root comes from `trading_days_per_year`. `n_trials` is documented as a **floor, not the
   real count** of specifications searched, so the DSR is **optimistic by construction**.
   `scripts/diagnostico_multihorizonte.py` is the Phase 2 gate: for N tickers x {1,3,6,12,24}-month
   horizons it retrains the ML with each horizon's own target and reports the event study's hit rates and
   `excess_difference`, a **Newey-West (HAC, lag = horizon_days) standard error** on that excess with its
   95% CI and p-value, and the **Information Coefficient** (Spearman between each date's walk-forward
   probability and that date's forward return). Two structural points: the features don't depend on the
   horizon (only the *target* does), so the expensive GARCH/ARIMA walk-forward is computed once per ticker
   and reused; and **every statistic is reported next to its own effective sample size**, with aggregates
   excluding cells that fail it (`grupos_suficientes`, `ic_fiable`, `hac_fiable`) — applying the overlap
   correction to the excess but not to the IC would commit, in the same table, the error the script exists
   to expose.

12. **`visualization_engine.ReportVisualizer`** — saves 2 PNGs per run to `config.output_dir` (default
   `output/`): the Monte Carlo terminal-price histogram (`mc_results['final_prices']`) and the GARCH
   conditional volatility series. Uses matplotlib's `Agg` backend (headless-safe, never opens a window) and
   follows the validated categorical palette in slot order (see the `dataviz` skill). Called from
   `main_pipeline.py` inside a `try/except` — a plotting failure logs a warning but never crashes an
   already-computed console report. `plot_equity_curve` still exists and `generate_all`'s `df_bt` parameter
   is optional: the equity curve plotted `run_strategy_backtest`, i.e. the retired signal, so the pipeline
   no longer computes one (asking for it would mean training the ML every run just to draw a chart), but a
   caller that has a `df_bt` still gets it drawn.

### The report, block by block

`run_production_system` prints, after the `FICHA DE ENTRADA — {TICKER} — {fecha}` header:

1. **PRECIO Y VALORACIÓN** (verbose only) — current price, 12-month range (a year of *sessions*, from
   `trading_days_per_year`), effective history, and the five point-in-time multiples with their own
   historical percentile, historical median and `n`, plus the `n_quarters`/`fiable` pair.
2. **PERFIL DE RIESGO** (verbose only) — today's conditional GARCH vol **and its percentile within its own
   history** (the level alone doesn't say whether the asset is calm or agitated *for it*), the horizon's
   expected vol, annualized vol vs. benchmark and relative, rolling 1-year beta, max drawdown, VaR 95%,
   Expected Shortfall, Sharpe, Sortino, and the Monte Carlo with **its two assumptions printed in the
   header line** and an explicit statement that it is a simulation under GBM, not a measurement.
3. **A QUÉ TE APUNTAS SI COMPRAS HOY** (always) — the central block, and the only one `--no-verbose`
   keeps. Three tables across 1/3/6/12/24 months: the entry-drawdown percentiles, the recovery-time
   distribution with `% no recupera` / `% nunca bajo agua`, and the two price-band fans (classical
   lognormal and quantile) with the quantile route's out-of-sample empirical coverage beside it. **Every
   row carries its `n`, its `n indep.` and a `fiable` flag, in the same row** — not in a footnote, because
   a footnote is read after you've already believed the number.
4. **SALUD FINANCIERA** (verbose only) — the descriptive card from `describe_quality`, including the
   `VENTANA` column with each value's real window coverage and the aggregate note.
5. **ADVERTENCIAS** (always) — every active warning, each naming its concrete cause and figure. Printed in
   both verbosity modes on purpose (see Commands).

### Key invariants to preserve when editing

- **No score, no verdict, no categorical label.** The report describes. If you find yourself computing a
  weighted average of several factors, or mapping a number to a word, stop and read **The Phase 3
  removal** above — including the guard tests that will fail.

- **Every number travels with its sample size.** This is the phase's acceptance criterion: *"the report
  contains no number that can't be justified with a sample whose size is printed next to it."* If you add
  a metric, print its `n` — and if it's built on forward-looking windows, its `n_independiente` too.

- **Overlapping windows are not independent observations.** With an `H`-day forward window, consecutive
  dates' windows overlap almost entirely, so a raw count overstates the sample by roughly `H`×.
  `n / horizon` is the (rough but honest) correction, and any window-based metric must report it alongside
  the raw count and gate its own reliability on it (`MIN_OBS_INDEPENDIENTES`,
  `MIN_INDEPENDENT_SIGNALS_FOR_CONCLUSIVE`) — a striking-looking figure over 4 independent windows is
  noise dressed up as a finding.

- **A missing number is never a measurement.** `n/d` for a non-finite value, `None` (not `False`) for a
  trigger that can't be evaluated, `nan` (not a fabricated value) for a DSR that can't be deflated,
  `reliable=False` for thin fundamentals, `regime_dependent=None` for a check that couldn't run, an empty
  card rather than a neutral 0.5. Three states, not two: `sí` / `NO` / `n/d`, where `NO` asserts it was
  measured and falls short and `n/d` asserts it couldn't be measured. If you add a metric that can fail to
  compute, give it an explicit not-evaluable state — and make sure the report *says which input was
  missing*, not just that something was.

- **A ratio whose denominator is a difference of large numbers needs a materiality floor, not a cap.**
  The ROIC case: `debt + equity − cash` can be positive but tiny, and then its relative error dominates.
  Nulling it (NaN) is right; clipping the ROIC to a ceiling would leave it reading as a measurement. See
  `_capital_invertido_material` and `config.roic_min_invested_capital_fraction`.

- **Windows that count rows are not windows that cover time.** `.rolling(4)` and `.pct_change(4)` count
  rows; with gappy fundamentals a "TTM" can span 15 months. `window_coverage()` measures it and the report
  publishes it per value. If you add another window-based accounting figure, report its real coverage
  too — 69% of real TTM windows fail the check, so assuming they're fine is assuming wrong.

- **Report a metric in the scale its label claims, and convert the WHOLE set of inputs.** Two live bugs
  came from mixed units, both invisible to the test suite: the Sharpe computed from a drift already shrunk
  50% toward `rf` yet printed as "histórico" (exactly half the real value), and the DSR mixing an
  annualized Sharpe with a daily observation count (pinned at 1.0000). When a formula combines a rate and
  a period count, convert every term to one frequency — converting one term and not the others just moves
  the bug.

- **Annualization and calendar come from `config.trading_days_per_year`; factor *shape* does not.**
  252 was a repeated literal and it is wrong for an asset trading 365 days a year (vol underestimated by
  `sqrt(365/252)` = 1.20, and a "6-month horizon" of 126 sessions is 4 calendar months). Everything that
  is a *unit conversion* — annualizing, or months→sessions — threads through the config field, including
  `trading_days_per_month` and `horizon_days`, which are derived from it rather than being separate
  fields. Everything that defines the *shape* of a methodology (the 21/252 of the academic 12-1 momentum,
  the 252-observation GARCH/ARIMA burn-in) stays literal, because changing those would redefine the factor
  rather than adapt the calendar. That's a documented limitation, not an oversight, and
  `tests/test_calendario_negociacion.py` pins it so reintroducing it has to be a conscious decision.

- **No lookahead bias.** Forward-looking targets are computed separately from factor computation; splits
  are chronological, not random; walk-forward features refit on an expanding window only. Any new feature
  or signal must respect this — don't let information from day T influence a decision dated before T.

- **Purged/embargoed splits for forward-looking targets.** `QuantileForecastEngine` (and
  `MachineLearningEngine`, still tested) predict a label that looks `horizon_days` into the future, so
  labels overlap heavily between neighboring rows. A plain chronological split would leak: training rows
  near the boundary would have labels computed from prices inside the test window. Both drop the last
  `embargo_days`/`horizon_days` rows of the training slice — and in the fan, **each horizon's embargo is
  sized to its own target**. If you add another forward-looking target anywhere, apply the same purge.

- **Walk-forward, not whole-series, for GARCH/ARIMA features.** `TimeSeriesFeatureExtractor` is the second
  place where lookahead can sneak back in: fitting once over the full return series estimates each date's
  parameters using data from *after* that date. Both methods must keep refitting periodically on an
  expanding window and propagating (not re-estimating) parameters between refits — see
  `test_extract_garch_volatility_has_no_lookahead_bias` for the property this preserves (changing data
  *after* a given date must not change that date's estimate).

- **Point-in-time everything.** Prices: `ffill` yes, `bfill` **never** — `bfill()` replicated a ticker's
  *first* known price backwards across everything before it listed, fabricating years of exactly-zero
  returns (artificially low vol, zero momentum, zero beta) and thousands of invented rows; trimming to the
  first real price is the rule, and the effective history obtained must be reported. Fundamentals:
  `_fetch_fundamental_data_edgar` indexes each quarter by its *publication* ("filed") date, not the
  accounting period's end — a Q4 balance sheet isn't public until the 10-K files months later — and
  resolves XBRL restatements and tag migrations by keeping the *first*-filed value per period, not the
  corrected one, because that's what the market actually knew. Valuation multiples align each reporting
  date to the price in effect then (`merge_asof`), not today's price. Don't "simplify" any of these.

- **Log-scale drift never gets a second `-0.5*sigma²` convexity correction.** Both
  `RiskSimulationEngine.run_monte_carlo` and `lognormal_price_bands` estimate their drift directly from
  *log* returns, which is already the log-price process's own drift — the `-0.5*sigma²` term exists only to
  convert an *arithmetic* drift into a log one via Itô's lemma, so applying it to a drift that's already
  log-scale double-subtracts it and silently shifts the whole distribution down. If you introduce another
  price simulation, estimate the drift in whichever scale matches the formula you feed it into.

- **The two projection routes share one drift and one sigma, by construction.** `extract_garch_forecast`
  is called once per run and both the Monte Carlo and the lognormal fan consume that value; the drift comes
  from `risk_engine.mu`, not a recomputation. The report prints the pair so it can be audited at a glance.
  `tests/test_coherencia_proyecciones.py` spies on what both routes actually receive.

- **Validate the object you publish.** The quantile rearrangement must be applied inside
  `evaluate_out_of_sample` and not only in `predict_price_quantiles`, or the reported coverage and pinball
  loss describe a different object than the bands on screen.

- **Every stochastic engine takes an explicit `random_state`.** `RiskSimulationEngine` seeds a
  `np.random.default_rng(random_state)` stored on `self.rng`, rather than drawing from numpy's global
  unseeded state — without this, identical back-to-back runs silently produced different `prob_loss`. Any
  new engine with randomness must take and propagate this same seed.

- **The same run twice must produce the same report.** A report that can't be reproduced can't be audited,
  and this was genuinely broken (evidence: `cache/ml_prob_history.json` held JD at both `0.66208637` and
  `0.66246547` on the *same date* with `random_state=42`). Four causes, all of which must stay fixed:
  `end_date` is the last **closed** session, never `date.today()`; the price panel is cached per
  `(tickers, start, end)`; ticker dedup preserves order via `dict.fromkeys`, never `list(set(...))` (string
  set iteration order varies *per process* under `PYTHONHASHSEED`); and tree models pin `n_jobs=1`/
  `nthread=1` (thread scheduling changes the order of floating-point summation, enough to flip a leaf).
  Note that an **in-process** double-run test cannot detect the first three, which is why
  `tests/test_reproducibilidad.py` also spawns real subprocesses; keep that if you touch this area.
  Verified on real data: two META runs in separate processes produce byte-identical reports.

- **The console report must survive a non-console stdout.** `main_pipeline` calls `_forzar_salida_utf8()`
  at import to reconfigure `sys.stdout`/`sys.stderr` to UTF-8. On Windows, a redirected or piped stdout
  defaults to cp1252, which cannot encode the characters the report emits throughout (`█` banner, `⚠`
  warnings, `─` separators), so `python main_pipeline.py > informe.txt` died with `UnicodeEncodeError`
  *after* paying for the GARCH walk-forward, the quantile fan and the Monte Carlo. The menu loop's own
  error handler prints `⚠` too, so when the failure *was* an encoding failure the handler re-raised and the
  traceback escaped anyway; that's why this is fixed at the root rather than by wrapping prints. This class
  of bug is invisible to the test suite by construction — pytest replaces stdout with its own capture
  object — so `tests/test_degradacion_fallos_datos.py` verifies it through a real subprocess writing to a
  file. If you add output to another entry-point script, apply the same reconfigure.

- **A data failure returns to the menu; it doesn't kill the session.** `data_engine` does `raise e` on a
  download failure. `run_menu_loop` wraps both branches in `try/except` — a data failure is an expected
  condition for a pipeline that lives off an external API, not a bug to abort on — and handles
  `KeyboardInterrupt` separately, because that's a user decision, not a failure.
  `run_portfolio_analysis` likewise catches `ValueError`/`ParserError`/`EmptyDataError`/`UnicodeDecodeError`
  from the IBKR parser (a file that exists but is unreadable) with a message distinct from the
  `FileNotFoundError` one, which keeps its Flex Query setup instructions.

- **Historical-percentile scoring**, not cross-sectional peer comparison: every percentile in this system
  ranks the asset against its *own* history (`(serie <= hoy).mean()`), so it is only meaningful in the
  context of that one ticker's own range. Percentiles are reported in **natural orientation** (high = high
  value) even where "less is better" — no inverted metrics, so one convention reads the whole table. A
  cross-sectional panel is the (optional, unstarted) Phase 6.

- **Spanish is the working language throughout the codebase** — docstrings, log messages, console output,
  and variable names mix Spanish and English. Match the existing convention when editing nearby code
  rather than switching a file to all-English.

- **`README.txt` documents what the report prints**, for a retail/semi-professional investor audience:
  what every printed metric means and its interpretation ranges. If you add a metric to the console
  output, update `README.txt` to match — this is an explicit standing requirement from the project owner
  (see `Ajustes y Arreglos.txt`). Likewise, `Ajustes y Arreglos.txt` is the running record of what changed
  and why; a removal from the output must be recorded there with its date and justification, because the
  owner's standing rule is **"only add to the output, never remove, unless I ask explicitly"**.

### Known open issues (from `Ajustes y Arreglos.txt`)

- **`shares_outstanding` missing for some EDGAR issuers** (META confirmed): no share count → no market cap
  → every valuation multiple `n/d` and the whole PRECIO Y VALORACIÓN block empty. The fix is to add
  `WeightedAverageNumberOfDilutedSharesOutstanding` as an alternative tag, but *which* share count to use
  (diluted weighted average vs. period-end outstanding) is a modeling decision that changes the entire
  historical multiple series, so it's open rather than silently patched.
- **TTM/YoY windows don't cover 12 months for most tickers** (69% / 78% measured; median TTM 455 days).
  Detected, flagged and published per value (point 4, `VENTANA` column), but *not* fixed: fixing it means
  reindexing the EDGAR extractor to fiscal quarters.
- **Long-horizon bands are extrapolation.** The `sqrt(t)` scaling of the lognormal fan assumes iid
  returns; at 12 and 24 months that ignores the vol mean-reversion the GJR-GARCH itself models. The
  quantile route's own out-of-sample coverage at 24 months is frequently 30-60% against a nominal 80-95%.
  Both are reported and warned about rather than hidden. Treat long horizons as a modeling caveat, not
  something to "fix" by raising a flag.
- **Crypto detection is manual.** `--trading-days-per-year 365` must be passed by hand; there's no
  auto-detection, and the momentum/GARCH *shape* windows still count sessions (see the calendar invariant).
- **Phase 5 (external analyst price targets) and Phase 6 (cross-sectional sector panel)** are specified in
  `PLAN_REFACTOR.md` and unstarted. Phase 6 has a non-negotiable entry condition: it only opens if a
  measured IC on the panel clears zero stably under the same yardstick that retired the ML in Phase 2
  (enough independent windows, stable sign across horizons, Newey-West). If the panel doesn't pass, no
  score is reintroduced.
