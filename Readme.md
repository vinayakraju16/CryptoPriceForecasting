# CryptoPriceForecasting

Forecasting, walk-forward backtesting and risk-adjusted screening for a basket of
cryptocurrencies - built so that a reviewer can clone it, run **one command**,
and see real numbers with **zero heavy dependencies installed**.

```bash
git clone https://github.com/vinayakraju16/CryptoPriceForecasting.git
cd CryptoPriceForecasting
python -m crypto_prediction screen --model drift        # rank every coin
python -m crypto_prediction recommend --capital 5000    # what to buy, and how much
python -m crypto_prediction.server                      # local dashboard on :8501
python -m unittest discover -s tests -v                 # 121 tests, stdlib only
```

No `pip install` is required for any of the above. The optional Flask UI and the
LSTM model pull in Flask / TensorFlow only when you actually use them.

---

## What this project does

| Command | What it answers |
|---|---|
| `python -m crypto_prediction validate` | Do all the datasets parse correctly, and what was rejected? |
| `python -m crypto_prediction screen --top 5` | Which coins look most attractive *relative to how wrong the model usually is*? |
| `python -m crypto_prediction forecast Bitcoin --steps 7` | Next price, confidence band, expected return, signal |
| `python -m crypto_prediction compare Bitcoin` | Benchmarks 7 models on one coin and picks the best |
| `python -m crypto_prediction backtest Bitcoin --model holt` | Walk-forward accuracy **and** a fee-aware trading simulation |
| `python -m crypto_prediction recommend --capital 5000` | Position-sized buy proposals |
| `python -m crypto_prediction report --out reports/latest.html` | Self-contained HTML report (no server needed) |
| `python -m crypto_prediction live Bitcoin` | Fetch recent bars from a public provider, then forecast |
| `python -m crypto_prediction.server --port 8501` | Local dashboard + JSON API, **no Flask required** |

Add `--json` to any command for machine-readable output.

---

## The honest part (read this first)

**You cannot reliably predict tomorrow's crypto price.** Anyone who claims
otherwise is either selling something or measuring it wrong. This project is
built around that fact rather than pretending otherwise:

1. **Every forecast ships with an error band.** `predicted_next_price` is
   accompanied by `predicted_low` / `predicted_high` derived from the model's
   own recent residuals. If the model has been inaccurate on a coin, the band
   is wide - visibly wide.
2. **Accuracy is measured out-of-sample, in order.** Models are evaluated with
   a walk-forward (expanding-window) backtest, never a random split.
3. **Signals are normalised by the model's own error.** A "buy" only fires when
   the expected move exceeds the model's typical error, so a noisy coin cannot
   produce a confident-looking signal by accident.
4. **A trading simulation with fees runs alongside every backtest**, so
   "accurate prediction" and "profitable strategy" are never conflated. Some
   models here are accurate and still lose money after fees; that is reported.
5. **`recommend` can legitimately return nothing.** Staying flat beats trading
   noise, and the tool says so in those words.

Sample of the trade-off being visible - Bitcoin, walk-forward, 2,841 out-of-sample steps:

| model | directional accuracy | sMAPE | simulated return (0.1% fee) |
|---|---|---|---|
| drift | 0.533 | 2.73 | +44021% |
| holt | 0.521 | 3.45 | +598% |
| ar | 0.485 | 3.97 | +1685% |
| ensemble | 0.499 | 3.68 | −74% |
| naive | 0.000 | 2.70 | 0% |

`naive` has the *best* raw price error and a **0%** directional accuracy - it
predicts "no change" every time. That is exactly the trap a single accuracy
number hides, and why direction is ranked first here.

---

## Architecture

```
crypto_prediction/
├── core/                 # dependency-free (stdlib only) - the real project
│   ├── dataset.py        # CSV discovery, tolerant parsing, validation report
│   ├── timeseries.py     # leak-free MinMax, windowing, chronological splits
│   ├── indicators.py     # SMA/EMA/RSI/MACD/Bollinger/ATR + risk statistics
│   ├── metrics.py        # MAE/RMSE/MAPE/sMAPE/R²/directional accuracy
│   ├── models.py         # 7 forecasters behind one interface + ensemble
│   ├── backtest.py       # walk-forward evaluation + fee-aware strategy sim
│   ├── signals.py        # confidence bands, signals, position sizing, recommend
│   └── sentiment.py      # lexicon sentiment (finance-tuned, negation-aware)
├── web/app.py            # Flask adapter (optional dependency)
├── server.py             # stdlib HTTP server + dashboard (no Flask needed)
├── report.py             # static HTML + JSON report generator
└── __main__.py           # CLI: the reproducible surface of the project

tests/                    # 121 stdlib tests, incl. the classic leakage traps
scripts/                  # Reddit sentiment fetch (credentials from env only)
notebooks are legacy     # see docs/AUDIT.md for what was wrong with them
```

The design rule: **the core never imports numpy, pandas, TensorFlow or Flask.**
It runs on a bare Python 3.9+ install, which is what makes it testable in CI and
reviewable by someone who does not want to install 2 GB of wheels.

There is also `core/data_source.py`: a live-data adapter for Binance and
CoinGecko public endpoints. The HTTP transport is injectable, so the parsing and
merge logic is fully tested offline; use it when you want current prices rather
than the bundled 2021 history.

### Models available

| model | family | notes |
|---|---|---|
| `naive` | baseline | persistence - the bar every other model must clear |
| `sma` | smoothing | simple moving average |
| `drift` | trend | last value + average per-step drift |
| `linear` | regression | OLS line extrapolated one step |
| `holt` | exponential smoothing | level + trend |
| `ar` | autoregressive | AR(p) fitted on **log-returns**, repriced from last close |
| `ensemble` | meta | equal-weight average of sma/drift/linear/holt |
| `keras_lstm` | deep learning | only registered if TensorFlow is importable |

`core.feature_matrix()` exposes the technical features (RSI-14, MACD,
EMA-12/26, SMA-20, ATR-14, 1-day return, price-vs-SMA) for the notebook
experiments and the `keras_lstm` path. The ten pre-trained `.h5` files that ship
with the repository can also be loaded via `core.models.load_saved_keras_model`
when TensorFlow is present - with a scaler fitted on the training slice, unlike
the original code.

---

## Web UI (optional - two ways)

**Option A: no dependencies at all.**

```bash
python -m crypto_prediction.server --port 8501   # http://127.0.0.1:8501
```

A standard-library `http.server` implementation exposing the same JSON API
(`/healthz`, `/api/datasets`, `/api/models`, `/api/forecast`, `/api/screen`,
`/api/recommend`, `/api/backtest`, `/api/stats`) plus a small interactive
dashboard. The server is covered by `tests/test_server.py`, including that error
responses are JSON and never leak filesystem paths.

**Option B: the Flask app** (template rendering, legacy-compatible routes).

```bash
pip install -r crypto_prediction/requirements.txt
python -m crypto_prediction.web.app
# or: FLASK_DEBUG=1 PORT=8501 python crypto_prediction/web/app.py
```

Routes: `/` and `/dashboard` (pages), plus the same `/api/*` JSON endpoints. The
original front-end contract (`/models`, `/data`, `/next_price`) is preserved with
the same payload shape, so the existing HTML keeps working.

Every error path returns JSON - never a stack trace, and never an internal path
or a raw exception string (see `docs/AUDIT.md`, finding S2).

---

## Sentiment (optional, and no longer dangerous)

The legacy notebook had **live Reddit API credentials committed in plaintext**.
Those are gone. The replacement:

```bash
export REDDIT_CLIENT_ID=...  REDDIT_CLIENT_SECRET=...  REDDIT_USER_AGENT="crypto-forecast/2.0"
pip install praw
python scripts/fetch_reddit_sentiment.py --subreddit CryptoCurrency --limit 100 \
    --out data/reddit_sentiment.csv --daily-out data/reddit_sentiment_daily.csv
```

Scores come from a finance-tuned lexicon (`core/sentiment.py`) that understands
negation ("not bullish") and intensifiers ("very bullish") and runs on the
standard library. The BERT + PyTorch classifier remains in the notebooks for
higher accuracy; it is no longer required to produce a sentiment number.

---

## Development

```bash
python -m unittest discover -s tests -v   # the whole suite
python -m crypto_prediction validate      # dataset sanity check
python -m crypto_prediction report        # build reports/latest.html
```

CI (`.github/workflows/tests.yml`) runs the suite on Python 3.9 / 3.11 / 3.12,
validates the datasets, screens the universe, smoke-tests the HTTP server, and
uploads the generated report as a build artefact. A second job installs Flask and
exercises the web routes, including the legacy `/data` and `/next_price`
endpoints.

---

## Live data (optional)

The bundled price history ends **2021-07-06**, so forecasts from it are
backtests, not a current view. To forecast on recent data:

```bash
python -m crypto_prediction live Bitcoin --provider binance
python -m crypto_prediction live Ethereum --provider coingecko --no-cache
```

Both providers are public and need no API key. Fetched bars are validated and
**merged onto** the cached history (newest revision wins per date), so today's
provisional candle does not corrupt the series. If the network is unreachable the
command fails with a clear message instead of silently forecasting stale data.

## Limitations, stated plainly

- **The bundled price data ends 2021-07-06.** Every number in the report is a
  historical backtest, not a live forecast. Use `python -m crypto_prediction live`
  to fetch current prices first.
- The models are univariate and price-only by default; there is no order-book,
  on-chain or macro input.
- No transaction-cost model beyond a flat per-trade fee, no slippage, no
  funding rates, no shorting.
- The lexicon sentiment is a rough proxy, not the BERT classifier.

### Next steps that would materially improve it

1. Persist fetched live data to disk so screening can run across a current
   universe rather than only the ten bundled coins.
2. Multivariate model that consumes the technical feature matrix and sentiment.
3. Probability calibration (quantile regression) so `confidence` is a real
   probability rather than a rescaled accuracy.
4. Portfolio-level construction across coins with a correlation-aware risk
   budget, instead of independent per-coin sizing.

---

## Disclaimer

This is a research and engineering project. Nothing in it is financial advice.
Cryptocurrency is extremely volatile and you can lose everything. Past
backtested performance does not predict future results. If you are considering
trading with real money, talk to a licensed professional.

## License

MIT
