# Audit: what was wrong, what was fixed, how each fix is proven

This document is the review trail. Every finding was reproduced against the
original code, fixed, and covered by a test in `tests/`. Findings are grouped by
severity.

Original state: three notebooks, a 145-line Flask app, a hardcoded-credential
Reddit notebook, 10 pre-trained `.h5` files, and a `requirements.txt` that could
not be installed.

---

## Critical

### C1 - Live Reddit API credentials committed in plaintext

**Where:** `reddit_sentiment_analysis.ipynb`, cell 2.

```python
reddit = praw.Reddit(
    client_id="h1oT9ECrvYgR8oipbCwXRg",
    client_secret="wq3QJIj0mcSB9TAIspKMG7pXTbhpLQ",
    user_agent="isreal_data",
)
```

**Why it matters:** anyone who has ever cloned or forked this public repository
holds a working credential. Secret-scanning bots find these within minutes.
Because it was committed to git history, **rotating the key is mandatory** -
deleting the line does not remove it from history.

**Fixed:** `scripts/fetch_reddit_sentiment.py` reads `REDDIT_CLIENT_ID`,
`REDDIT_CLIENT_SECRET` and `REDDIT_USER_AGENT` from the environment and refuses
to start with an explicit error when any are missing. `.env.example` documents
the names; `.gitignore` excludes `.env`.

**Action required by the owner:** revoke the exposed credentials in the Reddit
app console. This cannot be fixed from inside the repository.

### C2 - Test-set leakage through global normalisation

**Where:** `crypto_prediction/app.py` (both `/data` and `/next_price`).

```python
scaler = MinMaxScaler(feature_range=(0, 1))
df['scaled_price'] = scaler.fit_transform(df[['Close']].values)
```

**Why it matters:** the scaler is fitted on the **entire** series, including the
period later used as the test set. The min/max of the future therefore inform
the scaling of the past. Reported error becomes optimistic and the model is
never evaluated on a genuine out-of-sample distribution.

**Fixed:** `core/timeseries.MinMax` is fitted on the training slice only and
applied to everything after it. `test_no_leakage_from_future_values` asserts
that adding future data does not change the transform of past data.

### C3 - Random shuffling of a time series before splitting

**Where:** `model_training.ipynb` cell 16
(`train_test_split(..., shuffle=True)`) and the notebook workflow generally.

**Why it matters:** shuffling a time series lets the model train on day *t+1*
and be evaluated on day *t*. It manufactures accuracy that does not exist.

**Fixed:** `core/timeseries.sequential_split` produces chronological, contiguous
train/val/test slices. `test_split_is_chronological_and_non_overlapping` pins
the boundaries.

---

## High

### H1 - Predictions were not aligned with the data they predicted

**Where:** `app.py` `/data`.

```python
sequences = [...]
predictions = model.predict(sequences)
actual_prices = df['Close'].values[sequence_length:]
```

The window loop produces one fewer sequence than `len(df) - sequence_length`,
and the shifted `actual_prices` slice did not line up with the predictions being
plotted. The chart compared two differently-aligned series, so the visual "fit"
was an artefact.

**Fixed:** `core/backtest.walk_forward` walks index by index and records the
actual value for the exact step being predicted. `test_walk_forward_no_lookahead`
asserts the training window never contains its own target, and that
`len(actual) == len(predicted)`.

### H2 - `next_price` reused the whole series to scale a single window

**Where:** `app.py` `/next_price`. The scaler was re-fitted over the full series
on every request, so the "next price" moved whenever older data was included -
it was not a stable forecast.

**Fixed:** one fitted scaler per model run (`build_prediction`), reused to
inverse-transform the forecast. The scaler is fitted on the training split only.

### H3 - Malformed CSV rows silently produced `NaN`, which propagated everywhere

**Where:** `pd.read_csv(...)` with no validation. A single bad row makes the
scaled series contain `NaN`; every subsequent prediction is `NaN` and the UI
displays a blank chart with no error.

**Fixed:** `core/dataset.load_series` parses and validates row by row, returning
a `ValidationReport` (rows in, rows out, dropped-invalid, dropped-duplicates,
warnings, errors). Non-finite values, impossible dates (`2021-01-32`),
non-positive prices and `high < low` are all handled explicitly.
Covered by `test_handles_malformed_rows`, `test_impossible_date_is_rejected`,
`test_garbage_file_terminates_quickly`.

### H4 - No tests, so none of the above could be caught

**Fixed:** 86 standard-library tests across `tests/test_core.py` and
`tests/test_extras.py`, plus CI running them on three Python versions

---

## Medium

### M1 - Model files and data were loaded by directory listing with no contract

`list_models()` returned filenames; `get_csv_file()` derived the CSV name by
string-splitting the model filename (`model_name.split('_')[0]`). Any rename
silently broke the pairing.

**Fixed:** `core.dataset.discover_datasets` builds a symbol → path map, and
`web.app._symbol_from_model_name` normalises `Bitcoin_model.h5`,
`coin_Bitcoin.csv` and `Bitcoin` to the same key. Covered by
`test_symbol_from_model_name`.

### M2 - Internal paths and exception text leaked to the browser

`except Exception as e: return jsonify({"error": f"Error during prediction: {str(e)}"})`
returned raw exception messages (filesystem paths, library internals) to any
client.

**Fixed:** errors are mapped to typed responses; `DataError` → 404 with a safe
message, unexpected failures → 500 with a generic message. `MAX_CONTENT_LENGTH`
bounds request bodies.

### M3 - `debug=True` hardcoded in the app entry point

Running with `debug=True` unconditionally enables the Werkzeug debugger, which
allows arbitrary code execution from the browser if the server is reachable.

**Fixed:** debug is opt-in via `FLASK_DEBUG=1`; the app binds `127.0.0.1` by
default.

### M4 - No `MAPE` guard for near-zero prices

`MAPE` divides by the actual value. Dogecoin traded at `$0.000087`, so percentage
error explodes and the metric becomes meaningless - yet it was the headline
number.

**Fixed:** `smape` (symmetric, denominator is a sum of magnitudes) is the default
reported error, `mape` ignores zero actuals, and `value_at_risk`/`summary_stats`
return `null` instead of `NaN` so the JSON stays valid.

### M5 - Requirements files could not be installed

Both `requirements.txt` files are **UTF-16 encoded with a BOM**; `pip` cannot
parse them. `crypto_prediction/requirements.txt` also pins `tensorflow-cpu`
alongside `tensorboard`, `keras`, `h5py` and `protobuf` at mutually inconsistent
versions.

**Fixed:** `pyproject.toml` declares the core with **no dependencies** and groups
the heavy stack into extras (`ml`, `web`, `sentiment`, `dev`).

### M6 - Reproducibility: no seeds, no version pinning, no CI

`np.random` and TensorFlow were never seeded; `runtime.txt` pinned
`python-3.9.16` while the dev container used 3.11.

**Fixed:** deterministic models only in the core (no RNG), a seed set in the
Keras path, and CI on 3.9/3.11/3.12.

### M7 - `dummy.py` placeholder files shipped in `models/` and `CSV/`

Empty files committed to force directory creation. Confusing and a sign the
directories were never designed.

**Fixed:** removed; the loader tolerates a missing directory with a clear error.

---

## Low

- `app.run()` used the default port while `.devcontainer` forwarded 8501 →
  unified through `PORT`.
- The README documented the **upstream** repository (`AshishRShetty2000/...`),
  not this fork, and told users to run `npm install` for two CDN-loaded chart
  libraries. Rewritten.
- `.slugignore` excluded the very artefacts the app needed to serve.
- A stray backslash line-continuation in `get_csv_file` (`...\\`) made the
  following `raise` unreachable; the function returned `None` on a missing file.
- No license file. MIT added.

---

## What is still true (not fixed, by design)

- The bundled data ends **2021-07-06**. This is a historical backtest tool until
  a live data source is wired in.
- The models are univariate; sentiment is not an input to any forecast yet.
- `confidence` is a rescaled directional accuracy, not a calibrated probability.
- Fees are a flat per-trade charge; there is no slippage or funding model.
