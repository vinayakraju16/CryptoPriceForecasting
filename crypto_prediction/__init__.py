"""CryptoPriceForecasting - data pipeline, forecasting and backtesting package.

Layers:

* ``crypto_prediction.core``     - dependency-free (stdlib) data + modelling core.
* ``crypto_prediction.web``      - the Flask UI/HTTP layer (optional dependency).
* ``crypto_prediction.core.kat`` - optional Keras/TensorFlow LSTM wrapper.

Anything needing numpy/pandas/tensorflow is imported lazily so the forecasting
core keeps working on a bare Python interpreter.
"""

__version__ = "2.0.0"
