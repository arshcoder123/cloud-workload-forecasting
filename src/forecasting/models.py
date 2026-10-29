import numpy as np
import warnings
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.linear_model import Ridge

def persistence(train,h):
    return np.repeat(train[-1],h)

def moving_avg(window):
    def f(train,h):
        return np.repeat(np.mean(train[-window:]),h)
    f.__name__= f"ma{window}"
    return f

def train_mean(train,h):
    return np.repeat(np.mean(train),h)


def drift(train,h):
    n = len(train)
    slope = (train[-1] - train[0])/max(n-1,1)
    return train[-1]+slope* np.arange(1,h+1)

BASELINES = {
    "persistence":persistence,
    "ma6": moving_avg(6),
    "ma12": moving_avg(12),
    "train_mean": train_mean,
    "drift": drift
}

FALLBACKS = {"n": 0}

def _safe(fit_fn):
    def f(train, h):
        train = np.asarray(train, float)
        if np.std(train) < 1e-9 or len(train) < 10:
            return np.repeat(train[-1], h)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out = np.asarray(fit_fn(train, h), float)
            if np.all(np.isfinite(out)):
                return out
        except Exception as e:
            print("fit failed:", type(e).__name__, e)
        FALLBACKS["n"] += 1
        return np.repeat(train.mean(), h)
    return f

def arima(order):
    def fit(train, h):
        return ARIMA(train, order=order, trend="c" if order[1] == 0 else "n") \
                   .fit().forecast(h)
    f = _safe(fit); f.__name__ = f"arima{order}"; return f

def sarima(order, seasonal):
    def fit(train, h):
        return SARIMAX(train, order=order, seasonal_order=seasonal).fit(disp=False).forecast(h)
    f = _safe(fit); f.__name__ = f"sarima{order}x{seasonal}"; return f

def ets(trend=None, damped=False):
    def fit(train, h):
        return ExponentialSmoothing(train, trend=trend, damped_trend=damped
                                    if trend else False).fit().forecast(h)
    f = _safe(fit); f.__name__ = f"ets_{trend}_{'damped' if damped else 'plain'}"; return f

def ridge_direct(n_lags=12, alpha=1.0):
    """Direct multi-horizon: one ridge model per step ahead, on past lags only."""
    def fit(train, h):
        n = len(train)
        L = min(n_lags, max(n // 4, 3))
        preds = []
        for k in range(1, h + 1):
            rows = range(L, n - k + 1)
            if len(rows) < 8:
                return np.repeat(train.mean(), h)
            X = np.array([train[i - L:i] for i in rows])
            X = np.c_[X, X.mean(1)]
            y = np.array([train[i + k - 1] for i in rows])
            m = Ridge(alpha=alpha).fit(X, y)
            last = train[-L:]
            preds.append(m.predict(np.r_[last, last.mean()].reshape(1, -1))[0])
        return np.array(preds)
    f = _safe(fit); f.__name__ = f"ridge{n_lags}"; return f

STATISTICAL = {
    "ar1":        arima((1, 0, 0)),
    "ar2":        arima((2, 0, 0)),
    "arma11":     arima((1, 0, 1)),
    "arima111":   arima((1, 1, 1)),
    "ima11":      arima((0, 1, 1)),
    "ets_ses":    ets(),
    "ets_damped": ets("add", damped=True),
    "sarima_s12": sarima((1, 0, 0), (1, 0, 0, 12)),
    "ridge12":    ridge_direct(12),
}