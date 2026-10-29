from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import numpy as np
import pandas as pd


def mae(a,f):
    return float(mean_absolute_error(a,f))

def rmse(a,f):
    return float(root_mean_squared_error(a,f))


def mase(a, f, train):
    train = np.asarray(train, dtype=float)
    if len(train) < 2:
        return np.nan
    scale = np.mean(np.abs(np.diff(train)))
    if not np.isfinite(scale) or scale < 1e-9:
        return np.nan
    return mae(a, f) / scale


def rolling_origin(y,model,h,min_train=None,step=None):
    n = len(y)
    min_train = min_train or max(n//3,5 )
    step = step or max((n -h-min_train)//6,1)
    origins = list(range(min_train,n-h+1,step)) or [max(n - h, 1)]
    rows =[]

    for o in origins:
        f = np.asarray(model(y[:o], min(h, n - o) if o + h > n else h))
        a = y[o:o + len(f)]
        if len(a) == 0:
            continue

        rows.append({
            "origin":o,
            "mae": mae(a,f),
            "rmse": rmse(a,f),
            "mase": mase(a,f,y[:o])
        })
    return pd.DataFrame(rows)

def compare(y,models,h):
    out = {
        name: rolling_origin(y,m,h)[["mae","rmse","mase"]].mean()
        for name,m in models.items()
    }

    return pd.DataFrame(out).T.sort_values("mae", kind="stable")