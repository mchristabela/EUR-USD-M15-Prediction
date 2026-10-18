"""Fitur untuk model EUR/USD M15. Salinan dari notebook 02 (hanya memakai informasi sampai bar t)."""
import numpy as np
import pandas as pd

PIP = 1e-4


def build_features(d):
    """d: DataFrame OHLCV+Spread ber-index waktu. Return: DataFrame fitur (hanya info <= t)."""
    c = d['Close']
    f = pd.DataFrame(index=d.index)
    ret1 = c.diff() / PIP

    # 1) Momentum / return multi-horizon (15m, 30m, 45m, 1h15, 2h30, 5h, 12h, 24h)
    for k in (1, 2, 3, 5, 10, 20, 48, 96):
        f[f'ret{k}'] = c.diff(k) / PIP

    # 2) Struktur candle
    rng = (d.High - d.Low)
    f['body']       = (c - d.Open) / PIP
    f['range']      = rng / PIP
    f['upper_wick'] = (d.High - d[['Open', 'Close']].max(axis=1)) / PIP
    f['lower_wick'] = (d[['Open', 'Close']].min(axis=1) - d.Low) / PIP
    f['close_pos']  = ((c - d.Low) / rng.replace(0, np.nan)).fillna(0.5)

    # 3) Volatilitas
    f['vol20']     = ret1.rolling(20).std()
    f['vol96']     = ret1.rolling(96).std()
    f['vol_ratio'] = f['vol20'] / f['vol96']
    tr_ = pd.concat([rng, (d.High - c.shift()).abs(), (d.Low - c.shift()).abs()], axis=1).max(axis=1) / PIP
    f['atr14']     = tr_.rolling(14).mean()

    # 4) Mean-reversion / tren (dinormalisasi volatilitas)
    f['dev_sma20_z'] = (c - c.rolling(20).mean())  / PIP / f['vol96']
    f['dev_sma96_z'] = (c - c.rolling(96).mean())  / PIP / f['vol96']
    f['sma20_slope'] = (c.rolling(20).mean() - c.rolling(20).mean().shift(5)) / PIP
    delta = c.diff()
    up = delta.clip(lower=0).rolling(14).mean(); dn = (-delta.clip(upper=0)).rolling(14).mean()
    f['rsi14'] = 100 - 100 / (1 + up / dn.replace(0, np.nan))

    # 5) Volume & spread (likuiditas)
    f['logvol']     = np.log1p(d['Volume'])
    f['volume_z96'] = (d['Volume'] - d['Volume'].rolling(96).mean()) / d['Volume'].rolling(96).std()
    f['spread_pips'] = d['Spread'] / 10.0
    f['spread_z96']  = (d['Spread'] - d['Spread'].rolling(96).mean()) / d['Spread'].rolling(96).std()

    # 6) Waktu & sesi
    h = d.index.hour + d.index.minute / 60
    f['hour_sin'] = np.sin(2*np.pi*h/24); f['hour_cos'] = np.cos(2*np.pi*h/24)
    f['dow']      = d.index.dayofweek
    f['sess_asia']    = (d.index.hour < 7).astype(int)
    f['sess_london']  = ((d.index.hour >= 7)  & (d.index.hour < 16)).astype(int)
    f['sess_ny']      = ((d.index.hour >= 13) & (d.index.hour < 22)).astype(int)
    f['sess_overlap'] = ((d.index.hour >= 13) & (d.index.hour < 16)).astype(int)

    return f.replace([np.inf, -np.inf], np.nan)
