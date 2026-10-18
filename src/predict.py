"""Prediksi return bar M15 berikutnya (pips) dari CSV, memakai models/ridge_params.json (hanya butuh numpy + pandas).

Contoh:  python src/predict.py --csv data/raw/EURUSD_M15.csv
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from features import build_features

ROOT = Path(__file__).resolve().parents[1]


def load_bars(path):
    raw = pd.read_csv(path, sep='\t')
    if raw.shape[1] == 6:        # header kurang 1 kolom -> waktu menjadi index
        raw.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Spread']
        raw.index = pd.to_datetime(raw.index)
    else:
        raw['Time'] = pd.to_datetime(raw['Time'])
        raw = raw.set_index('Time')[['Open', 'High', 'Low', 'Close', 'Volume', 'Spread']]
    return raw.sort_index()[lambda d: ~d.index.duplicated()]


def predict_next_return(recent_df, params_path=ROOT / 'models' / 'ridge_params.json'):
    """recent_df: >= 120 bar M15 terakhir. Return: prediksi return bar berikutnya (pips)."""
    P = json.loads(Path(params_path).read_text())
    x = build_features(recent_df).iloc[-1][P['features']].to_numpy(dtype=float)
    if np.isnan(x).any():
        raise ValueError('Data terlalu pendek atau ada NaN pada fitur: berikan >= 120 bar.')
    z = (x - np.array(P['mean'])) / np.array(P['scale'])
    return float(np.dot(P['coef'], z) + P['b'])


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default=str(ROOT / 'data' / 'raw' / 'EURUSD_M15.csv'))
    ap.add_argument('--bars', type=int, default=200, help='jumlah bar terakhir yang dipakai')
    a = ap.parse_args()
    df = load_bars(a.csv).tail(a.bars)
    p = predict_next_return(df)
    print(f'Bar terakhir : {df.index[-1]}  (Close {df.Close.iloc[-1]:.5f}, spread {df.Spread.iloc[-1] / 10:.1f} pip)')
    print(f'Prediksi     : {p:+.3f} pips ({"naik" if p > 0 else "turun"})')
