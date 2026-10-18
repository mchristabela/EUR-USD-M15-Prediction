"""Latih model Ridge "1 bar" (hanya bentuk candle, jam/sesi, spread, volume dari SATU bar) dan ekspor ke models/ridge_1bar_params.json.

Walk-forward 5 fold dipakai untuk tabel keyakinan (kuantil |prediksi| + akurasi arah per desil); model final dilatih dengan seluruh data.
Contoh:  python src/train_1bar.py
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from features import build_features, PIP
from predict import load_bars

ROOT = Path(__file__).resolve().parents[1]
FEATS = ['body', 'range', 'upper_wick', 'lower_wick', 'close_pos',
         'hour_sin', 'hour_cos', 'dow', 'sess_asia', 'sess_london', 'sess_ny', 'sess_overlap',
         'spread_pips', 'logvol']
N_FOLDS, START = 5, 0.40

df = load_bars(ROOT / 'data' / 'raw' / 'EURUSD_M15.csv')
F = build_features(df)
nxt = (df.index.to_series().shift(-1) - df.index.to_series()) == pd.Timedelta('15min')
y = ((df['Close'].shift(-1) - df['Close']) / PIP).where(nxt)
data = F.assign(y=y).dropna()                      # sampel identik dengan notebook 02
mk = lambda: make_pipeline(StandardScaler(), Ridge(alpha=100.0))

n = len(data); step = int(n * (1 - START) / N_FOLDS); P, Y = [], []
for k in range(N_FOLDS):
    a = int(n * START) + k * step; b = n if k == N_FOLDS - 1 else a + step
    m = mk().fit(data.iloc[:a][FEATS], data.iloc[:a]['y'])
    P.append(m.predict(data.iloc[a:b][FEATS])); Y.append(data.iloc[a:b]['y'].values)
p, yv = np.concatenate(P), np.concatenate(Y); nz = yv != 0
hits = int((np.sign(p[nz]) == np.sign(yv[nz])).sum())
print(f'OOS: R2 vs naive {1 - np.sum((yv - p) ** 2) / np.sum(yv ** 2):+.5f} | dir acc {hits / nz.sum():.4f} | binom p {stats.binomtest(hits, int(nz.sum()), .5).pvalue:.4f}')

q = [round(float(np.quantile(np.abs(p), k / 100)), 5) for k in range(101)]
rk = pd.Series(np.abs(p)).rank(pct=True).values
acc = [round(float((np.sign(p[(rk > d / 10) & (rk <= (d + 1) / 10) & nz]) == np.sign(yv[(rk > d / 10) & (rk <= (d + 1) / 10) & nz])).mean()), 4) for d in range(10)]

final = mk().fit(data[FEATS], data['y']); sc, rg = final.steps[0][1], final.steps[1][1]
params = {'features': FEATS, 'mean': sc.mean_.tolist(), 'scale': sc.scale_.tolist(), 'coef': rg.coef_.tolist(),
          'b': float(rg.intercept_), 'q': q, 'acc': acc}
(ROOT / 'models' / 'ridge_1bar_params.json').write_text(json.dumps(params, separators=(',', ':')))
print('Tersimpan models/ridge_1bar_params.json | acc desil:', acc)
print('mean spread (pip):', round(sc.mean_[FEATS.index('spread_pips')], 3))

# referensi untuk uji JS: prediksi 1 bar pada 4 waktu yang sama dengan refs.json
refs = []
for ts, _ in json.loads((ROOT / 'web' / 'tests' / 'refs.json').read_text()):
    x = F.loc[pd.Timestamp(ts), FEATS].to_numpy(float)
    refs.append([ts, float(np.dot(params['coef'], (x - np.array(params['mean'])) / np.array(params['scale'])) + params['b'])])
(ROOT / 'web' / 'tests' / 'refs_1bar.json').write_text(json.dumps(refs))
