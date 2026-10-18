"""Suntikkan models/ridge_params.json (const P) dan models/ridge_1bar_params.json (const P1) ke web/index.html.

Contoh:  python src/update_web.py
"""
import argparse, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument('--html', default=str(ROOT / 'web' / 'index.html'))
a = ap.parse_args()

NEED = {'features', 'mean', 'scale', 'coef', 'b', 'q', 'acc'}
html = Path(a.html).read_text(); orig = html
for var, fname in [('P', 'ridge_params.json'), ('P1', 'ridge_1bar_params.json')]:
    src = ROOT / 'models' / fname
    if not src.exists():
        print(f'lewati {var}: {fname} tidak ada'); continue
    params = json.loads(src.read_text())
    assert NEED <= params.keys(), f'{fname} kurang kunci: {NEED - params.keys()}'
    html, n = re.subn(r'const ' + var + r'=\{[^\n]*\};', lambda m: f'const {var}=' + json.dumps(params, separators=(',', ':')) + ';', html, count=1)
    assert n == 1, f'Baris `const {var}={{...}};` tidak ditemukan di HTML'
Path(a.html).write_text(html)
print('web/index.html diperbarui:', 'ada perubahan' if html != orig else 'tidak ada perubahan (parameter sama)')
