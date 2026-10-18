// Uji: prediksi JS (web/core.js) harus sama dengan Python. Jalankan dari root proyek:  node web/tests/test_core.js
const fs = require('fs'), path = require('path');
const root = path.resolve(__dirname, '..', '..');
const { parseCSV, computeFeatures, predictAt } = require(path.join(root, 'web', 'core.js'));
const P = JSON.parse(fs.readFileSync(path.join(root, 'models', 'ridge_params.json')));
const refs = JSON.parse(fs.readFileSync(path.join(__dirname, 'refs.json')));   // [[waktu, prediksi Python], ...]
const { bars } = parseCSV(fs.readFileSync(path.join(root, 'data', 'raw', 'EURUSD_M15.csv'), 'utf8'));
const F = computeFeatures(bars), idx = new Map(bars.map((b, i) => [b.t, i]));
let worst = 0;
for (const [ts, ref] of refs) {
  const t = Date.UTC(+ts.slice(0, 4), +ts.slice(5, 7) - 1, +ts.slice(8, 10), +ts.slice(11, 13), +ts.slice(14, 16));
  const r = predictAt(F, idx.get(t), P), d = Math.abs(r.pred - ref);
  worst = Math.max(worst, d);
  console.log(`${ts}  python=${ref.toFixed(6)}  js=${r.pred.toFixed(6)}  selisih=${d.toExponential(1)}`);
}

// Model 1 bar: fitur dihitung hanya dari SATU bar (tanpa riwayat) lalu dibandingkan dengan Python
const P1 = JSON.parse(fs.readFileSync(path.join(root, 'models', 'ridge_1bar_params.json')));
const refs1 = JSON.parse(fs.readFileSync(path.join(__dirname, 'refs_1bar.json')));
for (const [ts, ref] of refs1) {
  const t = Date.UTC(+ts.slice(0, 4), +ts.slice(5, 7) - 1, +ts.slice(8, 10), +ts.slice(11, 13), +ts.slice(14, 16));
  const r = predictAt(computeFeatures([bars[idx.get(t)]]), 0, P1), d = Math.abs(r.pred - ref);
  worst = Math.max(worst, d);
  console.log(`${ts}  [1 bar] python=${ref.toFixed(6)}  js=${r.pred.toFixed(6)}  selisih=${d.toExponential(1)}`);
}
if (worst > 1e-9) { console.error('GAGAL: selisih terlalu besar'); process.exit(1); }
console.log('OK: JS cocok dengan Python.');
