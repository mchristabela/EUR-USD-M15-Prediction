# EUR/USD M15: Prediksi Return Bar Berikutnya

Proyek machine learning untuk memprediksi **return 15 menit berikutnya** EUR/USD, lengkap dengan evaluasi walk-forward, backtest berbiaya spread,
dan web untuk mengecek prediksi.

> **Ringkasan hasil (walk-forward 5 fold, ±59.800 bar uji):** tidak ada model yang mengalahkan naive pada RMSE;
> akurasi arah terbaik 51,5% (classifier), Ridge 50,8%. Setelah spread, semua konfigurasi backtest rugi.
> Ini proyek metodologi dan eksplorasi, **bukan sinyal trading**.

## Struktur

```
eurusd-ml/
├── README.md
├── requirements.txt
├── data/raw/EURUSD_M15.csv              data asli (100.000 bar M15)
├── notebooks/
│   ├── 01_mlr_baseline_original.ipynb   versi awal (MLR pada level harga)
│   └── 02_return_prediction_v2.ipynb    versi diperbaiki: target return, 32 fitur, walk-forward, backtest
├── models/
│   ├── eurusd_m15_ridge.joblib          Pipeline Ridge (dibuat oleh notebook 02)
│   ├── ridge_params.json                Ridge 32 fitur (riwayat 96 bar): koefisien + scaler + tabel keyakinan
│   └── ridge_1bar_params.json           Ridge 14 fitur dari SATU bar (dipakai mode "Isi manual")
├── src/
│   ├── features.py                      build_features()
│   ├── predict.py                       prediksi dari CSV lewat terminal
│   ├── train_1bar.py                    latih + ekspor model 1 bar (models/ridge_1bar_params.json)
│   └── update_web.py                    suntik kedua file parameter ke web/index.html
└── web/
    ├── index.html                       cek prediksi: isi OHLC 1 bar, atau tempel/unggah CSV (mandiri, buka langsung di browser)
    ├── report.html                      laporan hasil model
    ├── core.js                          mesin fitur JS (juga tertanam di index.html)
    └── tests/                           uji JS vs Python
```

## Mulai cepat

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

jupyter notebook notebooks/02_return_prediction_v2.ipynb   # jalankan dari folder notebooks/ (Run All)
python src/predict.py                                      # prediksi dari bar terakhir di CSV
```

Di Google Colab: upload `EURUSD_M15.csv` ke direktori kerja dan jalankan notebook 02 apa adanya.

## Memperbarui web setelah melatih ulang

1. Jalankan ulang notebook 02 sampai selesai. Cell terakhir menulis `models/ridge_params.json`.
2. `python src/train_1bar.py`: melatih ulang model 1 bar (sekaligus memperbarui `web/tests/refs_1bar.json`).
3. `python src/update_web.py`: menimpa parameter kedua model di `web/index.html`.
4. `node web/tests/test_core.js`: memastikan prediksi JS masih sama dengan Python. Perbarui `web/tests/refs.json` bila model 32 fitur berubah.

`web/report.html` berisi angka dari satu kali run dan **tidak** ikut diperbarui otomatis.

## Format data

Tab-separated: `Time, Open, High, Low, Close, Volume, Spread`. Waktu dianggap UTC, spread dalam point (broker 5-digit, 1 point = 0,1 pip).
Halaman web juga menerima CSV dengan pemisah koma/titik koma dan ekspor bar MT5.

## Catatan

- **Mode "Isi manual"** memakai model 1 bar: cukup Open, High, Low, Close (waktu, spread, volume opsional; yang kosong diisi rata-rata data latih sehingga tidak memengaruhi hasil). Akurasi arah walk-forward 51,0%, setara model 96 bar (50,8%).
- **Mode CSV** memakai model 32 fitur dan butuh minimal ±100 bar M15 berurutan.
- File `.joblib` bergantung versi scikit-learn. Untuk deployment, pakai `ridge_params.json` (hanya butuh numpy).
- Classifier HistGB (akurasi arah 51,5%) hanya ada di notebook; tidak ada di web karena butuh server.
