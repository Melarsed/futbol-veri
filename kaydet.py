"""GitHub Actions içinde çalışır: yaklaşan maçları tahmin eder ve
tahmin_kaydi.csv dosyasına ekler (her maçın İLK tahmini saklanır, sonradan değişmez)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("VERI_KLASORU", str(ROOT))

import pandas as pd  # noqa: E402

from futbol.data import load_fixtures, load_results  # noqa: E402
from futbol.predict import BETS, fit_all, predict_frame  # noqa: E402

KAYIT = ROOT / "tahmin_kaydi.csv"
KEYS = ["Div", "Date", "HomeTeam", "AwayTeam"]

results, w1 = load_results()
fixtures, w2 = load_fixtures()
for w in w1 + w2:
    print("uyarı:", w)
today = pd.Timestamp.now(tz="Europe/Istanbul").normalize().tz_localize(None)
fx = fixtures[fixtures["Date"] >= today].copy()
# Başlamış maçları kaydetme (fikstür saatleri İngiltere saatidir)
if "Time" in fx:
    t = pd.to_datetime(fx["Date"].dt.strftime("%Y-%m-%d") + " " + fx["Time"].fillna("23:59"), errors="coerce")
    t = t.dt.tz_localize("Europe/London", ambiguous="NaT", nonexistent="NaT")
    fx = fx[t.isna() | (t > pd.Timestamp.now(tz="Europe/London"))]
models = fit_all(results, today + pd.Timedelta(days=1))
pred = predict_frame(models, fx, 0.0)
if pred.empty:
    print("Yaklaşan maç yok")
    raise SystemExit(0)

cols = KEYS + [k for k in BETS if k in pred.columns] + ["Skor"]
new = pred[cols].copy()
new["kayit_zamani"] = pd.Timestamp.now(tz="Europe/Istanbul").strftime("%Y-%m-%d %H:%M")
for c in BETS:
    if c in new:
        new[c] = new[c].round(4)
if KAYIT.exists():
    old = pd.read_csv(KAYIT, parse_dates=["Date"])
    new = pd.concat([old, new]).drop_duplicates(subset=KEYS, keep="first")
new.sort_values("Date").to_csv(KAYIT, index=False, date_format="%Y-%m-%d")
print(f"{len(pred)} yaklaşan maç, kayıtta toplam {len(new)} maç")
