"""football-data.co.uk verilerini bu repoya kopyalar (GitHub Actions ile çalışır)."""
import time
from datetime import date
from pathlib import Path

import requests

BASE = "https://www.football-data.co.uk"
MAIN = ["E0", "E1", "E2", "E3", "EC", "SC0", "SC1", "SC2", "SC3", "D1", "D2", "I1", "I2",
        "SP1", "SP2", "F1", "F2", "N1", "B1", "P1", "T1", "G1"]
EXTRA = ["AUT", "SWZ", "DNK", "POL", "ROU", "NOR", "SWE", "FIN", "IRL", "RUS"]
UA = {"User-Agent": "Mozilla/5.0"}


def seasons(n=3):
    t = date.today()
    s = t.year if t.month >= 7 else t.year - 1
    return [f"{y % 100:02d}{(y + 1) % 100:02d}" for y in range(s - n + 1, s + 1)]


def fetch(path, force=True):
    out = Path(path)
    if out.exists() and not force:
        return
    for _ in range(3):
        try:
            r = requests.get(f"{BASE}/{path}", headers=UA, timeout=60)
            head = r.content[:200].decode("latin-1", "ignore")
            if r.status_code == 200 and ("Div" in head or "Country" in head):
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(r.content)
                print("ok  ", path)
                return
            print("yok ", path, r.status_code)
            return
        except requests.RequestException as e:
            print("hata", path, e)
            time.sleep(5)


ss = seasons()
for s in ss:
    for lg in MAIN:
        fetch(f"mmz4281/{s}/{lg}.csv", force=(s == ss[-1]))
for lg in EXTRA:
    fetch(f"new/{lg}.csv")
fetch("fixtures.csv")
fetch("new_league_fixtures.csv")
