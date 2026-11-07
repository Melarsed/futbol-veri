"""Veri indirme ve tek formata çevirme.

Ana kaynak: football-data.co.uk (sonuçlar, devre skorları, bahis oranları, fikstür)
Yedek kaynak: xgabora/Club-Football-Match-Data (GitHub, periyodik güncellenir)
"""
from __future__ import annotations

import io
import json
import os
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import requests

BASE = "https://www.football-data.co.uk"
DATA_DIR = Path(__file__).resolve().parent.parent / "veri"
DATA_DIR.mkdir(exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0 (FutbolTahmin kisisel kullanim)"}

# kod: (ülke grubu, görünen ad)
MAIN_LEAGUES = {
    "E0": ("ENG", "İngiltere Premier Lig"),
    "E1": ("ENG", "İngiltere Championship"),
    "E2": ("ENG", "İngiltere League One"),
    "E3": ("ENG", "İngiltere League Two"),
    "EC": ("ENG", "İngiltere National League"),
    "SC0": ("SCO", "İskoçya Premiership"),
    "SC1": ("SCO", "İskoçya Championship"),
    "SC2": ("SCO", "İskoçya League One"),
    "SC3": ("SCO", "İskoçya League Two"),
    "D1": ("GER", "Almanya Bundesliga"),
    "D2": ("GER", "Almanya 2. Bundesliga"),
    "I1": ("ITA", "İtalya Serie A"),
    "I2": ("ITA", "İtalya Serie B"),
    "SP1": ("ESP", "İspanya La Liga"),
    "SP2": ("ESP", "İspanya Segunda"),
    "F1": ("FRA", "Fransa Ligue 1"),
    "F2": ("FRA", "Fransa Ligue 2"),
    "N1": ("NED", "Hollanda Eredivisie"),
    "B1": ("BEL", "Belçika Pro League"),
    "P1": ("POR", "Portekiz Liga"),
    "T1": ("TUR", "Türkiye Süper Lig"),
    "G1": ("GRE", "Yunanistan Süper Lig"),
}
# football-data "ek ligler" (sadece maç sonucu + oran, devre skoru yok)
EXTRA_LEAGUES = {
    "AUT": "Avusturya Bundesliga",
    "SWZ": "İsviçre Super League",
    "DNK": "Danimarka Superliga",
    "POL": "Polonya Ekstraklasa",
    "ROU": "Romanya Liga 1",
    "NOR": "Norveç Eliteserien",
    "SWE": "İsveç Allsvenskan",
    "FIN": "Finlandiya Veikkausliiga",
    "IRL": "İrlanda Premier Division",
    "RUS": "Rusya Premier Lig",
}
# xgabora veri setindeki ek lig kodları -> football-data kodları
XG_EXTRA = {"AUT": "AUT", "SUI": "SWZ", "DEN": "DNK", "POL": "POL", "ROM": "ROU",
            "NOR": "NOR", "SWE": "SWE", "FIN": "FIN", "IRL": "IRL", "RUS": "RUS"}


def league_name(div: str) -> str:
    if div in MAIN_LEAGUES:
        return MAIN_LEAGUES[div][1]
    return EXTRA_LEAGUES.get(div, div)


def league_group(div: str) -> str:
    return MAIN_LEAGUES[div][0] if div in MAIN_LEAGUES else div


COLS = ["Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "HTHG", "HTAG",
        "OddH", "OddD", "OddA", "OddO25", "OddU25"]


def season_codes(today: date | None = None, n: int = 3):
    today = today or date.today()
    start = today.year if today.month >= 7 else today.year - 1
    return [f"{(y) % 100:02d}{(y + 1) % 100:02d}" for y in range(start - n + 1, start + 1)]


AYAR = DATA_DIR / "ayar.json"


DEFAULT_MIRROR = "Melarsed"
# GitHub Actions içinde veri dosyaları doğrudan repo klasöründen okunur
LOCAL_DIR = os.environ.get("VERI_KLASORU")


def get_mirror() -> str:
    """GitHub'daki veri kopyasının kullanıcı adı (Türkiye'den erişim için)."""
    try:
        return json.loads(AYAR.read_text()).get("github_kullanici", "").strip() or DEFAULT_MIRROR
    except Exception:
        return DEFAULT_MIRROR


def mirror_url(path: str) -> str:
    return f"https://raw.githubusercontent.com/{get_mirror()}/futbol-veri/main/{path}"


def set_mirror(user: str):
    AYAR.write_text(json.dumps({"github_kullanici": user.strip()}))


def _bases():
    u = get_mirror()
    b = [f"https://raw.githubusercontent.com/{u}/futbol-veri/main"] if u else []
    return b + [BASE]


def _valid(b: bytes) -> bool:
    # Engel sayfaları (HTML) CSV sanılmasın
    head = b[:300].decode("latin-1", "ignore")
    return ("Div" in head or "Country" in head) and "<html" not in head.lower()


def _get(path: str, timeout=20) -> bytes | None:
    urls = [path] if path.startswith("http") else [f"{b}/{path}" for b in _bases()]
    for url in urls:
        try:
            r = requests.get(url, headers=UA, timeout=timeout)
            if r.status_code == 200 and _valid(r.content):
                return r.content
        except requests.RequestException:
            continue
    return None


def _read_csv_bytes(b: bytes) -> pd.DataFrame:
    for enc in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(b), encoding=enc, on_bad_lines="skip", low_memory=False)
        except Exception:
            continue
    return pd.DataFrame()


def _first(df, names):
    for n in names:
        if n in df.columns:
            return pd.to_numeric(df[n], errors="coerce")
    return pd.Series(np.nan, index=df.index)


def _parse_date(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    d = pd.to_datetime(s, format="%d/%m/%Y", errors="coerce")
    d2 = pd.to_datetime(s, format="%d/%m/%y", errors="coerce")
    return d.fillna(d2)


def normalize_main(raw: pd.DataFrame) -> pd.DataFrame:
    """football-data ana lig / fixtures.csv formatı."""
    if raw.empty or "HomeTeam" not in raw.columns:
        return pd.DataFrame(columns=COLS)
    raw = raw.dropna(subset=["HomeTeam", "AwayTeam"])
    out = pd.DataFrame({
        "Div": raw["Div"].astype(str).str.strip(),
        "Date": _parse_date(raw["Date"]),
        "HomeTeam": raw["HomeTeam"].astype(str).str.strip(),
        "AwayTeam": raw["AwayTeam"].astype(str).str.strip(),
        "FTHG": _first(raw, ["FTHG", "HG"]),
        "FTAG": _first(raw, ["FTAG", "AG"]),
        "HTHG": _first(raw, ["HTHG"]),
        "HTAG": _first(raw, ["HTAG"]),
        "OddH": _first(raw, ["AvgH", "BbAvH", "B365H", "PSH"]),
        "OddD": _first(raw, ["AvgD", "BbAvD", "B365D", "PSD"]),
        "OddA": _first(raw, ["AvgA", "BbAvA", "B365A", "PSA"]),
        "OddO25": _first(raw, ["Avg>2.5", "BbAv>2.5", "B365>2.5", "P>2.5"]),
        "OddU25": _first(raw, ["Avg<2.5", "BbAv<2.5", "B365<2.5", "P<2.5"]),
    })
    if "Time" in raw.columns:
        out["Time"] = raw["Time"].astype(str).str.slice(0, 5)
    return out.dropna(subset=["Date"])


def normalize_extra(raw: pd.DataFrame, div: str) -> pd.DataFrame:
    """football-data 'new' ek lig formatı (Home/Away/HG/AG, kapanış oranları)."""
    if raw.empty or "Home" not in raw.columns:
        return pd.DataFrame(columns=COLS)
    raw = raw.dropna(subset=["Home", "Away"])
    out = pd.DataFrame({
        "Div": div,
        "Date": _parse_date(raw["Date"]),
        "HomeTeam": raw["Home"].astype(str).str.strip(),
        "AwayTeam": raw["Away"].astype(str).str.strip(),
        "FTHG": _first(raw, ["HG", "FTHG"]),
        "FTAG": _first(raw, ["AG", "FTAG"]),
        "HTHG": np.nan,
        "HTAG": np.nan,
        "OddH": _first(raw, ["AvgCH", "AvgH", "PSCH", "B365CH"]),
        "OddD": _first(raw, ["AvgCD", "AvgD", "PSCD", "B365CD"]),
        "OddA": _first(raw, ["AvgCA", "AvgA", "PSCA", "B365CA"]),
        "OddO25": _first(raw, ["AvgC>2.5", "Avg>2.5"]),
        "OddU25": _first(raw, ["AvgC<2.5", "Avg<2.5"]),
    })
    if "Time" in raw.columns:
        out["Time"] = raw["Time"].astype(str).str.slice(0, 5)
    return out.dropna(subset=["Date"])


def _cached(name: str, path: str, max_age_h: float) -> bytes | None:
    if LOCAL_DIR and not path.startswith("http"):
        f = Path(LOCAL_DIR) / path
        return f.read_bytes() if f.exists() else None
    f = DATA_DIR / name
    if f.exists() and (time.time() - f.stat().st_mtime) < max_age_h * 3600:
        return f.read_bytes()
    b = _get(path)
    if b:
        f.write_bytes(b)
        return b
    return f.read_bytes() if f.exists() else None   # internet yoksa eski kopya


def load_xgabora() -> pd.DataFrame:
    url = ("https://raw.githubusercontent.com/xgabora/"
           "Club-Football-Match-Data-2000-2025/main/data/Matches.csv")
    b = _cached("xgabora_Matches.csv", url, 24 * 7)
    if not b:
        return pd.DataFrame(columns=COLS)
    return normalize_xgabora(pd.read_csv(io.BytesIO(b), low_memory=False))


def normalize_xgabora(raw: pd.DataFrame) -> pd.DataFrame:
    div = raw["Division"].astype(str).replace(XG_EXTRA)
    out = pd.DataFrame({
        "Div": div,
        "Date": pd.to_datetime(raw["MatchDate"], errors="coerce"),
        "HomeTeam": raw["HomeTeam"].astype(str).str.strip(),
        "AwayTeam": raw["AwayTeam"].astype(str).str.strip(),
        "FTHG": pd.to_numeric(raw["FTHome"], errors="coerce"),
        "FTAG": pd.to_numeric(raw["FTAway"], errors="coerce"),
        "HTHG": pd.to_numeric(raw["HTHome"], errors="coerce"),
        "HTAG": pd.to_numeric(raw["HTAway"], errors="coerce"),
        "OddH": pd.to_numeric(raw["OddHome"], errors="coerce"),
        "OddD": pd.to_numeric(raw["OddDraw"], errors="coerce"),
        "OddA": pd.to_numeric(raw["OddAway"], errors="coerce"),
        "OddO25": pd.to_numeric(raw["Over25"], errors="coerce"),
        "OddU25": pd.to_numeric(raw["Under25"], errors="coerce"),
    })
    keep = set(MAIN_LEAGUES) | set(EXTRA_LEAGUES)
    return out[out["Div"].isin(keep)].dropna(subset=["Date"])


def load_results(leagues=None, progress=None) -> tuple[pd.DataFrame, list[str]]:
    """Son 3 sezonun sonuçları. Döner: (maçlar, uyarılar)."""
    leagues = leagues or list(MAIN_LEAGUES) + list(EXTRA_LEAGUES)
    frames, warn = [], []
    seasons = season_codes()
    current = seasons[-1]
    jobs = [(lg, s) for lg in leagues if lg in MAIN_LEAGUES for s in seasons]
    jobs += [(lg, None) for lg in leagues if lg in EXTRA_LEAGUES]
    for k, (lg, s) in enumerate(jobs):
        if progress:
            progress((k + 1) / len(jobs), f"{league_name(lg)} indiriliyor…")
        if s:
            age = 6 if s == current else 24 * 30
            b = _cached(f"{lg}_{s}.csv", f"mmz4281/{s}/{lg}.csv", age)
            if b:
                frames.append(normalize_main(_read_csv_bytes(b)))
            elif s == current:
                warn.append(f"{league_name(lg)}: güncel sezon indirilemedi")
        else:
            b = _cached(f"{lg}.csv", f"new/{lg}.csv", 6)
            if b:
                frames.append(normalize_extra(_read_csv_bytes(b), lg))
            else:
                warn.append(f"{league_name(lg)}: indirilemedi")
    df = pd.concat([f for f in frames if not f.empty], ignore_index=True) if frames else pd.DataFrame(columns=COLS)
    if df.empty:
        warn.append("Güncel veriye ulaşılamadı, GitHub yedek verisi kullanılıyor (güncel olmayabilir). "
                    "Soldaki 'Veri kaynağı' bölümünden GitHub kullanıcı adını gir.")
        df = load_xgabora()
        df = df[df["Div"].isin(leagues)]
    df = df.drop_duplicates(subset=["Div", "Date", "HomeTeam", "AwayTeam"])
    return df.sort_values("Date").reset_index(drop=True), warn


def load_fixtures() -> tuple[pd.DataFrame, list[str]]:
    """Önümüzdeki maçlar (oranlarla birlikte)."""
    warn, frames = [], []
    b = _cached("fixtures.csv", "fixtures.csv", 2)
    if b:
        frames.append(normalize_main(_read_csv_bytes(b)))
    else:
        warn.append("Ana lig fikstürü indirilemedi.")
    b = _cached("new_league_fixtures.csv", "new_league_fixtures.csv", 2)
    if b:
        raw = _read_csv_bytes(b)
        if "Country" in raw.columns:
            cmap = {"Austria": "AUT", "Switzerland": "SWZ", "Denmark": "DNK", "Poland": "POL",
                    "Romania": "ROU", "Norway": "NOR", "Sweden": "SWE", "Finland": "FIN",
                    "Ireland": "IRL", "Russia": "RUS"}
            raw = raw[raw["Country"].isin(cmap)]
            for c, code in cmap.items():
                part = raw[raw["Country"] == c]
                if len(part):
                    frames.append(normalize_extra(part, code))
    fx = pd.concat([f for f in frames if not f.empty], ignore_index=True) if frames else pd.DataFrame(columns=COLS)
    return fx, warn
