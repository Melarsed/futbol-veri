"""Geçmiş sezon testi: her hafta sadece o güne kadarki veriyle model kurulur,
o haftanın maçları tahmin edilir. Böylece gerçekçi isabet ölçülür."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import league_group
from .predict import LeagueModels, predict_match, BETS


def run(results: pd.DataFrame, start, end, step_days=7, xi=0.0019, weights=(1.0, 0.5, 0.0),
        groups=None, log=print):
    res = results.copy()
    res["Grp"] = res["Div"].map(league_group)
    groups = groups or sorted(res["Grp"].unique())
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    rows = []
    for g in groups:
        gd = res[res["Grp"] == g]
        t = start
        while t < end:
            t2 = t + pd.Timedelta(days=step_days)
            test = gd[(gd["Date"] >= t) & (gd["Date"] < t2) & gd["FTHG"].notna()]
            if len(test):
                try:
                    lm = LeagueModels(gd, t, xi)
                except ValueError:
                    t = t2
                    continue
                for _, r in test.iterrows():
                    odds = {k: r[k] for k in ["OddH", "OddD", "OddA", "OddO25", "OddU25"]}
                    base = r[["Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "HTHG", "HTAG",
                              "OddH", "OddD", "OddA", "OddO25", "OddU25"]].to_dict()
                    for w in weights:
                        p = predict_match(lm, r["HomeTeam"], r["AwayTeam"], odds, w)
                        if w < 1 and not p["piyasa"]:
                            continue
                        rows.append({**base, "w": w, **p})
            t = t2
        log(f"{g} tamam")
    return pd.DataFrame(rows)


def evaluate(bt: pd.DataFrame) -> pd.DataFrame:
    """Her bahis türü ve güven aralığı için tahmin sayısı ve isabet oranı."""
    out = []
    bins = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 1.01]
    for key, (name, check) in BETS.items():
        if key not in bt.columns:
            continue
        d = bt.dropna(subset=[key])
        if key.startswith("IY"):
            d = d.dropna(subset=["HTHG", "HTAG"])
        hit = check(d).astype(float)
        p = d[key]
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (p >= lo) & (p < hi)
            if m.sum() == 0:
                continue
            out.append({"w": None, "bahis": key, "ad": name, "alt": lo, "ust": hi,
                        "n": int(m.sum()), "ort_olasilik": float(p[m].mean()),
                        "isabet": float(hit[m].mean())})
    return pd.DataFrame(out)


def scores(bt: pd.DataFrame) -> dict:
    """1X2 log-loss ve Brier skoru (düşük = iyi)."""
    y = np.select([bt.FTHG > bt.FTAG, bt.FTHG == bt.FTAG], [0, 1], 2)
    P = bt[["MS1", "MSX", "MS2"]].to_numpy()
    ll = -np.log(np.clip(P[np.arange(len(y)), y], 1e-9, 1)).mean()
    Y = np.eye(3)[y]
    brier = ((P - Y) ** 2).sum(1).mean()
    o = (bt.FTHG + bt.FTAG >= 3).astype(float)
    ll_ou = -(o * np.log(bt.Ust25) + (1 - o) * np.log(1 - bt.Ust25)).mean()
    btts = ((bt.FTHG > 0) & (bt.FTAG > 0)).astype(float)
    ll_kg = -(btts * np.log(bt.KG_Var) + (1 - btts) * np.log(1 - bt.KG_Var)).mean()
    acc = (P.argmax(1) == y).mean()
    return {"n": len(bt), "logloss_1x2": ll, "brier_1x2": brier, "isabet_1x2": acc,
            "logloss_25": ll_ou, "logloss_kg": ll_kg}
