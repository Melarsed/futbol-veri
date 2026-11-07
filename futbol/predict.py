"""Modelleri eğitip maçlar için tüm bahis olasılıklarını üretir."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import league_group
from .model import DixonColes, implied_probs, market_rates, markets

# Model ile piyasa (bahis oranları) karışımı: 0 = sadece piyasa, 1 = sadece model.
DEFAULT_MODEL_WEIGHT = 0.0


class LeagueModels:
    """Bir ülke grubu için tam maç + ilk yarı modelleri."""

    def __init__(self, df: pd.DataFrame, ref_date, xi=0.0019):
        self.ft = DixonColes(xi=xi).fit(df, ref_date, "FTHG", "FTAG")
        ht = df.dropna(subset=["HTHG", "HTAG"])
        self.ht = None
        if len(ht[ht["Date"] < pd.Timestamp(ref_date)]) >= 150:
            try:
                self.ht = DixonColes(xi=xi).fit(ht, ref_date, "HTHG", "HTAG")
                h = ht[(ht["Date"] < pd.Timestamp(ref_date))].tail(2000)
                self.ht_share = float((h.HTHG + h.HTAG).sum() / max((h.FTHG + h.FTAG).sum(), 1))
            except ValueError:
                self.ht = None


def fit_all(results: pd.DataFrame, ref_date, groups=None, xi=0.0019, progress=None):
    res = results.copy()
    res["Grp"] = res["Div"].map(league_group)
    groups = groups or sorted(res["Grp"].unique())
    models = {}
    for k, g in enumerate(groups):
        if progress:
            progress((k + 1) / len(groups), f"Model eğitiliyor: {g}")
        try:
            models[g] = LeagueModels(res[res["Grp"] == g], ref_date, xi)
        except ValueError:
            continue
    return models


def predict_match(lm: LeagueModels, home, away, odds=None, w_model=DEFAULT_MODEL_WEIGHT):
    ft = lm.ft
    lam, mu, known = ft.rates(home, away)
    m_lam, m_mu = lam, mu
    used_market = False
    if odds is not None and w_model < 1:
        p = implied_probs(odds.get("OddH"), odds.get("OddD"), odds.get("OddA"))
        if p is not None:
            pou = implied_probs(odds.get("OddO25"), odds.get("OddU25"))
            k_lam, k_mu = market_rates(p[0], p[1], p[2], None if pou is None else pou[0],
                                       ft.rho, guess=(lam, mu))
            lam = float(np.exp(w_model * np.log(m_lam) + (1 - w_model) * np.log(k_lam)))
            mu = float(np.exp(w_model * np.log(m_mu) + (1 - w_model) * np.log(k_mu)))
            used_market = True
    ht_args = {}
    if lm.ht is not None:
        hl, hm, _ = lm.ht.rates(home, away)
        # İlk yarı gol payı: takıma özel pay ile lig ortalamasının karışımı (aşırı uçları önler)
        sh = np.clip(0.5 * hl / m_lam + 0.5 * lm.ht_share, 0.33, 0.55)
        sa = np.clip(0.5 * hm / m_mu + 0.5 * lm.ht_share, 0.33, 0.55)
        ht_args = dict(ht_lam=lam * sh, ht_mu=mu * sa, ht_rho=lm.ht.rho)
    out = markets(lam, mu, ft.rho, **ht_args)
    out["bilinen"] = known
    out["piyasa"] = used_market
    return out


def predict_frame(models, fixtures: pd.DataFrame, w_model=DEFAULT_MODEL_WEIGHT) -> pd.DataFrame:
    rows = []
    for _, r in fixtures.iterrows():
        lm = models.get(league_group(r["Div"]))
        if lm is None:
            continue
        odds = {k: r.get(k) for k in ["OddH", "OddD", "OddA", "OddO25", "OddU25"]}
        p = predict_match(lm, r["HomeTeam"], r["AwayTeam"], odds, w_model)
        rows.append({**r.to_dict(), **p})
    return pd.DataFrame(rows)


# Bahis türleri: (gösterilen ad, olasılık sütunu, sonuç kontrol fonksiyonu)
def _tot(r):
    return r["FTHG"] + r["FTAG"]


BETS = {
    "MS1": ("MS 1", lambda r: r["FTHG"] > r["FTAG"]),
    "MSX": ("MS X", lambda r: r["FTHG"] == r["FTAG"]),
    "MS2": ("MS 2", lambda r: r["FTHG"] < r["FTAG"]),
    "KG_Var": ("KG Var", lambda r: (r["FTHG"] > 0) & (r["FTAG"] > 0)),
    "KG_Yok": ("KG Yok", lambda r: (r["FTHG"] == 0) | (r["FTAG"] == 0)),
    "Ust25": ("2.5 Üst", lambda r: _tot(r) >= 3),
    "Alt25": ("2.5 Alt", lambda r: _tot(r) <= 2),
    "Ust15": ("1.5 Üst", lambda r: _tot(r) >= 2),
    "Alt15": ("1.5 Alt", lambda r: _tot(r) <= 1),
    "Gol_0_1": ("Toplam 0-1 Gol", lambda r: _tot(r) <= 1),
    "Gol_2_3": ("Toplam 2-3 Gol", lambda r: (_tot(r) >= 2) & (_tot(r) <= 3)),
    "Gol_4_6": ("Toplam 4-6 Gol", lambda r: (_tot(r) >= 4) & (_tot(r) <= 6)),
    "Gol_7p": ("Toplam 7+ Gol", lambda r: _tot(r) >= 7),
    "IY_Ust15": ("İY 1.5 Üst", lambda r: (r["HTHG"] + r["HTAG"]) >= 2),
    "IY_Alt15": ("İY 1.5 Alt", lambda r: (r["HTHG"] + r["HTAG"]) <= 1),
}
BET_GROUPS = {
    "Maç Sonucu": ["MS1", "MSX", "MS2"],
    "Karşılıklı Gol": ["KG_Var", "KG_Yok"],
    "2.5 Alt/Üst": ["Ust25", "Alt25"],
    "1.5 Alt/Üst": ["Ust15", "Alt15"],
    "Toplam Gol": ["Gol_0_1", "Gol_2_3", "Gol_4_6", "Gol_7p"],
    "İlk Yarı 1.5": ["IY_Ust15", "IY_Alt15"],
}
