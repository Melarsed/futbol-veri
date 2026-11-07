"""Dixon-Coles gol modeli ve bahis olasılıkları.

Her ülke için (tüm ligleri birlikte) takımların hücum/savunma gücü,
iç saha avantajı ve düşük skor düzeltmesi (rho) zamana göre ağırlıklı
olarak tahmin edilir. Yeni maçlar daha çok, eski maçlar daha az sayılır.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize, least_squares
from scipy.special import gammaln
from scipy.stats import poisson

MAX_GOALS = 10


class DixonColes:
    def __init__(self, xi: float = 0.0019, ridge: float = 1.0, window_days: int = 1100):
        self.xi = xi              # zaman sönümü (gün başına)
        self.ridge = ridge        # az maçı olan takımları ortalamaya çeker
        self.window_days = window_days
        self.fitted = False

    def fit(self, df: pd.DataFrame, ref_date, hg="FTHG", ag="FTAG") -> "DixonColes":
        ref_date = pd.Timestamp(ref_date)
        d = df[(df["Date"] < ref_date) & df[hg].notna() & df[ag].notna()]
        d = d[d["Date"] >= ref_date - pd.Timedelta(days=self.window_days)]
        if len(d) < 30:
            raise ValueError("Yetersiz maç verisi")
        teams = sorted(set(d["HomeTeam"]) | set(d["AwayTeam"]))
        idx = {t: i for i, t in enumerate(teams)}
        n = len(teams)
        hi = d["HomeTeam"].map(idx).to_numpy()
        ai = d["AwayTeam"].map(idx).to_numpy()
        x = d[hg].to_numpy(float)
        y = d[ag].to_numpy(float)
        days = (ref_date - d["Date"]).dt.days.to_numpy(float)
        w = np.exp(-self.xi * days)
        const = (w * (gammaln(x + 1) + gammaln(y + 1))).sum()

        m00 = (x == 0) & (y == 0)
        m01 = (x == 0) & (y == 1)
        m10 = (x == 1) & (y == 0)
        m11 = (x == 1) & (y == 1)
        ridge = self.ridge

        def nll(p):
            att, dfn = p[:n], p[n:2 * n]
            home, rho = p[2 * n], p[2 * n + 1]
            eh = att[hi] + dfn[ai] + home
            ea = att[ai] + dfn[hi]
            lam, mu = np.exp(eh), np.exp(ea)
            tau = np.ones_like(lam)
            tau[m00] = 1 - lam[m00] * mu[m00] * rho
            tau[m01] = 1 + lam[m01] * rho
            tau[m10] = 1 + mu[m10] * rho
            tau[m11] = 1 - rho
            if np.any(tau <= 0):
                return 1e12, np.zeros_like(p)
            ll = w * (x * eh - lam + y * ea - mu + np.log(tau))
            f = -(ll.sum() - const) + 0.5 * ridge * (att @ att + dfn @ dfn)
            # türevler
            geh = x - lam
            gea = y - mu
            grho = np.zeros_like(lam)
            t = tau
            geh[m00] += -lam[m00] * mu[m00] * rho / t[m00]
            gea[m00] += -lam[m00] * mu[m00] * rho / t[m00]
            grho[m00] = -lam[m00] * mu[m00] / t[m00]
            geh[m01] += lam[m01] * rho / t[m01]
            grho[m01] = lam[m01] / t[m01]
            gea[m10] += mu[m10] * rho / t[m10]
            grho[m10] = mu[m10] / t[m10]
            grho[m11] = -1 / t[m11]
            geh *= w
            gea *= w
            g = np.zeros_like(p)
            g[:n] += np.bincount(hi, geh, n) + np.bincount(ai, gea, n)
            g[n:2 * n] += np.bincount(ai, geh, n) + np.bincount(hi, gea, n)
            g[2 * n] = geh.sum()
            g[2 * n + 1] = (w * grho).sum()
            g = -g
            g[:n] += ridge * att
            g[n:2 * n] += ridge * dfn
            return f, g

        avg = np.log(max((w * (x + y)).sum() / (2 * w.sum()), 0.2))
        p0 = np.zeros(2 * n + 2)
        p0[:n] = avg / 2
        p0[n:2 * n] = avg / 2
        p0[2 * n] = 0.25
        p0[2 * n + 1] = -0.05
        bounds = [(None, None)] * (2 * n + 1) + [(-0.25, 0.25)]
        res = minimize(nll, p0, jac=True, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 500})
        p = res.x
        self.teams = idx
        self.att = p[:n]
        self.dfn = p[n:2 * n]
        self.home = p[2 * n]
        self.rho = p[2 * n + 1]
        # Az verisi olan takımlar için güvenilirlik (ağırlıklı maç sayısı)
        self.team_weight = np.bincount(hi, w, n) + np.bincount(ai, w, n)
        self.avg_att = np.average(self.att, weights=self.team_weight)
        self.avg_dfn = np.average(self.dfn, weights=self.team_weight)
        self.fitted = True
        return self

    def rates(self, home: str, away: str):
        """(ev sahibi gol beklentisi, deplasman gol beklentisi, bilinen_takım_mı)."""
        ih, ia = self.teams.get(home), self.teams.get(away)
        known = ih is not None and ia is not None
        ah = self.att[ih] if ih is not None else self.avg_att - 0.15
        dh = self.dfn[ih] if ih is not None else self.avg_dfn + 0.15
        aa = self.att[ia] if ia is not None else self.avg_att - 0.15
        da = self.dfn[ia] if ia is not None else self.avg_dfn + 0.15
        lam = float(np.exp(ah + da + self.home))
        mu = float(np.exp(aa + dh))
        return lam, mu, known


def score_matrix(lam: float, mu: float, rho: float = 0.0, max_goals: int = MAX_GOALS):
    g = np.arange(max_goals + 1)
    m = np.outer(poisson.pmf(g, lam), poisson.pmf(g, mu))
    m[0, 0] *= 1 - lam * mu * rho
    m[0, 1] *= 1 + lam * rho
    m[1, 0] *= 1 + mu * rho
    m[1, 1] *= 1 - rho
    m = np.clip(m, 0, None)
    return m / m.sum()


def markets(lam, mu, rho, ht_lam=None, ht_mu=None, ht_rho=0.0) -> dict:
    m = score_matrix(lam, mu, rho)
    g = np.arange(m.shape[0])
    tot = g[:, None] + g[None, :]
    out = {
        "xG_ev": lam,
        "xG_dep": mu,
        "MS1": float(np.tril(m, -1).sum()),
        "MSX": float(np.trace(m)),
        "MS2": float(np.triu(m, 1).sum()),
        "KG_Var": float(m[1:, 1:].sum()),
        "Ust25": float(m[tot >= 3].sum()),
        "Ust15": float(m[tot >= 2].sum()),
        "Ust35": float(m[tot >= 4].sum()),
        "Gol_0_1": float(m[tot <= 1].sum()),
        "Gol_2_3": float(m[(tot >= 2) & (tot <= 3)].sum()),
        "Gol_4_6": float(m[(tot >= 4) & (tot <= 6)].sum()),
        "Gol_7p": float(m[tot >= 7].sum()),
    }
    out["KG_Yok"] = 1 - out["KG_Var"]
    out["Alt25"] = 1 - out["Ust25"]
    out["Alt15"] = 1 - out["Ust15"]
    i, j = np.unravel_index(np.argmax(m), m.shape)
    out["Skor"] = f"{i}-{j}"
    out["Skor_p"] = float(m[i, j])
    if ht_lam is not None:
        h = score_matrix(ht_lam, ht_mu, ht_rho, 8)
        gh = np.arange(h.shape[0])
        htot = gh[:, None] + gh[None, :]
        out["IY_Ust15"] = float(h[htot >= 2].sum())
        out["IY_Alt15"] = 1 - out["IY_Ust15"]
        out["IY_Ust05"] = float(h[htot >= 1].sum())
    return out


def implied_probs(*odds):
    """Bahis oranlarından bahisçi payı çıkarılmış olasılıklar."""
    o = np.array(odds, dtype=float)
    if np.any(~np.isfinite(o)) or np.any(o <= 1.0):
        return None
    p = 1 / o
    return p / p.sum()


def market_rates(pH, pD, pA, pO25, rho, guess=(1.4, 1.1)):
    """Piyasa olasılıklarına en uygun (lam, mu) çiftini bulur."""
    def resid(v):
        lam, mu = np.exp(v)
        m = score_matrix(lam, mu, rho)
        g = np.arange(m.shape[0])
        tot = g[:, None] + g[None, :]
        r = [np.tril(m, -1).sum() - pH, np.trace(m) - pD, np.triu(m, 1).sum() - pA]
        if pO25 is not None:
            r.append(2 * (m[tot >= 3].sum() - pO25))
        return r
    s = least_squares(resid, np.log(guess), bounds=([-3, -3], [1.6, 1.6]))
    lam, mu = np.exp(s.x)
    return float(lam), float(mu)
