"""Futbol Tahmin Paneli (kişisel kullanım)."""
from __future__ import annotations

import io
import json
from datetime import date

import numpy as np
import pandas as pd
import requests
import streamlit as st

from futbol.data import (DATA_DIR, EXTRA_LEAGUES, MAIN_LEAGUES, get_mirror, league_name,
                         load_fixtures, load_results, mirror_url, set_mirror)
from futbol.predict import BET_GROUPS, BETS, fit_all, predict_frame

st.set_page_config(page_title="Futbol Tahmin", page_icon="⚽", layout="wide")

KALIB = DATA_DIR / "kalibrasyon.csv"
OZET = DATA_DIR / "test_ozeti.json"
ALL_LEAGUES = list(MAIN_LEAGUES) + list(EXTRA_LEAGUES)
ODDS_COL = {"MS1": "OddH", "MSX": "OddD", "MS2": "OddA", "Ust25": "OddO25", "Alt25": "OddU25"}
# 10 sezonluk testte maç sonucu en öngörülebilir ligler (favori isabeti %52+)
ONGORULEBILIR = ["P1", "N1", "I1", "E0", "SC0", "G1", "SP1", "T1", "D1", "F1", "B1"]
GUNLER = ["Pazartesi", "Salı", "Çarşamba", "Perşembe", "Cuma", "Cumartesi", "Pazar"]

# Kartta gösterilen bahis satırları: (başlık, [(seçenek, sütun), ...])
KART = [
    ("MS", [("1", "MS1"), ("X", "MSX"), ("2", "MS2")]),
    ("KG", [("Var", "KG_Var"), ("Yok", "KG_Yok")]),
    ("2.5 Gol", [("Üst", "Ust25"), ("Alt", "Alt25")]),
    ("1.5 Gol", [("Üst", "Ust15"), ("Alt", "Alt15")]),
    ("Toplam Gol", [("0-1", "Gol_0_1"), ("2-3", "Gol_2_3"), ("4-6", "Gol_4_6"), ("7+", "Gol_7p")]),
    ("İY 1.5", [("Üst", "IY_Ust15"), ("Alt", "IY_Alt15")]),
]
# Kombine bahisler (kartta açılır bölümde gösterilir)
KOMBINE = [
    ("MS & 2.5 Üst", [("1", "MS1_Ust25"), ("X", "MSX_Ust25"), ("2", "MS2_Ust25")]),
    ("MS & 1.5 Üst", [("1", "MS1_Ust15"), ("X", "MSX_Ust15"), ("2", "MS2_Ust15")]),
    ("MS & KG Var", [("1", "MS1_Var"), ("X", "MSX_Var"), ("2", "MS2_Var")]),
    ("2.5 Üst & KG Var", [("", "Ust25_Var")]),
]


# ---------------------------------------------------------------- veri / model
@st.cache_data(ttl=3 * 3600, show_spinner=False)
def get_results(leagues: tuple):
    bar = st.progress(0.0, "Sonuçlar indiriliyor…")
    df, warn = load_results(list(leagues), progress=lambda p, t: bar.progress(p, t))
    bar.empty()
    return df, warn


@st.cache_data(ttl=2 * 3600, show_spinner=False)
def get_fixtures():
    return load_fixtures()


@st.cache_resource(show_spinner=False)
def get_models(_results, key):
    with st.spinner("Modeller eğitiliyor…"):
        return fit_all(_results, pd.Timestamp(date.today()) + pd.Timedelta(days=1))


@st.cache_data(show_spinner=False)
def get_calibration():
    if KALIB.exists():
        return pd.read_csv(KALIB)
    return pd.DataFrame(columns=["bahis", "alt", "ust", "n", "isabet"])


@st.cache_data(ttl=3600, show_spinner=False)
def get_log():
    try:
        r = requests.get(mirror_url("tahmin_kaydi.csv"), timeout=20)
        if r.status_code == 200 and r.content.startswith(b"Div"):
            return pd.read_csv(io.BytesIO(r.content), parse_dates=["Date"])
    except requests.RequestException:
        pass
    return pd.DataFrame()


def hist_hit(cal: pd.DataFrame, key: str, p: float):
    """Bu bahis türünde, bu olasılık aralığında geçen sezonki gerçek tutma oranı."""
    r = cal[(cal["bahis"] == key) & (cal["alt"] <= p) & (cal["ust"] > p)]
    if r.empty or r["n"].iloc[0] < 30:
        return np.nan, 0
    return float(r["isabet"].iloc[0]), int(r["n"].iloc[0])


def level(h):
    if np.isnan(h):
        return "⚪ Az veri"
    if h >= 0.75:
        return "🟢 Yüksek"
    if h >= 0.62:
        return "🟡 Orta"
    return "🔴 Riskli"


def pct(p):
    return f"%{p * 100:.0f}"


def satirlar(r, cal, tanim):
    lines = []
    for baslik, secenekler in tanim:
        vals = [(ad, r.get(k), k) for ad, k in secenekler if pd.notna(r.get(k))]
        if not vals:
            continue
        best = max(vals, key=lambda v: v[1])
        h, _ = hist_hit(cal, best[2], best[1])
        parts = []
        for ad, p, k in vals:
            t = f"{ad} {pct(p)}".strip()
            if k == best[2]:
                t = f"**{t}**"
                if not np.isnan(h) and h >= 0.75:
                    t = f":green[{t}]"
                elif not np.isnan(h) and h >= 0.62:
                    t = f":orange[{t}]"
            parts.append(t)
        lines.append(f"**{baslik}** — " + " · ".join(parts))
    return "  \n".join(lines)


def kart(r, cal):
    with st.container(border=True):
        saat = r.get("Time") if isinstance(r.get("Time"), str) and r.get("Time") not in ("nan", "") else ""
        st.markdown(f"**{r['HomeTeam']} – {r['AwayTeam']}**  \n"
                    f":gray[{r['Lig']} · {r['Date'].strftime('%d.%m')} {saat}]")
        st.markdown(satirlar(r, cal, KART))
        with st.expander("Kombine bahisler"):
            st.markdown(satirlar(r, cal, KOMBINE))
        st.caption(f"Olası skor {r['Skor']} ({pct(r['Skor_p'])}) · "
                   f"Gol beklentisi {r['xG_ev']:.2f} – {r['xG_dep']:.2f}"
                   + ("" if r["piyasa"] else " · oran yok, sadece model"))


# ---------------------------------------------------------------- yan menü
with st.sidebar:
    st.header("⚙️ Ayarlar")
    w_model = st.slider(
        "Model ağırlığı", 0.0, 1.0, 0.0, 0.1,
        help="0 = bahis oranlarından türetilen olasılıklar (geçmiş testte en isabetlisi). "
             "1 = sadece kendi modelimiz. Oranı olmayan maçlarda her zaman model kullanılır.")
    with st.expander("🌐 Veri kaynağı"):
        kullanici = st.text_input("GitHub kullanıcı adı", value=get_mirror())
        if kullanici.strip() != get_mirror() and st.button("Kaydet"):
            set_mirror(kullanici)
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()
    if st.button("🔄 Verileri yenile"):
        st.cache_data.clear()
        st.cache_resource.clear()
        for f in DATA_DIR.glob("*.csv"):
            if f.name not in {KALIB.name, "xgabora_Matches.csv"}:
                f.unlink(missing_ok=True)
        st.rerun()
    st.caption("Renkler: :green[**yeşil**] = geçen sezon bu olasılıkta %75+ tuttu, "
               ":orange[**turuncu**] = %62–75.")

# ---------------------------------------------------------------- veri
st.title("⚽ Futbol Tahmin")
results, warn1 = get_results(tuple(ALL_LEAGUES))
fixtures, warn2 = get_fixtures()
if results.empty:
    st.error("Hiç veri indirilemedi. İnternet bağlantını kontrol edip 'Verileri yenile'ye bas.")
    st.stop()
models = get_models(results, (len(results), str(results["Date"].max())))
cal = get_calibration()

with st.expander("🔎 Filtre"):
    sadece = st.toggle("Sadece öngörülebilir ligler", value=True,
                       help="10 sezonluk testte en çok tutan ligler: Portekiz, Hollanda, İtalya, "
                            "Premier Lig, İskoçya, Yunanistan, İspanya, Türkiye, Almanya, Fransa, Belçika "
                            "(1. ligler). 2. liglerde favoriler çok daha az kazanıyor.")
    havuz = ONGORULEBILIR if sadece else ALL_LEAGUES
    secili = st.multiselect("Ligler", havuz, default=havuz, format_func=league_name)
    c1, c2 = st.columns(2)
    gun = c1.slider("Kaç gün ilerisi", 1, 10, 3)
    ara = c2.text_input("Takım ara")
if warn1 + warn2:
    with st.expander(f"⚠️ {len(warn1 + warn2)} veri uyarısı"):
        for w in warn1 + warn2:
            st.write("• " + w)

today = pd.Timestamp(date.today())
fx = fixtures[(fixtures["Date"] >= today) & (fixtures["Date"] <= today + pd.Timedelta(days=gun))]
fx = fx[fx["Div"].isin(secili)]
if ara.strip():
    q = ara.strip().lower()
    fx = fx[fx["HomeTeam"].str.lower().str.contains(q) | fx["AwayTeam"].str.lower().str.contains(q)]
pred = predict_frame(models, fx, w_model) if len(fx) else pd.DataFrame()
if len(pred):
    pred["Lig"] = pred["Div"].map(league_name)
    pred["sira"] = pred["Div"].map({d: i for i, d in enumerate(ALL_LEAGUES)})
    if "Time" not in pred:
        pred["Time"] = ""
    # Fikstür saatleri İngiltere saatidir -> Türkiye saatine çevir
    t = pd.to_datetime(pred["Date"].dt.strftime("%Y-%m-%d") + " " + pred["Time"].fillna(""), errors="coerce")
    t = t.dt.tz_localize("Europe/London", ambiguous="NaT", nonexistent="NaT").dt.tz_convert("Europe/Istanbul")
    pred["Time"] = t.dt.strftime("%H:%M").fillna("")
    pred = pred.sort_values(["Date", "sira", "Time"])

tab1, tab2, tab5, tab3, tab4 = st.tabs(["⚽ Maçlar", "🎯 En güvenli", "🧾 Kupon", "📒 Tahmin kaydı", "📊 Model başarısı"])

# ---- 1) Maç kartları
with tab1:
    if pred.empty:
        st.info("Bu aralıkta maç yok. Filtreden gün sayısını artır.")
    for gun_tarih, gd in pred.groupby("Date", sort=True) if len(pred) else []:
        st.subheader(f"{GUNLER[gun_tarih.weekday()]} {gun_tarih.strftime('%d.%m')}")
        for lig, ld in gd.groupby("sira", sort=True):
            st.markdown(f"##### {ld['Lig'].iloc[0]}")
            cols = st.columns(2)
            for i, (_, r) in enumerate(ld.iterrows()):
                with cols[i % 2]:
                    kart(r, cal)

# ---- 2) En güvenli seçimler
with tab2:
    if pred.empty:
        st.info("Bu aralıkta maç yok.")
    else:
        gruplar = st.multiselect("Bahis türleri", list(BET_GROUPS), default=list(BET_GROUPS))
        esik = st.slider("En düşük geçmiş tutma oranı", 0.5, 0.9, 0.70, 0.01, format="%.2f")
        rows = []
        for _, r in pred.iterrows():
            for g in gruplar:
                for key in BET_GROUPS[g]:
                    p = r.get(key)
                    if p is None or pd.isna(p):
                        continue
                    h, n = hist_hit(cal, key, p)
                    if np.isnan(h) or h < esik:
                        continue
                    rows.append({"Maç": f"{r['HomeTeam']} - {r['AwayTeam']}",
                                 "Tarih": r["Date"].strftime("%d.%m"), "Lig": r["Lig"],
                                 "Bahis": BETS[key][0], "Olasılık": p, "Geçmiş tutma": h,
                                 "Güven": level(h), "Adil oran": 1 / p})
        if rows:
            t = pd.DataFrame(rows).sort_values("Geçmiş tutma", ascending=False)
            st.dataframe(t, hide_index=True, width="stretch", column_config={
                "Olasılık": st.column_config.NumberColumn(format="percent"),
                "Geçmiş tutma": st.column_config.NumberColumn(format="percent"),
                "Adil oran": st.column_config.NumberColumn(format="%.2f")})
            st.caption("Adil oran = programın olasılığına göre olması gereken oran. "
                       "İddaa'daki oran bundan düşükse o bahis uzun vadede kaybettirir.")
        else:
            st.info("Bu eşikte seçim yok. Eşiği düşür.")
        st.caption("Kombinede olasılıklar çarpılır: %75'lik üç seçimin hepsinin tutma ihtimali yaklaşık %42.")


# ---- Kupon analizi
with tab5:
    st.markdown("Maç ve bahis seç, İddaa'daki oranı yaz. Program her seçimin ve kuponun "
                "gerçek tutma ihtimalini ve oranın adil olup olmadığını hesaplar.")
    if "kupon" not in st.session_state:
        st.session_state.kupon = []
    if pred.empty:
        st.info("Bu aralıkta maç yok.")
    else:
        pred["etiket"] = pred["Date"].dt.strftime("%d.%m") + " " + pred["HomeTeam"] + " - " + pred["AwayTeam"]
        mac = st.selectbox("Maç", pred["etiket"].tolist(), index=None, placeholder="Maç ara…")
        if mac:
            r = pred[pred["etiket"] == mac].iloc[0]
            secenek = [k for k in BETS if k in r and pd.notna(r[k])]
            bahis = st.selectbox("Bahis", secenek,
                                 format_func=lambda k: f"{BETS[k][0]}  ·  {pct(r[k])}  ·  adil oran {1 / r[k]:.2f}")
            if st.button("➕ Kupona ekle"):
                st.session_state.kupon.append({"mac": mac, "bahis": bahis, "p": float(r[bahis]),
                                               "oran": round(max(1.01, 1 / r[bahis]), 2)})
                st.rerun()
    kp = st.session_state.kupon
    if kp:
        st.divider()
        prob, oran_carp = 1.0, 1.0
        maclar = [k["mac"] for k in kp]
        for i, k in enumerate(kp):
            with st.container(border=True):
                st.markdown(f"**{k['mac']}**  \n{BETS[k['bahis']][0]} · tutma ihtimali **{pct(k['p'])}** "
                            f"· adil oran **{1 / k['p']:.2f}**")
                c1, c2 = st.columns([3, 1])
                k["oran"] = c1.number_input("İddaa oranı", 1.01, 100.0, float(k["oran"]), 0.01,
                                            key=f"oran_{i}_{k['mac']}_{k['bahis']}")
                deger = k["p"] * k["oran"] - 1
                if deger >= 0:
                    c1.markdown(f":green[✅ Oran adil veya iyi ({int(round(deger * 100)):+d}%)]")
                elif deger >= -0.07:
                    c1.markdown(f":orange[⚠️ Oran biraz düşük ({int(round(deger * 100)):+d}%)]")
                else:
                    c1.markdown(f":red[❌ Oran çok düşük ({int(round(deger * 100)):+d}%)]")
                if c2.button("Sil", key=f"sil_{i}"):
                    kp.pop(i)
                    st.rerun()
            prob *= k["p"]
            oran_carp *= k["oran"]
        if len(set(maclar)) < len(maclar):
            st.warning("Aynı maçtan birden fazla seçim var. Bu seçimler birbirine bağlı olduğu için "
                       "kupon ihtimali hesabı yanlış olur; aynı maçı tek seçimle oyna.")
        beklenti = prob * oran_carp - 1
        st.subheader("Kupon özeti")
        c1, c2, c3 = st.columns(3)
        c1.metric("Tutma ihtimali", f"%{prob * 100:.1f}", f"{1 / prob:.0f} kuponda 1" if prob < 0.5 else None,
                  delta_color="off")
        c2.metric("İddaa kupon oranı", f"{oran_carp:.2f}")
        c3.metric("Adil kupon oranı", f"{1 / prob:.2f}")
        if beklenti >= 0:
            st.success(f"Uzun vadede her 100 TL için beklenen sonuç: **{int(round(beklenti * 100)):+d} TL**")
        else:
            st.error(f"Uzun vadede her 100 TL için beklenen sonuç: **{int(round(beklenti * 100)):+d} TL**")
        if prob < 0.2:
            st.caption("Kupon ihtimali %20'nin altında. Maç sayısını azaltmak hem ihtimali hem beklentiyi iyileştirir.")
        if st.button("🗑️ Kuponu temizle"):
            st.session_state.kupon = []
            st.rerun()

# ---- 3) Tahmin kaydı (GitHub'da otomatik tutulur)
with tab3:
    log = get_log()
    if log.empty:
        st.info("Kayıt henüz yok. GitHub her 4 saatte bir yaklaşan maçların tahminini kaydeder.")
    else:
        done = log.merge(results[["Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "HTHG", "HTAG"]],
                         on=["Div", "Date", "HomeTeam", "AwayTeam"], how="inner").dropna(subset=["FTHG"])
        st.write(f"Kayıtlı tahmin: **{len(log)}** maç · sonuçlanan: **{len(done)}** maç")
        if len(done):
            esik2 = st.slider("Sadece bu olasılığın üstündeki tahminler", 0.5, 0.9, 0.65, 0.05)
            rows = []
            for key, (name, check) in BETS.items():
                if key not in done:
                    continue
                d = done.dropna(subset=[key])
                if key.startswith("IY"):
                    d = d.dropna(subset=["HTHG", "HTAG"])
                d = d[d[key] >= esik2]
                if len(d):
                    rows.append({"Bahis": name, "Tahmin": len(d), "Ort. olasılık": d[key].mean(),
                                 "Tutan": float(check(d).mean())})
            if rows:
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
                    "Ort. olasılık": st.column_config.NumberColumn(format="percent"),
                    "Tutan": st.column_config.NumberColumn(format="percent")})
            st.caption("Birkaç yüz maç birikmeden bu sonuçlara fazla güvenme.")

# ---- 4) Model başarısı
with tab4:
    if OZET.exists():
        oz = json.loads(OZET.read_text())
        st.caption(f"Geçmiş test · {oz.get('donem', '')} · 22 ana lig")
        names = {"1.0": "Sadece model", "0.5": "Yarı yarıya", "0.0": "Bahis oranları"}
        rows = [{"Yöntem": names[k], "Maç": int(v["n"]), "1X2 isabet": v["isabet_1x2"],
                 "1X2 log-loss": v["logloss_1x2"], "2.5 log-loss": v["logloss_25"],
                 "KG log-loss": v["logloss_kg"]} for k, v in oz.items() if k in names]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
            "1X2 isabet": st.column_config.NumberColumn(format="percent"),
            **{c: st.column_config.NumberColumn(format="%.4f") for c in ["1X2 log-loss", "2.5 log-loss", "KG log-loss"]}})
        st.info("Log-loss düşük = daha doğru olasılık. Testte bahis oranları kendi modelimizden iyi çıktı; "
                "varsayılan ayar bu yüzden oranları kullanır.")
    if len(cal):
        b = st.selectbox("Bahis türü", [k for k in BETS if k in set(cal["bahis"])], format_func=lambda k: BETS[k][0])
        c = cal[cal["bahis"] == b][["alt", "ust", "n", "ort_olasilik", "isabet"]]
        c = c.rename(columns={"alt": "Olasılık ≥", "ust": "Olasılık <", "n": "Maç",
                              "ort_olasilik": "Tahmin edilen", "isabet": "Gerçekleşen"})
        st.dataframe(c, hide_index=True, width="stretch", column_config={
            "Tahmin edilen": st.column_config.NumberColumn(format="percent"),
            "Gerçekleşen": st.column_config.NumberColumn(format="percent")})
