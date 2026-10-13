"""EURUSD multi-timeframe : price action + ICT + Volume Profile + VWAP ancrés."""
import time, datetime as dt
import numpy as np, pandas as pd, yfinance as yf
import fundamentals

SYMBOL, PIP = "EURUSD=X", 0.0001
TFS = ["1d", "4h", "1h", "15m", "5m"]
PER = {"1d": "2y", "4h": "730d", "1h": "730d", "15m": "60d", "5m": "30d"}
SWK = {"1d": 3, "4h": 3, "1h": 3, "15m": 2, "5m": 2}
_cache = {}

PLAYBOOK = [
    "1. Top-down : D1/H4 donnent la direction, H1 donne la zone (POI), M15/M5 donnent la confirmation d'entrée.",
    "2. Biais fondamental (Fed vs BCE, spread US-Bund, VIX) : ne trader que dans son sens ou rester à l'écart.",
    "3. Structure : BOS = continuation, CHoCH = premier signe de retournement. Jamais contre D1 + H4 alignés.",
    "4. Premium/Discount du range H4 : acheter sous 50%, vendre au-dessus. Zone OTE 62-79%.",
    "5. Liquidité : le prix cible PDH/PDL, PWH/PWL, high/low d'Asie et equal highs/lows. Sweep + rejet = entrée de qualité.",
    "6. Judas swing : en London, un sweep du high/low d'Asie suivi d'un retour dans le range donne le sens de la journée.",
    "7. PD Arrays : entrer sur FVG / Order Block H1-H4 non mitigé, dans le sens du biais, confirmé par CHoCH M15/M5.",
    "8. Volume Profile : POC = aimant, VAH/VAL = bords de la value area. Rejet de VAH/VAL = retour vers POC ; acceptation hors value = tendance.",
    "9. VWAP du jour/semaine : au-dessus = acheteurs dominants, en dessous = vendeurs. Les écarts au VWAP servent de zones de retour.",
    "10. Timing : killzones London 07-10 UTC et New York 12-15 UTC. Éviter 10-12 UTC et la fin de séance NY.",
    "11. Risque : SL derrière la zone + 0.5 ATR H1, RR minimum 1.5, risque <= 1% par trade.",
]

def load(tf):
    c = _cache.get(tf)
    if c and time.time() - c[0] < 90: return c[1]
    raw = yf.download(SYMBOL, period=PER[tf], interval="1h" if tf == "4h" else tf, progress=False, auto_adjust=False)
    if isinstance(raw.columns, pd.MultiIndex): raw.columns = raw.columns.get_level_values(0)
    df = raw[["Open", "High", "Low", "Close", "Volume"]].dropna()
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None) if tf == "1d" else df.index.tz_convert("UTC").tz_localize(None)
    if tf == "4h":
        df = df.resample("4h").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna()
    if len(df) < 60: raise ValueError(f"Données insuffisantes ({tf}). Réessayez dans un instant.")
    _cache[tf] = (time.time(), df)
    return df

def atr(df, n=14):
    pc = df.Close.shift()
    tr = pd.concat([df.High - df.Low, (df.High - pc).abs(), (df.Low - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def swings(df, k):
    H, L, hi, lo = df.High.values, df.Low.values, [], []
    for i in range(k, len(df) - k):
        if H[i] == H[i - k:i + k + 1].max(): hi.append(i)
        if L[i] == L[i - k:i + k + 1].min(): lo.append(i)
    return hi, lo

def avwap(df):
    tp = (df.High + df.Low + df.Close) / 3
    v = df.Volume.replace(0, np.nan).fillna(1)
    return float(((tp * v).cumsum() / v.cumsum()).iloc[-1])

def profile(df, bins=50):
    """Volume profile (si volume = 0 sur le forex, profil basé sur le temps passé par prix, façon TPO)."""
    if len(df) < 8: return None
    lo, hi = df.Low.min(), df.High.max()
    if hi - lo < 1e-9: return None
    edges = np.linspace(lo, hi, bins + 1); vol = np.zeros(bins)
    v = df.Volume.replace(0, np.nan).fillna(1).values
    for l, h, x in zip(df.Low.values, df.High.values, v):
        a = max(np.searchsorted(edges, l, side="right") - 1, 0)
        b = min(np.searchsorted(edges, h, side="right") - 1, bins - 1)
        vol[a:b + 1] += x / (b - a + 1)
    p = int(vol.argmax()); acc = vol[p]; i = j = p
    while acc < 0.7 * vol.sum() and (i > 0 or j < bins - 1):
        up = vol[j + 1] if j < bins - 1 else -1
        dn = vol[i - 1] if i > 0 else -1
        if up >= dn: j += 1; acc += vol[j]
        else: i -= 1; acc += vol[i]
    return {"poc": (edges[p] + edges[p + 1]) / 2, "vah": edges[j + 1], "val": edges[i]}

def fvgs(H, L, a):
    out = []
    for i in range(2, len(H)):
        if np.isnan(a[i]): continue
        if L[i] > H[i - 2] and L[i] - H[i - 2] > 0.2 * a[i] and L[i + 1:].min(initial=np.inf) > H[i - 2]:
            out.append(("bull", L[i], H[i - 2], i - 1))
        if H[i] < L[i - 2] and L[i - 2] - H[i] > 0.2 * a[i] and H[i + 1:].max(initial=-np.inf) < L[i - 2]:
            out.append(("bear", L[i - 2], H[i], i - 1))
    return out

def order_blocks(O, H, L, C, a):
    out = []
    for i in range(1, len(H)):
        if np.isnan(a[i]): continue
        body = C[i] - O[i]
        if body > 1.5 * a[i] and C[i - 1] < O[i - 1] and C[i + 1:].min(initial=np.inf) > L[i - 1]:
            out.append(("bull", H[i - 1], L[i - 1], i - 1))
        if body < -1.5 * a[i] and C[i - 1] > O[i - 1] and C[i + 1:].max(initial=-np.inf) < H[i - 1]:
            out.append(("bear", H[i - 1], L[i - 1], i - 1))
    return out

def f(x): return None if x is None or pd.isna(x) else round(float(x), 5)

def tfa(df, tf):
    df = df.tail(1500)
    a = atr(df).values; A = float(a[-1])
    O, H, L, C = (df[c].values for c in ("Open", "High", "Low", "Close"))
    price = C[-1]; hi, lo = swings(df, SWK[tf])
    if len(hi) < 2 or len(lo) < 2: raise ValueError(f"Structure insuffisante ({tf})")
    trend = "range"
    if H[hi[-1]] > H[hi[-2]] and L[lo[-1]] > L[lo[-2]]: trend = "haussier"
    elif H[hi[-1]] < H[hi[-2]] and L[lo[-1]] < L[lo[-2]]: trend = "baissier"
    event = None
    if C[hi[-1] + 1:].max(initial=-np.inf) > H[hi[-1]]:
        event = "BOS haussier" if trend == "haussier" else "CHoCH haussier"
    elif C[lo[-1] + 1:].min(initial=np.inf) < L[lo[-1]]:
        event = "BOS baissier" if trend == "baissier" else "CHoCH baissier"
    rh, rl = H[hi[-1]], L[lo[-1]]; eq = (rh + rl) / 2
    pct = round(float((price - rl) / (rh - rl) * 100), 1) if rh > rl else 50.0
    sweep = None
    for i in range(max(len(df) - 8, 0), len(df)):
        for h in hi[-6:]:
            if h < i - 1 and H[i] > H[h] and C[i] < H[h]: sweep = "bear"
        for l in lo[-6:]:
            if l < i - 1 and L[i] < L[l] and C[i] > L[l]: sweep = "bull"
    ts = lambda i: int(df.index[i].timestamp())
    arrs = []
    for kind, lst in (("FVG", fvgs(H, L, a)), ("OB", order_blocks(O, H, L, C, a))):
        for typ in ("bull", "bear"):
            for t, top, bot, i in [x for x in lst if x[0] == typ][-2:]:
                arrs.append({"tf": tf, "kind": kind, "type": t, "top": f(top), "bottom": f(bot), "time": ts(i)})
    eqh = [H[x] for x in hi[-6:] if any(abs(H[x] - H[y]) < 0.15 * A for y in hi[-6:] if y != x)]
    eql = [L[x] for x in lo[-6:] if any(abs(L[x] - L[y]) < 0.15 * A for y in lo[-6:] if y != x)]
    return {"trend": trend, "event": event, "zone": "premium" if price > eq else "discount", "zone_pct": pct,
            "sweep": sweep, "atr": A, "atr_pips": round(A / PIP, 1), "range_high": f(rh), "range_low": f(rl),
            "arrays": arrs, "eq_highs": [float(x) for x in eqh], "eq_lows": [float(x) for x in eql],
            "swing_highs": [float(H[x]) for x in hi[-4:]], "swing_lows": [float(L[x]) for x in lo[-4:]]}

def dirn(t): return 1 if t["trend"] == "haussier" else -1 if t["trend"] == "baissier" else 0

def run():
    D = {tf: load(tf) for tf in TFS}
    T = {tf: tfa(D[tf], tf) for tf in TFS}
    m15, h1, d1 = D["15m"], D["1h"], D["1d"]
    price = float(m15.Close.iloc[-1]); A1 = T["1h"]["atr"]
    day = m15.index[-1].normalize(); monday = day - pd.Timedelta(days=day.weekday())

    # Niveaux jour / semaine
    dd = d1[d1.index < day]; L_ = {}
    if len(dd): L_["PDH"], L_["PDL"] = dd.High.iloc[-1], dd.Low.iloc[-1]
    w = dd.resample("W").agg({"High": "max", "Low": "min"}).dropna(); w = w[w.index < monday]
    if len(w): L_["PWH"], L_["PWL"] = w.High.iloc[-1], w.Low.iloc[-1]
    today = m15[m15.index >= day]
    if len(today): L_["DO"] = today.Open.iloc[0]; L_["VWAP_D"] = avwap(today)
    wk = h1[h1.index >= monday]
    if len(wk): L_["WO"] = wk.Open.iloc[0]; L_["VWAP_W"] = avwap(wk)

    # Sessions : range d'Asie + Judas swing
    asia = today[today.index.hour < 7]; judas = None
    if len(asia) >= 4:
        ah, al = asia.High.max(), asia.Low.min(); L_["ASIA_H"], L_["ASIA_L"] = ah, al
        lon = today[today.index.hour >= 7]
        if len(lon):
            if lon.High.max() > ah and price < ah: judas = "bear"
            elif lon.Low.min() < al and price > al: judas = "bull"

    # Volume profile : jour, jour précédent, composite H1
    for name, part in (("D", today), ("PD", m15[(m15.index >= (sorted(set(m15.index.normalize()))[-2])) & (m15.index < day)]),
                       ("W", h1.tail(200))):
        p = profile(part)
        if p: L_[f"POC_{name}"], L_[f"VAH_{name}"], L_[f"VAL_{name}"] = p["poc"], p["vah"], p["val"]
    levels = {k: f(v) for k, v in L_.items()}

    # Fondamental
    fund = fundamentals.analyse(SYMBOL); b = fund["bias_score"]

    # Score top-down
    score, why = b * 0.25, [f"Biais fondamental {fund['bias']} ({b:+.0f}) : {b*0.25:+.0f} pts"]
    def add(p, txt):
        nonlocal score; p = round(p); score += p; why.append(f"{txt} : {p:+d} pts")
    for tf, w_, n in (("1d", 12, "D1"), ("4h", 14, "H4"), ("1h", 8, "H1")):
        if dirn(T[tf]): add(w_ * dirn(T[tf]), f"Structure {T[tf]['trend']} {n}")
    add(6 if T["4h"]["zone_pct"] < 50 else -6, f"Prix en {T['4h']['zone']} H4 ({T['4h']['zone_pct']}%)")
    add(4 if T["1h"]["zone_pct"] < 50 else -4, f"Prix en {T['1h']['zone']} H1")
    for tf, w_ in (("15m", 8), ("5m", 4)):
        e = T[tf]["event"]
        if e: add(w_ if "haussier" in e else -w_, f"{e} {tf.upper()}")
    for tf in ("15m", "1h"):
        s = T[tf]["sweep"]
        if s: add(8 if s == "bull" else -8, f"Sweep de liquidité {'sell-side' if s=='bull' else 'buy-side'} {tf.upper()}"); break
    if judas: add(8 if judas == "bull" else -8, f"Judas swing {'haussier' if judas=='bull' else 'baissier'} (sweep range Asie)")
    if "VWAP_D" in L_: add(4 if price > L_["VWAP_D"] else -4, "Prix vs VWAP du jour")
    if "POC_W" in L_: add(3 if price > L_["POC_W"] else -3, "Prix vs POC composite")
    arrays = [x for tf in ("4h", "1h") for x in T[tf]["arrays"]]
    near = lambda typ: [x for x in arrays if x["type"] == typ and
                        ((typ == "bull" and x["top"] <= price + 0.3 * A1 and price - x["top"] < 2 * A1) or
                         (typ == "bear" and x["bottom"] >= price - 0.3 * A1 and x["bottom"] - price < 2 * A1))]
    bz, sz = near("bull"), near("bear")
    if bz: add(8, f"PD Array haussier proche ({bz[-1]['kind']} {bz[-1]['tf'].upper()})")
    if sz: add(-8, f"PD Array baissier proche ({sz[-1]['kind']} {sz[-1]['tf'].upper()})")
    score = round(score)
    direction = "LONG" if score >= 40 else "SHORT" if score <= -40 else "ATTENDRE"
    note = None
    if direction != "ATTENDRE" and dirn(T["4h"]) == (-1 if direction == "LONG" else 1):
        note = "Signal contre la structure H4 : trade interdit par les règles"; direction = "ATTENDRE"
    grade = "A" if abs(score) >= 60 and direction != "ATTENDRE" else "B" if direction != "ATTENDRE" else None

    setup = None
    if direction != "ATTENDRE":
        lg = direction == "LONG"; zs = bz if lg else sz
        if zs:
            z = zs[-1]; entry = (z["top"] + z["bottom"]) / 2
            sl = z["bottom"] - 0.5 * A1 if lg else z["top"] + 0.5 * A1; poi = f"{z['kind']} {z['tf'].upper()}"
        else:
            sw = T["1h"]["swing_lows"][-1] if lg else T["1h"]["swing_highs"][-1]
            entry = price; sl = sw - 0.5 * A1 if lg else sw + 0.5 * A1; poi = "Marché (pas de PD Array proche)"
        risk = abs(entry - sl)
        liq = {k: v for k, v in L_.items() if k in ("PDH", "PDL", "PWH", "PWL", "ASIA_H", "ASIA_L", "VAH_D", "VAL_D", "VAH_W", "VAL_W")}
        for tf in ("1h", "4h"):
            for v in T[tf]["swing_highs"] + T[tf]["eq_highs"]: liq[f"High {tf.upper()}"] = v
            for v in T[tf]["swing_lows"] + T[tf]["eq_lows"]: liq[f"Low {tf.upper()}"] = v
        c = sorted([(v, k) for k, v in liq.items() if (v > entry + 1.5 * risk if lg else v < entry - 1.5 * risk)], reverse=not lg is False and False)
        c = sorted(c, key=lambda x: x[0], reverse=not lg)
        tp1, n1 = c[0] if c else ((entry + 2 * risk if lg else entry - 2 * risk), "2R")
        tp2, n2 = c[1] if len(c) > 1 else ((entry + 3 * risk if lg else entry - 3 * risk), "3R")
        pips = lambda x: round(abs(x - entry) / PIP, 1)
        setup = {"side": direction, "poi": poi, "entry": f(entry), "sl": f(sl), "tp1": f(tp1), "tp2": f(tp2),
                 "tp1_name": n1, "tp2_name": n2, "risk_pips": pips(sl), "tp1_pips": pips(tp1),
                 "rr_tp1": round(abs(tp1 - entry) / risk, 2) if risk else None}

    h = dt.datetime.utcnow().hour
    kz = "London" if 7 <= h < 10 else "New York" if 12 <= h < 15 else "Hors killzone"
    mtf = {tf: {k: v for k, v in T[tf].items() if k in ("trend", "event", "zone", "zone_pct", "sweep", "atr_pips", "range_high", "range_low")} for tf in TFS}
    candles = {tf: [[int(t.timestamp()), f(r.Open), f(r.High), f(r.Low), f(r.Close)] for t, r in D[tf].tail(200).iterrows()] for tf in TFS}
    return {"symbol": "EURUSD", "price": f(price), "last_bar_utc": str(m15.index[-1]), "fundamental": fund, "mtf": mtf,
            "sessions": {"judas": judas, "killzone": kz, "asia_high": levels.get("ASIA_H"), "asia_low": levels.get("ASIA_L")},
            "levels": levels, "pd_arrays": arrays, "signal": {"direction": direction, "grade": grade, "score": score, "reasons": why, "note": note, "setup": setup},
            "playbook": PLAYBOOK, "candles": candles}
