"""Analyse fondamentale : FRED (sans clé), Banque mondiale, FMI DataMapper, yfinance."""
import io, requests, pandas as pd, yfinance as yf

H = {"User-Agent": "Mozilla/5.0"}

def fred(sid):
    r = requests.get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}", timeout=15, headers=H)
    df = pd.read_csv(io.StringIO(r.text))
    s = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    s.index = pd.to_datetime(df.iloc[:, 0])
    return s.dropna()

def worldbank(ind="NY.GDP.MKTP.KD.ZG", c="USA"):
    j = requests.get(f"https://api.worldbank.org/v2/country/{c}/indicator/{ind}?format=json&per_page=6", timeout=15).json()
    for x in j[1]:
        if x["value"] is not None:
            return round(x["value"], 2), x["date"]

def imf(ind="PCPIPCH", c="USA"):
    j = requests.get(f"https://www.imf.org/external/datamapper/api/v1/{ind}/{c}", timeout=15, headers=H).json()
    d = j["values"][ind][c]
    y = sorted(d)[-1]
    return round(d[y], 2), y

def at(s, k):
    """Valeur il y a k observations (bornée au début de la série)."""
    return s.iloc[max(-len(s), -k)]

def macro():
    """Retourne scores USD et Risk (-100..100) + détails."""
    out, usd, risk, notes = {}, 0, 0, []
    def safe(name, fn):
        try: return fn()
        except Exception: out[name] = "indisponible"; return None

    cpi = safe("cpi_yoy", lambda: fred("CPIAUCSL"))
    if cpi is not None:
        y = cpi.pct_change(12).iloc[-1] * 100; out["cpi_yoy"] = round(y, 2)
        if y > 3: usd += 1; notes.append("Inflation > 3% : Fed restrictive, soutien USD")
        elif y < 2.2: usd -= 1; notes.append("Inflation < 2.2% : marge d'assouplissement, USD faible")
    ff = safe("fed_funds", lambda: fred("FEDFUNDS"))
    if ff is not None:
        out["fed_funds"] = round(ff.iloc[-1], 2); d = ff.iloc[-1] - at(ff, 7)
        if d > 0.25: usd += 1; notes.append("Taux Fed en hausse sur 6 mois : USD+")
        elif d < -0.25: usd -= 1; notes.append("Taux Fed en baisse sur 6 mois : USD-")
    u = safe("unemployment", lambda: fred("UNRATE"))
    if u is not None:
        out["unemployment"] = round(u.iloc[-1], 2); d = u.iloc[-1] - at(u, 13)
        if d > 0.3: usd -= 1; risk -= 1; notes.append("Chômage en hausse : risque de ralentissement")
        elif d < -0.3: usd += 1; risk += 1; notes.append("Chômage en baisse : économie solide")
    u10 = safe("us10y_val", lambda: fred("DGS10"))
    cv = safe("curve_10y2y", lambda: fred("T10Y2Y"))
    if cv is not None:
        out["curve_10y2y"] = round(cv.iloc[-1], 2)
        if cv.iloc[-1] < 0: risk -= 1; notes.append("Courbe inversée : signal risk-off")
        elif cv.iloc[-1] > 0.5: risk += 1; notes.append("Courbe pentue : contexte risk-on")
    v = safe("vix", lambda: fred("VIXCLS"))
    if v is not None:
        out["vix"] = round(v.iloc[-1], 2)
        if v.iloc[-1] > 25: risk -= 2; notes.append("VIX > 25 : stress, risk-off")
        elif v.iloc[-1] < 16: risk += 1; notes.append("VIX < 16 : appétit pour le risque")
    ecb = safe("ecb_rate", lambda: fred("ECBDFR"))
    if ecb is not None and ff is not None:
        out["ecb_rate"] = round(ecb.iloc[-1], 2); out["spread_fed_bce"] = round(ff.iloc[-1] - ecb.iloc[-1], 2)
        d = ecb.iloc[-1] - at(ecb, 90)
        if d > 0.2: usd -= 1; notes.append("BCE en resserrement : soutien EUR")
        elif d < -0.2: usd += 1; notes.append("BCE en assouplissement : pression sur EUR")
    us10 = u10
    if u10 is not None: out["us10y"] = round(u10.iloc[-1], 2)
    out.pop("us10y_val", None); de10 = safe("bund10y", lambda: fred("IRLTLT01DEM156N"))
    if us10 is not None and de10 is not None:
        sp = us10.iloc[-1] - de10.iloc[-1]; sp0 = at(us10, 60) - at(de10, 3) if len(de10) > 3 else sp
        out["spread_US10_Bund10"] = round(sp, 2)
        if sp - sp0 > 0.15: usd += 1; notes.append("Écart US-Bund en hausse : USD favorisé")
        elif sp - sp0 < -0.15: usd -= 1; notes.append("Écart US-Bund en baisse : EUR favorisé")
    ez = safe("gdp_growth_ez", lambda: worldbank("NY.GDP.MKTP.KD.ZG", "EMU"))
    if ez: out["gdp_growth_ez"] = f"{ez[0]}% ({ez[1]}, Banque mondiale)"
    iez = safe("imf_inflation_ez", lambda: imf("PCPIPCH", "EURO"))
    if iez: out["imf_inflation_ez"] = f"{iez[0]}% ({iez[1]}, FMI)"
    g = safe("gdp_growth_us", worldbank)
    if g: out["gdp_growth_us"] = f"{g[0]}% ({g[1]}, Banque mondiale)"
    i = safe("imf_inflation_us", imf)
    if i: out["imf_inflation_us"] = f"{i[0]}% ({i[1]}, FMI)"
    return {"usd": max(-100, min(100, usd * 20)), "risk": max(-100, min(100, risk * 25)), "data": out, "notes": notes}

def bias(symbol, m):
    s = symbol.upper()
    if s in ("DX-Y.NYB",) or (s.startswith("USD") and s.endswith("=X")): v = m["usd"]
    elif s.endswith("USD=X") and len(s) == 8: v = -m["usd"]
    elif s in ("GC=F", "SI=F", "XAUUSD=X"): v = -m["usd"] * 0.6
    elif s.endswith("=X"): v = 0
    else: v = m["risk"]
    return v

def company(symbol):
    try:
        i = yf.Ticker(symbol).info
        keys = ["trailingPE", "forwardPE", "revenueGrowth", "profitMargins", "returnOnEquity", "debtToEquity"]
        d = {k: i.get(k) for k in keys if i.get(k) is not None}
        return d or None
    except Exception:
        return None

def analyse(symbol):
    m = macro(); b = bias(symbol, m)
    label = "HAUSSIER" if b >= 25 else "BAISSIER" if b <= -25 else "NEUTRE"
    return {"bias_score": b, "bias": label, "macro": m["data"], "notes": m["notes"], "company": None}
