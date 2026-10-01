"""
Reel Match - a classic golden-age cinema front end for the user-based
collaborative-filtering movie recommender.

Run:  python -m streamlit run app.py
Data: data/Movie_Recommendation_System.xlsx  (or upload another file in the sidebar)
"""
import hashlib
import html
import io
import random
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------------------------------------
# 1. Settings - the sample dataset's columns. For other datasets use the sidebar mapping.
# --------------------------------------------------------------------------
DATA_PATH = "data/Movie_Recommendation_System.xlsx"
SHEET = "Movie_Data"

# internal names (also the sample dataset's real column names)
USER, TITLE, GENRE, LANG = "User_ID", "Movie_Title", "Genre", "Language"
YEAR, RATING, RUNTIME = "Release_Year", "Rating", "Runtime_Minutes"

REQUIRED = [USER, TITLE, RATING]
OPTIONAL = [GENRE, LANG, YEAR, RUNTIME]
LABELS = {USER: "Viewer / user ID", TITLE: "Movie title", RATING: "Rating", GENRE: "Genre",
          LANG: "Language", YEAR: "Release year", RUNTIME: "Runtime (minutes)"}
GUESS = {USER: ["user"], TITLE: ["title", "name", "movie"], RATING: ["rating", "score", "star"],
         GENRE: ["genre", "category"], LANG: ["language", "lang"], YEAR: ["year"],
         RUNTIME: ["runtime", "duration", "length"]}

TOP_USERS = 5     # how many similar users ("taste twins") to look at
TOP_MOVIES = 5    # how many movies to recommend


# --------------------------------------------------------------------------
# 2. Data + recommender (same logic as your notebook)
# --------------------------------------------------------------------------
def read_raw(src, filename, sheet):
    if isinstance(src, bytes):
        src = io.BytesIO(src)
    if str(filename).lower().endswith(".csv"):
        return pd.read_csv(src)
    return pd.read_excel(src, sheet_name=sheet)


def guess_column(columns, canon):
    """Pick the most likely source column for an internal field."""
    low = {c: str(c).lower() for c in columns}
    for c in columns:                                    # exact match first
        if low[c].replace(" ", "_") == canon.lower():
            return c
    bad = ("number", "count", "num", "total", "id")
    for key in GUESS[canon]:
        for c in columns:
            if key in low[c] and not (canon in (RATING, TITLE) and any(b in low[c] for b in bad)):
                return c
    return None


def prepare(raw, mapping):
    """Clean data, build user x movie matrix and user-user cosine similarity.
    mapping: internal field -> column name in the file (or None for optional fields)."""
    used = {canon: src for canon, src in mapping.items() if src}
    df = raw[list(used.values())].copy()
    df.columns = list(used.keys())

    if GENRE not in df: df[GENRE] = "Unknown"
    if LANG not in df: df[LANG] = "Unknown"
    if YEAR not in df: df[YEAR] = float("nan")
    if RUNTIME not in df: df[RUNTIME] = float("nan")

    df[GENRE] = df[GENRE].fillna("Unknown")
    df[LANG] = df[LANG].fillna("Unknown")
    df[RATING] = pd.to_numeric(df[RATING], errors="coerce")
    df[YEAR] = pd.to_numeric(df[YEAR], errors="coerce")
    df[RUNTIME] = pd.to_numeric(df[RUNTIME], errors="coerce")
    df = df.dropna(subset=[USER, TITLE, RATING])
    df[TITLE] = df[TITLE].astype(str).str.strip()
    df = df.drop_duplicates(subset=[USER, TITLE])
    if df.empty:
        raise ValueError("No usable rows after cleaning - check that Rating is a number column.")

    matrix = df.pivot_table(index=USER, columns=TITLE, values=RATING, aggfunc="mean")
    if len(matrix) < 2:
        raise ValueError("Need at least 2 different users to find similar users.")
    sim = cosine_similarity(matrix.fillna(0))
    sim_df = pd.DataFrame(sim, index=matrix.index, columns=matrix.index)

    meta = df.groupby(TITLE).agg(
        avg_rating=(RATING, "mean"),
        runtime=(RUNTIME, "mean"),
        year=(YEAR, "median"),
    )
    top = df[RATING].max()
    scale = 5 if top <= 5 else 10 if top <= 10 else float(top)   # rating scale of this dataset
    return df, matrix, sim_df, meta, scale


def recommend(user_id, matrix, sim_df, k=TOP_USERS, n=TOP_MOVIES):
    """Returns (twins, watched, recs).
    twins   : Series  similar user -> cosine similarity
    watched : Series  movie -> rating given by this user
    recs    : DataFrame index=movie, columns=score, backers
    """
    k = min(k, len(sim_df) - 1)
    twins = sim_df[user_id].sort_values(ascending=False).drop(user_id).head(k)
    watched = matrix.loc[user_id].dropna()

    twin_ratings = matrix.loc[twins.index]
    scores = twin_ratings.mean(axis=0)
    scores = scores.drop(watched.index, errors="ignore").dropna()
    scores = scores.sort_values(ascending=False).head(n)

    backers = twin_ratings.notna().sum(axis=0).loc[scores.index]
    recs = pd.DataFrame({"score": scores, "backers": backers})
    return twins, watched, recs


# --------------------------------------------------------------------------
# 3. Design: CSS
# --------------------------------------------------------------------------
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,500;0,600;0,700;1,500;1,600;1,700&family=Jost:wght@300;400;500&display=swap');

:root{
  --ebony:#0e0b0a; --oxblood:#5a1420; --gold:#c9a45c; --gold-hi:#f0d590; --gold-dim:rgba(201,164,92,.35);
  --ivory:#f1e8d6; --muted:#a89a84;
}
#MainMenu, header[data-testid="stHeader"], footer{visibility:hidden; height:0;}
.stApp{
  color:var(--ivory); font-family:'Jost',sans-serif; font-weight:300;
  background:
    url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='180' height='180'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='.85' numOctaves='2' stitchTiles='stitch'/><feColorMatrix values='0 0 0 0 1 0 0 0 0 .9 0 0 0 0 .7 0 0 0 .07 0'/></filter><rect width='100%' height='100%' filter='url(%23n)'/></svg>"),
    radial-gradient(90% 60% at 50% 0%, #2a0d13 0%, rgba(14,11,10,0) 70%),
    #0e0b0a;
}
.block-container{max-width:1060px; padding-top:2.2rem; padding-bottom:5rem;}
.stApp p, .stApp label, .stApp li, .stApp .stMarkdown{font-family:'Jost',sans-serif;}
/* keep Streamlit's own icons (upload, arrows, sidebar toggle) on their icon font */
[data-testid="stIconMaterial"], span[class*="material-symbols"], span[class*="material-icons"]{
  font-family:'Material Symbols Rounded','Material Icons'!important; font-weight:400!important; font-style:normal!important;
  letter-spacing:normal!important; text-transform:none!important; font-feature-settings:'liga'!important;}
section[data-testid="stSidebar"]{background:#120d0c; border-right:1px solid var(--gold-dim);}
section[data-testid="stSidebar"] *{color:var(--ivory);}
[data-testid="stFileUploaderDropzone"]{background:#17110f!important; border:1px dashed var(--gold-dim)!important; border-radius:2px!important;}

/* ---------- curtain: the one entrance moment ---------- */
.curtain{position:fixed; inset:0; z-index:99999; pointer-events:none; animation:cur-gone 0s 2.6s forwards;}
.curtain i{position:absolute; top:0; bottom:0; width:50.4%;
  background:
    linear-gradient(90deg, rgba(0,0,0,.45), rgba(0,0,0,0) 25%, rgba(0,0,0,0) 75%, rgba(0,0,0,.45)),
    repeating-linear-gradient(90deg,#5a1420 0 26px,#7d1f2f 26px 50px,#450e18 50px 78px);
  animation-duration:1.9s; animation-delay:.55s; animation-fill-mode:forwards;
  animation-timing-function:cubic-bezier(.75,0,.2,1);}
.curtain .l{left:0; border-right:5px solid var(--gold); animation-name:open-l;}
.curtain .r{right:0; border-left:5px solid var(--gold); animation-name:open-r;}
@keyframes open-l{to{transform:translateX(-102%)}}
@keyframes open-r{to{transform:translateX(102%)}}
@keyframes cur-gone{to{visibility:hidden}}

/* ---------- masthead ---------- */
.masthead{position:relative; text-align:center; padding:46px 24px 40px; margin:6px 6px 2.4rem;
  border:1px solid var(--gold); outline:1px solid var(--gold-dim); outline-offset:7px;
  background:
    radial-gradient(70% 90% at 50% 100%, rgba(240,213,144,.16), rgba(240,213,144,0) 70%),
    repeating-conic-gradient(from -90deg at 50% 118%, rgba(201,164,92,.10) 0 4deg, rgba(201,164,92,0) 4deg 9deg);}
.masthead .orn{display:flex; align-items:center; justify-content:center; gap:14px; color:var(--gold);}
.masthead .orn::before,.masthead .orn::after{content:""; height:1px; width:min(120px,18vw); background:var(--gold);}
.masthead .orn b{width:9px; height:9px; background:var(--gold); transform:rotate(45deg);}
.masthead h1{font-family:'Cormorant Garamond',serif; font-style:italic; font-weight:600; margin:14px 0 6px; padding:0;
  font-size:clamp(3rem,9vw,6.2rem); line-height:1; letter-spacing:.01em;
  background:linear-gradient(180deg,#fbeab9 10%,#d9b060 55%,#a67c2e 100%); -webkit-background-clip:text; background-clip:text; color:transparent;}
.masthead p{margin:0 auto 14px; max-width:31em; font-size:1.08rem; line-height:1.6; color:var(--muted);}

/* ---------- section titles ---------- */
.sec{display:flex; align-items:center; gap:18px; margin:3.2rem 0 .35rem;}
.sec span{font-family:'Cormorant Garamond',serif; font-style:italic; font-weight:600; font-size:2.15rem; line-height:1.1; color:var(--gold-hi); white-space:nowrap;}
.sec::after{content:""; flex:1; height:1px; background:linear-gradient(90deg,var(--gold),rgba(201,164,92,0));}
.sec-sub{margin:0 0 1.5rem; color:var(--muted); font-size:.98rem;}

/* ---------- widgets ---------- */
.stSelectbox label p{color:var(--muted)!important; font-size:.95rem;}
div[data-baseweb="select"] > div{background:#17110f!important; border:1px solid var(--gold-dim)!important; border-radius:2px!important; color:var(--ivory)!important;}
.stButton > button{background:transparent; color:var(--gold-hi); border:1px solid var(--gold); border-radius:2px;
  font-family:'Cormorant Garamond',serif; font-style:italic; font-weight:700; font-size:1.2rem; padding:.45rem 1.2rem;
  transition:background .2s ease,color .2s ease;}
.stButton > button:hover{background:var(--gold); color:#1a0f08; border-color:var(--gold);}
.stButton > button:focus-visible{outline:2px solid var(--gold-hi); outline-offset:3px;}

/* ---------- viewer ledger ---------- */
.ledger{display:grid; grid-template-columns:1.5fr repeat(4,1fr); margin-top:1.6rem;
  border-top:1px solid var(--gold); border-bottom:1px solid var(--gold);}
.ledger > div{padding:18px 22px; border-left:1px solid var(--gold-dim);}
.ledger > div:first-child{border-left:0; padding-left:4px;}
.ledger .name{font-family:'Cormorant Garamond',serif; font-style:italic; font-weight:600; font-size:2.6rem; line-height:1.05; color:var(--ivory);}
.ledger small{display:block; color:var(--muted); font-size:.85rem; margin-bottom:2px;}
.ledger b{display:block; font-family:'Cormorant Garamond',serif; font-weight:700; font-size:1.9rem; line-height:1.15; color:var(--gold-hi);}
.ledger span{color:var(--muted); font-size:.88rem;}
.seen{margin-top:16px; display:flex; flex-wrap:wrap; gap:8px; align-items:center;}
.seen em{font-family:'Cormorant Garamond',serif; color:var(--muted); font-size:1.1rem; margin-right:4px;}
.seen span{border:1px solid var(--gold-dim); padding:3px 12px; font-size:.9rem;}
.seen span i{font-style:normal; color:var(--gold-hi); margin-left:6px; font-weight:500;}

/* ---------- posters ---------- */
.poster{position:relative; aspect-ratio:3/4; overflow:hidden; border-radius:2px;
  box-shadow:0 22px 44px rgba(0,0,0,.6), 0 0 0 1px var(--gold-dim);}
.poster .sun{position:absolute; left:50%; top:60%; width:54%; aspect-ratio:1; transform:translate(-50%,-50%); border-radius:50%;
  background:radial-gradient(circle at 50% 38%,#fdf0c4,#dcb262 58%,#a67c2e); box-shadow:0 0 46px rgba(240,205,120,.55);}
.poster .horizon{position:absolute; left:0; right:0; bottom:0; height:40%; border-top:2px solid var(--gold);}
.poster .frame{position:absolute; inset:9px; border:1px solid rgba(201,164,92,.85); pointer-events:none;}
.poster .frame::after{content:""; position:absolute; inset:4px; border:1px solid rgba(201,164,92,.35);}
.poster .num{position:absolute; top:20px; left:0; right:0; text-align:center; font-family:'Cormorant Garamond',serif; font-weight:700; font-size:1.55rem; color:var(--gold-hi); letter-spacing:.12em;}
.poster .cap{position:absolute; left:18px; right:18px; bottom:20px; text-align:center; font-family:'Cormorant Garamond',serif; font-style:italic; font-weight:700;
  color:var(--ivory); font-size:clamp(1.15rem,2.4vw,1.6rem); line-height:1.05; text-shadow:0 2px 10px rgba(0,0,0,.7);}

/* ---------- feature presentation ---------- */
.feature{display:grid; grid-template-columns:minmax(220px,290px) 1fr; gap:44px; align-items:center;}
.feature h3{font-family:'Cormorant Garamond',serif; font-style:italic; font-weight:700; margin:0 0 6px; padding:0;
  font-size:clamp(2.4rem,5vw,3.8rem); line-height:1; color:var(--ivory);}
.feature .lead{color:var(--gold); font-family:'Cormorant Garamond',serif; font-style:italic; font-size:1.25rem; margin-bottom:6px;}
.facts{display:flex; gap:0; margin:22px 0; border-top:1px solid var(--gold-dim); border-bottom:1px solid var(--gold-dim);}
.facts > div{flex:1; padding:14px 18px; border-left:1px solid var(--gold-dim);}
.facts > div:first-child{border-left:0; padding-left:0;}
.facts b{display:block; font-family:'Cormorant Garamond',serif; font-weight:700; font-size:1.9rem; color:var(--ivory); line-height:1.1;}
.facts span{color:var(--muted); font-size:.86rem;}
.pred{display:flex; align-items:baseline; gap:14px; flex-wrap:wrap;}
.pred .pn{font-family:'Cormorant Garamond',serif; font-weight:700; font-size:3.4rem; line-height:1; color:var(--gold-hi);}
.pred .pl{color:var(--muted); font-size:.95rem;}
.gauge{height:3px; background:rgba(201,164,92,.18); margin:10px 0 14px; max-width:420px;}
.gauge i{display:block; height:100%; background:linear-gradient(90deg,var(--gold),var(--gold-hi));}
.why{margin:0; color:var(--muted); font-size:.98rem;}

/* ---------- shelf of the other picks ---------- */
.shelf{display:grid; grid-template-columns:repeat(4,1fr); gap:26px; margin-top:3rem;}
.pcard .pn{font-family:'Cormorant Garamond',serif; font-weight:700; font-size:1.8rem; color:var(--gold-hi); line-height:1; margin-top:16px;}
.pcard .pn small{font-family:'Jost',sans-serif; font-weight:300; font-size:.85rem; color:var(--muted); margin-left:8px;}
.pcard .gauge{margin:8px 0 8px; max-width:none;}
.pcard .why{font-size:.9rem;}

/* ---------- credits (taste twins) ---------- */
.credits{max-width:560px;}
.cr{display:flex; align-items:baseline; margin:.75rem 0; font-family:'Cormorant Garamond',serif; font-size:1.5rem;}
.cr .nm{color:var(--ivory); font-style:italic; font-weight:600;}
.cr .dots{flex:1; margin:0 10px; border-bottom:2px dotted rgba(201,164,92,.5); transform:translateY(-5px);}
.cr .pc{color:var(--gold-hi); font-weight:700;}

/* ---------- genre bars ---------- */
.gbar{display:grid; grid-template-columns:130px 1fr 44px; gap:16px; align-items:center; margin:12px 0; max-width:640px;}
.gbar .nm{font-family:'Cormorant Garamond',serif; font-size:1.3rem; font-style:italic; font-weight:600;}
.gbar .t{height:3px; background:rgba(201,164,92,.18);}
.gbar .t i{display:block; height:100%; background:linear-gradient(90deg,var(--gold),var(--gold-hi));}
.gbar em{font-style:normal; font-family:'Cormorant Garamond',serif; font-weight:700; font-size:1.3rem; color:var(--gold-hi); text-align:right;}

.empty{border:1px dashed var(--gold-dim); padding:28px; text-align:center; color:var(--muted); font-family:'Cormorant Garamond',serif; font-size:1.3rem; font-style:italic;}

@media (max-width:900px){
  .shelf{grid-template-columns:repeat(2,1fr);}
  .ledger{grid-template-columns:1fr 1fr;}
  .ledger > div{border-top:1px solid var(--gold-dim);}
  .ledger > div:first-child{grid-column:1 / -1; border-top:0;}
  .ledger > div:nth-child(2n){border-left:0;}
}
@media (max-width:720px){
  .feature{grid-template-columns:1fr; gap:26px;}
  .feature .poster{width:100%; max-width:300px; margin:0 auto;}
  .facts b{font-size:1.5rem;}
  .gbar{grid-template-columns:96px 1fr 40px; gap:10px;}
  .sec span{font-size:1.75rem;}
}
@media (prefers-reduced-motion:reduce){
  .curtain{display:none;}
}
"""


# --------------------------------------------------------------------------
# 4. Design: HTML builders
# --------------------------------------------------------------------------
ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]


def flat(markup: str) -> str:
    """One line of HTML, so Streamlit's markdown never mistakes indented HTML for a code block."""
    return " ".join(line.strip() for line in markup.splitlines() if line.strip())


def esc(x) -> str:
    return html.escape(str(x))


def hue_of(title: str) -> int:
    return int(hashlib.md5(title.encode()).hexdigest()[:4], 16) % 360


def pct(value, scale=1.0) -> int:
    return int(max(0, min(100, value * scale)))


def masthead_html() -> str:
    return flat("""
    <div class="curtain" aria-hidden="true"><i class="l"></i><i class="r"></i></div>
    <div class="masthead">
      <div class="orn"><b></b></div>
      <h1>Reel Match</h1>
      <p>An evening's programme, chosen for you by the viewers who share your taste.</p>
      <div class="orn"><b></b></div>
    </div>""")


def sec_html(title: str, sub: str = "") -> str:
    out = f'<div class="sec"><span>{esc(title)}</span></div>'
    if sub:
        out += f'<p class="sec-sub">{esc(sub)}</p>'
    return flat(out)


def poster_html(title, caption, numeral) -> str:
    h = hue_of(title)
    rays = (f"repeating-conic-gradient(from -90deg at 50% 60%,"
            f"hsl({h} 48% 27%) 0 5deg,hsl({h} 52% 18%) 5deg 10deg)")
    horizon = f"linear-gradient(180deg,hsl({h} 40% 10%),#0b0807)"
    return flat(f"""
    <div class="poster" style="background:{rays}">
      <div class="sun"></div>
      <div class="horizon" style="background:{horizon}"></div>
      <div class="frame"></div>
      <div class="num">{esc(numeral)}</div>
      <div class="cap">{esc(caption)}</div>
    </div>""")


def na(x, fmt="{:.0f}") -> str:
    return "n/a" if pd.isna(x) else fmt.format(x)


def viewer_html(uid, df, watched) -> str:
    mine = df[df[USER] == uid]
    genre_avg = mine.groupby(GENRE)[RATING].mean().sort_values(ascending=False)
    best_genre = genre_avg.index[0] if len(genre_avg) else "Unknown"
    fav_lang = mine[LANG].mode().iloc[0] if len(mine) else "Unknown"
    best_genre = "n/a" if best_genre == "Unknown" else best_genre
    fav_lang = "n/a" if fav_lang == "Unknown" else fav_lang
    avg_given = f"{watched.mean():.1f}" if len(watched) else "n/a"

    seen = "".join(f"<span>{esc(t)}<i>{r:.1f}</i></span>" for t, r in watched.items())
    return flat(f"""
    <div class="ledger">
      <div><small>Now showing to</small><div class="name">Viewer {esc(uid)}</div></div>
      <div><b>{len(watched)}</b><span>films rated</span></div>
      <div><b>{avg_given}</b><span>average rating given</span></div>
      <div><b>{esc(best_genre)}</b><span>best-rated genre</span></div>
      <div><b>{esc(fav_lang)}</b><span>most watched language</span></div>
    </div>
    <div class="seen"><em>Already seen</em>{seen}</div>""")


def feature_html(title, score, backers, meta, scale) -> str:
    m = meta.loc[title]
    return flat(f"""
    <div class="feature">
      {poster_html(title, "The top pick", ROMAN[0])}
      <div>
        <div class="lead">Your feature presentation</div>
        <h3>{esc(title)}</h3>
        <div class="facts">
          <div><b>{m.avg_rating:.1f}</b><span>audience rating</span></div>
          <div><b>{na(m.runtime)}</b><span>minutes</span></div>
          <div><b>{na(m.year)}</b><span>released around</span></div>
        </div>
        <div class="pred"><div class="pn">{score:.1f}</div><div class="pl">predicted rating for you (out of {scale:g})</div></div>
        <div class="gauge"><i style="width:{pct(score, 100 / scale)}%"></i></div>
        <p class="why">Recommended by {int(backers)} of your {TOP_USERS} taste twins.</p>
      </div>
    </div>""")


def shelf_html(rows, scale) -> str:
    cards = ""
    for rank, title, score, backers in rows:
        cards += f"""
        <div class="pcard">
          {poster_html(title, title, ROMAN[rank - 1])}
          <div class="pn">{score:.1f}<small>predicted</small></div>
          <div class="gauge"><i style="width:{pct(score, 100 / scale)}%"></i></div>
          <p class="why">{int(backers)} of {TOP_USERS} taste twins recommend it.</p>
        </div>"""
    return flat(f'<div class="shelf">{cards}</div>')


def twins_html(twins) -> str:
    rows = "".join(
        f'<div class="cr"><span class="nm">Viewer {esc(u)}</span><span class="dots"></span>'
        f'<span class="pc">{pct(s, 100)}%</span></div>'
        for u, s in twins.items()
    )
    return flat(f'<div class="credits">{rows}</div>')


def genre_bars_html(uid, df, scale) -> str:
    g = df[df[USER] == uid].groupby(GENRE)[RATING].mean().sort_values(ascending=False)
    if g.empty or set(g.index) == {"Unknown"}:
        return flat('<div class="empty">This dataset has no genre column.</div>')
    rows = "".join(
        f'<div class="gbar"><span class="nm">{esc(name)}</span>'
        f'<div class="t"><i style="width:{pct(val, 100 / scale)}%"></i></div><em>{val:.1f}</em></div>'
        for name, val in g.items()
    )
    return flat(f"<div>{rows}</div>")


# --------------------------------------------------------------------------
# 5. Streamlit page
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def cached_raw(src, filename, sheet):
    return read_raw(src, filename, sheet)


@st.cache_data(show_spinner=False)
def cached_sheets(data: bytes):
    return pd.ExcelFile(io.BytesIO(data)).sheet_names


@st.cache_data(show_spinner=False)
def cached_prepare(raw, mapping_items):
    return prepare(raw, dict(mapping_items))


def sidebar_dataset():
    """Returns (raw_dataframe, mapping) for the sample file or an uploaded one."""
    with st.sidebar:
        st.markdown("**Dataset**")
        uploaded = st.file_uploader("Test another Excel or CSV file", type=["xlsx", "csv"])

        if uploaded is None:
            st.caption("Using the sample dataset.")
            if not Path(DATA_PATH).exists():
                return None, f"Could not find {DATA_PATH}. Put the Excel file there, or upload one in the sidebar."
            raw = cached_raw(DATA_PATH, DATA_PATH, SHEET)
            return raw, {c: c for c in REQUIRED + OPTIONAL}

        data, name = uploaded.getvalue(), uploaded.name
        sheet = None
        if not name.lower().endswith(".csv"):
            sheets = cached_sheets(data)
            default = sheets.index(SHEET) if SHEET in sheets else 0
            sheet = st.selectbox("Sheet", sheets, index=default, key=f"sheet_{name}") if len(sheets) > 1 else sheets[0]
        raw = cached_raw(data, name, sheet)

        st.markdown("**Match your columns**")
        cols = list(raw.columns)
        mapping = {}
        for canon in REQUIRED + OPTIONAL:
            guess = guess_column(cols, canon)
            options = cols if canon in REQUIRED else ["(none)"] + cols
            index = options.index(guess) if guess in options else 0
            pick = st.selectbox(LABELS[canon], options, index=index, key=f"map_{canon}_{name}_{sheet}")
            mapping[canon] = None if pick == "(none)" else pick
        return raw, mapping


def main():
    st.set_page_config(page_title="Reel Match", page_icon="🎞️", layout="wide")
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)
    st.markdown(masthead_html(), unsafe_allow_html=True)      # always shown, even if the data fails

    try:
        raw, mapping = sidebar_dataset()
    except Exception as exc:
        st.error(f"Could not open that file: {exc}")
        st.stop()
    if raw is None:
        st.error(mapping)
        st.stop()

    if len({mapping[c] for c in REQUIRED}) < len(REQUIRED) or any(mapping[c] is None for c in REQUIRED):
        st.warning("Pick three different columns in the sidebar for user ID, movie title and rating.")
        st.stop()

    try:
        df, matrix, sim_df, meta, scale = cached_prepare(raw, tuple(mapping.items()))
    except Exception as exc:
        st.error(f"This dataset could not be used: {exc}")
        st.stop()

    ids = matrix.index.tolist()
    if st.session_state.get("uid") not in ids:
        st.session_state["uid"] = ids[0]

    def surprise():
        st.session_state["uid"] = random.choice(ids)

    c1, c2 = st.columns([3, 1.2], vertical_alignment="bottom")
    with c1:
        uid = st.selectbox("Choose a viewer", ids, key="uid")
    with c2:
        st.button("Surprise me", on_click=surprise, use_container_width=True)

    twins, watched, recs = recommend(uid, matrix, sim_df)

    st.markdown(viewer_html(uid, df, watched), unsafe_allow_html=True)

    st.markdown(sec_html("Tonight's programme",
                         "Films this viewer has not rated yet, ranked by what their taste twins gave them."),
                unsafe_allow_html=True)
    if recs.empty:
        st.markdown('<div class="empty">This viewer has already rated everything their twins rated. '
                    'Choose another viewer.</div>', unsafe_allow_html=True)
    else:
        first = recs.iloc[0]
        st.markdown(feature_html(recs.index[0], first.score, first.backers, meta, scale), unsafe_allow_html=True)
        rest = [(i + 2, t, r.score, r.backers) for i, (t, r) in enumerate(recs.iloc[1:].iterrows())]
        if rest:
            st.markdown(shelf_html(rest, scale), unsafe_allow_html=True)

    st.markdown(sec_html("Viewers who share their taste",
                         f"The {len(twins)} viewers whose ratings are closest to this one, by cosine similarity."),
                unsafe_allow_html=True)
    st.markdown(twins_html(twins), unsafe_allow_html=True)

    st.markdown(sec_html("What this viewer rates highest", "Average rating given, by genre."),
                unsafe_allow_html=True)
    st.markdown(genre_bars_html(uid, df, scale), unsafe_allow_html=True)

    st.write("")
    with st.expander("How the picks are made"):
        st.markdown(
            "1. Every viewer's ratings become a row of numbers (missing = 0).\n"
            "2. Cosine similarity finds the 5 viewers closest to the chosen one.\n"
            "3. Their ratings are averaged per film, films already rated are removed, "
            "and the top 5 become the programme."
        )


if __name__ == "__main__":
    main()