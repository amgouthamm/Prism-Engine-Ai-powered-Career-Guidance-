"""PRISM Engine backend: Flask + scikit-learn + Supabase.
SETUP : pip install -r requirements.txt
        copy .env.example to .env and fill in your Supabase details
RUN   : python app.py      then open http://localhost:5000   (index.html must sit next to this file)
"""
import os, re
import numpy as np, requests
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

load_dotenv()
URL = os.getenv("SUPABASE_URL", "").rstrip("/")
KEY = os.getenv("SUPABASE_KEY", "")
STUDENT_TABLE = os.getenv("STUDENT_TABLE", "students")
MARKET_TABLE = os.getenv("MARKET_TABLE", "market")
HERE = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)

SUBJ = ["maths", "physics", "chemistry", "biology", "cs", "humanities", "commerce"]
# fee multiplier per state (the base fee below is multiplied by this)
STATES = {"Kerala": 1.0, "Karnataka": 1.15, "Tamil Nadu": 1.1, "Telangana": 1.05, "Maharashtra": 1.2, "Andhra Pradesh": 0.95}

def C(w, cut, rng):  # w = subject weights (sum 1), cut = cutoff 0-1, rng = (min, max) fee in Rs lakh for 4 years
    return {"w": w, "cut": cut, "fee": rng}

# ---- EDIT ME: placeholder cutoffs / fees / market values. Replace with your real numbers. ----
COURSES = {
    "CS - AI and ML": C({"maths": .4, "cs": .4, "physics": .2}, .55, (8, 24)),
    "CS - AI and Robotics": C({"maths": .35, "cs": .35, "physics": .3}, .55, (8, 24)),
    "CS - Data Science": C({"maths": .4, "cs": .4, "commerce": .2}, .50, (7, 22)),
    "CS - Cybersecurity": C({"cs": .5, "maths": .3, "physics": .2}, .50, (7, 22)),
    "Mechanical Engineering": C({"maths": .35, "physics": .45, "chemistry": .2}, .50, (5, 18)),
    "Civil Engineering": C({"maths": .35, "physics": .4, "chemistry": .25}, .45, (4, 15)),
    "Electrical Engineering": C({"maths": .35, "physics": .5, "cs": .15}, .50, (5, 18)),
    "Chemical Engineering": C({"chemistry": .45, "maths": .3, "physics": .25}, .50, (5, 18)),
    "Environmental Engineering": C({"chemistry": .3, "biology": .3, "maths": .2, "physics": .2}, .45, (4, 14)),
    "Fine Arts": C({"humanities": .7, "biology": .2, "commerce": .1}, .40, (2, 8)),
    "Architecture": C({"maths": .3, "physics": .2, "humanities": .3, "cs": .2}, .50, (6, 20)),
    "Design": C({"humanities": .5, "cs": .3, "maths": .2}, .45, (6, 20)),
    "Media Communication": C({"humanities": .6, "commerce": .2, "cs": .2}, .40, (3, 12)),
    "BSc Physics": C({"physics": .55, "maths": .45}, .50, (1.5, 6)),
    "BSc Chemistry": C({"chemistry": .6, "physics": .2, "maths": .2}, .45, (1.5, 6)),
    "BSc Biology": C({"biology": .7, "chemistry": .3}, .45, (1.5, 6)),
    "BSc Maths": C({"maths": .8, "cs": .2}, .50, (1.5, 6)),
    "BSc Environmental Science": C({"biology": .4, "chemistry": .3, "humanities": .3}, .40, (1.5, 6)),
}
LABELS = list(COURSES)

# ---- EDIT ME: placeholder market estimates = (median starting salary in Rs LPA, demand growth % per year) ----
SAL = {"CS - AI and ML": (9, 22), "CS - AI and Robotics": (8, 20), "CS - Data Science": (8, 21), "CS - Cybersecurity": (7.5, 18),
       "Mechanical Engineering": (4.5, 6), "Civil Engineering": (4, 7), "Electrical Engineering": (5, 8), "Chemical Engineering": (4.5, 7),
       "Environmental Engineering": (4, 10), "Fine Arts": (3, 6), "Architecture": (4.5, 7), "Design": (5, 12), "Media Communication": (3.5, 8),
       "BSc Physics": (3.5, 5), "BSc Chemistry": (3.5, 5), "BSc Biology": (3.2, 6), "BSc Maths": (4, 8), "BSc Environmental Science": (3.2, 9)}
SAL_ST = {"Kerala": .92, "Karnataka": 1.2, "Tamil Nadu": 1.05, "Telangana": 1.12, "Maharashtra": 1.15, "Andhra Pradesh": .95}
GROW_ST = {"Kerala": .9, "Karnataka": 1.15, "Tamil Nadu": 1.05, "Telangana": 1.1, "Maharashtra": 1.05, "Andhra Pradesh": 1.0}
MKT = {}  # (course, state) -> numbers from your Supabase market table; they override the estimates above

def fee(c, st):  # (min, max) 4-year fee in Rs lakh
    o, base = MKT.get((c, st), {}), COURSES[c]["fee"]
    return (o.get("fee_min") or round(base[0] * STATES[st], 1), o.get("fee_max") or round(base[1] * STATES[st], 1))

def market(c, st):  # median salary (LPA) and yearly growth (%) for this course in this state
    o = MKT.get((c, st), {})
    return dict(salary=o.get("salary") or round(SAL[c][0] * SAL_ST[st], 1), growth=o.get("growth") or round(SAL[c][1] * GROW_ST[st], 1))

def msc(c, st):  # market score 0-1 used in ranking
    m = market(c, st)
    return min(1, m["salary"] / 12) * .5 + min(1, m["growth"] / 30) * .5

# ---- EDIT ME: extra ranking weight a state gives to a course group (0.08 = a clear push, 0.15 = strong; negative lowers it).
# Groups: "Engineering" (CS and all engineering), "BSc", "Arts and Design". These starting values are a guess, change them freely.
GROUP_BOOST = {"Kerala": {}, "Karnataka": {"Engineering": .08}, "Tamil Nadu": {"Engineering": .08},
               "Telangana": {"Engineering": .08}, "Maharashtra": {}, "Andhra Pradesh": {"Engineering": .08}}

def group(c):
    return "BSc" if c.startswith("BSc") else "Engineering" if c.startswith("CS -") or c.endswith("Engineering") else "Arts and Design"

def boost(c, st):
    return GROUP_BOOST.get(st, {}).get(group(c), 0)

def load_market():
    MKT.clear()
    try:
        norm = lambda v: re.sub(r"[^a-z0-9]", "", str(v).lower())
        cn, sn = {norm(c): c for c in COURSES}, {norm(s): s for s in STATES}
        num = lambda v: float(v) if v not in (None, "") else None
        for r in fetch(MARKET_TABLE):
            low = {k.lower(): v for k, v in r.items()}
            get = lambda *ks: next((v for k, v in low.items() if any(x in k for x in ks)), None)
            c, s = cn.get(norm(get("course"))), sn.get(norm(get("state")))
            if c and s:
                MKT[(c, s)] = dict(salary=num(get("salary")), growth=num(get("growth")), fee_min=num(get("min")), fee_max=num(get("max")))
        STATE["notes"].append(f"market table: {len(MKT)} course-state rows loaded")
    except Exception as e:
        STATE["notes"].append(f"market table not used, built-in estimates instead: {e}")
W = np.array([[d["w"].get(s, 0) for s in SUBJ] for d in COURSES.values()])  # course x subject weights

# ---------------- Supabase loading ----------------
SUB_TOK = {"maths": {"maths", "math", "mathematics"}, "physics": {"physics", "phy"}, "chemistry": {"chemistry", "chem"},
           "biology": {"biology", "bio"}, "cs": {"cs", "computer", "comp"}, "humanities": {"humanities", "human", "hum"},
           "commerce": {"commerce", "com"}}
ATT_TOK = {"mark": {"mark", "marks", "score"}, "skill": {"skill", "skills"}, "interest": {"interest", "interests"}}
LABEL_COLS = {"course", "recommended_course", "best_course", "selected_course", "chosen_course", "label", "target"}

def fetch(table):
    rows, start = [], 0
    while True:  # Supabase returns max 1000 rows per request, so page through
        r = requests.get(f"{URL}/rest/v1/{table}?select=*", timeout=30,
                         headers={"apikey": KEY, "Range-Unit": "items", "Range": f"{start}-{start + 999}"})
        r.raise_for_status()
        chunk = r.json()
        rows += chunk
        if len(chunk) < 1000:
            return rows
        start += 1000

def tokens(col):
    return set(re.findall(r"[a-z]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", col).lower()))

def map_columns(cols):
    m = {}
    for s, st in SUB_TOK.items():
        for a, at in ATT_TOK.items():
            for c in cols:
                if tokens(c) & st and tokens(c) & at:
                    m[(s, a)] = c
                    break
    return m

def load_students():
    rows = fetch(STUDENT_TABLE)
    if not rows:
        raise RuntimeError("0 rows returned: check STUDENT_TABLE and that a row-level-security SELECT policy allows the publishable key")
    cols = list(rows[0])
    m = map_columns(cols)
    miss = [f"{s}_{a}" for s in SUBJ for a in ATT_TOK if (s, a) not in m]
    if miss:
        raise RuntimeError(f"no column found for {miss}; table columns are {cols}")
    num = lambda v: float(v) if v not in (None, "") else 0.0
    X = np.clip(np.array([[.4 * num(r[m[(s, "mark")]]) / 100 + .3 * num(r[m[(s, "skill")]]) / 3 + .3 * num(r[m[(s, "interest")]]) / 100
                           for s in SUBJ] for r in rows]), 0, 1)
    y, note = rule_labels(X), "no usable course-label column: model learns the rule-based best course"
    lc = next((c for c in cols if c.lower() in LABEL_COLS), None)
    if lc:
        norm = lambda v: re.sub(r"[^a-z0-9]", "", str(v).lower())
        names = {norm(c): c for c in COURSES}
        lab = np.array([names.get(norm(r[lc])) for r in rows], dtype=object)
        ok = np.array([v is not None for v in lab])
        if ok.mean() > .5:
            X, y, note = X[ok], lab[ok], f"trained on your '{lc}' column"
    return X, y, len(rows), note

def rule_labels(X):
    return np.array([LABELS[i] for i in (X @ W.T).argmax(1)], dtype=object)

def demo():
    rng = np.random.default_rng(0)
    X = rng.beta(2, 2, (4000, 7))
    X[np.arange(4000), rng.integers(0, 7, 4000)] += .3
    X = np.clip(X, 0, 1)
    return X, rule_labels(X), 4000, "demo data"

# ---------------- Training ----------------
STATE = {"model": None, "source": "none", "rows": 0, "acc": None, "notes": []}

def init():
    STATE["notes"] = []
    try:
        if not (URL and KEY):
            raise RuntimeError("SUPABASE_URL / SUPABASE_KEY missing in .env")
        X, y, n, note = load_students()
        STATE["source"] = "supabase"
    except Exception as e:
        STATE["notes"].append(f"Supabase not used, running on demo data: {e}")
        X, y, n, note = demo()
        STATE["source"] = "demo"
    STATE["notes"].append(note)
    load_market() if (URL and KEY) else MKT.clear()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=1)
    clf = RandomForestClassifier(n_estimators=150, min_samples_leaf=2, n_jobs=-1, random_state=1).fit(Xtr, ytr)
    STATE["acc"] = round(float(clf.score(Xte, yte)), 3)
    STATE["model"] = clf.fit(X, y)
    STATE["rows"] = n
    print(f"[PRISM] source={STATE['source']} rows={n} accuracy={STATE['acc']} | {STATE['notes']}")

# ---------------- PRISM rules ----------------
def parent_risk(total, loan, p):
    """Parent risk factor from loan share, repayment burden and fee headroom for the top match p."""
    if not p:
        return dict(level="High", points=6, lines=["The total budget is below the minimum 4-year fee of every shortlisted course in every state."],
                    advice="Raise the budget or loan, look for scholarships, or choose lower-fee courses.")
    c, st, sal = p["course"], p["state"], p["salary"]
    share, years = loan / total, loan / sal
    fee_pts = 0 if total >= p["fee_max"] else 1 if total >= (p["fee_min"] + p["fee_max"]) / 2 else 2
    pts = (0 if share < .2 else 1 if share < .4 else 2) + (0 if years < .75 else 1 if years < 1.5 else 2) + fee_pts
    level = "Low" if pts <= 1 else "Moderate" if pts <= 3 else "High"
    burden = f", about {years:.1f} years of the median starting salary (₹{sal} LPA)." if loan else ". No loan, so no repayment burden."
    cover = ["The budget covers even the top-end fee (₹%g L)." % p["fee_max"],
             "The budget covers the typical fee but not the top-end fee (₹%g L), so a fee rise needs extra funding." % p["fee_max"],
             "The budget sits close to the minimum fee (₹%g L), so any fee rise needs extra funding." % p["fee_min"]][fee_pts]
    advice = {"Low": "Finances look comfortable. Keep a buffer for living costs.",
              "Moderate": "Manageable, but compare scholarships and education-loan interest rates before committing.",
              "High": "High exposure. Consider lower-fee states, scholarships or a larger budget before committing."}[level]
    return dict(level=level, points=pts, lines=[f"Based on the top match: {c} in {st}.", f"The loan is {share:.0%} of the ₹{total:g} L total budget{burden}", cover], advice=advice)

def recommend(d):
    sub = d["subjects"]
    comp = {s: .4 * float(sub[s]["mark"]) / 100 + .3 * float(sub[s]["skill"]) / 3 + .3 * float(sub[s]["interest"]) / 100 for s in SUBJ}
    x = np.array([[comp[s] for s in SUBJ]])
    R = dict(zip(LABELS, (x @ W.T)[0]))                       # rule score per course
    P = dict.fromkeys(LABELS, 0.0)
    P.update(zip(STATE["model"].classes_, STATE["model"].predict_proba(x)[0]))  # ML probability per course
    lo, hi, pm = min(R.values()), max(R.values()), max(P.values()) or 1
    S = {c: .7 * (R[c] - lo) / ((hi - lo) or 1) + .3 * P[c] / pm for c in LABELS}
    top = sorted(LABELS, key=lambda c: (R[c] >= COURSES[c]["cut"], S[c]), reverse=True)[:10]  # best 10 courses

    total = float(d["budget"]) + float(d["loan"])
    afford = {c: [st for st in STATES if total >= fee(c, st)[0]] for c in top}
    sp, pp = d.get("sp") or None, d.get("pp") or None
    same = bool(sp and sp == pp)

    states = []
    for st in STATES:
        ok = sorted([c for c in top if st in afford[c]], key=lambda c: S[c] + .15 * msc(c, st) + boost(c, st), reverse=True)
        if same and sp in ok and ok.index(sp) > 0:            # sp == pp: move up one place
            i = ok.index(sp)
            ok[i - 1], ok[i] = ok[i], ok[i - 1]
        rows = []
        for c in ok + [c for c in top if st not in afford[c]]:
            lo_, hi_ = fee(c, st)
            if c in ok:
                status, note = "ok", ""
            elif not afford[c]:
                status, note = "financial", "Financial instability: total budget is below the minimum fee in every state"
            else:
                status, note = "blocked", f"Total budget is below the minimum fee here. Affordable in: {', '.join(afford[c])}"
            gaps = [s for s, w in COURSES[c]["w"].items() if w >= .3 and comp[s] < COURSES[c]["cut"]]
            rows.append(dict(rank=len(rows) + 1, course=c, score=round(100 * min(1, S[c]), 1), eligible=bool(R[c] >= COURSES[c]["cut"]),
                             fee_min=lo_, fee_max=hi_, **market(c, st), boost=boost(c, st), status=status, note=note, gaps=gaps,
                             boosted=bool(same and c == sp)))
        states.append(dict(state=st, rows=rows, best=rows[0]["score"] if rows[0]["status"] == "ok" else 0))

    pref = list(dict.fromkeys(s for s in (d.get("student_state"), d.get("parent_state")) if s in STATES))
    states.sort(key=lambda o: (pref.index(o["state"]) if o["state"] in pref else len(pref), -o["best"]))  # preferred states first
    for o in states:
        o["preferred"] = o["state"] in pref
    ix = lambda c: top.index(c) if c in top else 10
    conflict = round(abs(ix(sp) - ix(pp)) / 10, 2) if sp and pp and not same else 0
    picks = sorted(((S[c] + .15 * msc(c, st) + boost(c, st) + (.05 if same and c == sp else 0), c, st) for c in top for st in afford[c]), reverse=True)[:5]
    picks = [dict(course=c, state=st, score=round(100 * min(1, v), 1), fee_min=fee(c, st)[0], fee_max=fee(c, st)[1], **market(c, st))
             for v, c, st in picks]  # best course + location pairs across all states
    return dict(total_budget=total, composite={s: round(comp[s] * 100, 1) for s in SUBJ}, match=same, conflict=conflict,
                states=states, picks=picks, risk=parent_risk(total, float(d["loan"]), picks[0] if picks else None), market_source="your Supabase market table" if MKT else "built-in estimates",
                source=STATE["source"], accuracy=STATE["acc"])

# ---------------- Routes ----------------
@app.get("/")
def home():
    return send_from_directory(HERE, "index.html")

@app.get("/api/meta")
def meta():
    return jsonify(subjects=SUBJ, states=list(STATES), courses=LABELS, source=STATE["source"], rows=STATE["rows"],
                   accuracy=STATE["acc"], notes=STATE["notes"])

@app.post("/api/recommend")
def api_recommend():
    try:
        return jsonify(recommend(request.get_json(force=True)))
    except (KeyError, ValueError, TypeError) as e:
        return jsonify(error=f"Invalid input: {e}"), 400

@app.post("/api/retrain")
def retrain():
    init()
    return meta()

if __name__ == "__main__":
    init()
    app.run(host="127.0.0.1", port=5000, debug=False)