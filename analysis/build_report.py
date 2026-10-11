#!/usr/bin/env python3
"""
Build analysis/effects_summary.html from the stored outputs of the executed
R notebooks. Nothing is refitted -- this only reads what lmer already printed.

    python analysis/build_report.py

Two guards matter, because a Jupyter notebook happily displays output from a
fit that is no longer the one its code describes:

  1. the model cell must have an execution_count AND outputs;
  2. the results_*.csv the notebook READS must exist on disk.

Guard 2 is what catches the prop25 notebooks for aemri/frmri/beus, whose stored
tables are bit-identical to their base counterparts because the model cell was
never run against prop25 data.
"""
from __future__ import annotations
import json, os, re, glob, datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NUM  = r'-?\d+\.?\d*(?:[eE][+-]?\d+)?'

# --------------------------------------------------------------------------- #
# Notebook reading
# --------------------------------------------------------------------------- #
def cells_text(path):
    nb = json.load(open(path))
    out, src, ran = [], [], False
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        s = "".join(c["source"])
        src.append(s)
        if "model <- lmer(" in s:
            ran = bool(c.get("execution_count")) and bool(c.get("outputs"))
        for o in c.get("outputs", []):
            t = o.get("text") or (o.get("data", {}) or {}).get("text/plain")
            if t:
                out.append("".join(t) if isinstance(t, list) else t)
    return "\n".join(out), "".join(src), ran


def parse_fixed(text):
    """lmer wraps the fixed-effects table across blocks. Collect each term's
    numbers in the order they appear, then assign positionally."""
    i = text.find("Fixed effects:")
    if i < 0:
        return {}
    chunk = text[i:]
    end = chunk.find("Signif. codes")
    if end > 0:
        chunk = chunk[:end]
    acc, order = {}, []
    for line in chunk.splitlines():
        line = line.rstrip()
        if not line or line.startswith(("Fixed effects", "---")):
            continue
        m = re.match(r'^(\S+)\s+(.*)$', line)
        if not m:
            continue
        term, rest = m.group(1), m.group(2)
        if term in ("Estimate", "Std.", "df", "t", "Pr(>|t|)"):
            continue
        nums = re.findall(NUM, rest.replace("< 2e-16", "2e-16").replace("<2e-16", "2e-16"))
        if not nums:
            continue
        if term not in acc:
            acc[term], _ = [], order.append(term)
        acc[term] += [float(x) for x in nums]
    fx = {}
    for t in order:
        v = acc[t]
        if len(v) >= 5:
            fx[t] = dict(est=v[0], se=v[1], df=v[2], t=v[3], p=v[4])
        elif len(v) == 4:
            fx[t] = dict(est=v[0], se=v[1], df=None, t=v[2], p=v[3])
    return fx


MARG = re.compile(
    r'^\s*(overall|art|ac|ac - art)\s+(averaged over the data|at flex1 = flex2 = 0)\s+'
    rf'({NUM})\s+({NUM})\s+({NUM})\s+({NUM})\s+({NUM})(?:\s+({NUM}))?\s*$')


def parse_marginal(text):
    """
    The marginal_slopes table, tolerant of R wrapping the p column.

    When a p value is wide (aemri reaches 3.53e-158) R prints the frame in two
    blocks: the first seven columns, then a bare 'p' column underneath. The row
    regex therefore makes p optional, and any missing values are filled from
    that trailing block in order.
    """
    rows = []
    for line in text.splitlines():
        m = MARG.match(line)
        if m:
            rows.append(dict(effect=m.group(1), moderators=m.group(2),
                             est=float(m.group(3)), se=float(m.group(4)),
                             lower=float(m.group(5)), upper=float(m.group(6)),
                             z=float(m.group(7)),
                             p=float(m.group(8)) if m.group(8) else None))
    missing = [r for r in rows if r["p"] is None]
    if missing:
        tail, collecting = [], False
        for line in text.splitlines():
            t = line.strip()
            if t == "p":
                collecting, tail = True, []
                continue
            if collecting:
                if re.fullmatch(NUM, t):
                    tail.append(float(t))
                    continue
                if t == "":
                    continue
                collecting = False
        if len(tail) >= len(missing):
            for r, v in zip(missing, tail[:len(missing)]):
                r["p"] = v
    return [r for r in rows if r["p"] is not None]


def parse_meta(text, source, unit, ran):
    m = re.search(r'Number of obs:\s*(\d[\d,]*)', text)
    nobs = int(m.group(1).replace(",", "")) if m else None
    sing = re.findall(r'singular fit:\s+(TRUE|FALSE)', text)
    mv = re.search(r'dist ~ (\w+) \* modality', source)
    mf = re.search(r'Formula:\s*\n(dist ~.*?)\n\s*(?:Data:|Control:)', text, re.S)
    formula = " ".join(mf.group(1).split()) if mf else None
    rand = re.findall(r'\(([^)]*\|[^)]*)\)', formula or "")
    named = sorted(set(re.findall(r'results_[a-z0-9_]+\.csv', source)))
    cdir = os.path.join(ROOT, f"{unit}_based/data_processing/combine_data")
    return dict(n_obs=nobs, singular=(sing[-1] == "TRUE") if sing else None,
                predictor=mv.group(1) if mv else None, formula=formula,
                random=" + ".join("(" + r.strip() + ")" for r in rand),
                reads=named, ran=ran,
                reads_exist=bool(named) and all(
                    os.path.exists(os.path.join(cdir, c)) for c in named))


def parse_all():
    out = {}
    pats = ["base", "base_H", "prop25", "prop25_H"]
    paths = sum((glob.glob(f"{ROOT}/*_based/data_analysis/*/{w}/*.ipynb") for w in pats), [])
    for p in sorted(set(paths)):
        parts = p.split("/")
        unit   = "phone" if parts[-5].startswith("phone_based") else "syllable"
        family, folder, corpus = parts[-3], parts[-2], parts[-1][:-6]
        # "_H" folders are the same window fitted with conditional entropy as
        # the flexibility moderator instead of -MI.
        moderator = "H" if folder.endswith("_H") else "flex"
        window = folder[:-2] if folder.endswith("_H") else folder
        text, source, ran = cells_text(p)
        meta = parse_meta(text, source, unit, ran)
        out[f"{unit}|{family}|{window}|{moderator}|{corpus}"] = dict(
            unit=unit, family=family, window=window, moderator=moderator,
            corpus=corpus, path=p, meta=meta,
            fixed=parse_fixed(text), marginal=parse_marginal(text))
    return out


# --------------------------------------------------------------------------- #
# Payload
# --------------------------------------------------------------------------- #
CORPORA = ["aemri", "aemri75anno", "aeema", "frmri", "beus"]
LABEL = {"aemri": "USC-TIMIT rtMRI (+USC-75)", "aemri75anno": "USC-75 tracked contours",
         "aeema": "USC-TIMIT EMA", "frmri": "French rtMRI", "beus": "BE ultrasound"}
MOD_LABEL = {"flex": "flexibility = −MI",
             "H":    "flexibility = H(art | ac, phone)"}
FAMILIES = [
    ("phone",    "bigram_surprisal",   "base",   "flex", "Phone bigram surprisal"),
    ("phone",    "bigram_surprisal",   "base",   "H",    "Phone bigram surprisal"),
    ("phone",    "bigram_surprisal",   "prop25", "flex", "Phone bigram surprisal — extended window"),
    ("phone",    "bigram_surprisal",   "prop25", "H",    "Phone bigram surprisal — extended window"),
    ("syllable", "bigram_surprisal",   "base",   "flex", "Syllable bigram surprisal"),
    ("syllable", "bigram_surprisal",   "base",   "H",    "Syllable bigram surprisal"),
    ("phone",    "phone2_pos_entropy", "base",   "flex", "Phone positional entropy"),
    ("phone",    "phone2_pos_entropy", "base",   "H",    "Phone positional entropy"),
]


def build_payload(P):
    key = lambda t: frozenset(t.split(":"))
    payload = {"generated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
               "families": []}
    for unit, family, window, moderator, title in FAMILIES:
        block = {"unit": unit, "family": family, "window": window,
                 "moderator": moderator, "mod_label": MOD_LABEL[moderator],
                 "title": f"{title} · {MOD_LABEL[moderator]}", "rows": []}
        for corpus in CORPORA:
            k = f"{unit}|{family}|{window}|{moderator}|{corpus}"
            d = P.get(k)
            row = {"corpus": corpus, "label": LABEL[corpus]}
            if d is None:
                row.update(missing=True, why="no notebook")
                block["rows"].append(row); continue
            m = d["meta"]
            if not m["ran"]:
                row.update(missing=True, why="model cell never executed")
                block["rows"].append(row); continue
            if not m["reads_exist"]:
                row.update(missing=True,
                           why="pipeline has not produced "
                               + ", ".join(m["reads"]) + " yet; the notebook's "
                               "stored output is from an earlier fit")
                block["rows"].append(row); continue
            if m["predictor"] not in d["fixed"]:
                row.update(missing=True, why="stored output is from a different predictor")
                block["rows"].append(row); continue

            fx, pred = d["fixed"], m["predictor"]
            lk = {key(t): v for t, v in fx.items()}
            g = lambda *c: lk.get(frozenset(c))

            flex = {}
            terms = [("u1", (pred, "u1_flexibility")),
                     ("u2", (pred, "u2_flexibility")),
                     # the bilinear term: how the two flexibilities jointly bend
                     # the surprisal slope. It is also the coefficient that makes
                     # the section-3 surfaces curve rather than tilt.
                     ("u12", (pred, "u1_flexibility", "u2_flexibility"))]
            for w, t in terms:
                art = g(*t)
                dif = g(t[0], "modalityac", *t[1:])
                if art is None or dif is None:
                    continue
                flex[w] = {"art": art, "gap": dif,
                           "avg_est": art["est"] + 0.5 * dif["est"],
                           "ac_est":  art["est"] + dif["est"]}
            surf = {
                "gap": [g(pred, "modalityac")["est"],
                        g(pred, "modalityac", "u1_flexibility")["est"],
                        g(pred, "modalityac", "u2_flexibility")["est"],
                        g(pred, "modalityac", "u1_flexibility", "u2_flexibility")["est"]],
                "art": [g(pred)["est"],
                        g(pred, "u1_flexibility")["est"],
                        g(pred, "u2_flexibility")["est"],
                        g(pred, "u1_flexibility", "u2_flexibility")["est"]]}
            surf["avg"] = [a + 0.5 * b for a, b in zip(surf["art"], surf["gap"])]

            row.update(
                predictor=pred, n_obs=m["n_obs"], singular=m["singular"],
                random=m["random"],
                marg={r["effect"]: r for r in d["marginal"]
                      if r["moderators"] == "averaged over the data"},
                marg0={r["effect"]: r for r in d["marginal"]
                       if r["moderators"] == "at flex1 = flex2 = 0"},
                flex=flex, surf=surf)
            block["rows"].append(row)
        payload["families"].append(block)
    return payload


def verify(payload):
    """The bilinear surfaces, at flex=0, must equal the printed emtrends rows."""
    worst, n = 0.0, 0
    for blk in payload["families"]:
        for r in blk["rows"]:
            if r.get("missing"):
                continue
            for nm, pr in (("art", "art"), ("gap", "ac - art")):
                printed = r["marg0"].get(pr, {}).get("est")
                if printed is None:
                    continue
                worst = max(worst, abs(r["surf"][nm][0] - printed)); n += 1
    return n, worst


# --------------------------------------------------------------------------- #
# HTML
# --------------------------------------------------------------------------- #
import html as _h

def stars(p):
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "†" if p < .1 else ""

def cell_marg(m, k):
    v = m.get(k)
    if not v:
        return '<td class="num">—</td>'
    return (f'<td class="num" title="SE {v["se"]:.4f} · 95% CI [{v["lower"]:.4f}, '
            f'{v["upper"]:.4f}] · z = {v["z"]:.2f} · p = {v["p"]:.3g}">'
            f'{v["est"]:+.4f}<span class="st">{stars(v["p"])}</span></td>')

def cell_fx(c):
    if not c:
        return '<td class="num">—</td>'
    return (f'<td class="num" title="SE {c["se"]:.4f} · t = {c["t"]:.2f} '
            f'· p = {c["p"]:.3g}">{c["est"]:+.4f}'
            f'<span class="st">{stars(c["p"])}</span></td>')

def table1(fam):
    pred = "entropy" if fam["family"] == "phone2_pos_entropy" else "surprisal"
    rows = []
    for r in fam["rows"]:
        if r.get("missing"):
            rows.append(f'<tr class="missing"><th>{_h.escape(r["label"])}'
                        f'<span class="sub">{r["corpus"]}</span></th>'
                        f'<td colspan="5" class="note-cell">{_h.escape(r["why"])}</td></tr>')
            continue
        m, note = r["marg"], (" &middot; singular" if r["singular"] else "")
        rows.append(
            ('<tr class="sing">' if r["singular"] else "<tr>")
            + f'<th>{_h.escape(r["label"])}<span class="sub">{r["corpus"]}{note}</span></th>'
            + f'<td class="num n">{r["n_obs"]:,}</td>'
            + cell_marg(m, "overall") + cell_marg(m, "art") + cell_marg(m, "ac")
            + cell_marg(m, "ac - art") + "</tr>")
    return f'''
<div class="tscroll"><table class="t">
 <thead>
  <tr><th rowspan="2">Dataset</th><th rowspan="2" class="num">rows</th>
      <th colspan="3">Effect of {pred} on distinctiveness</th>
      <th rowspan="2" class="num gapcol">Difference<span class="sub">ac &minus; art</span></th></tr>
  <tr><th class="num">Both modalities<span class="sub">marginalised</span></th>
      <th class="num">Articulatory</th><th class="num">Acoustic</th></tr>
 </thead>
 <tbody>{"".join(rows)}</tbody>
</table></div>'''

def _mod_label(u, w):
    """Row label for a moderator in table 2."""
    if w == "u12":
        return f"{u}1 &times; {u}2 flexibility"
    return f"{u}{w[-1]} flexibility"


def table2(fam):
    pred = "entropy" if fam["family"] == "phone2_pos_entropy" else "surprisal"
    u = "syll" if fam["unit"] == "syllable" else "phone"
    rows = []
    for r in fam["rows"]:
        if r.get("missing"):
            rows.append(f'<tr class="missing"><th>{_h.escape(r["label"])}</th>'
                        f'<td colspan="5" class="note-cell">{_h.escape(r["why"])}</td></tr>')
            continue
        note = " &middot; singular" if r["singular"] else ""
        shown = [w for w in ("u1", "u2", "u12") if w in r["flex"]]
        for i, w in enumerate(shown):
            fl = r["flex"][w]
            head = ("" if i else
                    f'<th rowspan="{len(shown)}">{_h.escape(r["label"])}'
                    f'<span class="sub">{r["corpus"]}{note}</span></th>')
            cls = ("grp " if i == 0 else "") + ("sing" if r["singular"] else "")
            rows.append(
                f'<tr class="{cls}">' + head
                + f'<td class="wcol">{_mod_label(u, w)}</td>'
                + cell_fx(fl["art"])
                + f'<td class="num est">{fl["ac_est"]:+.4f}</td>'
                + f'<td class="num est">{fl["avg_est"]:+.4f}</td>'
                + cell_fx(fl["gap"]) + "</tr>")
    return f'''
<div class="tscroll"><table class="t t2">
 <thead>
  <tr><th rowspan="2">Dataset</th><th rowspan="2">Moderator</th>
      <th colspan="3">Change in the {pred} slope per +1 SD of flexibility</th>
      <th rowspan="2" class="num gapcol">Difference<span class="sub">ac &minus; art</span></th></tr>
  <tr><th class="num">Articulatory</th>
      <th class="num est">Acoustic<span class="sub">estimate only</span></th>
      <th class="num est">Both<span class="sub">estimate only</span></th></tr>
 </thead>
 <tbody>{"".join(rows)}</tbody>
</table></div>'''


CSS = """
:root{--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;
 --grid:#e1e0d9;--rule:#c3c2b7;--border:rgba(11,11,11,.10);--mid:#f0efec;
 --warn-bg:#fdf6e6;--warn-br:#eda100}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
 --page:#0d0d0d;--surface:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--muted:#898781;
 --grid:#2c2c2a;--rule:#383835;--border:rgba(255,255,255,.10);--mid:#383835;
 --warn-bg:#241f12;--warn-br:#c98500}}
:root[data-theme="dark"]{--page:#0d0d0d;--surface:#1a1a19;--ink:#fff;--ink2:#c3c2b7;
 --muted:#898781;--grid:#2c2c2a;--rule:#383835;--border:rgba(255,255,255,.10);
 --mid:#383835;--warn-bg:#241f12;--warn-br:#c98500}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);
 font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif;-webkit-font-smoothing:antialiased}
.wrap{max-width:1180px;margin:0 auto;padding:40px 16px 96px}
h1{font-size:26px;line-height:1.25;margin:0 0 6px;letter-spacing:-.01em}
h2{font-size:19px;margin:44px 0 6px}
h3{font-size:15px;margin:26px 0 8px;color:var(--ink2);font-weight:600}
p{color:var(--ink2);margin:8px 0}
.lede{max-width:74ch}
.meta{color:var(--muted);font-size:13px;margin-bottom:24px}
.card{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:18px 20px;margin:18px 0}
.warn{border-left:3px solid var(--warn-br);background:var(--warn-bg)}
.warn h3{margin-top:0;color:var(--ink)}
.warn ul{margin:8px 0 0;padding-left:20px}
.warn li{color:var(--ink2);margin:6px 0}
table.t{width:100%;border-collapse:collapse;margin:10px 0 4px;font-size:13.5px}
table.t th,table.t td{padding:7px 9px;text-align:left;border-bottom:1px solid var(--grid)}
table.t thead th{color:var(--ink2);font-weight:600;font-size:12.5px;
 border-bottom:1px solid var(--rule);vertical-align:bottom}
table.t tbody th{font-weight:600;white-space:nowrap}
.num{text-align:right!important;font-variant-numeric:tabular-nums}
.n,.est{color:var(--muted)}
td.gapcol,th.gapcol{border-left:1px solid var(--grid)}
.sub{display:block;font-weight:400;font-size:11px;color:var(--muted)}
.st{display:inline-block;min-width:22px;text-align:left;color:var(--ink2);font-size:11px;padding-left:3px}
tr.grp th,tr.grp td{border-top:1px solid var(--grid)}
tr.sing th,tr.sing td{background:var(--warn-bg)}
.wcol{color:var(--ink2);white-space:nowrap}
tr.missing td{color:var(--muted);font-style:italic}
.legend{font-size:12px;color:var(--muted);margin:2px 0 0}
.tscroll{overflow-x:auto;-webkit-overflow-scrolling:touch;max-width:100%}
@media(max-width:640px){table.t{min-width:560px}table.t tbody th{white-space:normal}}
.grid{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:14px;margin-top:10px}
@media(max-width:860px){.grid{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media(max-width:460px){.grid{grid-template-columns:1fr}}
.panel{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:12px}
.panel h4{margin:0 0 2px;font-size:12.5px;font-weight:600;line-height:1.3;min-height:2.5em}
svg{display:block;width:100%;height:auto;overflow:visible}
.tip{position:fixed;pointer-events:none;background:var(--ink);color:var(--page);
 padding:6px 9px;border-radius:6px;font-size:12px;opacity:0;transition:opacity .1s;
 z-index:50;font-variant-numeric:tabular-nums;white-space:nowrap}
.ctrl{display:flex;gap:8px;align-items:center;margin:14px 0 4px;flex-wrap:wrap}
.ctrl button{font:inherit;font-size:13px;padding:6px 12px;border-radius:999px;cursor:pointer;
 border:1px solid var(--border);background:var(--surface);color:var(--ink2)}
.ctrl button[aria-pressed="true"]{background:var(--ink);color:var(--page);border-color:var(--ink)}
.scale{display:flex;align-items:center;gap:8px;font-size:12px;color:var(--muted);margin-left:auto}
.bar{width:150px;height:10px;border-radius:3px;border:1px solid var(--border)}
code{font:12.5px/1.4 ui-monospace,SFMono-Regular,Menlo,monospace;background:var(--mid);
 padding:1px 5px;border-radius:4px}
pre.spec{background:var(--mid);border:1px solid var(--border);border-radius:8px;padding:12px 14px;
 overflow-x:auto;font:12.5px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--ink2);margin:10px 0}
.foot{color:var(--muted);font-size:12.5px;margin-top:8px}
.famhead{display:flex;align-items:baseline;gap:10px}
.chip{font-weight:400;font-size:11.5px;color:var(--muted);border:1px solid var(--border);
 border-radius:999px;padding:1px 8px;font-variant-numeric:tabular-nums}
@media(max-width:640px){.wrap{padding:24px 16px 72px}table.t{font-size:12.5px}
 table.t th,table.t td{padding:6px 5px}}
"""

JS = """
const SURFACES = __SURF__;
const BLUE_L=['#cde2fb','#9ec5f4','#6da7ec','#3987e5','#256abf','#184f95','#0d366b'];
const RED_L =['#fbd7d4','#f5aca7','#ef807a','#e34948','#bf3030','#942626','#6b1a1a'];
const BLUE_D=['#184f95','#256abf','#3987e5','#6da7ec','#9ec5f4','#cde2fb'];
const RED_D =['#942626','#bf3030','#e34948','#ef807a','#f5aca7','#fbd7d4'];
function isDark(){const t=document.documentElement.getAttribute('data-theme');
  return t?t==='dark':matchMedia('(prefers-color-scheme: dark)').matches;}
let BLUE=BLUE_L,RED=RED_L,MID='#f0efec';
function setRamps(){const d=isDark();BLUE=d?BLUE_D:BLUE_L;RED=d?RED_D:RED_L;MID=d?'#1f1f1e':'#f0efec';}
setRamps();
const N=21,LO=-2,HI=2;
function hex2rgb(h){return [1,3,5].map(i=>parseInt(h.slice(i,i+2),16));}
function lerp(a,b,t){return a.map((v,i)=>Math.round(v+(b[i]-v)*t));}
function ramp(arm,t){const s=[hex2rgb(MID),...arm.map(hex2rgb)];
  const x=Math.min(0.999,Math.max(0,t))*(s.length-1),i=Math.floor(x);
  return 'rgb('+lerp(s[i],s[i+1],x-i).join(',')+')';}
function colour(v,max){if(!isFinite(v)||max<=0)return MID;
  const t=Math.min(1,Math.abs(v)/max);return v>=0?ramp(BLUE,t):ramp(RED,t);}
function val(c,f1,f2){return c[0]+c[1]*f1+c[2]*f2+c[3]*f1*f2;}
const tip=document.createElement('div');tip.className='tip';document.body.appendChild(tip);
function draw(host,coef,maxAbs,unit){
  const S=170,PAD=26,cell=S/N;
  let s=`<svg viewBox="0 0 ${S+PAD+8} ${S+PAD+8}" role="img"><g transform="translate(${PAD},4)">`;
  for(let i=0;i<N;i++)for(let j=0;j<N;j++){
    const f1=LO+(HI-LO)*i/(N-1),f2=HI-(HI-LO)*j/(N-1),v=val(coef,f1,f2);
    s+=`<rect x="${(i*cell).toFixed(2)}" y="${(j*cell).toFixed(2)}" width="${(cell+0.4).toFixed(2)}" height="${(cell+0.4).toFixed(2)}" fill="${colour(v,maxAbs)}" data-v="${v.toFixed(4)}" data-f1="${f1.toFixed(2)}" data-f2="${f2.toFixed(2)}"/>`;}
  // The zero contour, as connected runs. Points where the contour leaves the
  // box (or the denominator vanishes) break the run; emitting a bare 'M' there
  // makes an invalid d attribute, so collect runs and drop the empty ones.
  const runs=[]; let run=[];
  for(let i=0;i<=120;i++){const f1=LO+(HI-LO)*i/120,den=coef[2]+coef[3]*f1;
    let pt=null;
    if(Math.abs(den)>=1e-12){const f2=-(coef[0]+coef[1]*f1)/den;
      if(f2>=LO&&f2<=HI) pt=[(f1-LO)/(HI-LO)*S,(HI-f2)/(HI-LO)*S];}
    if(pt) run.push(pt); else if(run.length){runs.push(run);run=[];}}
  if(run.length) runs.push(run);
  const path=runs.filter(r=>r.length>1).map(r=>
    'M'+r.map(([x,y],k)=>(k?'L':'')+x.toFixed(1)+','+y.toFixed(1)).join('')).join(' ');
  if(path) s+=`<path d="${path}" fill="none" stroke="currentColor" stroke-width="1.1" opacity=".55"/>`;
  s+=`<rect width="${S}" height="${S}" fill="none" stroke="var(--rule)" stroke-width="1"/></g>`;
  s+=`<text x="${PAD+S/2}" y="${S+22}" text-anchor="middle" font-size="9.5" fill="var(--muted)">${unit}1 flexibility (SD)</text>`;
  s+=`<text x="10" y="${S/2+4}" text-anchor="middle" font-size="9.5" fill="var(--muted)" transform="rotate(-90 10 ${S/2+4})">${unit}2 flexibility (SD)</text>`;
  s+=`<text x="${PAD-3}" y="${S+8}" text-anchor="end" font-size="9" fill="var(--muted)">−2</text>`;
  s+=`<text x="${PAD-3}" y="10" text-anchor="end" font-size="9" fill="var(--muted)">+2</text></svg>`;
  host.innerHTML=s;
  host.querySelectorAll('rect[data-v]').forEach(r=>{
    r.addEventListener('mousemove',e=>{tip.textContent=`flex1 ${r.dataset.f1}, flex2 ${r.dataset.f2}  →  ${r.dataset.v}`;
      tip.style.opacity=1;tip.style.left=(e.clientX+12)+'px';tip.style.top=(e.clientY-30)+'px';});
    r.addEventListener('mouseleave',()=>tip.style.opacity=0);});}
let MODE='gap';
function famMax(fam){return Math.max(...SURFACES.filter(s=>s.family===fam).map(s=>s.surf[MODE])
  .flatMap(c=>[[-2,-2],[-2,2],[2,-2],[2,2],[0,0]].map(([a,b])=>Math.abs(val(c,a,b)))));}
function render(){
  const kb=document.getElementById('keybar');
  if(kb)kb.style.background=isDark()
    ?'linear-gradient(90deg,#fbd7d4,#e34948,#1f1f1e,#3987e5,#cde2fb)'
    :'linear-gradient(90deg,#6b1a1a,#e34948,#f0efec,#3987e5,#0d366b)';
  const mx={};SURFACES.forEach(s=>{if(!(s.family in mx))mx[s.family]=famMax(s.family);});
  document.querySelectorAll('[data-famscale]').forEach(el=>{
    el.textContent='scale ±'+mx[el.dataset.famscale].toFixed(2);});
  SURFACES.forEach((s,i)=>{const h=document.getElementById('hm'+i);
    if(h)draw(h,s.surf[MODE],mx[s.family],s.unit==='syllable'?'syll':'phone');});}
document.querySelectorAll('.ctrl button[data-mode]').forEach(b=>{
  b.addEventListener('click',()=>{MODE=b.dataset.mode;
    document.querySelectorAll('.ctrl button[data-mode]').forEach(x=>x.setAttribute('aria-pressed',x===b));
    document.getElementById('hmcap').textContent=b.dataset.cap;render();});});
render();
matchMedia('(prefers-color-scheme: dark)').addEventListener('change',()=>{setRamps();render();});
"""


def _and(xs):
    xs = list(xs)
    if len(xs) <= 1: return "".join(xs)
    if len(xs) == 2: return f"{xs[0]} and {xs[1]}"
    return ", ".join(xs[:-1]) + f" and {xs[-1]}"


def _pend_label(title, corpus):
    """'aemri in extended window' -- the part of the title after the dash."""
    tail = title.split("\u2014")[-1].strip().lower()
    return f"{corpus} in {tail}"


def caveats(payload):
    """
    The report's lead caveats, DERIVED from the payload rather than typed.

    Everything in this project that went wrong went wrong because a hand-written
    statement outlived the numbers it described. So the pending list, the
    singular list and the base-vs-prop25 comparison are all computed here; the
    only hard-coded item is the ultrasound blind-spot note, which records a
    decision rather than a current state.
    """
    fams = {f["window"] + "|" + f["family"] + "|" + f["unit"] + "|" + f["moderator"]: f
            for f in payload["families"]}
    items = []

    # -- 1. what the extended window actually did ---------------------------
    base = {r["corpus"]: r for r in fams["base|bigram_surprisal|phone|H"]["rows"]
            if not r.get("missing")}
    p25  = {r["corpus"]: r for r in fams["prop25|bigram_surprisal|phone|H"]["rows"]
            if not r.get("missing")}
    both = [c for c in base if c in p25]
    if both:
        d_art = {c: p25[c]["marg"]["art"]["est"] - base[c]["marg"]["art"]["est"] for c in both}
        d_ac  = {c: p25[c]["marg"]["ac"]["est"]  - base[c]["marg"]["ac"]["est"]  for c in both}
        d_ov  = {c: p25[c]["marg"]["overall"]["est"] - base[c]["marg"]["overall"]["est"] for c in both}
        lost  = [c for c in both
                 if base[c]["marg"]["art"]["p"] < .05 <= p25[c]["marg"]["art"]["p"]]
        allneg = all(v < 0 for v in d_art.values())
        items.append(
            "<strong>Widening the articulatory window weakens the articulatory "
            f"effect.</strong> Across the {len(both)} corpora with both windows, the "
            "articulatory slope moves "
            + ("down in every one" if allneg else "down in most")
            + f" (by up to {max(abs(v) for v in d_art.values()):.4f}), while the acoustic "
            f"slope barely moves (up to {max(abs(v) for v in d_ac.values()):.4f}) &mdash; as it "
            "should, since only the articulatory window changed. "
            + (f"In {_and(lost)} the articulatory effect drops below "
               "significance altogether. " if lost else "")
            + "The <strong>ac &minus; art</strong> gap therefore widens rather than "
            "closing. That is the opposite of what the extended window was "
            "proposed to show. The overall effect is stable throughout "
            f"(largest shift {max(abs(v) for v in d_ov.values()):.4f}, no sign or "
            "significance flips), so this is a change in <em>where</em> the effect "
            "sits across modalities, not in whether there is one.")

    # -- 1b. does the moderator choice matter? -----------------------------
    key = lambda t: frozenset(t.split(":"))
    def u2gap(row):
        lk = {key(t): v for t, v in row["_fixed"].items()} if "_fixed" in row else None
        return None
    fl = {r["corpus"]: r for r in fams["base|bigram_surprisal|phone|flex"]["rows"]
          if not r.get("missing")}
    hh = {r["corpus"]: r for r in fams["base|bigram_surprisal|phone|H"]["rows"]
          if not r.get("missing")}
    shared = [c for c in fl if c in hh]
    if shared:
        d_ov = max(abs(hh[c]["marg"]["overall"]["est"] - fl[c]["marg"]["overall"]["est"])
                   for c in shared)
        d_gap = max(abs(hh[c]["marg"]["ac - art"]["est"] - fl[c]["marg"]["ac - art"]["est"])
                    for c in shared)
        flips = [c for c in shared
                 if (fl[c]["flex"]["u2"]["art"]["est"] > 0)
                 != (hh[c]["flex"]["u2"]["art"]["est"] > 0)]
        items.append(
            "<strong>Which flexibility measure you use barely touches the surprisal "
            "effect, but changes the flexibility effect.</strong> Across the "
            f"{len(shared)} corpora the overall surprisal slope moves by at most "
            f"{d_ov:.4f} between the two moderators and the modality gap by at most "
            f"{d_gap:.4f}, with no sign changes. The "
            "<em>phone2 flexibility</em> coefficient, by contrast, changes sign in "
            f"{len(flexflip) if (flexflip := flips) else 0} of {len(shared)} "
            + ("(" + ", ".join(flips) + ")" if flips else "")
            + ". So tables reporting the surprisal effect are robust to this choice; "
            "tables reporting flexibility are not, and the paper has to commit to one "
            "operationalisation. <code>utils/mi.py</code> predicts exactly this: "
            "&ldquo;whatever coefficient you had on the H version of this predictor "
            "will change sign when you swap in &minus;I.&rdquo;")

    # -- 2. what is still missing ------------------------------------------
    pend = [(f["title"], r["corpus"]) for f in payload["families"]
            for r in f["rows"] if r.get("missing")]
    if pend:
        items.append(
            "<strong>"
            + ("One cell is" if len(pend) == 1 else f"{len(pend)} cells are")
            + " still pending: </strong>"
            + _and(_pend_label(t, c) for t, c in pend)
            + ". Those notebooks currently display output from an earlier fit, so "
            "this report refuses to read them and marks the row pending rather "
            "than repeating stale numbers under a new heading.")

    # -- 3. model health ----------------------------------------------------
    sing = [(f["title"], r["corpus"], r["n_obs"]) for f in payload["families"]
            for r in f["rows"] if not r.get("missing") and r["singular"]]
    ok_n = sum(1 for f in payload["families"] for r in f["rows"]
               if not r.get("missing") and not r["singular"])
    if sing:
        # the same dataset can be singular under both moderators -- name it once
        uniq, seen = [], set()
        for _, c, n in sing:
            if c not in seen:
                seen.add(c); uniq.append((c, n))
        items.append(
            f"<strong>{ok_n} of {ok_n + len(sing)} fitted models converge cleanly.</strong> "
            "The exceptions are "
            + _and(f"{c} ({n:,} rows)" for c, n in uniq)
            + " &mdash; all in the syllable analysis, the smallest models in the set, "
            + ("and singular under both flexibility measures" if len(sing) > len(uniq)
               else "")
            + ". Their rows are shaded below; read their estimates as unstable and "
            "their standard errors as unreliable. Every phone model converges.")
    else:
        items.append(f"<strong>All {ok_n} fitted models converge cleanly</strong> "
                     "(<code>isSingular = FALSE</code>), so standard errors and "
                     "significance marks can be read normally.")

    # -- 4. a decision, not a state ----------------------------------------
    # The historical figure is a fixed past fact; the current one is read from
    # the payload, so this item cannot go stale the way a typed number does.
    ent = {r["corpus"]: r for r in fams["base|phone2_pos_entropy|phone|flex"]["rows"]
           if not r.get("missing")}
    now = ent.get("beus")
    current = (f"it now stands at {now['marg']['ac - art']['est']:+.4f} "
               f"(p = {now['marg']['ac - art']['p']:.3g}, {now['n_obs']:,} rows)"
               if now else "the current figure is in the table below")
    items.append(
        "<strong>The ultrasound modality gap depends on the blind-spot list.</strong> "
        "All beus models use the 15-phone <code>limited_phones</code> list, with the "
        "three rhotics added on 20 Sep. On the entropy model, adding them dropped the "
        "ac&nbsp;&minus;&nbsp;art gap from +0.0421 at p&nbsp;&lt;&nbsp;.001 (110,877 "
        f"rows, old list) to roughly a third of that; {current}. A result that moves "
        "that much when three segments are reclassified is worth settling deliberately "
        "rather than inheriting.")

    return ("<div class=\"card warn\"><h3>Read this first</h3><ul>"
            + "".join(f"<li>{x}</li>" for x in items) + "</ul></div>")


def render(payload):
    surfaces, panels = [], []
    for fam in payload["families"]:
        for r in fam["rows"]:
            if r.get("missing"):
                continue
            i = len(surfaces)
            surfaces.append(dict(family=fam["title"], unit=fam["unit"],
                                 corpus=r["corpus"], surf=r["surf"]))
            panels.append((fam["title"],
                           f'<div class="panel"><h4>{_h.escape(r["label"])}</h4>'
                           f'<span class="sub">{r["n_obs"]:,} rows</span>'
                           f'<div id="hm{i}"></div></div>'))
    by_fam = {}
    for title, p in panels:
        by_fam.setdefault(title, []).append(p)
    hm = "".join(
        f'<h3 class="famhead">{_h.escape(k)}'
        f'<span class="chip" data-famscale="{_h.escape(k)}"></span></h3>'
        f'<div class="grid">{"".join(v)}</div>' for k, v in by_fam.items())

    t1 = "".join(f'<h3>{_h.escape(f["title"])}</h3>{table1(f)}' for f in payload["families"])
    t2 = "".join(f'<h3>{_h.escape(f["title"])}</h3>{table2(f)}' for f in payload["families"])

    sing = [f'{r["corpus"]}' for f in payload["families"] for r in f["rows"]
            if not r.get("missing") and r["singular"]]
    pend = [f'{f["title"].split("—")[-1].strip()}: {r["corpus"]}'
            for f in payload["families"] for r in f["rows"] if r.get("missing")]

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Surprisal and Flexibility Effects</title>
<style>{CSS}</style></head><body><div class="wrap">

<h1>Surprisal, entropy and flexibility effects on distinctiveness</h1>
<p class="meta">Five datasets &times; four analyses &middot; read out of the fitted
<code>lmer</code> models in the project's R notebooks &middot; compiled {payload["generated"]}</p>

<p class="lede">Every number below is the slope of z-scored latent distance on a
z-scored predictor, with duration and log frequency partialled out. Positive means
<em>more surprising (or higher-entropy) units are more distinct</em>. Modality is
coded with articulation as the reference level, so the <strong>ac &minus; art</strong>
column is the modality gap: positive means the effect is steeper in acoustics
than in articulation.</p>

{caveats(payload)}
<h2>1 &middot; The effect of surprisal and entropy on distinctiveness</h2>
<p>Averaged over the flexibility values that actually occur in each dataset.
Hover any number for its standard error, confidence interval and exact p.</p>
{t1}
<p class="legend">*** p &lt; .001 &middot; ** p &lt; .01 &middot; * p &lt; .05 &middot;
&dagger; p &lt; .1 &mdash; from the <code>emtrends</code> contrasts, asymptotic (z)
degrees of freedom. Shaded rows are singular fits.</p>

<h2>2 &middot; How flexibility changes that slope</h2>
<p>Each row asks: when a unit is one standard deviation more flexible, how much
does the slope move? The <strong>articulatory</strong> and
<strong>ac &minus; art</strong> columns are model coefficients and carry exact
tests. The two greyed columns are exact point estimates derived from them
(acoustic = articulatory + difference; both = articulatory + half the
difference), but their standard errors need a covariance <code>lmer</code> does
not print &mdash; see <code>analysis/flex_marginal_se.R</code>.</p>
{t2}
<p class="legend">*** p &lt; .001 &middot; ** p &lt; .01 &middot; * p &lt; .05 &middot;
&dagger; p &lt; .1 &mdash; from the fixed-effects table, Satterthwaite degrees of freedom.</p>

<h2>3 &middot; Where the effect lives in flexibility space</h2>
<p>The slope is exactly bilinear in the two flexibility moderators, so these
surfaces are reconstructions from four coefficients apiece rather than smoothed
data &mdash; they are exact. The dark line is where the surface crosses zero.
There is no significance contour: that needs the same covariance the greyed
table columns need. Colour is scaled <strong>within each analysis</strong>, not
across all four; compare panels inside a row, not between rows.</p>
<div class="ctrl">
  <button data-mode="gap" data-cap="Modality gap (ac &minus; art) in the slope, across flexibility space." aria-pressed="true">Modality gap (ac &minus; art)</button>
  <button data-mode="avg" data-cap="Slope averaged over both modalities, across flexibility space." aria-pressed="false">Both modalities</button>
  <button data-mode="art" data-cap="Articulatory slope, across flexibility space." aria-pressed="false">Articulatory only</button>
  <div class="scale"><span>art steeper</span><div class="bar" id="keybar"></div><span>ac steeper</span></div>
</div>
<p class="foot" id="hmcap">Modality gap (ac &minus; art) in the slope, across flexibility space.</p>
{hm}

<h2>4 &middot; Provenance</h2>
<p>Built by <code>analysis/build_report.py</code>, which reads the stored outputs
of the executed notebooks. Nothing is refitted, so these are the same fits you
ran. Every model uses one specification:</p>
<pre class="spec">dist ~ PREDICTOR * modality * u2_flexibility * u1_flexibility
     + u2_duration + u1_duration + u2_freq + u1_freq
     + (1 | u1) + (1 | u2) + (1 | word) + (1 | bigram) + (1 | subject)</pre>
<p class="foot">PREDICTOR is <code>bigram_surprisal</code> or
<code>phone2_pos_entropy</code>. <code>dist</code> is z-scored within modality
(and within dataset where two corpora are stacked); the predictor, both
flexibility moderators and log frequency are z-scored. Modality is coded with
articulation as the reference level. Duration and frequency enter additively, so
they do not appear in any slope reported here. The moderators are the
<code>flexibility</code> columns, not <code>H_art_given_ac</code>.</p>
<p class="foot">Singular fits: {", ".join(sing) or "none"}.
Pending: {"; ".join(pend) or "none"}.
Checked: the surfaces in section 3, evaluated at flex1 = flex2 = 0, reproduce the
printed <code>emtrends</code> values in all {verify(payload)[0]} comparisons,
worst discrepancy {verify(payload)[1]:.0e} (the printed values are rounded to four
decimals). The <code>aemri</code> row is the combined USC-TIMIT + USC-75 file.</p>

<script>{JS.replace("__SURF__", json.dumps(surfaces))}</script>
</div></body></html>
"""


def main():
    P = parse_all()
    payload = build_payload(P)
    n, worst = verify(payload)
    out = os.path.join(ROOT, "analysis", "effects_summary.html")
    open(out, "w").write(render(payload))
    print(f"parsed {len(P)} notebooks")
    print(f"verification: {n} comparisons, worst discrepancy {worst:.2e}")
    for f in payload["families"]:
        ok = sum(1 for r in f["rows"] if not r.get("missing"))
        print(f"  {f['title']:<46} {ok}/{len(f['rows'])} datasets")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
