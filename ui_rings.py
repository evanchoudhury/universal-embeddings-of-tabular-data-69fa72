"""Animated HUD of the headline results (added by the Ex_Senior agent).

Rings = our model / XGBoost / baseline (lit share = score). Streaks = test predictions from
artifacts/sample_predictions.csv (length = confidence, red = wrong). Look comes from .streamlit/hud.json.
"""
import csv
import json
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

HERE = Path(__file__).parent


def _samples(limit=300):
    """[(strength 0..1, wrong?)] from sample_predictions.csv, whatever its column names are."""
    try:
        with open(HERE / "artifacts" / "sample_predictions.csv", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))[:limit]
    except Exception:
        return []
    if not rows:
        return []
    cols = list(rows[0])
    find = lambda *keys: next((c for c in cols if any(k in c.lower() for k in keys)), None)
    t, p, c = find("true", "actual", "label", "target", "y_test"), find("pred"), find("conf", "prob", "score")
    out = []
    for r in rows:
        try:
            tv, pv = r.get(t), r.get(p)
            try:                                        # numbers: regression or numeric classes
                tf, pf = float(tv), float(pv)
                err = abs(tf - pf) / (abs(tf) + 1e-9)
                wrong = (err > .25) if not (tf.is_integer() and pf.is_integer()) else tf != pf
                strength = float(r[c]) if c else max(0.0, 1 - min(err, 1))
            except (TypeError, ValueError):
                wrong = str(tv).strip() != str(pv).strip()
                strength = float(r[c]) if c else (.4 if wrong else .9)
            out.append([round(max(0.05, min(1.0, strength)), 3), bool(wrong)])
        except Exception:
            continue
    return out


def hero():
    try:
        m = json.loads((HERE / "artifacts" / "metrics.json").read_text(encoding="utf-8"))
        v, b = float(m["value"]), float(m["baseline"])
    except Exception:
        return                                          # no results yet: the app explains that itself
    try:
        look = json.loads((HERE / ".streamlit" / "hud.json").read_text(encoding="utf-8"))
    except Exception:
        look = {}
    up = bool(m.get("higher_is_better", True))
    x = (m.get("metrics") or {}).get("xgboost")
    x = x.get(m.get("primary_metric")) if isinstance(x, dict) else None
    rings = [["Our model", v], ["XGBoost", float(x)] if isinstance(x, (int, float)) else None, ["Baseline", b]]
    rings = [r for r in rings if r]
    vals = [r[1] for r in rings]
    top = 1.0 if all(0 <= t <= 1 for t in vals) else (max(abs(t) for t in vals) or 1.0)
    worst = max(abs(t) for t in vals) * 1.15 or 1.0
    for r in rings:                                     # lit share of each ring: better score = more lit
        r.append(abs(r[1]) / top if up else (1 - abs(r[1]) if top == 1.0 else 1 - abs(r[1]) / worst))
    gain = ((v - b) if up else (b - v)) / abs(b) if b else 0.0
    data = {"rings": rings, "samples": _samples(), "metric": str(m.get("primary_metric", "score")).replace("_", " "),
            "up": up, "gain": gain, "look": look,
            "kpis": [["test rows", m.get("n_test")], ["train rows", m.get("n_train")],
                     ["train time", f"{m.get('train_seconds', 0):.0f} s" if m.get("train_seconds") else None],
                     ["vs baseline", f"{gain * 100:+.0f}%"]]}
    components.html(HTML.replace("__DATA__", json.dumps(data)), height=560)


HTML = r"""<!doctype html><html><head><meta charset="utf-8"><style>
html,body{margin:0;background:transparent;font-family:ui-monospace,"JetBrains Mono",Consolas,monospace;color:#cfe3f2}
.hud{position:relative;height:552px;border-radius:20px;overflow:hidden;background:#04070d;border:1px solid rgba(255,255,255,.08)}
canvas{position:absolute;inset:0;width:100%;height:100%}
.bar{position:absolute;left:0;right:0;top:0;display:flex;gap:28px;flex-wrap:wrap;padding:12px 18px;font-size:10.5px;
  letter-spacing:.06em;text-transform:uppercase;color:#7f93a8;background:linear-gradient(#04070df0,#04070d00)}
.bar b{display:block;font-size:17px;color:#e8f3ff;letter-spacing:0;text-transform:none;margin-top:2px}
.legend{position:absolute;left:18px;bottom:14px;font-size:10.5px;line-height:1.9;color:#7f93a8}
.legend i{display:inline-block;width:9px;height:9px;margin-right:8px;vertical-align:-1px}
</style></head><body><div class="hud"><canvas id="c"></canvas><div class="bar" id="bar"></div><div class="legend" id="leg"></div></div>
<script>
const D=__DATA__, L=D.look||{}, P=L.colors||["#5EEAD4","#60A5FA","#A78BFA"], BAD=L.bad||"#FF5A6E", BG=L.bg||"#04070d";
document.querySelector(".hud").style.background=BG;
const VAR=L.variant||{segs:84,spin:1,style:"lines",gap:.28,offset:.6};
const fmt=v=>Math.abs(v)<10?v.toFixed(3):v.toLocaleString(undefined,{maximumFractionDigits:1});
document.getElementById("bar").innerHTML=D.kpis.filter(k=>k[1]!=null).map(k=>`<div>${k[0]}<b>${typeof k[1]=="number"?k[1].toLocaleString():k[1]}</b></div>`).join("");
document.getElementById("leg").innerHTML=D.rings.map((r,i)=>`<div><i style="background:${P[i%P.length]}"></i>${r[0]} &nbsp;${fmt(r[1])}</div>`).join("")
  +`<div><i style="background:${BAD}"></i>missed: wrong class or over 25% off (${D.samples.filter(s=>s[1]).length} of ${D.samples.length} test rows)</div>`;
const cv=document.getElementById("c"), g=cv.getContext("2d"); let W,H,R,cx,cy;
function size(){const d=devicePixelRatio||1;W=cv.clientWidth;H=cv.clientHeight;cv.width=W*d;cv.height=H*d;g.setTransform(d,0,0,d,0,0);
  cx=W*(W<620?.5:VAR.offset);cy=H*.55;R=Math.min(H*.40,W*.42);}
size(); addEventListener("resize",size);
const n=D.rings.length, still=matchMedia("(prefers-reduced-motion: reduce)").matches;
let t=0, grow=0, run=true;
new IntersectionObserver(e=>run=e[0].isIntersecting).observe(cv);
function frame(){
  if(run||t===0){
  t+=1/60; grow=Math.min(1,grow+.012); const ease=1-Math.pow(1-grow,3);
  g.globalCompositeOperation="source-over"; g.fillStyle=BG; g.fillRect(0,0,W,H);
  const core=g.createRadialGradient(cx,cy,0,cx,cy,R*.55); core.addColorStop(0,P[n>2?2:0]+"55"); core.addColorStop(1,"transparent");
  g.fillStyle=core; g.beginPath(); g.arc(cx,cy,R*.55,0,7); g.fill();
  g.globalCompositeOperation="lighter";
  // streaks: one per test prediction
  const S=D.samples, rot=t*.05*VAR.spin;
  S.forEach((s,i)=>{ const a=i/S.length*Math.PI*2+rot, fl=.82+.18*Math.sin(t*2.2+i*1.7);
    const r0=R*1.04, len=(R*.15+s[0]*R*.75)*ease*fl;
    g.strokeStyle=s[1]?BAD:P[i%2?0:1]; g.globalAlpha=s[1]?.9:.45+.4*s[0];
    if(VAR.style=="dots"){ g.fillStyle=g.strokeStyle; for(let q=0;q<len;q+=7){ g.globalAlpha=(s[1]?.9:.7)*(1-q/len);
      g.beginPath(); g.arc(cx+Math.cos(a)*(r0+q),cy+Math.sin(a)*(r0+q),1.3,0,7); g.fill(); } }
    else if(VAR.style=="bars"){ g.lineWidth=3; g.beginPath(); g.moveTo(cx+Math.cos(a)*r0,cy+Math.sin(a)*r0);
      g.lineTo(cx+Math.cos(a)*(r0+len),cy+Math.sin(a)*(r0+len)); g.stroke(); }
    else { const grd=g.createLinearGradient(cx+Math.cos(a)*r0,cy+Math.sin(a)*r0,cx+Math.cos(a)*(r0+len),cy+Math.sin(a)*(r0+len));
      grd.addColorStop(0,g.strokeStyle); grd.addColorStop(1,"transparent"); g.strokeStyle=grd; g.lineWidth=1.3;
      g.beginPath(); g.moveTo(cx+Math.cos(a)*r0,cy+Math.sin(a)*r0); g.lineTo(cx+Math.cos(a)*(r0+len),cy+Math.sin(a)*(r0+len)); g.stroke(); } });
  // segmented rings: lit share = score
  D.rings.forEach((r,k)=>{ const rad=R*(1-k*.2), segs=Math.round(VAR.segs*(1-k*.18)), th=R*.085, col=P[k%P.length];
    const spin=t*(.08+k*.05)*(k%2?-1:1)*VAR.spin, lit=r[2]*ease, step=Math.PI*2/segs;
    for(let j=0;j<segs;j++){ const a=j*step+spin, on=j/segs<lit, fl=on?.6+.4*Math.abs(Math.sin(t*1.3+j*.9+k)):.09;
      g.globalAlpha=fl; g.strokeStyle=col; g.lineWidth=th; g.beginPath(); g.arc(cx,cy,rad,a,a+step*(1-VAR.gap)); g.stroke();
      if(on&&j%6==0){ g.globalAlpha=.8; g.lineWidth=1; g.beginPath(); g.arc(cx,cy,rad+th*.9,a,a+step*.6); g.stroke(); } }
    g.globalAlpha=.25; g.lineWidth=1; g.beginPath(); g.arc(cx,cy,rad-th*.85,0,7); g.stroke(); });
  // centre readout
  g.globalCompositeOperation="source-over"; g.globalAlpha=1; g.textAlign="center"; g.fillStyle="#eef7ff";
  g.font=`700 ${Math.round(R*.2)}px ui-monospace,Consolas,monospace`; g.fillText(fmt(D.rings[0][1]),cx,cy+R*.05);
  g.font=`11px ui-monospace,Consolas,monospace`; g.fillStyle="#8aa0b6";
  g.fillText(D.metric.slice(0,34)+(D.up?"  (higher is better)":"  (lower is better)"),cx,cy+R*.05+22);
  g.fillStyle=D.gain>=0?P[0]:BAD; g.fillText((D.gain>=0?"+":"")+(D.gain*100).toFixed(0)+"% vs baseline",cx,cy+R*.05+40);
  }
  if(!still||grow<1) requestAnimationFrame(frame);
}
frame();
</script></body></html>"""
