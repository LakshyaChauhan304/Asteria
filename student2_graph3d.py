"""Interactive 3D attack-graph view for the isolated Asteria demo.

Design contract
---------------
* Input is ONLY ``report["attack_graph"]`` and ``report["graph_exposure"]`` from
  ``DemoOrchestrator.export_audit_report()``. Nothing is invented: every node,
  edge and status below is copied or derived from those two fields.
* The view is read-only. It cannot advance the workflow. The blue "sever"
  animation replays recorded state and always ends on exactly
  ``graph_exposure["blocked_edge_ids"]``.
* Layout (x/y/z positions) is presentation only: x = distance from the entry
  node, z = trust zone depth, y = spread to avoid overlap.
* No external scripts or network access: a small dependency-free perspective
  renderer on a 2D canvas, so it works offline and on Streamlit Community Cloud.
* Accessible: shapes (not just colour) mark entry/target/decoy, a text table
  lists every node and edge with status words, and ``prefers-reduced-motion``
  disables all animation.
"""

from __future__ import annotations

import json
from html import escape
from typing import Mapping

ZONE_ORDER = ("untrusted", "dmz", "management", "internal", "restricted")
EMPTY_MESSAGE = "No simulated graph exists yet. Run the Red simulation to create one."


def _zone_rank(zone: str, extra: list[str]) -> int:
    if zone in ZONE_ORDER:
        return ZONE_ORDER.index(zone)
    if zone not in extra:
        extra.append(zone)
    return len(ZONE_ORDER) + extra.index(zone)


def graph_payload(report: Mapping[str, object], mode: str) -> dict[str, object] | None:
    """Build the renderer input from backend state, or ``None`` if there is nothing to draw."""
    if mode not in {"red", "blue"}:
        raise ValueError("mode must be 'red' or 'blue'")
    graph = report.get("attack_graph") or {}
    exposure = report.get("graph_exposure") or {}
    nodes, edges = graph.get("nodes") or [], graph.get("edges") or []
    if not nodes or not edges or not exposure:
        return None

    path = graph.get("candidate_path") or {}
    path_nodes, path_edges = list(path.get("node_ids", [])), list(path.get("edge_ids", []))
    route = set(exposure.get("route_node_ids", []))
    decoys = set(exposure.get("decoy_node_ids", []))
    reachable = set(exposure.get("reachable_node_ids", []))
    blocked = set(exposure.get("blocked_edge_ids", []))
    opened = set(exposure.get("open_edge_ids", []))
    control_id = str(graph.get("control_id", ""))
    entry = path_nodes[0] if path_nodes else nodes[0]["node_id"]
    target = path_nodes[-1] if path_nodes else nodes[-1]["node_id"]

    # Layer = shortest structural distance from the entry node (presentation only).
    adjacency: dict[str, list[str]] = {}
    for edge in edges:
        adjacency.setdefault(edge["source_node_id"], []).append(edge["target_node_id"])
    layer = {entry: 0}
    queue = [entry]
    while queue:
        current = queue.pop(0)
        for neighbour in adjacency.get(current, []):
            if neighbour not in layer:
                layer[neighbour] = layer[current] + 1
                queue.append(neighbour)
    fallback = max(layer.values(), default=0) + 1
    extra_zones: list[str] = []
    by_layer: dict[int, list[dict]] = {}
    for node in nodes:
        by_layer.setdefault(layer.get(node["node_id"], fallback), []).append(node)
    positions: dict[str, tuple[float, float, float]] = {}
    layers = sorted(by_layer)
    zone_ranks = sorted({_zone_rank(n["trust_zone"], extra_zones) for n in nodes})
    for column in layers:
        members = sorted(by_layer[column], key=lambda n: (_zone_rank(n["trust_zone"], extra_zones), n["node_id"]))
        for slot, node in enumerate(members):
            x = (layers.index(column) - (len(layers) - 1) / 2) * 2.6
            y = (slot - (len(members) - 1) / 2) * 1.5
            z = (zone_ranks.index(_zone_rank(node["trust_zone"], extra_zones)) - (len(zone_ranks) - 1) / 2) * 2.4
            positions[node["node_id"]] = (round(x, 3), round(y, 3), round(z, 3))

    def node_kind(node_id: str) -> str:
        if node_id == entry:
            return "entry"
        if node_id == target:
            return "target"
        return "decoy" if node_id in decoys else "route"

    out_nodes = []
    for node in nodes:
        node_id = node["node_id"]
        out_nodes.append({
            "id": node_id, "name": node["display_name"], "kind": node_kind(node_id),
            "zone": node["trust_zone"], "criticality": node["criticality"],
            "asset_type": node["asset_type"], "description": node["description"],
            "reachable": node_id in reachable, "pos": positions[node_id],
        })
    out_edges = []
    for edge in edges:
        edge_id = edge["edge_id"]
        out_edges.append({
            "id": edge_id, "source": edge["source_node_id"], "target": edge["target_node_id"],
            "relationship": edge["relationship"],
            "controlled": control_id in edge.get("blocking_control_ids", []),
            "status": "blocked" if edge_id in blocked else ("open" if edge_id in opened else "inactive"),
            "on_path": edge_id in path_edges,
        })
    zones = [
        {"name": zone, "z": positions[next(n["id"] for n in out_nodes if n["zone"] == zone)][2]}
        for zone in dict.fromkeys(n["zone"] for n in sorted(out_nodes, key=lambda n: n["pos"][2]))
    ]
    return {
        "mode": mode, "nodes": out_nodes, "edges": out_edges, "zones": zones,
        "path_edge_ids": path_edges, "control_id": control_id,
        "target_reachable": bool(exposure.get("target_reachable")),
        "blocked_edge_ids": sorted(blocked), "route_node_ids": sorted(route),
    }


def fallback_table_html(payload: Mapping[str, object]) -> str:
    """Text equivalent of the canvas: every node and edge with status in words."""
    node_name = {n["id"]: n["name"] for n in payload["nodes"]}
    node_rows = "".join(
        "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
            escape(n["name"]), escape(n["zone"]), escape(n["criticality"]),
            {"entry": "Entry point", "target": "Target", "decoy": "Decoy (no route to target)",
             "route": "On a route to target"}[n["kind"]],
            "reachable" if n["reachable"] else "not reachable",
        ) for n in payload["nodes"]
    )
    edge_rows = "".join(
        "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
            escape(e["id"]), escape(node_name.get(e["source"], e["source"])),
            escape(node_name.get(e["target"], e["target"])),
            "BLOCKED" if e["status"] == "blocked" else e["status"].upper(),
            "attack path" if e["on_path"] else ("controlled" if e["controlled"] else "-"),
        ) for e in payload["edges"]
    )
    return (
        "<section class='fb'><h3>Text equivalent of the graph</h3>"
        "<table><caption>Nodes</caption><thead><tr><th>Name</th><th>Trust zone</th><th>Criticality</th>"
        "<th>Role</th><th>Reachability</th></tr></thead><tbody>" + node_rows + "</tbody></table>"
        "<table><caption>Edges</caption><thead><tr><th>Edge</th><th>From</th><th>To</th><th>Status</th>"
        "<th>Note</th></tr></thead><tbody>" + edge_rows + "</tbody></table></section>"
    )


def headline(payload: Mapping[str, object]) -> str:
    """One factual sentence describing the drawn state."""
    blocked, total = len(payload["blocked_edge_ids"]), len(payload["edges"])
    if payload["mode"] == "red":
        state = "reachable" if payload["target_reachable"] else "NOT reachable"
        return f"Red view: the target is {state} in the simulated graph. {blocked} of {total} edges are blocked by controls."
    if not payload["blocked_edge_ids"]:
        return f"Blue view: no control has been applied yet. 0 of {total} edges are blocked."
    state = "still reachable" if payload["target_reachable"] else "no longer reachable"
    return f"Blue view: {blocked} of {total} edges are blocked; the target is {state}."


_TEMPLATE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--bg:#060b15;--panel:#0b1422;--line:#263243;--text:#dce9f8;--muted:#8096aa;--red:#ff5968;--green:#00d9a0;
--amber:#ffbd39;--cyan:#00d9f5;--slate:#5b6b80}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:13px/1.45 ui-sans-serif,system-ui,sans-serif}
.bar{display:flex;gap:12px;align-items:center;justify-content:space-between;padding:10px 14px;border:1px solid var(--line);
border-bottom:0;background:var(--panel);border-radius:8px 8px 0 0}
.bar b{letter-spacing:.08em;font-size:11px;text-transform:uppercase;color:var(--cyan)}
.hl{color:var(--text);flex:1}.hint{color:var(--muted);font-size:11px;white-space:nowrap}
.stage{position:relative;border:1px solid var(--line);background:radial-gradient(ellipse at 50% 35%,#0f2233 0,#060b15 72%)}
canvas{display:block;width:100%;height:460px;cursor:grab;touch-action:none}canvas:active{cursor:grabbing}
#tip{position:absolute;display:none;pointer-events:none;max-width:250px;padding:9px 11px;background:#0b1422f2;
border:1px solid #3a4a60;border-radius:6px;font-size:12px;z-index:3}#tip b{display:block;margin-bottom:2px}
#tip .m{color:var(--muted)}
.legend{display:flex;flex-wrap:wrap;gap:14px;padding:9px 14px;border:1px solid var(--line);border-top:0;
background:var(--panel);border-radius:0 0 8px 8px;color:var(--muted);font-size:11.5px}
.legend span{display:inline-flex;gap:6px;align-items:center}
.sw{width:18px;height:0;border-top:3px solid}.sw.dash{border-top-style:dashed}
.ctl{position:absolute;right:10px;top:10px;display:flex;gap:6px;z-index:2}
.ctl button{background:#0b1422cc;color:var(--text);border:1px solid var(--line);border-radius:5px;padding:4px 9px;
font:11px ui-monospace,monospace;cursor:pointer}.ctl button:hover,.ctl button:focus-visible{border-color:var(--cyan);outline:none}
.empty{padding:28px;color:var(--muted)}
.fb{padding:14px;margin-top:12px;border:1px solid var(--line);border-radius:8px;background:var(--panel)}
.fb h3{margin:0 0 6px;font-size:13px}table{border-collapse:collapse;width:100%;margin:8px 0 14px;font-size:12px}
caption{text-align:left;font-weight:700;padding:4px 0;color:var(--muted)}
th,td{text-align:left;border:1px solid #2b3a4f;padding:5px 7px}th{color:var(--cyan);font-weight:600}
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
</style></head><body>
<div class="bar"><b id="mode"></b><span class="hl" id="hl"></span><span class="hint">drag to orbit · scroll to zoom · click a node</span></div>
<div class="stage"><canvas id="c" role="img" aria-label="__ARIA__"></canvas>
<div class="ctl"><button id="rot" aria-pressed="true">auto-rotate</button><button id="reset">reset view</button>__REPLAY__</div>
<div id="tip"></div></div>
<div class="legend" id="legend"></div>
__FALLBACK__
<script>
const P=__PAYLOAD__;
const reduce=matchMedia('(prefers-reduced-motion: reduce)').matches;
const C=getComputedStyle(document.documentElement);const col=n=>C.getPropertyValue(n).trim();
const colors={red:col('--red'),green:col('--green'),amber:col('--amber'),cyan:col('--cyan'),slate:col('--slate'),text:col('--text'),muted:col('--muted')};
const cv=document.getElementById('c'),ctx=cv.getContext('2d'),tip=document.getElementById('tip');
document.getElementById('mode').textContent=P.mode==='red'?'Red agent view':'Blue agent view';
document.getElementById('hl').textContent=__HEADLINE__;
const byId={};P.nodes.forEach(n=>byId[n.id]=n);
let yaw=0.62,pitch=0.34,zoom=1,auto=!reduce,drag=null,hover=null,pinned=null,W=0,H=0,t0=performance.now();
const home={yaw,pitch,zoom};
// Blue replay: severing sweeps along the recorded path, ending on the recorded blocked set.
let sever=1; // 0..1 progress; 1 = final recorded state
const blockedFinal=new Set(P.blocked_edge_ids);
const sweepOrder=[...P.path_edge_ids.filter(e=>blockedFinal.has(e)),...P.edges.map(e=>e.id).filter(e=>blockedFinal.has(e)&&!P.path_edge_ids.includes(e))];
function startReplay(){if(reduce||P.mode!=='blue'||!sweepOrder.length){sever=1;return}sever=0;const s=performance.now();
 (function step(){sever=Math.min(1,(performance.now()-s)/(900*Math.max(1,sweepOrder.length)*0.5));if(sever<1)requestAnimationFrame(step)})()}
const isBlocked=id=>{if(P.mode==='red')return blockedFinal.has(id);const i=sweepOrder.indexOf(id);return blockedFinal.has(id)&&(i<0||i<sever*sweepOrder.length)};
function resize(){const r=cv.getBoundingClientRect(),d=Math.min(devicePixelRatio||1,2);W=r.width;H=r.height;cv.width=W*d;cv.height=H*d;ctx.setTransform(d,0,0,d,0,0)}
addEventListener('resize',resize);resize();
function proj(p){const[x,y,z]=p;const cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch);
 const x1=x*cy+z*sy,z1=-x*sy+z*cy;const y1=y*cp-z1*sp,z2=y*sp+z1*cp;
 const cam=20,f=Math.min(W/15,H/8.2)*zoom,d=-z2,k=cam/(cam+d);
 return{x:W/2+x1*f*k,y:H/2-y1*f*k,s:k,d:d}}
function poly(pts,fill,stroke){ctx.beginPath();pts.forEach((p,i)=>i?ctx.lineTo(p.x,p.y):ctx.moveTo(p.x,p.y));ctx.closePath();if(fill){ctx.fillStyle=fill;ctx.fill()}if(stroke){ctx.strokeStyle=stroke;ctx.lineWidth=1;ctx.stroke()}}
function status(e){if(isBlocked(e.id))return'blocked';return e.status==='blocked'?'open':e.status}
function edgeStyle(e){const s=status(e);
 if(s==='blocked')return{c:P.mode==='blue'?colors.green:colors.slate,w:2.2,dash:[6,5],x:true};
 if(P.mode==='red')return e.on_path&&P.target_reachable?{c:colors.red,w:3,dash:[],arrow:true}:{c:colors.slate,w:1.5,dash:[],arrow:false};
 const exposed=e.status==='open'&&P.target_reachable&&P.route_node_ids.includes(e.source)&&P.route_node_ids.includes(e.target);
 return exposed?{c:colors.amber,w:2.4,dash:[],arrow:true}:{c:colors.slate,w:1.3,dash:[],arrow:false}}
function nodeColor(n){if(n.kind==='decoy')return'#64748b';if(P.mode==='red')return n.kind==='target'?colors.amber:colors.red;
 return P.target_reachable?(n.kind==='target'?colors.amber:colors.cyan):colors.green}
function shape(n,p,r,fill){ctx.beginPath();
 if(n.kind==='entry'){for(let i=0;i<3;i++){const a=-Math.PI/2+i*2*Math.PI/3;ctx.lineTo(p.x+Math.cos(a)*r*1.25,p.y+Math.sin(a)*r*1.25)}ctx.closePath()}
 else if(n.kind==='target'){ctx.rect(p.x-r,p.y-r,2*r,2*r)}
 else if(n.kind==='decoy'){ctx.moveTo(p.x,p.y-r);ctx.lineTo(p.x+r,p.y);ctx.lineTo(p.x,p.y+r);ctx.lineTo(p.x-r,p.y);ctx.closePath()}
 else ctx.arc(p.x,p.y,r,0,7);
 ctx.fillStyle=fill;ctx.fill()}
function draw(now){ctx.clearRect(0,0,W,H);
 if(auto&&!drag&&!pinned)yaw+=0.0025;
 // trust-zone planes
 P.zones.forEach(z=>{const xs=P.nodes.map(n=>n.pos[0]),ys=P.nodes.map(n=>n.pos[1]);const x0=Math.min(...xs)-1.6,x1=Math.max(...xs)+1.6,y0=Math.min(...ys)-1.2,y1=Math.max(...ys)+1.2;
  const q=[[x0,y0],[x1,y0],[x1,y1],[x0,y1]].map(([x,y])=>proj([x,y,z.z]));poly(q,'rgba(60,90,130,.07)','rgba(90,130,180,.28)');
  const l=proj([x0,y1,z.z]);ctx.fillStyle=colors.muted;ctx.font='10px ui-monospace,monospace';ctx.fillText(z.name.toUpperCase(),l.x+6,l.y+13)});
 const items=[];
 P.edges.forEach(e=>{const a=proj(byId[e.source].pos),b=proj(byId[e.target].pos);items.push({d:(a.d+b.d)/2,e,a,b})});
 P.nodes.forEach(n=>items.push({d:proj(n.pos).d,n,p:proj(n.pos)}));
 items.sort((u,v)=>v.d-u.d);
 const pulse=((now-t0)/1400)%1;
 items.forEach(it=>{
  if(it.e){const e=it.e,st=edgeStyle(e),a=it.a,b=it.b;ctx.save();ctx.globalAlpha=Math.max(.45,Math.min(1,1-it.d*.03));
   ctx.strokeStyle=st.c;ctx.lineWidth=st.w;ctx.setLineDash(st.dash);ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.stroke();ctx.setLineDash([]);
   const mx=(a.x+b.x)/2,my=(a.y+b.y)/2,ang=Math.atan2(b.y-a.y,b.x-a.x);
   if(st.arrow){ctx.fillStyle=st.c;ctx.beginPath();ctx.moveTo(mx+Math.cos(ang)*7,my+Math.sin(ang)*7);ctx.lineTo(mx-Math.cos(ang)*5+Math.sin(ang)*5,my-Math.sin(ang)*5-Math.cos(ang)*5);ctx.lineTo(mx-Math.cos(ang)*5-Math.sin(ang)*5,my-Math.sin(ang)*5+Math.cos(ang)*5);ctx.fill()}
   if(st.x){ctx.strokeStyle=st.c;ctx.lineWidth=2.4;ctx.beginPath();ctx.moveTo(mx-5,my-5);ctx.lineTo(mx+5,my+5);ctx.moveTo(mx+5,my-5);ctx.lineTo(mx-5,my+5);ctx.stroke()}
   if(!reduce&&P.mode==='red'&&e.on_path&&P.target_reachable&&st.arrow){ctx.fillStyle='#fff';ctx.beginPath();ctx.arc(a.x+(b.x-a.x)*pulse,a.y+(b.y-a.y)*pulse,3,0,7);ctx.fill()}
   ctx.restore()}
  else{const n=it.n,p=it.p,r=(n.kind==='target'?11:9)*p.s*1.15,fill=nodeColor(n);
   const on=hover===n.id||pinned===n.id;
   ctx.save();ctx.globalAlpha=n.kind==='decoy'?.75:1;
   if(n.kind!=='decoy'||on){ctx.shadowColor=fill;ctx.shadowBlur=on?26:12}
   shape(n,p,r,fill);ctx.restore();
   if(n.kind==='target'){ctx.strokeStyle=fill;ctx.lineWidth=1.5;ctx.strokeRect(p.x-r-4,p.y-r-4,2*r+8,2*r+8)}
   if(on){ctx.strokeStyle='#fff';ctx.lineWidth=2;ctx.beginPath();ctx.arc(p.x,p.y,r+7,0,7);ctx.stroke()}
   ctx.fillStyle=n.kind==='decoy'?colors.muted:colors.text;ctx.font=(on?'600 ':'')+'11.5px ui-sans-serif,system-ui,sans-serif';ctx.textAlign='center';
   ctx.fillText(n.name,p.x,p.y+r+15);ctx.textAlign='left'}});
 requestAnimationFrame(draw)}
function pick(ev){const r=cv.getBoundingClientRect(),mx=ev.clientX-r.left,my=ev.clientY-r.top;let best=null,bd=18;
 P.nodes.forEach(n=>{const p=proj(n.pos),d=Math.hypot(p.x-mx,p.y-my);if(d<bd){bd=d;best=n.id}});return best}
function showTip(id,ev){if(!id){tip.style.display='none';return}const n=byId[id],r=cv.getBoundingClientRect();
 const role={entry:'Entry point',target:'Target',decoy:'Decoy: no route to target',route:'On a route to target'}[n.kind];
 const e=document.createElement('div');tip.textContent='';const b=document.createElement('b');b.textContent=n.name;tip.appendChild(b);
 [n.description,'Zone: '+n.zone+' · Criticality: '+n.criticality,role+' · '+(n.reachable?'reachable':'not reachable')].forEach((t,i)=>{const d=document.createElement('div');d.textContent=t;if(i)d.className='m';tip.appendChild(d)});
 tip.style.display='block';tip.style.left=Math.min(r.width-260,ev.clientX-r.left+14)+'px';tip.style.top=Math.max(4,ev.clientY-r.top+14)+'px'}
cv.addEventListener('pointerdown',e=>{drag={x:e.clientX,y:e.clientY,m:false};cv.setPointerCapture(e.pointerId)});
cv.addEventListener('pointermove',e=>{if(drag){const dx=e.clientX-drag.x,dy=e.clientY-drag.y;if(Math.abs(dx)+Math.abs(dy)>2)drag.m=true;yaw+=dx*.008;pitch=Math.max(-.2,Math.min(1.2,pitch+dy*.006));drag.x=e.clientX;drag.y=e.clientY}
 hover=pick(e);if(!pinned)showTip(hover,e)});
cv.addEventListener('pointerup',e=>{if(drag&&!drag.m){const id=pick(e);pinned=id&&id!==pinned?id:null;showTip(pinned||hover,e)}drag=null});
cv.addEventListener('pointerleave',()=>{hover=null;if(!pinned)tip.style.display='none'});
cv.addEventListener('wheel',e=>{zoom=Math.max(.6,Math.min(2.4,zoom*(1-e.deltaY*.001)));e.preventDefault()},{passive:false});
cv.tabIndex=0;cv.addEventListener('keydown',e=>{const k={ArrowLeft:()=>yaw-=.12,ArrowRight:()=>yaw+=.12,ArrowUp:()=>pitch=Math.min(1.2,pitch+.08),ArrowDown:()=>pitch=Math.max(-.2,pitch-.08),'+':()=>zoom=Math.min(2.4,zoom*1.1),'-':()=>zoom=Math.max(.6,zoom/1.1)}[e.key];if(k){k();e.preventDefault()}});
const rot=document.getElementById('rot');rot.setAttribute('aria-pressed',String(auto));rot.onclick=()=>{auto=!auto;rot.setAttribute('aria-pressed',String(auto))};
document.getElementById('reset').onclick=()=>{yaw=home.yaw;pitch=home.pitch;zoom=home.zoom;pinned=null;tip.style.display='none'};
const rp=document.getElementById('replay');if(rp)rp.onclick=startReplay;
const L=document.getElementById('legend');
const items=P.mode==='red'?[['solid','red','Open attack path'],['solid','slate','Other open edge'],['dash','slate','Blocked by a control (✕)']]
 :[['solid','amber','Exposed route'],['dash','green','Severed by control (✕)'],['solid','slate','Other edge']];
items.concat([['','','▲ entry   ■ target   ◆ decoy   ● on a route']]).forEach(([k,c,t])=>{const s=document.createElement('span');
 if(k){const w=document.createElement('i');w.className='sw'+(k==='dash'?' dash':'');w.style.borderColor=colors[c];s.appendChild(w)}s.appendChild(document.createTextNode(t));L.appendChild(s)});
if(P.mode==='blue')startReplay();
requestAnimationFrame(draw);
</script></body></html>"""


def graph_document_html(payload: Mapping[str, object]) -> str:
    """Complete self-contained HTML for the graph view (also used for previews and tests)."""
    source = json.dumps(payload).replace("</", "<\\/")
    replay = (
        "<button id='replay'>replay sever</button>"
        if payload["mode"] == "blue" and payload["blocked_edge_ids"] else ""
    )
    line = headline(payload)
    return (
        _TEMPLATE.replace("__PAYLOAD__", source)
        .replace("__HEADLINE__", json.dumps(line).replace("</", "<\\/"))
        .replace("__ARIA__", escape(line, quote=True))
        .replace("__REPLAY__", replay)
        .replace("__FALLBACK__", fallback_table_html(payload))
    )


def empty_document_html() -> str:
    return (
        "<body style='margin:0;background:#060b15;color:#8096aa;font:13px system-ui'>"
        f"<div style='padding:28px;border:1px dashed #263243;border-radius:8px'>{escape(EMPTY_MESSAGE)}</div></body>"
    )


def render_graph_3d(report: Mapping[str, object], mode: str, height: int = 1100) -> None:
    """Render the read-only 3D graph inside Streamlit. Shows an empty state before Red has run."""
    import streamlit.components.v1 as components  # imported lazily so the builders stay testable

    payload = graph_payload(report, mode)
    if payload is None:
        components.html(empty_document_html(), height=90)
        return
    rows = len(payload["nodes"]) + len(payload["edges"])
    components.html(graph_document_html(payload), height=min(height, 640 + rows * 30), scrolling=True)
