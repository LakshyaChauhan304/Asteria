"""Celestial Cybernetics presentation views for the isolated Asteria demo."""
from __future__ import annotations

import html
from typing import Mapping


def _s(value: object, fallback: str = "—") -> str:
    return html.escape(str(fallback if value is None or value == "" else value))


def _m(report: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = report.get(key)
    return value if isinstance(value, Mapping) else {}


CSS = """<style>
*{box-sizing:border-box}html,body{margin:0;background:#060b15;color:#f3f5f7;font-family:Georgia,serif}body{background-image:radial-gradient(#8c73351e 1px,transparent 1px);background-size:16px 16px}.top{height:56px;border-bottom:1px solid #1e2632;display:flex;align-items:center;padding:0 40px;gap:28px;font-family:ui-monospace,Consolas,monospace}.brand{font:26px Georgia,serif;white-space:nowrap}.star{color:#f2b632;font-size:23px;margin-right:8px}.brand small{font:9px ui-monospace,Consolas,monospace;color:#9eabbc;letter-spacing:1.4px;margin-left:10px}.top-status{font-size:10px;letter-spacing:1.5px;color:#e7ae35}.top-status.mint{color:#37e0ba}.nav{margin-left:auto;display:flex;gap:27px;height:56px;align-items:center}.nav span{height:56px;padding-top:22px;color:#aeb8c7;font-size:10px;letter-spacing:1.3px;white-space:nowrap}.nav .on{color:#f0b62e;border-bottom:2px solid #e9ae27}.shell{min-height:944px;padding:38px 40px 28px}.layout{display:grid;grid-template-columns:300px minmax(580px,1fr) 340px;gap:24px;min-height:815px}.kicker{font:10px ui-monospace,Consolas,monospace;letter-spacing:1.8px;color:#a8b7c8}.display{font:50px Georgia,serif;line-height:.96;letter-spacing:-1.5px;margin:10px 0 17px}.copy{color:#c2b6a5;font-size:17px;line-height:1.45;max-width:270px}.rule{height:1px;background:#46505d;width:64px;margin:27px 0}.fact{margin:22px 0}.fact label,.right-title,.section-label{display:block;color:#a8b7c8;font:10px ui-monospace,Consolas,monospace;letter-spacing:1.7px;margin-bottom:8px}.fact b{display:block;font:19px Georgia,serif;font-weight:400}.fact b.mint{color:#33d7b5}.fact b.red{color:#ff6972}.fact b.mono{font:14px ui-monospace,Consolas,monospace}.center{position:relative;min-height:760px}.right{padding-left:0}.right-title{margin:2px 0 23px}.button{display:block;border:1px solid #e6ad2a;color:#e9b334;font:10px ui-monospace,Consolas,monospace;letter-spacing:1.6px;padding:15px 16px;margin-top:25px;text-decoration:none}.button i{float:right;font-style:normal}.footer{position:fixed;bottom:0;left:0;right:0;height:35px;border-top:1px solid #1e2632;padding:11px 40px;color:#9dacbc;font:9px ui-monospace,Consolas,monospace;letter-spacing:1.3px;display:flex;justify-content:space-between;background:#060b15}.footer b{color:#38e1bf;font-weight:400}.orb{position:absolute;left:50%;top:46%;width:530px;height:530px;transform:translate(-50%,-50%);border:1px solid #1b8d8a;border-radius:50%;box-shadow:0 0 0 45px #0a12211c}.orb:before,.orb:after{content:"";position:absolute;border:1px dotted #334359;border-radius:50%;inset:44px}.orb:after{inset:132px;border-style:solid;border-color:#293443}.node{position:absolute;white-space:nowrap;text-align:center}.node .pin{display:block;width:26px;height:26px;margin:auto;border:1px solid #a9b9c9;border-radius:50%;background:#07101d;box-shadow:0 0 0 5px #07101d}.node .pin:after{content:"";display:block;width:7px;height:7px;margin:8px;border:1px solid #f2f4f7;border-radius:50%}.node b{display:block;font:25px Georgia,serif;font-weight:400;margin-top:2px}.node small{display:block;color:#8e9dad;font:italic 12px Georgia,serif}.node.amber b{color:#f1b22e}.node.amber .pin{border-color:#e2a62b}.node.amber .pin:after{background:#e2a62b}.node.red b{color:#ff6b73}.node.red .pin{border-color:#ff6770}.node.red .pin:after{background:#ff6770}.node.mint b{color:#34d7b6}.node.mint .pin{border-color:#35dbba}.node.mint .pin:after{background:#35dbba}.n-api{left:14%;top:39%}.n-id{left:54%;top:55%}.n-store{left:68%;top:17%}.n-ext{left:9%;top:72%}.n-top{left:43%;top:-10%}.n-bottom{left:43%;bottom:-13%}.n-left{left:-3%;top:52%}.n-right{right:-9%;top:60%}.link{position:absolute;height:1px;border-top:2px dashed #ef6671;transform-origin:left}.l1{left:31%;top:46%;width:27%;transform:rotate(25deg)}.l2{left:62%;top:51%;width:25%;transform:rotate(-59deg)}.l3{left:18%;top:69%;width:27%;transform:rotate(-45deg)}.selector{display:flex;gap:9px;margin:16px 0 20px}.selector span{border:1px solid #2a3442;color:#afbdca;font:10px ui-monospace,Consolas,monospace;padding:10px 12px}.selector .on{background:#e4ac2d;border-color:#e4ac2d;color:#07101b}.stage-name{font:46px Georgia,serif;margin:13px 0 27px}.inspect{border-top:1px solid #202936;padding:16px 0}.inspect label{color:#a8b7c8;font:10px ui-monospace,Consolas,monospace;letter-spacing:1.7px}.inspect p{font:19px Georgia,serif;color:#d9d0c2;margin:9px 0 0;line-height:1.4}.hyp{position:absolute;bottom:42px;border-top:1px solid #202936;padding-top:17px;color:#c2b6a5;font-size:16px;line-height:1.45}.score-orb{position:absolute;left:50%;top:43%;transform:translate(-50%,-50%);width:490px;height:490px;border:3px solid #ecaf29;border-radius:50%;display:grid;place-items:center}.score-orb:before,.score-orb:after{content:"";position:absolute;border:2px dotted #69748b;border-radius:50%;inset:60px}.score-orb:after{inset:125px}.score{position:relative;text-align:center;font:54px Georgia,serif}.score small{display:block;color:#9dacbc;font:10px ui-monospace,Consolas,monospace;letter-spacing:1.8px;margin:12px}.model-label{position:absolute;top:-31px;left:50%;transform:translateX(-50%);color:#efb128;font:10px ui-monospace,Consolas,monospace;letter-spacing:1.7px}.eclass{border-top:1px solid #202936;padding:17px 0}.eclass b{display:block;font:10px ui-monospace,Consolas,monospace;letter-spacing:1.6px;color:#efb128}.eclass p{font:18px Georgia,serif;color:#c7bcad;line-height:1.35;margin:10px 0}.eclass.mint b{color:#39dcbc}.eclass.red b{color:#ff6d74}.queue{padding:150px 0 0}.case{border-top:1px solid #283241;padding:16px 0 25px;display:grid;grid-template-columns:1fr 150px 130px;gap:20px}.case h2{font:43px Georgia,serif;margin:0}.case p{color:#c0b5a5;margin:7px 0}.case div{align-self:center}.case b{font:12px ui-monospace,Consolas,monospace;color:#eef2f5}.case .verified{color:#33d7b6;font:22px Georgia,serif}.case.empty{border:2px dashed #68707d;color:#78818d;opacity:.75}.case.empty h2{color:#78818d}.mini-stages{display:flex;gap:7px;margin-top:16px}.mini-stages span{border:1px solid #35d9b8;color:#35d9b8;padding:8px 10px;font:10px ui-monospace,Consolas,monospace}.progress{width:190px;height:190px;border:6px dashed #35d9b8;border-radius:50%;display:grid;place-items:center;text-align:center;margin:120px auto 55px}.progress b{font:45px Georgia,serif;font-weight:400}.progress small{display:block;color:#aab8c8;font:10px ui-monospace,Consolas,monospace;letter-spacing:1.4px}.path-list{border-top:1px solid #202936}.path-list div{padding:16px 0;border-bottom:1px solid #202936}.path-list b{color:#f0b32c;font:10px ui-monospace,Consolas,monospace;letter-spacing:1.5px}.path-list p{font:22px Georgia,serif;margin:5px 0 0}.path-list small{font:italic 13px Georgia,serif;color:#8b98a9}.hand{position:absolute;right:0;bottom:50px;color:#e5ad2e;font:10px ui-monospace,Consolas,monospace;letter-spacing:1.8px}.control-ring{position:absolute;left:40%;top:42%;height:80px;width:80px;border:2px solid #38dab9;border-radius:50%;box-shadow:inset 0 0 0 14px #07101d}.steps{position:absolute;left:31%;bottom:70px;display:flex;align-items:center;gap:36px;color:#35d9ba;font:10px ui-monospace,Consolas,monospace}.steps span{width:33px;height:33px;border:1px solid #35d9ba;padding:10px 12px;position:relative}.steps span:not(:last-child):after{content:"";position:absolute;height:1px;width:36px;background:#35d9ba;left:32px;top:16px}.spec{border-top:1px solid #202936;padding:16px 0}.spec p{font:20px Georgia,serif;margin:8px 0}.checks{margin-top:13px}.checks div{border-top:1px solid #202936;padding:18px 0;color:#c5b9aa;font-size:18px;line-height:1.4}.checks b{color:#36dbba;margin-right:15px}.before-after{display:grid;grid-template-columns:1fr 90px 1fr;gap:18px;align-items:center;margin-top:190px}.mini-orb{height:310px;border:1px solid #2b3543;border-radius:50%;position:relative}.mini-orb:before{content:"";position:absolute;inset:55px;border:1px dotted #344257;border-radius:50%}.mini-orb .dot{position:absolute;width:18px;height:18px;border:1px solid #a8b7c8;border-radius:50%;background:#07101d}.mini-orb .red-dot{left:15%;bottom:23%;border-color:#ff6c75}.mini-orb .amber-dot{right:13%;top:20%;border-color:#eeae2d}.mini-orb .mid-dot{left:45%;top:45%}.control-arrow{text-align:center;color:#39dbba;font:10px ui-monospace,Consolas,monospace;letter-spacing:1.5px}.control-arrow b{display:block;font-size:36px;font-weight:400;margin-top:12px}.under{text-align:center;font:10px ui-monospace,Consolas,monospace;letter-spacing:1.5px}.under.red{color:#ff6c75}.under.mint{color:#36dbba}.trace{padding-top:150px}.trace-row{display:grid;grid-template-columns:62px 30px 1fr 200px 200px;border-top:1px solid #202936;padding:14px 0;align-items:center}.trace-row .num{font:10px ui-monospace,Consolas,monospace;color:#e9b132}.trace-row .bullet{width:10px;height:10px;border-radius:50%;background:#eaf1f6;box-shadow:0 0 11px #eaf1f6}.trace-row h2{font:26px Georgia,serif;margin:0}.trace-row small{color:#9dacbc;font:12px ui-monospace,Consolas,monospace}.trace-row b{font:10px ui-monospace,Consolas,monospace;letter-spacing:1.2px}.trace-row.red .bullet{background:#ff6972;box-shadow:0 0 11px #ff6972}.trace-row.mint .bullet{background:#35d9b8;box-shadow:0 0 11px #35d9b8}.trace-row.mint .num,.trace-row.mint b{color:#35d9b8}.trace-row.red .num,.trace-row.red b{color:#ff6972}.report{margin-top:38px}.report .fact{margin:30px 0}@media(max-width:1100px){.layout{grid-template-columns:230px 1fr}.right{display:none}.nav{gap:12px}.shell{padding:25px}.orb{transform:translate(-50%,-50%) scale(.8)}.queue{padding-top:80px}}@media(max-width:720px){.top{padding:0 18px}.top-status,.nav{display:none}.shell{padding:22px}.layout{display:block}.center{min-height:590px}.display{font-size:42px}.orb{transform:translate(-50%,-50%) scale(.62)}.footer{padding:11px 18px;font-size:7px}.queue{padding-top:35px}.case{grid-template-columns:1fr}.before-after{margin-top:70px;grid-template-columns:1fr}.control-arrow{transform:rotate(90deg)}}
</style>"""

CSS = CSS.replace(
    "</style>",
    """.interactive{cursor:pointer}.interactive:hover{filter:brightness(1.3)}
.dialog{display:none;position:fixed;z-index:20;inset:0;background:#02050bd9;place-items:center;padding:24px}
.dialog.open{display:grid}.dialog-card{position:relative;width:min(560px,94vw);border:1px solid #e8ae2b;background:#07101c;padding:30px;box-shadow:0 0 55px #000}
.dialog-card h2{font:34px Georgia,serif;margin:0 0 14px}.dialog-card p{color:#c8bdaf;font:18px Georgia,serif;line-height:1.45}.dialog-card button{position:absolute;right:13px;top:10px;background:none;border:0;color:#e8ae2b;font:22px monospace;cursor:pointer}
.red-center .orb,.blue-center .orb{top:42%;transform:translate(-50%,-50%) scale(.84);transform-origin:center}.verify-center{min-height:670px;overflow:hidden}.verify-center .before-after{margin-top:115px}.signal-console{padding-top:145px;display:grid;grid-template-columns:repeat(3,1fr);gap:16px}.signal-card{min-height:250px;border:1px solid #2c3645;background:#09111e;padding:20px;position:relative}.signal-card:before{content:"";position:absolute;top:0;left:0;width:100%;height:3px;background:#e8ae2b}.signal-card:nth-child(2):before{background:#a3b2c2}.signal-card:nth-child(3):before{background:#3adaba}.signal-card label{font:10px ui-monospace,Consolas,monospace;color:#9dacbc;letter-spacing:1.5px}.signal-card h2{font:26px Georgia,serif;margin:38px 0 10px}.signal-card b{font:36px Georgia,serif;color:#e8ae2b}.signal-card:nth-child(2) b{color:#eef3f7}.signal-card:nth-child(3) b{color:#3adaba}.signal-card p{color:#b9ad9e;font-size:15px;line-height:1.4}.signal-link{height:1px;background:#314353;margin:44px 11% 0}.signal-summary{border-top:1px solid #263242;padding-top:20px;margin-top:28px;color:#c6b9aa;font:18px Georgia,serif}
@media(max-width:900px){.red-center .orb,.blue-center .orb{transform:translate(-50%,-50%) scale(.62)}.signal-console{grid-template-columns:1fr;padding-top:35px}}
</style>""",
)
CSS = CSS.replace("</style>", ".top{position:sticky;top:0;z-index:100;background:#060b15eF;backdrop-filter:blur(12px)}.shell{scroll-margin-top:58px}</style>")

CSS = CSS.replace(".nav span{", ".nav span,.nav a{").replace(".nav .on{", ".nav .on{")
CSS = CSS.replace("</style>", ".top-action{border:1px solid #e8ae2b;color:#e8ae2b;text-decoration:none;padding:8px 11px;font:9px ui-monospace,Consolas,monospace;letter-spacing:1.2px;white-space:nowrap}.top-action:hover{background:#e8ae2b;color:#07101b}</style>")
CSS = CSS.replace("</style>", ".flow-map{width:100%;margin-top:165px;display:grid;grid-template-columns:1fr 42px 1fr;gap:12px 8px;align-items:center}.flow-card{min-height:135px;border:1px solid #3a495c;background:#09121f;padding:16px;position:relative;cursor:pointer}.flow-card:before{content:'';position:absolute;left:0;top:0;height:3px;width:100%;background:#94a9bd}.flow-card.red:before{background:#ff6873}.flow-card.amber:before{background:#e7ae2c}.flow-card.mint:before{background:#39daba}.flow-card.blue:before{background:#6da9ff}.flow-card label{display:block;color:#a8b7c8;font:10px ui-monospace,Consolas,monospace;letter-spacing:1.5px}.flow-card h2{font:24px Georgia,serif;margin:14px 0 7px}.flow-card p{font:13px Georgia,serif;color:#bcae9e;line-height:1.35;margin:0}.flow-card .state{font:10px ui-monospace,Consolas,monospace;letter-spacing:1px;color:#39daba}.flow-arrow{color:#9fb0bf;text-align:center;font:25px Georgia,serif}.flow-map .wide{grid-column:1/4;min-height:105px}.flow-map .wide:after{content:'';position:absolute;left:50%;top:-25px;height:25px;border-left:1px dashed #405268}.flow-caption{margin-top:23px;border-top:1px solid #243040;padding:15px 0;color:#acbdca;font:11px ui-monospace,Consolas,monospace;letter-spacing:1px}@media(max-width:900px){.flow-map{margin-top:55px}.flow-card h2{font-size:19px}}</style>")

# Desktop sizing belongs to the viewport, not a fixed mockup canvas. These
# overrides preserve the smaller breakpoints above while using wide monitors.
CSS = CSS.replace(
    "</style>",
    ".shell{min-height:calc(100vh - 56px);padding:clamp(24px,3vh,48px) clamp(28px,4vw,72px) 52px}.layout{grid-template-columns:minmax(280px,.8fr) minmax(680px,2fr) minmax(300px,.9fr);gap:clamp(24px,2vw,46px);min-height:calc(100vh - 156px)}.center{min-height:clamp(620px,calc(100vh - 170px),900px)}.flow-map{margin-top:clamp(52px,8vh,118px)}.blue-center .flow-map,.red-center .flow-map{margin-top:clamp(52px,9vh,128px)}@media(min-width:1600px){.display{font-size:clamp(52px,3.3vw,78px)}.copy{max-width:330px;font-size:clamp(17px,1vw,21px)}.flow-card{min-height:clamp(150px,17vh,220px);padding:clamp(18px,1.2vw,28px)}.flow-card h2{font-size:clamp(24px,1.55vw,34px)}.flow-card p{font-size:clamp(13px,.85vw,17px)}}</style>",
)


def _action_link(report: Mapping[str, object]) -> str:
    state = str(report.get("workflow_state", "created"))
    actions = {
        "created": ("initialize", "Detection Engine", "INITIALIZE MODELS"),
        "models_ready": ("detect", "Detection Engine", "RUN DETECTION"),
        "detected": ("investigate", "Investigations", "START INVESTIGATION"),
        "investigated": ("red", "Red Simulation", "RUN RED SIMULATION"),
        "red_simulated": ("blue", "Blue Response", "GENERATE BLUE PROPOSAL"),
        "blue_recommended": ("approve", "Blue Response", "APPROVE SANDBOX CONTROL"),
        "awaiting_approval": ("apply", "Verification", "APPLY SIMULATION"),
        "remediation_in_progress": ("verify", "Verification", "RUN VERIFIER"),
        "verified": ("", "Audit & Reports", "OPEN AUDIT"),
    }
    action, view, label = actions.get(state, ("", "Command Center", "VIEW COMMAND"))
    query = f"?view={view.replace(' ', '%20')}" + (f"&action={action}" if action else "")
    return f'<a class="top-action" href="/{query}" target="_top">{label}</a>'


def _route_link(view: str, label: str) -> str:
    query = view.replace(" ", "%20").replace("&", "%26")
    return f'<a class="button" href="/?view={query}" target="_top">{label}<i>→</i></a>'


def _case_link(event_id: str, label: str) -> str:
    return f'<a class="button" href="/?view=Investigations&action=investigate&event={_s(event_id)}" target="_top">{label}<i>→</i></a>'


def _header(active: str, report: Mapping[str, object]) -> str:
    pages = (("Command Center", "COMMAND"), ("Detection Engine", "DETECTIONS"), ("Investigations", "INVESTIGATIONS"), ("Red Simulation", "RED"), ("Blue Response", "BLUE"), ("Verification", "VERIFICATION"), ("Audit & Reports", "AUDIT"))
    nav = "".join(
        f'<a href="/?view={name.replace(" ", "%20").replace("&", "%26")}" target="_top" class="{"on" if name == active else ""}">{short}</a>'
        for name, short in pages
    )
    verified = _m(report, "verification").get("status") == "verified"
    script = '''<div class="dialog" id="detail-dialog"><div class="dialog-card"><button onclick="closeDetail()">×</button><h2 id="detail-title"></h2><p id="detail-body"></p></div></div><script>function openDetail(title,body){document.getElementById('detail-title').textContent=title;document.getElementById('detail-body').textContent=body;document.getElementById('detail-dialog').classList.add('open')}function closeDetail(){document.getElementById('detail-dialog').classList.remove('open')}document.addEventListener('click',function(event){if(event.target.id==='detail-dialog'){closeDetail()}});document.addEventListener('DOMContentLoaded',function(){document.querySelectorAll('.node,.selector span,.path-list div,.spec,.checks div,.trace-row,.eclass,.case,.signal-card,.mini-orb').forEach(function(item){item.classList.add('interactive');item.addEventListener('click',function(){var title=item.textContent.trim().replace(/\s+/g,' ').slice(0,70)||'Asteria detail';openDetail(title,'Recorded demo detail. This view presents evidence, simulation, approval, or verification state only. It does not perform live infrastructure actions.')})})})</script>'''
    return f'''{CSS}<header class="top"><div class="brand"><span class="star">✦</span>Asteria <small>AUTONOMOUS THREAT<br>INTERCEPT ENGINE</small></div><span class="top-status">● SIMULATION · ISOLATED GRAPH</span><span class="top-status mint">● {"TRACE VERIFIED" if verified else "TRACE PENDING"}</span><nav class="nav">{nav}</nav>{_action_link(report)}</header>{script}'''


def _footer() -> str:
    return '<footer class="footer"><span>REASONING ≠ AUTHORIZATION ≠ EXECUTION</span><b>● TOPOLOGY_MESH ACTIVE · LAT 42.109 · RNG 840MS</b><span>[ SIMULATION ONLY · NO LIVE INFRASTRUCTURE ]</span></footer>'


def _orbital(report: Mapping[str, object], mode: str = "command") -> str:
    finding = _m(report, "red_finding")
    verified = _m(report, "verification").get("status") == "verified"
    links = '<i class="link l1"></i><i class="link l2"></i><i class="link l3"></i>' if mode in {"red", "verify"} else ""
    top = '<div class="node amber n-top"><span class="pin"></span><b>Telemetry</b><small>UNSW-NB15-8318</small></div>' if mode == "command" else ""
    bottom = '<div class="node mint n-bottom"><span class="pin"></span><b>Blue agent</b><small>control proposed</small></div>' if mode == "command" else ""
    left = '<div class="node mint n-left"><span class="pin"></span><b>Remediation</b><small>ingress_restricted</small></div>' if mode == "command" else ""
    right = '<div class="node red n-right"><span class="pin"></span><b>Red agent</b><small>candidate path</small></div>' if mode == "command" else ""
    status = "passed" if verified else ("reachable" if _m(finding, "candidate_path").get("reachable") else "pending")
    return f'''<div class="orb">{links}{top}{bottom}{left}{right}<div class="node n-api"><span class="pin"></span><b>Public API</b><small>simulated entry point</small></div><div class="node n-id"><span class="pin"></span><b>Workload identity</b><small>simulated trust hop</small></div><div class="node amber n-store"><span class="pin"></span><b>Sensitive data store</b><small>protected target</small></div><div class="node red n-ext"><span class="pin"></span><b>External actor</b><small>untrusted origin</small></div><span style="display:none">{status}</span></div>'''


def _flow_card(tone: str, label: str, title: str, detail: str, state: str = "") -> str:
    suffix = f'<span class="state">{state}</span>' if state else ""
    return f'<article class="flow-card {tone}" onclick="openDetail(\'{_s(title)}\', \'{_s(detail)}\')"><label>{label}</label><h2>{title}</h2><p>{detail}</p>{suffix}</article>'


def _box_flow(kind: str) -> str:
    if kind == "red":
        left = _flow_card("red", "01 · ORIGIN", "External actor", "Untrusted origin in the isolated attack graph.", "SIMULATED")
        right = _flow_card("amber", "02 · ENTRY", "Public API", "Simulated external entry point.")
        end = _flow_card("wide red", "03 → 04 · PATH", "Workload identity → Sensitive data store", "Candidate path is reachable only in the isolated graph.", "REVIEW WITH BLUE AGENT")
        caption = "Candidate path only. This simulation does not scan or touch live systems."
    elif kind == "blue":
        left = _flow_card("blue", "01 · INPUT", "Red finding", "Blue Agent receives a provenanced simulated finding.")
        right = _flow_card("mint", "02 · PROPOSAL", "Ingress restriction", "Control proposal targets the simulated Public API.", "PROPOSAL ONLY")
        end = _flow_card("wide amber", "03 · HUMAN GATE", "Approval required before the sandbox transition", "The proposal does not authorize itself and cannot change infrastructure.", "APPROVED IN DEMO")
        caption = "Policy, human approval, and sandbox application remain distinct stages."
    else:
        left = _flow_card("red", "BEFORE", "Path reachable", "Verifier records the original simulated reachability result.")
        right = _flow_card("mint", "CONTROL", "Ingress restriction active", "The sandbox control is active in the in-memory demo environment.")
        end = _flow_card("wide mint", "INDEPENDENT RECHECK", "Path unreachable", "The verifier recomputes reachability after the control without trusting agent assertions.", "VERIFIED")
        caption = "Verification evaluates approval linkage, audit provenance, control state, and the recomputed path."
    return f'<div class="flow-map">{left}<span class="flow-arrow">→</span>{right}{end}</div><div class="flow-caption">{caption}</div>'


def _left(title: str, copy: str, facts: tuple[tuple[str, str, str], ...]) -> str:
    details = "".join(f'<div class="fact"><label>{label}</label><b class="{tone}">{value}</b></div>' for label, value, tone in facts)
    return f'<aside><span class="kicker">{title.upper()}</span><h1 class="display">{title}</h1><p class="copy">{copy}</p><div class="rule"></div>{details}</aside>'


def _telemetry_feed(report: Mapping[str, object]) -> str:
    items = report.get("telemetry")
    if not isinstance(items, list) or not items:
        return '<section class="telemetry-feed"><span class="kicker">REAL UNSW-NB15 TELEMETRY · INITIALIZE MODELS TO LOAD 20 RECORDS</span></section>'
    rows = "".join(
        f'<tr><td>{_s(item.get("event_id"))}</td><td>{_s(item.get("protocol"))}</td><td>{_s(item.get("service"))}</td><td>{_s(item.get("state"))}</td><td>{_s(item.get("duration"))}</td><td>{_s(item.get("source_bytes"))} / {_s(item.get("destination_bytes"))}</td></tr>'
        for item in items[:20]
        if isinstance(item, Mapping)
    )
    return f'''<section class="telemetry-feed"><span class="kicker">REAL UNSW-NB15 TELEMETRY · 20 SAMPLED RECORDS · READ ONLY</span><table><thead><tr><th>EVENT</th><th>PROTO</th><th>SERVICE</th><th>STATE</th><th>DURATION</th><th>BYTES S / D</th></tr></thead><tbody>{rows}</tbody></table></section>'''


def command_center_html(report: Mapping[str, object]) -> str:
    inv = _m(report, "investigation")
    event = _s(report.get("selected_event_id"), "UNSW-NB15-8318")
    left = _left("AST–0001", "Exposed API path to a sensitive data store, intercepted at ingress.", (("DETECTION EVENT", event, "mono"), ("SIGNALS", "3 independent outputs", ""), ("HUMAN GATE", _s(_m(report, "approval").get("decision"), "Approved"), "mint"), ("VERIFICATION", "Passed · path unreachable", "mint")))
    stages = "".join(f'<span class="{"on" if index == 1 else ""}">{label}</span>' for index, label in enumerate(("01", "02", "03", "04A", "04B", "05", "06", "07"), 1))
    right = f'''<section class="right"><span class="right-title">STAGE INSPECTOR · READ ONLY</span><div class="selector">{stages}</div><h2 class="stage-name">Telemetry</h2><div class="inspect"><label>INPUT</label><p>Network flow record ingested for scoring. Read only.</p></div><div class="inspect"><label>OUTPUT</label><p>{event}</p></div><div class="inspect"><label>AUTHORITY</label><p>None · data only</p></div>{_route_link("Audit & Reports", "OPEN IN AUDIT")}<div class="hyp"><span class="kicker">HYPOTHESIS · UNCONFIRMED</span><br>A detection signal may warrant investigation. It is not proof of compromise and never grants authorization.</div></section>'''
    return _header("Command Center", report) + f'<main class="shell"><div class="layout">{left}<section class="center">{_orbital(report)}</section>{right}</div></main>' + _footer()


def detection_html(report: Mapping[str, object]) -> str:
    inv = _m(report, "investigation")
    evidence = _m(inv, "evidence")
    score = _s(evidence.get("model_score"), "0.9998386")
    left = _left("Signals, not verdicts", "Three independent outputs are recorded for one event. A signal may warrant investigation; it is not proof of compromise.", (("EVENT", _s(report.get("selected_event_id"), "UNSW-NB15-8318"), "mono"), ("SELECTED MODEL", "XGBoost", ""), ("INDEPENDENT OUTPUTS", "3", "")))
    center = f'''<section class="center"><div class="signal-console"><article class="signal-card"><label>MODEL 01 · BASELINE</label><h2>Logistic regression</h2><b>DETECTED</b><p>Supervised baseline output retained with source provenance.</p></article><article class="signal-card"><label>MODEL 02 · ANOMALY</label><h2>Isolation forest</h2><b>ANOMALY</b><p>Unsupervised anomaly signal remains independent from other models.</p></article><article class="signal-card"><label>MODEL 03 · SELECTED</label><h2>XGBoost runtime</h2><b>{score}</b><p>Artifact-backed score for {_s(report.get("selected_event_id"), "UNSW-NB15-8318")}.</p></article></div><div class="signal-link"></div><p class="signal-summary">The investigator receives three model signals as recorded facts. It keeps model outputs separate from inferences, hypotheses, authorization, and execution.</p>{_telemetry_feed(report)}</section>'''
    right = f'''<section class="right"><span class="right-title">EVIDENCE CLASSES</span><div class="eclass"><b>I · FACT</b><p>Each model output remains a recorded detection signal with its own provenance.</p></div><div class="eclass mint"><b>II · INFERENCE</b><p>The investigator may request more read-only context after reviewing the signals.</p></div><div class="eclass red"><b>III · HYPOTHESIS</b><p>Potential compromise stays unconfirmed until independent evidence supports it.</p></div>{_route_link("Investigations", "OPEN AST–0001")}</section>'''
    return _header("Detection Engine", report) + f'<main class="shell"><div class="layout">{left}{center}{right}</div></main>' + _footer()


def investigations_html(report: Mapping[str, object]) -> str:
    event = _s(report.get("selected_event_id"), "UNSW-NB15-8318")
    cases = report.get("detection_cases")
    case_items = cases if isinstance(cases, list) else []
    left = _left("Case queue", "Twenty real UNSW-NB15 records become individually selectable investigation cases after detection.", (("LOADED", str(len(case_items)), "mint"), ("SELECTED", event, "mono"), ("FILTER", "ALL   SIGNALS   REVIEW", "mono")))
    stages = ''.join(f'<span>{item}</span>' for item in ("01", "02", "03", "04A", "04B", "05", "06", "07"))
    queue_rows = "".join(
        f'<article class="case"><div><h2>AST–{index:04d}</h2><p>UNSW-NB15 detection record · {item.get("signal_count", 0)} model signals</p><div class="mini-stages">{stages}</div></div><div><label class="kicker">EVENT</label><b>{_s(item.get("event_id"))}</b></div><div><label class="kicker">XGBOOST</label><p class="verified">{_s(item.get("xgboost_probability"))}</p>{_case_link(str(item.get("event_id")), "SELECT CASE")}</div></article>'
        for index, item in enumerate(case_items, 1)
        if isinstance(item, Mapping)
    ) or '<article class="case empty"><div><h2>NO DETECTION RECORDS YET</h2><p>Run detection to create the 20-case queue.</p></div></article>'
    center = f'''<section class="center queue"><span class="kicker">CASE QUEUE · REAL UNSW-NB15 RUN · {len(case_items)} RECORDS</span>{queue_rows}</section>'''
    selected_title = "AST–0001" if report.get("selected_event_id") else "SELECT A CASE"
    right = f'<section class="right"><span class="right-title">SELECTED CASE</span><h2 class="stage-name">{selected_title}</h2><div class="progress"><div><b>{"7 / 7" if report.get("selected_event_id") else "0 / 7"}</b><small>STAGES</small></div></div><div class="fact"><label>EVENT</label><b class="mono">{event}</b></div><div class="fact"><label>WORKFLOW</label><b class="mint">{_s(report.get("workflow_state"))}</b></div>{_route_link("Audit & Reports", "OPEN CASE")}</section>'
    return _header("Investigations", report) + f'<main class="shell"><div class="layout">{left}{center}{right}</div></main>' + _footer()


def red_html(report: Mapping[str, object]) -> str:
    left = _left("Candidate path", "A simulated adversary route in an isolated graph. It holds no authority and touches no live system.", (("SCENARIO", "Exposed API", ""), ("AGENT", "Red agent · 04A", ""), ("AUTHORITY", "None · simulated", "red"), ("RESULT", "Reachable in the isolated graph", "")))
    center = f'<section class="center red-center">{_box_flow("red")}<span class="hand">HAND TO BLUE AGENT</span></section>'
    right = '<section class="right"><span class="right-title">PATH STEPS</span><div class="path-list"><div><b>01</b><p>External actor</p><small>untrusted origin</small></div><div><b>02</b><p>Public API</p><small>simulated entry point</small></div><div><b>03</b><p>Workload identity</p><small>simulated trust hop</small></div><div><b>04</b><p>Sensitive data store</p><small>protected target</small></div></div></section>'
    return _header("Red Simulation", report) + f'<main class="shell"><div class="layout">{left}{center}{right}</div></main>' + _footer()


def blue_html(report: Mapping[str, object]) -> str:
    items = report.get("recommendations")
    rec = items[0] if isinstance(items, list) and items and isinstance(items[0], Mapping) else {}
    left = _left("Control proposed", "A defensive control is proposed against the simulated path. A proposal grants nothing; the human gate decides.", (("CONTROL", _s(rec.get("target_control_id"), "ingress_restricted"), "mint mono"), ("TARGET", "Public API · entry point", ""), ("AUTHORITY", "None · proposal only", ""), ("HUMAN GATE", _s(_m(report, "approval").get("decision"), "Approved"), "mint")))
    center = f'<section class="center blue-center">{_box_flow("blue")}</section>'
    right = f'<section class="right"><span class="right-title">CONTROL SPECIFICATION</span><div class="spec"><label>TYPE</label><p>Ingress restriction</p></div><div class="spec"><label>SCOPE</label><p>Public API</p></div><div class="spec"><label>EXECUTION</label><p>Sandbox only</p></div>{_route_link("Verification", "VIEW VERIFICATION")}</section>'
    return _header("Blue Response", report) + f'<main class="shell"><div class="layout">{left}{center}{right}</div></main>' + _footer()


def verification_html(report: Mapping[str, object]) -> str:
    left = _left("Path unreachable", "The verifier re-ran the path query after the approved sandbox control. It is independent of the agents that proposed it.", (("RESULT", "Passed · path unreachable", "mint"), ("CONTROL", "ingress_restricted", "mono"), ("VERIFIER", "07 · independent", "")))
    center = '''<section class="center verify-center">''' + _box_flow("verification") + '''</section>'''
    right = f'<section class="right"><span class="right-title">VERIFIER CHECKS</span><div class="checks"><div><b>■</b>Path query re-run after the control</div><div><b>■</b>External actor to sensitive data store: unreachable</div><div><b>■</b>Result recorded as verified state</div></div><div class="fact"><label>VERIFIED STATE</label><b class="mint">Passed</b></div>{_route_link("Audit & Reports", "OPEN IN AUDIT")}</section>'
    return _header("Verification", report) + f'<main class="shell"><div class="layout">{left}{center}{right}</div></main>' + _footer()


def audit_html(report: Mapping[str, object]) -> str:
    left = _left("Recorded trace", "Reasoning, authorization and execution stay separate at every step.", (("LAYERS", "• REASONING", ""), ("", "• AUTHORIZATION", "mono"), ("", "• EXECUTION", ""), ("", "• VERIFICATION", "mint mono")))
    rows = (("01", "Telemetry", _s(report.get("selected_event_id"), "UNSW-NB15-8318"), "DATA", ""), ("02", "Detection", "3 independent outputs", "REASONING", ""), ("03", "Investigator", "AST-0001", "REASONING", ""), ("04A", "Red agent", "candidate path", "REASONING · SIM", "red"), ("04B", "Blue agent", "control proposed", "REASONING · PROPOSAL", "mint"), ("05", "Approval", "approved", "AUTHORIZATION", ""), ("06", "Remediation", "ingress_restricted", "EXECUTION", ""), ("07", "Verifier", "passed · path unreachable", "VERIFICATION", "mint"))
    trace = ''.join(f'<div class="trace-row {tone}"><span class="num">{num}</span><i class="bullet"></i><h2>{name}</h2><small>{detail}</small><b>{layer}</b></div>' for num, name, detail, layer, tone in rows)
    center = f'<section class="center trace"><span class="kicker">RECORDED TRACE · 8 EVENTS</span>{trace}</section>'
    right = '<section class="right report"><span class="right-title">REPORT</span><div class="fact"><label>CASE</label><b>AST–0001</b></div><div class="fact"><label>INTEGRITY</label><b class="mint">Trace verified</b></div><div class="fact"><label>TRACE ID</label><b class="mono">[TRACE ID]</b></div><a class="button">EXPORT PDF <i>↓</i></a><a class="button">EXPORT MARKDOWN <i>↓</i></a><a class="button">EXPORT JSON <i>↓</i></a></section>'
    return _header("Audit & Reports", report) + f'<main class="shell"><div class="layout">{left}{center}{right}</div></main>' + _footer()


def screen_html(report: Mapping[str, object], view: str) -> str:
    views = {"Command Center": command_center_html, "Detection Engine": detection_html, "Investigations": investigations_html, "Red Simulation": red_html, "Blue Response": blue_html, "Verification": verification_html, "Audit & Reports": audit_html}
    return views.get(view, command_center_html)(report)


def pipeline_svg(state: str) -> str:
    return f'<div style="padding:12px;color:#e9b132;font:11px monospace">RECORDED PIPELINE · {_s(state).upper()}</div>'


def graph_svg(finding: Mapping[str, object] | None, verified: bool) -> str:
    graph = _orbital({"red_finding": finding or {}, "verification": {"status": "verified"} if verified else {}})
    return graph + '<span class="kicker">NO LIVE NETWORK CONNECTION</span>'


def spiral_html(recommendations: list[Mapping[str, object]], verified: bool, paused: bool) -> str:
    state = "VERIFIED" if verified else "PENDING"
    return f'<div style="height:220px;display:grid;place-items:center;background:#060b15;color:#35d9b8;font:18px Georgia">SIMULATION {state}</div>'
