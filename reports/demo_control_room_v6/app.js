
const state = {
  actions: [],
  replay: null,
  traces: [],
  timer: null,
  ticking: false,
  tabId: Math.random().toString(36).slice(2)
};
const speedMs = {"0.5x": 1600, "1x": 900, "2x": 520, "5x": 240};
async function api(path, body=null){
  try{
    setText("connectionStatus", "Bağlanıyor");
    const res = await fetch(path, {method: body ? "POST" : "GET", headers: {"Content-Type": "application/json"}, body: body ? JSON.stringify(body) : undefined});
    if(!res.ok){ throw new Error(await res.text()); }
    const payload = await res.json();
    setText("connectionStatus", "Hazır");
    showError("");
    return payload;
  }catch(err){
    setText("connectionStatus", "Hata");
    showError(`Bağlantı hatası: ${err.message || err}`);
    throw err;
  }
}
function fmtPct(v){ return `${Math.round(Number(v||0)*1000)/10}%`; }
function setText(id, text){ const el=document.getElementById(id); if(el) el.textContent=text; }
function showError(text){ const el=document.getElementById("errorPanel"); if(!el) return; el.textContent=text || ""; el.classList.toggle("hidden", !text); }
function clamp(n, lo, hi){ return Math.max(lo, Math.min(Number(n||0), hi)); }
function bar(label, value, max=1){
  const n=clamp(value, 0, max);
  const pct=Math.round((n/max)*100);
  return `<div class="bar-row"><span>${label}</span><span class="bar-track"><i class="bar-fill" style="width:${pct}%"></i></span><b>${pct}%</b></div>`;
}
function routePoints(route){
  const raw = route.path || [{x:route.x1, y:route.y1}, {x:route.x2, y:route.y2}];
  return raw.map(p=>`${p.x},${p.y}`).join(" ");
}
function renderMap(s){
  const svg=document.getElementById("mapSvg"); svg.innerHTML="";
  for(let i=0;i<=100;i+=10){
    svg.insertAdjacentHTML("beforeend",`<line class="grid-line" x1="${i}" y1="0" x2="${i}" y2="100"/><line class="grid-line" x1="0" y1="${i}" x2="100" y2="${i}"/>`);
  }
  (s.zones||[]).forEach(z=>svg.insertAdjacentHTML("beforeend",`<circle class="zone" cx="${z.x}" cy="${z.y}" r="${z.r}"><title>${z.label}</title></circle>`));
  (s.routes||[]).forEach(r=>svg.insertAdjacentHTML("beforeend",`<polyline class="route path ${r.selected?'selected':''}" points="${routePoints(r)}"><title>${r.family_tr || r.label || r.id}</title></polyline>`));
  (s.hubs||[]).forEach(h=>svg.insertAdjacentHTML("beforeend",`<rect class="hub" x="${h.x-2.4}" y="${h.y-2.4}" width="4.8" height="4.8" rx=".8"/><text class="map-label" x="${h.x}" y="${h.y+6}" text-anchor="middle">${h.id}</text>`));
  (s.orders||[]).forEach(o=>svg.insertAdjacentHTML("beforeend",`<circle class="order ${o.status}" cx="${o.x}" cy="${o.y}" r="2.2"><title>${o.id} ${o.status_tr}</title></circle><text class="map-label" x="${o.x}" y="${o.y-4}" text-anchor="middle">${o.id}</text>`));
  (s.vehicles||[]).forEach(v=>svg.insertAdjacentHTML("beforeend",`<polygon class="vehicle" points="${v.x},${v.y-3} ${v.x+2.8},${v.y+2.4} ${v.x-2.8},${v.y+2.4}"><title>${v.id} ${v.status_tr}</title></polygon><text class="map-label" x="${v.x}" y="${v.y+6}" text-anchor="middle">${v.id}</text>`));
}
function renderPpoHistory(s){
  const summary = s.ppo_variance_summary || {};
  const rows = Object.entries(summary).map(([name, row]) => {
    const min = Number(row.min ?? 0), max = Number(row.max ?? 1), current = Number(row.current ?? 0);
    const width = max > 0 ? Math.round((max / Math.max(max, 1)) * 100) : 0;
    const dot = max > min ? Math.round(((current - min) / (max - min)) * 100) : 50;
    return `<div class="sparkline"><span>${name}</span><span class="sparkline-track"><i class="sparkline-range" style="width:${clamp(width, 3, 100)}%"></i><i class="sparkline-dot" style="left:${clamp(dot, 0, 100)}%"></i></span><b>${current.toFixed(2)}</b></div>`;
  });
  document.getElementById("ppoHistory").innerHTML = rows.join("") || `<p class="muted-line">PPO geçmiş özeti bu modda kullanılmıyor.</p>`;
}
function syncTimeline(s){
  const slider=document.getElementById("timelineSlider");
  const maxStep=Math.max(0, Number(s.trace_step_count||1)-1);
  if(slider){
    slider.max=String(maxStep);
    slider.value=String(clamp(s.step, 0, maxStep));
  }
  setText("traceProgress", `${s.step} / ${maxStep}`);
  const traceSelect=document.getElementById("traceSelect");
  if(traceSelect && s.loaded_trace_id && traceSelect.value!==s.loaded_trace_id){ traceSelect.value=s.loaded_trace_id; }
  const scenarioSelect=document.getElementById("scenarioSelect");
  if(scenarioSelect && s.selected_scenario && scenarioSelect.value!==s.selected_scenario){ scenarioSelect.value=s.selected_scenario; }
}
function renderState(s){
  setText("topScenario", s.scenario_name_tr);
  setText("topMode", `${s.mode_label_tr} · ${s.operation_phase || "trace"}`);
  setText("topStatus", s.running ? "Oynatılıyor" : "Duraklatıldı");
  setText("topTime", s.simulated_time);
  setText("topStep", s.step);
  setText("mapNotice", s.map_notice_tr);
  setText("presentationNotice", s.presentation_notice_tr || "");
  setText("ppoSource", `${s.ppo_source_label_tr || s.ppo_source || "PPO kaynağı"} · ${s.ppo_explanation_tr || ""}`);
  syncTimeline(s);
  const modeSelect=document.getElementById("modeSelect"); if(modeSelect && modeSelect.value!==s.mode){ modeSelect.value=s.mode; }
  const speedSelect=document.getElementById("speedSelect"); if(speedSelect && s.speed && speedSelect.value!==s.speed && speedMs[s.speed]){ speedSelect.value=s.speed; }
  const policyToggle=document.getElementById("policyToggle"); if(policyToggle){ policyToggle.checked=!!s.production_policy_read_only; policyToggle.disabled=s.mode==="SUNUM_MODU"; }
  setText("actionId", `DQN ${s.dqn_action.action_id}`);
  setText("actionLabel", s.dqn_action.interpretation_tr);
  setText("actionWatch", s.dqn_action.watch_tr || "Normal aksiyon");
  renderMap(s);
  document.getElementById("ppoBars").innerHTML = Object.entries(s.ppo_controls||{}).map(([k,v])=>bar(k, v, k==="speed_multiplier"?1.35:k==="safety_stock_multiplier"?2:1)).join("");
  renderPpoHistory(s);
  document.getElementById("stressBars").innerHTML = Object.entries(s.scenario_stress||{}).map(([k,v])=>bar(k, v, 1)).join("");
  const k = s.kpis || {};
  document.getElementById("kpiGrid").innerHTML = [
    ["Servis", fmtPct(k.service_level)],["Gecikme", fmtPct(k.lateness)],["Sevk oranı", fmtPct(k.dispatch_rate)],["Sevk başarısı", fmtPct(k.dispatch_success)],["Faydasız iş", k.no_work],["Rota hatası", k.route_failure],["Stok/ikmal baskısı", fmtPct(k.stockout_reorder_pressure)]
  ].map(([a,b])=>`<article class="metric"><b>${a}</b><strong>${b}</strong></article>`).join("");
  document.getElementById("eventLog").innerHTML = (s.event_log||[]).map(e=>`<li><b>${e.time}</b> - ${e.text}</li>`).join("");
  document.getElementById("actionHistory").innerHTML = (s.action_history||[]).map(a=>`<span class="chip">#${a.action_id} ${a.watch_tr || ""}</span>`).join("");
}
async function refresh(){ const payload = await api("/api/playback/state"); renderState(payload.state); if(payload.state.running) schedule(); }
async function tick(){
  if(state.ticking) return;
  state.ticking=true;
  try{
    const payload = await api("/api/playback/tick", {});
    renderState(payload.state);
    if(payload.state.running) schedule();
  } finally { state.ticking=false; }
}
function stopTimer(){ if(state.timer){ clearTimeout(state.timer); state.timer=null; } }
function schedule(){ stopTimer(); state.timer = setTimeout(tick, speedMs[document.getElementById("speedSelect").value] || 900); }
async function post(path, body={}){
  const payload=await api(path, body);
  renderState(payload.state);
  if(payload.state.running) schedule(); else stopTimer();
}
async function loadTraces(){
  const payload=await api("/api/traces");
  state.traces=payload.traces || [];
  const options=state.traces.map(t=>`<option value="${t.trace_id}">${t.scenario_name_tr} · ${t.step_count} adım</option>`).join("");
  document.getElementById("traceSelect").innerHTML=options;
  document.getElementById("scenarioSelect").innerHTML=state.traces.map(t=>`<option value="${t.trace_id}">${t.scenario_name_tr}</option>`).join("");
}
async function loadActions(){
  const payload=await api("/api/actions");
  state.actions=payload.actions;
  state.replay=payload.replay;
  renderActions();
  renderReplay();
}
function renderActions(){
  const q=(document.getElementById("actionSearch")?.value||"").toLowerCase();
  document.getElementById("actionCards").innerHTML=state.actions.filter(a=>JSON.stringify(a).toLowerCase().includes(q)).map(a=>`<article class="action-item ${a.watch_tr?'watch':''}"><b>#${a.action_id}</b><h3>${a.interpretation_tr}</h3><p>${a.watch_tr || "normal"}</p><p>Kamu replay: ${a.global_count}</p></article>`).join("");
}
function renderReplay(){
  if(!state.replay) return;
  const r=state.replay;
  document.getElementById("replayStats").innerHTML=`<article class="metric"><b>Toplam</b><strong>${r.global_total.toLocaleString("tr-TR")}</strong></article><article class="metric"><b>LaDe</b><strong>${r.dataset_totals.LaDe.toLocaleString("tr-TR")}</strong></article><article class="metric"><b>NYC HVFHS</b><strong>${r.dataset_totals["NYC HVFHS"].toLocaleString("tr-TR")}</strong></article><article class="metric"><b>Olist</b><strong>${r.dataset_totals.Olist.toLocaleString("tr-TR")}</strong></article>`;
  document.getElementById("replayTop").innerHTML=r.top_10_actions.map(a=>`<article class="action-item"><b>#${a.action_id}</b><p>${a.global_count.toLocaleString("tr-TR")} kayıt</p></article>`).join("");
}
function openPanelFromHash(){
  const panels={"#aksiyonlar":"actionsPanel","#kamu-replay":"replayPanel","#kanitlar":"evidencePanel"};
  const id=panels[window.location.hash];
  if(id){ document.getElementById(id)?.showModal(); }
}
function currentStep(){ return Number(document.getElementById("timelineSlider")?.value || 0); }
document.getElementById("startBtn").onclick=()=>post("/api/playback/play");
document.getElementById("pauseBtn").onclick=()=>post("/api/playback/pause");
document.getElementById("resetBtn").onclick=()=>post("/api/playback/reset");
document.getElementById("rewindBtn").onclick=()=>post("/api/playback/seek", {step: Math.max(0, currentStep()-1)});
document.getElementById("stepBtn").onclick=()=>post("/api/playback/seek", {step: currentStep()+1});
document.getElementById("timelineSlider").oninput=e=>post("/api/playback/seek", {step:Number(e.target.value)});
document.getElementById("traceSelect").onchange=e=>post("/api/playback/load", {trace_id:e.target.value});
document.getElementById("scenarioSelect").onchange=e=>post("/api/playback/load", {trace_id:e.target.value});
document.getElementById("speedSelect").onchange=e=>post("/api/control/set_speed", {speed:e.target.value});
document.getElementById("modeSelect").onchange=e=>post("/api/control/set_mode", {mode:e.target.value});
document.getElementById("policyToggle").onchange=e=>post("/api/control/set_policy", {enabled:e.target.checked});
document.querySelectorAll(".secondary button").forEach(b=>b.onclick=()=>document.getElementById(b.dataset.panel).showModal());
document.querySelectorAll("dialog .close").forEach(b=>b.onclick=()=>b.closest("dialog").close());
document.getElementById("actionSearch").oninput=renderActions;
function showTabWarning(){ document.getElementById("tabWarning")?.classList.remove("hidden"); }
function setupTabWarning(){
  try{
    const bc = new BroadcastChannel("control-room-v6-tabs");
    bc.onmessage = ev => { if(ev.data && ev.data.tabId !== state.tabId){ showTabWarning(); bc.postMessage({tabId:state.tabId, type:"ack"}); } };
    bc.postMessage({tabId:state.tabId, type:"hello"});
    window.addEventListener("beforeunload", ()=>bc.postMessage({tabId:state.tabId, type:"bye"}));
  }catch(_err){
    const key="control-room-v6-last-tab";
    const previous=localStorage.getItem(key);
    localStorage.setItem(key, `${state.tabId}:${Date.now()}`);
    if(previous && !previous.startsWith(state.tabId)){ showTabWarning(); }
  }
}
(async function init(){ setupTabWarning(); await loadTraces(); await loadActions(); await refresh(); openPanelFromHash(); })();
