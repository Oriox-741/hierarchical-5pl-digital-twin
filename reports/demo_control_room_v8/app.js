
const state = { actions: [], replay: null, analytics: null, timer: null, ticking: false, tabId: Math.random().toString(36).slice(2) };
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
function setText(id, text){ const el=document.getElementById(id); if(el) el.textContent=text ?? ""; }
function showError(text){ const el=document.getElementById("errorPanel"); if(!el) return; el.textContent=text || ""; el.classList.toggle("hidden", !text); }
function clamp(n, lo, hi){ return Math.max(lo, Math.min(Number(n||0), hi)); }
function fmtPct(v){ return `${(Number(v||0)*100).toFixed(1)}%`; }
function fmtCount(v){ return Number(v||0).toLocaleString("tr-TR"); }
function option(label, value){ return `<option value="${value}">${label}</option>`; }
function metric(label, value){ return `<article class="metric"><b>${label}</b><strong>${value}</strong></article>`; }
function bar(label, value, max=1){
  const pct = Math.round((clamp(value, 0, max) / max) * 100);
  return `<div class="bar-row"><span>${label}</span><span class="bar-track"><i class="bar-fill" style="width:${pct}%"></i></span><b>${pct}%</b></div>`;
}
function renderSummary(s){
  const d=s.dashboard || {};
  const summary=d.operation_summary || {};
  setText("operationPhase", summary.operation_phase_tr || s.operation_phase_tr || "");
  document.getElementById("summaryMetrics").innerHTML = [
    metric("Model", summary.model_id_short || s.model_id_short),
    metric("Gözlem", `${summary.obs_dim || 73} boyut`),
    metric("PPO", `${summary.continuous_controls || 5} kontrol`),
    metric("DQN", `${summary.discrete_actions || 48} aksiyon`),
    metric("Servis", fmtPct((s.kpis||{}).service_level)),
    metric("Gecikme", fmtPct((s.kpis||{}).lateness)),
    metric("Dispatch", fmtPct((s.kpis||{}).dispatch_rate)),
    metric("Route fail", fmtPct((s.kpis||{}).route_failure))
  ].join("");
  document.getElementById("operationPipeline").innerHTML = (d.operation_pipeline||[]).map(stage =>
    `<article class="stage"><b>${stage.label_tr}</b><strong>${stage.count}</strong><small>${(stage.order_ids||[]).join(", ") || "boş"}</small></article>`
  ).join("");
  document.getElementById("actionFamilyFlow").innerHTML = (d.action_family_flow||[]).map(item =>
    `<article class="flow-step"><b>${item.label_tr}</b><span>${item.value_tr}</span></article>`
  ).join("");
}
function renderPpo(s){
  const controls = s.ppo_controls || {};
  setText("ppoSource", `${s.ppo_source_label_tr || s.ppo_source || "Kaynak yok"} · ${s.production_policy_read_only ? "policy read-only" : "sunum izi"}`);
  const maxByKey = {speed_multiplier:1.35, safety_stock_multiplier:2, capacity_buffer_fraction:.5};
  document.getElementById("ppoBars").innerHTML = Object.entries(controls).map(([name,value]) => bar(name, value, maxByKey[name] || 1)).join("");
  const summary = s.ppo_variance_summary || {};
  document.getElementById("ppoHistory").innerHTML = Object.entries(summary).map(([name,row]) => {
    const min=Number(row.min||0), max=Number(row.max||1), current=Number(row.current||0);
    const pct = max > min ? Math.round(((current-min)/(max-min))*100) : 50;
    return `<div><div class="sparkline"><span>${name}</span><span class="spark-track"><i class="spark-fill" style="width:${clamp(pct,2,100)}%"></i></span><b>${min.toFixed(2)} / ${max.toFixed(2)} / ${current.toFixed(2)}</b></div>${row.message_tr ? `<p class="stable-note">${row.message_tr}</p>` : ""}</div>`;
  }).join("");
}
function renderDqn(s){
  const card=(s.dashboard||{}).decision_card || {};
  setText("decisionEventId", card.decision_event_id || "DEC");
  setText("dqnActionId", `DQN ${card.dqn_action_id ?? (s.dqn_action||{}).action_id ?? 0}`);
  setText("dqnInterpretation", card.interpretation_tr || (s.dqn_action||{}).interpretation_tr || "");
  setText("decisionStatusMessage", card.status_message_tr || "Yeni karar olayı işlendi.");
  document.getElementById("dqnComponents").innerHTML = [
    ["Dispatch", card.dispatch_tr],["Rota", card.route_family_tr],["Filo", card.fleet_mode_tr],["İkmal", card.reorder_mode_tr],["Atama", card.assignment_id],["Karar", card.decision_event_id]
  ].filter(([,v])=>v).map(([k,v])=>`<span><b>${k}</b><br>${v}</span>`).join("");
  const watch=document.getElementById("actionWatch");
  const active=Boolean(card.watch_tr);
  watch.textContent = active ? `${card.watch_tr}: 24/32 izlemesi` : "Normal aksiyon";
  watch.classList.toggle("active", active);
}
function renderKpis(s){
  const k=s.kpis || {};
  setText("kpiStepLabel", `adım ${s.step}`);
  document.getElementById("kpiCards").innerHTML = [
    metric("Service level", fmtPct(k.service_level)),
    metric("Lateness", fmtPct(k.lateness)),
    metric("Dispatch rate", fmtPct(k.dispatch_rate)),
    metric("Dispatch success", fmtPct(k.dispatch_success)),
    metric("No-work", fmtPct(k.no_work)),
    metric("Route failure", fmtPct(k.route_failure)),
    metric("Stockout / ikmal baskısı", fmtPct(k.stockout_reorder_pressure))
  ].join("");
  const rows=((s.dashboard||{}).kpi_timeline || []);
  document.getElementById("kpiTimeline").innerHTML = rows.map(row => {
    const service = clamp(row.service_level,0,1)*100;
    const late = clamp(row.lateness*8,0,1)*100;
    const dispatch = clamp(row.dispatch_rate,0,1)*100;
    const route = clamp(row.route_failure*80,0,1)*100;
    return `<div class="timeline-bar" title="${row.simulated_time} step ${row.step}"><span class="timeline-service" style="height:${service}%"></span><span class="timeline-late" style="height:${late}%"></span><span class="timeline-dispatch" style="height:${dispatch}%"></span><span class="timeline-route" style="height:${route}%"></span></div>`;
  }).join("");
}
function fillFilter(id, values){
  const el=document.getElementById(id);
  const first=el.options[0]?.outerHTML || option("Tümü","");
  el.innerHTML = first + (values||[]).map(v=>option(v, v)).join("");
}
function renderActions(){
  const q=(document.getElementById("actionSearch")?.value||"").toLowerCase();
  const dispatch=document.getElementById("dispatchFilter")?.value || "";
  const route=document.getElementById("routeFilter")?.value || "";
  const fleet=document.getElementById("fleetFilter")?.value || "";
  const reorder=document.getElementById("reorderFilter")?.value || "";
  const rows=state.actions.filter(a =>
    (!q || JSON.stringify(a).toLowerCase().includes(q)) &&
    (!dispatch || a.dispatch===dispatch) &&
    (!route || a.route_family===route) &&
    (!fleet || a.fleet_mode===fleet) &&
    (!reorder || a.reorder_mode===reorder)
  );
  document.getElementById("actionCoverage").textContent = `${rows.length} / 48 görünür`;
  document.getElementById("actionCards").innerHTML = rows.map(a =>
    `<article class="action-card-mini ${a.watch_tr ? "watch" : ""}"><b>#${a.action_id}</b><h3>${a.interpretation_tr}</h3><p>${fmtCount(a.global_count)} public replay · ${fmtPct(a.global_rate)}</p></article>`
  ).join("");
  document.getElementById("topActions").innerHTML = (state.analytics?.top_10_actions||[]).map(a =>
    `<div class="rank-item"><b>#${a.action_id}</b><span>${a.interpretation_tr || `${a.dispatch}/${a.route_family}/${a.fleet_mode}/${a.reorder_mode}`}</span><strong>${fmtCount(a.global_count)}</strong></div>`
  ).join("");
  document.getElementById("zeroActions").innerHTML = (state.analytics?.zero_count_action_ids||[]).map(id=>`<span class="chip">#${id}</span>`).join("");
  setText("actionWatchExplanation", state.analytics?.watch_explanation_tr || "");
}
function renderReplay(){
  const r=state.replay; if(!r) return;
  setText("replayTotal", fmtCount(r.global_total));
  document.getElementById("replayTotals").innerHTML = [
    metric("Toplam", fmtCount(r.global_total)),
    metric("LaDe", fmtCount(r.dataset_totals.LaDe)),
    metric("NYC HVFHS", fmtCount(r.dataset_totals["NYC HVFHS"])),
    metric("Olist", fmtCount(r.dataset_totals.Olist))
  ].join("");
  const families = r.family_distribution || {};
  const rows = [];
  for(const [family, values] of Object.entries(families)){
    for(const [label, count] of Object.entries(values)){
      rows.push({family,label,count});
    }
  }
  document.getElementById("replayDistribution").innerHTML = rows.map(row => {
    const pct = r.global_total ? Math.round((row.count/r.global_total)*100) : 0;
    return `<div class="dist-row"><span>${row.family}: ${row.label}</span><span class="dist-track"><i class="dist-fill ${pct<5?'alt':''}" style="width:${Math.max(1,pct)}%"></i></span><b>${fmtCount(row.count)}</b></div>`;
  }).join("");
}
function renderScenarios(s){
  const rows=(s.dashboard||{}).scenario_comparison || [];
  document.getElementById("scenarioComparison").innerHTML = rows.map(row =>
    `<article class="scenario-item ${row.current ? "current" : ""}"><h3>${row.name_tr}</h3><p>${row.runnable ? "çalıştırılabilir" : "dokümante" } · ${row.stress_type_tr}</p><p>${row.expected_pressure_tr}</p><div class="mini-kpi"><span>Servis ${fmtPct(row.kpis.service_level)}</span><span>Gecikme ${fmtPct(row.kpis.lateness)}</span><span>Dispatch ${fmtPct(row.kpis.dispatch_rate)}</span><span>Başarı ${fmtPct(row.kpis.dispatch_success)}</span></div></article>`
  ).join("");
}
function renderScorecard(s){
  const card=(s.dashboard||{}).evidence_scorecard || {};
  document.getElementById("scorecardGrid").innerHTML = (card.dimensions||[]).map(row =>
    `<article class="score-row"><b>${row.label_tr}<br><small>${row.dimension}</small></b><strong>${Number(row.score).toFixed(2)}</strong><p>${row.evidence_tr} Sınır: ${row.limiter_tr}</p></article>`
  ).join("");
}
function renderCompanyRequirements(s){
  const req=(s.dashboard||{}).company_data_requirements || {};
  document.getElementById("companyRequirements").innerHTML = (req.table_families||[]).map(row =>
    `<article class="requirement"><h3>${row.label_tr}</h3><p>${row.purpose_tr}</p><p>${(row.required_fields||[]).map(f=>`<code>${f}</code>`).join(" ")}</p></article>`
  ).join("");
}
function renderEvents(s){
  document.getElementById("eventLog").innerHTML = (s.event_log||[]).map(e=>`<li><b>${e.time}</b> · ${e.text}</li>`).join("");
  const rows=(s.dashboard||{}).decision_timeline || [];
  document.getElementById("decisionTimeline").innerHTML = rows.map(item=>`<span class="chip">step ${item.step} · DQN ${item.action_id}${item.watch_tr ? " · izlenen" : ""}</span>`).join("");
  document.getElementById("claimBoundaries").innerHTML = ((s.dashboard||{}).claim_boundaries||[]).map(item=>`<li>${item}</li>`).join("");
}
function syncControls(s){
  setText("topScenario", s.scenario_name_tr);
  setText("topMode", s.mode_label_tr);
  setText("topStatus", s.running ? "Oynatılıyor" : "Duraklatıldı");
  setText("topTime", s.simulated_time);
  setText("topStep", s.step);
  setText("topModel", s.model_id_short);
  const contractEl=document.getElementById("topContract");
  if(contractEl){
    contractEl.textContent = "v5 route candidate visibility · obs 73 · action 48";
    contractEl.title = s.contract || "";
  }
  const slider=document.getElementById("timelineSlider");
  const maxStep=Math.max(0, Number(s.trace_step_count||1)-1);
  slider.max=String(maxStep);
  slider.value=String(clamp(s.step,0,maxStep));
  setText("traceProgress", `${s.step} / ${maxStep}`);
  const scenarioSelect=document.getElementById("scenarioSelect"); if(scenarioSelect && s.selected_scenario && scenarioSelect.value!==s.selected_scenario){ scenarioSelect.value=s.selected_scenario; }
  const modeSelect=document.getElementById("modeSelect"); if(modeSelect && [...modeSelect.options].some(o=>o.value===s.mode)){ modeSelect.value=s.mode; }
  const speedSelect=document.getElementById("speedSelect"); if(speedSelect && s.speed && speedMs[s.speed]){ speedSelect.value=s.speed; }
  const policyToggle=document.getElementById("policyToggle"); if(policyToggle){ policyToggle.checked=!!s.production_policy_read_only; }
}
function renderState(s){
  syncControls(s);
  renderSummary(s);
  renderPpo(s);
  renderDqn(s);
  renderKpis(s);
  renderScenarios(s);
  renderScorecard(s);
  renderCompanyRequirements(s);
  renderEvents(s);
}
async function refresh(){ const payload=await api("/api/state"); renderState(payload.state); if(payload.state.running) schedule(); }
async function tick(){ if(state.ticking) return; state.ticking=true; try{ const payload=await api("/api/control/tick", {}); renderState(payload.state); if(payload.state.running) schedule(); } finally { state.ticking=false; } }
function stopTimer(){ if(state.timer){ clearTimeout(state.timer); state.timer=null; } }
function schedule(){ stopTimer(); state.timer=setTimeout(tick, speedMs[document.getElementById("speedSelect").value] || 900); }
async function post(path, body={}){ const payload=await api(path, body); renderState(payload.state); if(payload.state.running) schedule(); else stopTimer(); }
async function loadScenarios(){
  const payload=await api("/api/scenarios");
  document.getElementById("scenarioSelect").innerHTML=(payload.scenarios||[]).map(s=>option(`${s.name_tr} · ${s.status_tr}`, s.scenario_id)).join("");
}
async function loadReplay(){
  const payload=await api("/api/replay");
  state.actions=payload.actions || [];
  state.replay=payload.replay;
  state.analytics=payload.analytics;
  fillFilter("dispatchFilter", state.analytics?.filters?.dispatch || []);
  fillFilter("routeFilter", state.analytics?.filters?.route_family || []);
  fillFilter("fleetFilter", state.analytics?.filters?.fleet_mode || []);
  fillFilter("reorderFilter", state.analytics?.filters?.reorder_mode || []);
  renderActions();
  renderReplay();
}
function setupTabWarning(){
  try{
    const bc = new BroadcastChannel("control-room-v8-tabs");
    bc.onmessage = ev => { if(ev.data && ev.data.tabId !== state.tabId){ document.getElementById("tabWarning")?.classList.remove("hidden"); } };
    bc.postMessage({tabId:state.tabId, type:"hello"});
  }catch(_err){
    const key="control-room-v8-last-tab";
    const previous=localStorage.getItem(key);
    localStorage.setItem(key, `${state.tabId}:${Date.now()}`);
    if(previous && !previous.startsWith(state.tabId)){ document.getElementById("tabWarning")?.classList.remove("hidden"); }
  }
}
document.getElementById("startBtn").onclick=()=>post("/api/control/start");
document.getElementById("pauseBtn").onclick=()=>post("/api/control/pause");
document.getElementById("resetBtn").onclick=()=>post("/api/control/reset");
document.getElementById("stepBtn").onclick=()=>post("/api/control/step");
document.getElementById("timelineSlider").oninput=e=>post("/api/playback/seek", {step:Number(e.target.value)});
document.getElementById("scenarioSelect").onchange=e=>post("/api/control/set_scenario", {scenario_id:e.target.value});
document.getElementById("speedSelect").onchange=e=>post("/api/control/set_speed", {speed:e.target.value});
document.getElementById("modeSelect").onchange=e=>post("/api/control/set_mode", {mode:e.target.value});
document.getElementById("policyToggle").onchange=e=>post("/api/control/set_policy", {enabled:e.target.checked});
for(const id of ["actionSearch","dispatchFilter","routeFilter","fleetFilter","reorderFilter"]){ document.getElementById(id).oninput=renderActions; document.getElementById(id).onchange=renderActions; }
(async function init(){ setupTabWarning(); await loadScenarios(); await loadReplay(); await refresh(); })();
