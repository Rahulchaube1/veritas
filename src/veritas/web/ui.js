"use strict";
const el = id => document.getElementById(id);
const node = (tag, text) => { const n = document.createElement(tag); n.textContent = text; return n; };
el("mode").addEventListener("change", () => { el("steps").value = el("mode").selectedIndex ? 5 : 3; });
el("form").addEventListener("submit", async event => {
  event.preventDefault();
  el("submit").disabled = true;
  el("result").hidden = true;
  el("status").textContent = "Checking the local collection…";
  try {
    const headers = {"Content-Type":"application/json"};
    if (el("token").value) headers.Authorization = "Bearer " + el("token").value;
    const response = await fetch("/v1/search", {method:"POST",headers,body:JSON.stringify({
      follow_links:el("mode").selectedIndex===1, query:el("query").value, as_of:el("date").value || null,max_search_steps:Number(el("steps").value)})});
    if (!response.ok) throw Error("Request failed (" + response.status + "). " + (response.status === 401 ? "Enter the API token." : "Check the query and service status."));
    const r = await response.json();
    el("verdict").textContent = r.claims[0].verdict;
    el("answer").textContent = r.answer;
    el("stats").textContent = r.documents_examined + " documents · " + r.search_budget_used + " search steps · " + r.latency_ms.toFixed(1) + " ms · " +
      (r.confidence === null ? "Factual confidence uncalibrated" : r.confidence.toFixed(2) + " · " + r.confidence_target);
    el("evidence").replaceChildren();
    r.evidence.forEach((e,i) => {
      const card = node("article","");
      card.append(node("strong", (i+1) + ". " + (e.title || e.doc_id)),
        node("p",e.passage),node("small",[e.stance,e.temporal,e.method,"group "+e.group].join(" · ")));
      if (e.url) {const a=node("a"," Open source"); a.href=e.url;a.rel="noreferrer";card.append(a);}
      el("evidence").append(card);
    });
    el("edges").replaceChildren();
    if (!r.source_dependencies.length) el("edges").append(node("p","No dependency links detected among the examined documents."));
    r.source_dependencies.forEach(edge => el("edges").append(node("p",edge.source+" → "+edge.target+" · "+edge.relation+" · "+edge.basis)));
    el("result").hidden=false;
    el("status").textContent=r.abstained ? "Abstained from a decisive answer." : "Verification complete.";
  } catch (error) {el("status").textContent=error.message;}
  finally {el("submit").disabled=false;}
});
fetch("/assets/benchmark.json").then(r => r.json()).then(data=>{
  el("bench-status").textContent=data.description;
  const table=node("table",""); const head=node("tr","");
  ["Retrieval method","Recall@10","nDCG@10"].forEach(t=>head.append(node("th",t)));table.append(head);
  data.rows.forEach(row=>{const tr=node("tr","");row.forEach(v=>tr.append(node("td",v)));table.append(tr);});
  el("tables").append(table,node("p",data.controlled));
}).catch(()=>{el("bench-status").textContent="Recorded benchmark results unavailable.";});
