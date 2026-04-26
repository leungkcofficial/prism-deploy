/* PRISM Clinical Decision Aid — Frontend Logic */

// ── CCI Definitions ────────────────────────────────────────────────────────────
// Weights from Quan et al. (2005) CCI mapping
const CCI_ITEMS = [
  { key: "myocardial_infarction",       label: "Myocardial infarction",         weight: 1 },
  { key: "congestive_heart_failure",    label: "Congestive heart failure",       weight: 1 },
  { key: "peripheral_vascular_disease", label: "Peripheral vascular disease",    weight: 1 },
  { key: "cerebrovascular_disease",     label: "Cerebrovascular disease",        weight: 1 },
  { key: "dementia",                    label: "Dementia",                       weight: 1 },
  { key: "chronic_pulmonary_disease",   label: "Chronic pulmonary disease",      weight: 1 },
  { key: "peptic_ulcer_disease",        label: "Peptic ulcer disease",           weight: 1 },
  { key: "mild_liver_disease",          label: "Mild liver disease",             weight: 1 },
  { key: "diabetes_wo_complication",    label: "Diabetes (uncomplicated)",       weight: 1 },
  { key: "diabetes_w_complication",     label: "Diabetes (with complications)",  weight: 2 },
  { key: "hemiplegia_paraplegia",       label: "Hemiplegia / paraplegia",        weight: 2 },
  { key: "any_malignancy",              label: "Any malignancy (non-metastatic)",weight: 2 },
  { key: "metastatic_cancer",           label: "Metastatic cancer",              weight: 6 },
];

// Hierarchy rules: if key_a is checked, uncheck key_b
const CCI_HIERARCHY = [
  ["diabetes_w_complication", "diabetes_wo_complication"],
  ["metastatic_cancer",       "any_malignancy"],
];

// ── Init ──────────────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  buildCCIGrid();
});

function buildCCIGrid() {
  const grid = document.getElementById("cci-grid");
  CCI_ITEMS.forEach(item => {
    const div = document.createElement("div");
    div.className = "cci-item";
    div.innerHTML = `
      <input type="checkbox" id="cci_${item.key}" data-key="${item.key}"
             onchange="onCCIChange('${item.key}')">
      <label for="cci_${item.key}">
        <span>${item.label}</span>
        <span class="weight">+${item.weight}</span>
      </label>`;
    grid.appendChild(div);
  });
}

function toggleCCI() {
  const body = document.getElementById("cci-body");
  const chevron = document.getElementById("cci-chevron");
  body.classList.toggle("open");
  chevron.classList.toggle("open");
}

function onCCIChange(changedKey) {
  // Enforce hierarchy
  CCI_HIERARCHY.forEach(([dominant, subordinate]) => {
    if (changedKey === dominant) {
      const domEl = document.getElementById(`cci_${dominant}`);
      const subEl = document.getElementById(`cci_${subordinate}`);
      if (domEl && domEl.checked && subEl) subEl.checked = false;
    }
    if (changedKey === subordinate) {
      const subEl = document.getElementById(`cci_${subordinate}`);
      const domEl = document.getElementById(`cci_${dominant}`);
      if (subEl && subEl.checked && domEl && domEl.checked) subEl.checked = false;
    }
  });
  updateCCIScore();
}

function updateCCIScore() {
  let score = 0;
  CCI_ITEMS.forEach(item => {
    const el = document.getElementById(`cci_${item.key}`);
    if (el && el.checked) score += item.weight;
  });
  document.getElementById("cci-pill").textContent = score;
  return score;
}

function getCCIFlags() {
  const flags = {};
  CCI_ITEMS.forEach(item => {
    const el = document.getElementById(`cci_${item.key}`);
    if (el) flags[item.key] = el.checked ? 1 : 0;
  });
  // Only return flags if at least one is set (enables subgroup model)
  const hasAny = Object.values(flags).some(v => v === 1);
  return hasAny ? flags : null;
}

// ── Age / DOB ─────────────────────────────────────────────────────────────────
function computeAge() {
  const dob = document.getElementById("dob").value;
  if (!dob) { document.getElementById("age-badge").textContent = "—"; return; }
  const today = new Date();
  const birth = new Date(dob);
  let age = today.getFullYear() - birth.getFullYear();
  const m = today.getMonth() - birth.getMonth();
  if (m < 0 || (m === 0 && today.getDate() < birth.getDate())) age--;
  document.getElementById("age-badge").textContent = `${age} yrs`;
  document.getElementById("age").value = age;
}

function getAge() {
  const dob = document.getElementById("dob").value;
  if (dob) {
    const today = new Date();
    const birth = new Date(dob);
    let age = today.getFullYear() - birth.getFullYear();
    const m = today.getMonth() - birth.getMonth();
    if (m < 0 || (m === 0 && today.getDate() < birth.getDate())) age--;
    return age;
  }
  return null;
}

// ── Sex ───────────────────────────────────────────────────────────────────────
function setSex(val) {
  document.getElementById("female").value = val;
  document.getElementById("btn-male").classList.toggle("active", val === 0);
  document.getElementById("btn-female").classList.toggle("active", val === 1);
}

// ── Sample patient ────────────────────────────────────────────────────────────
async function loadSample() {
  try {
    const r = await fetch("/api/sample");
    const data = await r.json();
    const inp = data.input;
    // Set age via manual input (no DOB for sample)
    document.getElementById("dob").value = "";
    document.getElementById("age-badge").textContent = `${inp.age} yrs`;
    document.getElementById("age").value = inp.age;
    setSex(inp.female);
    document.getElementById("creatinine").value = inp.creatinine;
    document.getElementById("haemoglobin").value = inp.haemoglobin ?? "";
    document.getElementById("phosphate").value   = inp.phosphate   ?? "";
    // CCI
    if (inp.cci_flags) {
      Object.entries(inp.cci_flags).forEach(([k, v]) => {
        const el = document.getElementById(`cci_${k}`);
        if (el) el.checked = v === 1;
      });
    }
    updateCCIScore();
    // Open CCI accordion
    document.getElementById("cci-body").classList.add("open");
    document.getElementById("cci-chevron").classList.add("open");
  } catch (e) {
    console.error(e);
  }
}

// ── Prediction ────────────────────────────────────────────────────────────────
async function runPrediction() {
  const age = getAge() || parseFloat(document.getElementById("age").value);
  if (!age || age < 18 || age > 110) { alert("Please enter a valid age (18–110)."); return; }

  const cr = parseFloat(document.getElementById("creatinine").value);
  if (!cr || cr <= 0) { alert("Creatinine is required."); return; }

  const female = parseInt(document.getElementById("female").value);
  const hb  = parseFloat(document.getElementById("haemoglobin").value) || null;
  const po4 = parseFloat(document.getElementById("phosphate").value)   || null;
  const cci = updateCCIScore();
  const cci_flags = getCCIFlags();

  const btn     = document.getElementById("predict-btn");
  const spinner = document.getElementById("spinner");
  const btnText = document.getElementById("btn-text");

  btn.disabled = true;
  spinner.style.display = "block";
  btnText.textContent = "Analysing...";
  document.getElementById("error-box").style.display = "none";
  document.getElementById("results-panel").classList.remove("visible");
  document.getElementById("empty-state").style.display = "none";

  try {
    const resp = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ age, female, creatinine: cr, haemoglobin: hb,
                              phosphate: po4, cci_total: cci, cci_flags }),
    });

    if (!resp.ok) {
      const err = await resp.json();
      throw new Error(err.detail || "Server error");
    }

    const data = await resp.json();
    renderResults(data);
    document.getElementById("results-panel").classList.add("visible");

  } catch (e) {
    document.getElementById("error-msg").textContent = e.message;
    document.getElementById("error-box").style.display = "block";
    document.getElementById("empty-state").style.display = "block";
  } finally {
    btn.disabled = false;
    spinner.style.display = "none";
    btnText.textContent = "Run PRISM Analysis";
  }
}

// ── Render results ────────────────────────────────────────────────────────────
const ZONE_COLORS = { A: "#66BB6A", B: "#29B6F6", C: "#7E57C2", D: "#FFA726" };

function renderResults(d) {
  const zone = d.zone;

  // Zone card
  const zcard = document.getElementById("zone-card");
  zcard.style.borderLeftColor = ZONE_COLORS[zone];
  zcard.style.background = ZONE_COLORS[zone] + "18";
  zcard.style.color = "inherit";
  document.getElementById("zone-title").textContent = `Zone ${zone} — ${d.zone_name}`;
  document.getElementById("zone-rec").textContent   = d.recommendation;
  document.getElementById("zone-badge").textContent = zone;

  // Caveat
  const cavBox = document.getElementById("caveat-box");
  if (d.caveat) {
    document.getElementById("caveat-text").textContent = d.caveat;
    cavBox.style.display = "block";
  } else {
    cavBox.style.display = "none";
  }

  // ACMM gauge
  const pct = (d.acmm_prob * 100);
  document.getElementById("gauge-needle").style.left = `${pct}%`;
  document.getElementById("gauge-value").innerHTML =
    `<span style="color:${pct >= 30 ? "var(--red)" : "var(--green)"};font-size:18px">
       ${pct.toFixed(1)}%
     </span>
     <span style="font-size:12px;color:var(--text-muted);font-weight:400;margin-left:6px">
       1-year mortality risk · ${d.acmm_risk_level === "high" ? "HIGH risk group" : "Low risk group"}
     </span>`;

  // Survival table
  const tbody = document.getElementById("surv-tbody");
  tbody.innerHTML = "";
  const R0 = d.rsf_R0, R1 = d.rsf_R1;
  [1,2,3,4,5].forEach((yr, i) => {
    const benefit = R1[i] - R0[i];
    const benefitPct = (benefit * 100).toFixed(1);
    const cls = benefit < -0.02 ? "benefit-neg" : (benefit > 0.02 ? "benefit-pos" : "");
    const arrow = benefit < -0.02 ? "▼ " : (benefit > 0.02 ? "▲ " : "");
    tbody.innerHTML += `<tr>
      <td>${yr}y</td>
      <td>${(R0[i]*100).toFixed(1)}%</td>
      <td>${(R1[i]*100).toFixed(1)}%</td>
      <td class="benefit-cell ${cls}">${arrow}${benefitPct > 0 ? "+" : ""}${benefitPct}pp</td>
    </tr>`;
  });

  // ITE bar chart
  const chartDiv = document.getElementById("ite-chart");
  chartDiv.innerHTML = "";
  const maxAbs = Math.max(...d.rsf_ITE.map(Math.abs), 0.05);
  [1,2,3,4,5].forEach((yr, i) => {
    const ite = d.rsf_ITE[i];
    const pctBar = Math.min(Math.abs(ite) / maxAbs * 45, 45); // max 45% width from centre
    const isBenefit = ite < 0;
    const color = isBenefit ? "var(--green)" : "var(--red)";
    const itePct = (ite * 100).toFixed(1);

    chartDiv.innerHTML += `
      <div class="ite-row">
        <div class="ite-yr">${yr}y</div>
        <div class="ite-bar-wrap">
          <div class="ite-zero"></div>
          <div class="ite-bar ${isBenefit ? "benefit" : "harm"}"
               style="width:${pctBar}%"></div>
        </div>
        <div class="ite-val" style="color:${color}">
          ${ite < 0 ? "▼ " : "▲ "}${Math.abs(itePct)}pp
        </div>
      </div>`;
  });

  // CF 1-year
  const cf1 = d.cf_ite[0], cfLo = d.cf_lo[0], cfHi = d.cf_hi[0];
  const cfSig = !(cfLo < 0 && cfHi > 0);
  document.getElementById("cf-val").innerHTML =
    `<span style="color:${cf1 < 0 ? "var(--green)" : "var(--red)"}">
       ${(cf1*100).toFixed(1)}pp
     </span>`;
  document.getElementById("cf-ci").textContent = `95% CI [${(cfLo*100).toFixed(1)}, ${(cfHi*100).toFixed(1)}]pp`;
  document.getElementById("cf-sig").className = `sig-badge ${cfSig ? "sig-yes" : "sig-no"}`;
  document.getElementById("cf-sig").textContent = cfSig ? "Significant ✓" : "Non-significant";

  // RL 1-year
  const rl1 = d.rl_ite[0], rlLo = d.rl_lo[0], rlHi = d.rl_hi[0];
  const rlSig = !(rlLo < 0 && rlHi > 0);
  document.getElementById("rl-val").innerHTML =
    `<span style="color:${rl1 < 0 ? "var(--green)" : "var(--red)"}">
       ${(rl1*100).toFixed(1)}pp
     </span>`;
  document.getElementById("rl-ci").textContent = `95% CI [${(rlLo*100).toFixed(1)}, ${(rlHi*100).toFixed(1)}]pp`;
  document.getElementById("rl-sig").className = `sig-badge ${rlSig ? "sig-yes" : "sig-no"}`;
  document.getElementById("rl-sig").textContent = rlSig ? "Significant ✓" : "Non-significant";

  // Overlap
  document.getElementById("overlap-div").innerHTML = `
    <div class="overlap-pill ${d.in_overlap ? "overlap-yes" : "overlap-no"}">
      ${d.in_overlap ? "✓ In propensity overlap" : "⚠ Outside propensity overlap"}
      — propensity ${(d.propensity*100).toFixed(1)}%
      (${d.in_overlap ? "CF/RL estimates reliable" : "RSF fallback used for zone"})
    </div>`;
}
