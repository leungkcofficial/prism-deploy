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
      let detail = `Server error (HTTP ${resp.status})`;
      try { const err = await resp.json(); detail = err.detail || detail; } catch (_) {}
      throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
    }

    let data;
    try {
      data = await resp.json();
    } catch (parseErr) {
      // Show raw response body to help diagnose serialisation issues
      const raw = await resp.text().catch(() => "(unreadable)");
      throw new Error(`Response is not valid JSON. Server returned: ${raw.slice(0, 200)}`);
    }
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

  // Patient-friendly RMST summary
  renderRMST(d);

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


// ── RMST Patient-Friendly Summary ────────────────────────────────────────────

function renderRMST(d) {
  const el = document.getElementById("rmst-summary");
  if (!d.rmst_diff_months || !el) return;

  const months = d.rmst_diff_months;
  const a0 = d.rmst_A0;
  const a1 = d.rmst_A1;
  const horizons = [0, 2, 4]; // 1y, 3y, 5y
  const labels = ["1 year", "3 years", "5 years"];

  // Headline: 5-year difference
  const d5 = months[4];
  const absD5 = Math.abs(d5);
  const direction = d5 > 0 ? "longer" : "shorter";
  let headline;
  if (absD5 >= 1.0) {
    headline = `~${absD5.toFixed(1)} months ${direction}`;
  } else {
    headline = `~${Math.round(absD5 * 30.44)} days ${direction}`;
  }

  const ciLo = d.rmst_ci_lo_months;
  const ciHi = d.rmst_ci_hi_months;
  const hasCi = ciLo !== null && ciHi !== null;

  let rows = "";
  horizons.forEach((idx, i) => {
    const m0 = (a0[idx] / 30.44).toFixed(1);
    const m1 = (a1[idx] / 30.44).toFixed(1);
    const dm = months[idx];
    const absDm = Math.abs(dm);
    let diffStr;
    if (absDm >= 1.0) {
      diffStr = `${dm > 0 ? "+" : ""}${dm.toFixed(1)} mo`;
    } else {
      diffStr = `${dm > 0 ? "+" : ""}${Math.round(dm * 30.44)} days`;
    }
    // Add CI (only for zone A/B)
    let ciStr = "";
    if (hasCi && ciLo.length > idx && ciHi.length > idx) {
      const lo = ciLo[idx], hi = ciHi[idx];
      ciStr = `<br><span style="font-size:11px;font-weight:400;opacity:0.8">[${lo > 0 ? "+" : ""}${lo.toFixed(1)}, +${hi.toFixed(1)}] mo</span>`;
    }
    const diffColor = dm > 0 ? "var(--green)" : "var(--red)";
    rows += `<tr>
      <td style="font-weight:500">${labels[i]}</td>
      <td>${m0} mo</td>
      <td>${m1} mo</td>
      <td style="color:${diffColor};font-weight:600">${diffStr}${ciStr}</td>
    </tr>`;
  });

  // CI for headline (only for zone A/B)
  let ciNote = "";
  if (hasCi && ciLo.length > 4 && ciHi.length > 4) {
    const lo5 = ciLo[4], hi5 = ciHi[4];
    ciNote = `<div style="font-size:13px;color:var(--text-muted);margin-top:2px">95% CI: [${lo5 > 0 ? "+" : ""}${lo5.toFixed(1)}, +${hi5.toFixed(1)}] months</div>`;
  } else {
    ciNote = `<div style="font-size:11px;color:var(--text-muted);margin-top:4px">(95% CI not available for RSF-based RMST)</div>`;
  }

  el.innerHTML = `
    <div class="rmst-headline">
      Over the next 5 years, early dialysis is associated with<br>
      <span class="rmst-number" style="color:${d5 > 0 ? 'var(--green)' : 'var(--red)'}">
        ${headline}
      </span>
      of expected life
      ${ciNote}
    </div>
    <table class="rmst-table">
      <thead>
        <tr>
          <th>Horizon</th>
          <th>Without dialysis</th>
          <th>With dialysis</th>
          <th>Difference</th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>
    <div class="rmst-note">
      RMST = Restricted Mean Survival Time. These estimates reflect expected survival
      within each horizon based on similar patients, not total life expectancy.
    </div>`;
  el.style.display = "block";
}
