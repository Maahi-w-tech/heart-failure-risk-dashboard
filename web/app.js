/*
 * Gradient Descent Lab
 * --------------------
 * This is intentionally written without a framework so every part of the
 * learning loop is visible: standardize inputs, calculate a sigmoid, measure
 * cross-entropy loss, calculate the gradient, and update the weights.
 */

const FEATURE_LIBRARY = [
  {key: 'age', label: 'Age', min: 40, max: 100, step: 1, decimals: 0, unit: 'years', help: 'patient age'},
  {key: 'ejection_fraction', label: 'Ejection fraction', min: 10, max: 80, step: 1, decimals: 0, unit: '%', help: 'heart pumping percentage'},
  {key: 'serum_creatinine', label: 'Serum creatinine', min: .4, max: 9.5, step: .1, decimals: 1, unit: 'mg/dL', help: 'kidney function marker'},
  {key: 'serum_sodium', label: 'Serum sodium', min: 110, max: 150, step: 1, decimals: 0, unit: 'mEq/L', help: 'blood sodium level'},
  {key: 'time', label: 'Follow-up time', min: 1, max: 300, step: 1, decimals: 0, unit: 'days', help: 'observed follow-up duration'},
  {key: 'platelets', label: 'Platelets', min: 25000, max: 850000, step: 1000, decimals: 0, unit: '/µL', help: 'platelet count'}
];

const CASES = {
  typical: {age: 52, ejection_fraction: 48, serum_creatinine: .8, serum_sodium: 140, time: 230, platelets: 270000},
  higher: {age: 82, ejection_fraction: 20, serum_creatinine: 2.6, serum_sodium: 128, time: 24, platelets: 155000},
  borderline: {age: 67, ejection_fraction: 34, serum_creatinine: 1.3, serum_sodium: 135, time: 105, platelets: 230000}
};

const state = {
  rows: [],
  target: 'DEATH_EVENT',
  classes: [0, 1],
  featureDefs: FEATURE_LIBRARY,
  patient: {...CASES.typical},
  activeCase: 'typical',
  dataset: null,
  model: null,
  history: [],
  iteration: 0,
  maxIterations: 180,
  learningRate: .12,
  playing: false,
  playTimer: null,
  charts: {}
};

const $ = id => document.getElementById(id);
const clamp = (value, min, max) => Math.max(min, Math.min(max, value));
const sigmoid = value => 1 / (1 + Math.exp(-clamp(value, -35, 35)));
const fmt = (value, decimals = 3) => Number.isFinite(Number(value)) ? Number(value).toFixed(decimals) : '—';
const pct = value => `${Math.round(clamp(Number(value) || 0, 0, 1) * 100)}%`;
const esc = value => String(value).replace(/[&<>"']/g, char => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;'}[char]));

function csvParse(text) {
  const lines = text.trim().split(/\r?\n/).filter(Boolean);
  if (lines.length < 2) throw new Error('The CSV must contain a header and at least one data row.');
  const parseLine = line => {
    const output = [];
    let current = '';
    let quoted = false;
    for (let i = 0; i < line.length; i += 1) {
      const char = line[i];
      if (char === '"' && line[i + 1] === '"') { current += '"'; i += 1; }
      else if (char === '"') quoted = !quoted;
      else if (char === ',' && !quoted) { output.push(current.trim()); current = ''; }
      else current += char;
    }
    output.push(current.trim());
    return output.map(value => value.replace(/^"|"$/g, ''));
  };
  const headers = parseLine(lines[0]);
  return lines.slice(1).map(line => {
    const values = parseLine(line);
    return Object.fromEntries(headers.map((header, index) => [header, values[index] || '']));
  });
}

function targetNumber(value, classes) {
  const normalized = String(value ?? '').trim().toLowerCase();
  if (['0', 'no', 'false', 'negative', 'survived', 'survival', 'none', 'alive'].includes(normalized)) return 0;
  if (['1', 'yes', 'true', 'positive', 'death', 'event', 'dead'].includes(normalized)) return 1;
  return classes.indexOf(value) === 1 ? 1 : Number(value) > 0 ? 1 : 0;
}

function detectTarget(headers) {
  return headers.find(header => /death.?event|target|outcome|label|class/i.test(header)) || headers[headers.length - 1];
}

function numeric(value) {
  return value !== '' && Number.isFinite(Number(value));
}

function getFeatureDefs(rows, target) {
  const headers = Object.keys(rows[0] || {});
  const known = FEATURE_LIBRARY.filter(feature => headers.includes(feature.key) && rows.some(row => numeric(row[feature.key])));
  if (known.length >= 3) return known;
  return headers.filter(header => header !== target && rows.some(row => numeric(row[header]))).slice(0, 6).map((key, index) => ({
    key, label: key.replace(/_/g, ' ').replace(/\b\w/g, char => char.toUpperCase()), min: 0, max: 100, step: 1, decimals: 1, unit: '', help: `numeric feature ${index + 1}`
  }));
}

function featureValue(feature, raw) {
  const value = Number(raw);
  return Number.isFinite(value) ? value : (feature.min + feature.max) / 2;
}

function displayFeatureValue(feature, value) {
  const decimals = feature.decimals ?? 1;
  const formatted = Number(value).toLocaleString(undefined, {minimumFractionDigits: decimals, maximumFractionDigits: decimals});
  return `${formatted}${feature.unit ? ` ${feature.unit}` : ''}`;
}

function prepareDataset(rows, target, featureDefs) {
  const classes = [...new Set(rows.map(row => row[target]))];
  const usable = rows.filter(row => featureDefs.every(feature => numeric(row[feature.key])) && row[target] !== '');
  if (classes.length !== 2 || usable.length < 10) throw new Error('The target needs two classes and at least 10 complete numeric rows.');
  const raw = usable.map(row => featureDefs.map(feature => Number(row[feature.key])));
  const y = usable.map(row => targetNumber(row[target], classes));
  const means = featureDefs.map((_, index) => raw.reduce((sum, row) => sum + row[index], 0) / raw.length);
  const sds = featureDefs.map((_, index) => Math.sqrt(raw.reduce((sum, row) => sum + (row[index] - means[index]) ** 2, 0) / raw.length) || 1);
  const X = raw.map(row => [1, ...row.map((value, index) => (value - means[index]) / sds[index])]);
  return {usable, raw, y, X, means, sds, classes};
}

function dot(left, right) {
  return left.reduce((sum, value, index) => sum + value * right[index], 0);
}

function modelLoss(params, dataset) {
  const tiny = 1e-12;
  return dataset.X.reduce((sum, row, index) => {
    const probability = clamp(sigmoid(dot(params, row)), tiny, 1 - tiny);
    return sum - (dataset.y[index] * Math.log(probability) + (1 - dataset.y[index]) * Math.log(1 - probability));
  }, 0) / dataset.X.length;
}

function gradients(params, dataset) {
  const values = Array(params.length).fill(0);
  dataset.X.forEach((row, index) => {
    const error = sigmoid(dot(params, row)) - dataset.y[index];
    row.forEach((value, column) => { values[column] += error * value; });
  });
  return values.map(value => value / dataset.X.length);
}

function gradientSize(values) {
  return Math.sqrt(values.reduce((sum, value) => sum + value ** 2, 0));
}

function recordHistory() {
  const params = state.model.params;
  const currentGradients = gradients(params, state.dataset);
  state.history.push({iteration: state.iteration, params: [...params], gradients: currentGradients, loss: modelLoss(params, state.dataset)});
}

function resetTraining(render = true) {
  stopPlaying();
  state.iteration = 0;
  state.model = {params: Array(state.featureDefs.length + 1).fill(0)};
  state.history = [];
  recordHistory();
  if (render) renderAll();
}

function trainStep(amount = 1, render = true) {
  if (!state.dataset || !state.model) return;
  for (let step = 0; step < amount && state.iteration < state.maxIterations; step += 1) {
    const currentGradients = gradients(state.model.params, state.dataset);
    state.model.params = state.model.params.map((weight, index) => weight - state.learningRate * currentGradients[index]);
    state.iteration += 1;
    recordHistory();
  }
  if (state.iteration >= state.maxIterations) stopPlaying();
  if (render) renderAll();
}

function stopPlaying() {
  if (state.playTimer) window.clearInterval(state.playTimer);
  state.playTimer = null;
  state.playing = false;
  const button = $('playTraining');
  if (button) button.innerHTML = '<span class="play-icon">▶</span> Play training';
}

function playTraining() {
  if (!state.dataset) return;
  if (state.iteration >= state.maxIterations) resetTraining(false);
  if (state.playing) { stopPlaying(); renderTraining(); return; }
  state.playing = true;
  $('playTraining').innerHTML = '<span class="play-icon">Ⅱ</span> Pause training';
  state.playTimer = window.setInterval(() => {
    trainStep(3);
    if (state.iteration >= state.maxIterations) stopPlaying();
  }, 55);
}

function renderLossChart() {
  if (!window.Chart) return;
  if (state.charts.loss) state.charts.loss.destroy();
  state.charts.loss = new Chart($('lossChart'), {
    type: 'line',
    data: {labels: state.history.map(point => point.iteration), datasets: [{data: state.history.map(point => point.loss), borderColor: '#3867ee', backgroundColor: '#3867ee16', borderWidth: 2.5, pointRadius: 0, fill: true, tension: .25}]},
    options: {responsive: true, maintainAspectRatio: false, plugins: {legend: {display: false}, tooltip: {displayColors: false, callbacks: {label: context => ` loss: ${fmt(context.parsed.y)}`}}}, scales: {x: {title: {display: true, text: 'training iteration', color: '#8b98ac', font: {size: 10}}, grid: {display: false}, ticks: {color: '#8b98ac', maxTicksLimit: 7}}, y: {beginAtZero: false, title: {display: true, text: 'cross-entropy loss', color: '#8b98ac', font: {size: 10}}, grid: {color: '#edf0f5'}, ticks: {color: '#8b98ac', maxTicksLimit: 5}}}}
  });
}

function renderWeightChart() {
  if (!window.Chart || !state.history.length) return;
  if (state.charts.weight) state.charts.weight.destroy();
  const featureIndex = Math.max(0, Number($('trackedFeature').value || 0));
  const feature = state.featureDefs[featureIndex];
  state.charts.weight = new Chart($('weightChart'), {
    type: 'line',
    data: {labels: state.history.map(point => point.iteration), datasets: [{label: feature?.label || 'weight', data: state.history.map(point => point.params[featureIndex + 1]), borderColor: '#12a99b', backgroundColor: '#12b7a514', borderWidth: 2, pointRadius: 0, fill: true, tension: .24}]},
    options: {responsive: true, maintainAspectRatio: false, plugins: {legend: {display: false}}, scales: {x: {grid: {display: false}, ticks: {color: '#8b98ac', maxTicksLimit: 6}}, y: {grid: {color: '#edf0f5'}, ticks: {color: '#8b98ac', maxTicksLimit: 5}}}}
  });
}

function renderTraining() {
  if (!state.history.length) return;
  const latest = state.history[state.history.length - 1];
  const size = gradientSize(latest.gradients);
  $('trainingIteration').textContent = state.iteration;
  $('trainedStepLabel').textContent = state.iteration;
  $('lossValue').textContent = fmt(latest.loss);
  $('currentLoss').textContent = fmt(latest.loss);
  $('startLoss').textContent = fmt(state.history[0].loss);
  $('gradientSize').textContent = fmt(size);
  $('gradientBadge').textContent = `step ${state.iteration}`;
  $('stepLabel').textContent = state.iteration === 0 ? 'Weights are ready to learn' : state.iteration >= state.maxIterations ? 'Training is complete' : 'Weights just moved downhill';
  $('stepExplanation').innerHTML = state.iteration === 0
    ? 'At the start, all weights are zero, so every patient receives a probability of 0.50. Press <b>Play training</b> to see the loss fall.'
    : `The model used the gradient to update ${state.featureDefs.length} feature weights. Its average error is now <b>${fmt(latest.loss)}</b>. ${state.iteration >= state.maxIterations ? 'Try a case below to use the learned weights.' : 'Keep going to watch it settle.'}`;
  const gradientMax = Math.max(...latest.gradients.slice(1).map(Math.abs), .001);
  $('gradientList').innerHTML = state.featureDefs.map((feature, index) => {
    const value = latest.gradients[index + 1];
    const width = Math.max(3, Math.abs(value) / gradientMax * 50);
    return `<div class="gradient-row"><label title="${esc(feature.label)}">${esc(feature.label)}</label><div class="gradient-track"><span class="gradient-center"></span><i class="${value >= 0 ? 'positive' : 'negative'}" style="width:${width}%"></i></div><span class="gradient-value">${value >= 0 ? '+' : ''}${fmt(value)}</span></div>`;
  }).join('');
  const trackedValue = Number($('trackedFeature').value || 0);
  $('trackedFeature').value = String(Math.min(trackedValue, Math.max(0, state.featureDefs.length - 1)));
  renderLossChart();
  renderWeightChart();
}

function renderPatientInputs() {
  $('patientInputs').innerHTML = state.featureDefs.map((feature, index) => {
    const value = state.patient[feature.key] ?? (feature.min + feature.max) / 2;
    return `<div class="patient-field"><label for="input-${esc(feature.key)}"><span>${esc(feature.label)}</span><output id="output-${esc(feature.key)}">${esc(displayFeatureValue(feature, value))}</output></label><input id="input-${esc(feature.key)}" data-feature="${esc(feature.key)}" data-index="${index}" type="range" min="${feature.min}" max="${feature.max}" step="${feature.step}" value="${value}" /><small class="field-help">${esc(feature.help)}</small></div>`;
  }).join('');
  state.featureDefs.forEach(feature => {
    const input = $(`input-${feature.key}`);
    if (input) input.addEventListener('input', event => {
      state.patient[feature.key] = Number(event.target.value);
      $(`output-${feature.key}`).textContent = displayFeatureValue(feature, state.patient[feature.key]);
      state.activeCase = null;
      document.querySelectorAll('.case-button').forEach(button => button.classList.remove('active'));
      renderPrediction();
    });
  });
}

function normalizedPatient() {
  return state.featureDefs.map((feature, index) => (Number(state.patient[feature.key]) - state.dataset.means[index]) / state.dataset.sds[index]);
}

function renderPrediction() {
  if (!state.dataset || !state.model) return;
  const normalized = normalizedPatient();
  const params = state.model.params;
  const contributions = normalized.map((value, index) => value * params[index + 1]);
  const z = params[0] + contributions.reduce((sum, value) => sum + value, 0);
  const probability = sigmoid(z);
  const threshold = Number($('thresholdSlider').value);
  const flagged = probability >= threshold;
  $('zValue').textContent = fmt(z, 2);
  $('probabilityValue').textContent = pct(probability);
  $('thresholdValue').textContent = pct(threshold);
  $('riskMeterFill').style.width = `${probability * 100}%`;
  $('riskMeterMarker').style.left = `${probability * 100}%`;
  const classification = $('classificationValue');
  classification.className = `classification ${flagged ? 'flag' : 'clear'}`;
  classification.textContent = flagged ? 'Event flag at this threshold' : 'No event flag at this threshold';
  $('predictionExplanation').textContent = flagged
    ? `The probability is above the ${pct(threshold)} cutoff, so this example is flagged for an event.`
    : `The probability is below the ${pct(threshold)} cutoff, so this example is not flagged for an event.`;
  const maxContribution = Math.max(...contributions.map(Math.abs), .001);
  $('contributionList').innerHTML = state.featureDefs.map((feature, index) => {
    const value = contributions[index];
    const width = Math.max(3, Math.abs(value) / maxContribution * 50);
    return `<div class="contribution-row"><label title="${esc(feature.label)}">${esc(feature.label)}</label><div class="contribution-track"><i class="${value >= 0 ? 'positive' : 'negative'}" style="width:${width}%"></i></div><span class="contribution-value">${value >= 0 ? '+' : ''}${fmt(value, 2)}</span></div>`;
  }).join('');
}

function renderDataset() {
  const dataset = state.dataset;
  if (!dataset) return;
  const positive = dataset.y.filter(Boolean).length;
  $('datasetStatus').textContent = `${dataset.usable.length} rows · ${state.featureDefs.length} visible features`;
  $('datasetCards').innerHTML = [['Rows', dataset.usable.length], ['Features', state.featureDefs.length], ['Events', positive], ['Missing', state.rows.length - dataset.usable.length]].map(([label, value]) => `<div class="dataset-stat"><b>${value}</b><span>${label}</span></div>`).join('');
}

function renderAll() {
  renderTraining();
  renderPrediction();
}

function setCase(name) {
  const preset = CASES[name];
  if (!preset) return;
  state.activeCase = name;
  state.featureDefs.forEach(feature => {
    const fallback = state.dataset.means[state.featureDefs.indexOf(feature)];
    state.patient[feature.key] = clamp(preset[feature.key] ?? fallback, feature.min, feature.max);
  });
  renderPatientInputs();
  document.querySelectorAll('.case-button').forEach(button => button.classList.toggle('active', button.dataset.case === name));
  renderPrediction();
}

function loadRows(rows, isDemo = false) {
  if (!rows.length) throw new Error('No rows were found.');
  const headers = Object.keys(rows[0]);
  state.target = detectTarget(headers);
  state.featureDefs = getFeatureDefs(rows, state.target);
  if (state.featureDefs.length < 3) throw new Error('At least three numeric predictor columns are needed.');
  state.rows = rows;
  state.dataset = prepareDataset(rows, state.target, state.featureDefs);
  state.classes = state.dataset.classes;
  state.patient = {};
  state.featureDefs.forEach((feature, index) => {
    const defaults = isDemo ? CASES.typical[feature.key] : state.dataset.means[index];
    state.patient[feature.key] = clamp(defaults ?? state.dataset.means[index], feature.min, feature.max);
  });
  $('datasetStatus').textContent = isDemo ? 'Demo dataset ready' : 'Uploaded CSV ready';
  $('trackedFeature').innerHTML = state.featureDefs.map((feature, index) => `<option value="${index}">${esc(feature.label)} weight</option>`).join('');
  renderPatientInputs();
  renderDataset();
  resetTraining(false);
  trainStep(state.maxIterations, false);
  renderAll();
}

async function loadDemo() {
  try {
    const response = await fetch('data/heart_failure_clinical_records_dataset.csv');
    if (!response.ok) throw new Error('Bundled dataset unavailable.');
    loadRows(csvParse(await response.text()), true);
  } catch (error) {
    $('datasetStatus').textContent = 'Could not load demo dataset';
    $('patientInputs').innerHTML = `<p class="input-note">${esc(error.message)} Upload the CSV to start the lab.</p>`;
    $('stepExplanation').textContent = 'Upload the supplied CSV to activate the interactive training loop.';
  }
}

function bindEvents() {
  $('resetTraining').addEventListener('click', () => resetTraining());
  $('stepTraining').addEventListener('click', () => trainStep(10));
  $('playTraining').addEventListener('click', playTraining);
  $('trackedFeature').addEventListener('change', renderWeightChart);
  $('thresholdSlider').addEventListener('input', renderPrediction);
  $('learningRate').addEventListener('input', event => {
    state.learningRate = Number(event.target.value);
    $('learningRateValue').textContent = state.learningRate.toFixed(2);
    resetTraining();
  });
  document.querySelectorAll('.case-button').forEach(button => button.addEventListener('click', () => setCase(button.dataset.case)));
  $('csvFile').addEventListener('change', event => {
    const file = event.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      try { loadRows(csvParse(reader.result), false); document.querySelector('#learn').scrollIntoView({behavior: 'smooth'}); }
      catch (error) { window.alert(`Could not analyze CSV: ${error.message}`); }
    };
    reader.readAsText(file);
  });
}

bindEvents();
loadDemo();
