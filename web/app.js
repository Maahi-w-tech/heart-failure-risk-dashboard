/*
 * Heart Failure Risk Lab — explainability frontend
 *
 * This file deliberately does not train a model. The Python project pipeline
 * writes web/data/dashboard_data.json, and this page visualizes that artifact
 * plus the bundled source CSV. That keeps probabilities, metrics, thresholds,
 * and explanations tied to the existing fitted Gradient Boosting model.
 */

const state = {
  artifact: null,
  rows: [],
  sourceRows: [],
  threshold: 0.35,
  selectedRecord: 0,
  selectedFeature: '',
  selectedMetric: 'recall',
  selectedCell: null,
  mode: 'client',
  currentStage: 'dataset',
  tourStep: 0,
  charts: {}
};

const $ = id => document.getElementById(id);
const clamp = (value, min, max) => Math.max(min, Math.min(max, value));
const fmt = (value, digits = 3) => Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : '—';
const pct = (value, digits = 1) => Number.isFinite(Number(value)) ? `${(Number(value) * 100).toFixed(digits)}%` : '—';
const esc = value => String(value).replace(/[&<>"']/g, char => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;'}[char]));
const titleize = value => String(value).replace(/_/g, ' ').replace(/\b\w/g, char => char.toUpperCase());

function parseCsv(text) {
  const lines = text.trim().split(/\r?\n/).filter(Boolean);
  if (lines.length < 2) throw new Error('The CSV needs a header and at least one data row.');
  const parseLine = line => {
    const values = [];
    let current = '';
    let quoted = false;
    for (let index = 0; index < line.length; index += 1) {
      const char = line[index];
      if (char === '"' && line[index + 1] === '"') { current += '"'; index += 1; }
      else if (char === '"') quoted = !quoted;
      else if (char === ',' && !quoted) { values.push(current.trim()); current = ''; }
      else current += char;
    }
    values.push(current.trim());
    return values.map(value => value.replace(/^"|"$/g, ''));
  };
  const headers = parseLine(lines[0]);
  return lines.slice(1).map(line => {
    const values = parseLine(line);
    return Object.fromEntries(headers.map((header, index) => [header, values[index] || '']));
  });
}

function featureInfo(name) {
  return state.artifact.dataset.features.find(feature => feature.name === name) || {
    name,
    label: titleize(name),
    kind: 'numeric',
    description: 'Predictor supplied by the dataset.'
  };
}

function selectedRecord() {
  return state.artifact.test_predictions[state.selectedRecord] || state.artifact.test_predictions[0];
}

function modelResult(name) {
  return state.artifact.models.find(model => model.name === name);
}

function selectedMetrics() {
  const model = modelResult(state.artifact.selected_model);
  return model.metrics_at_035;
}

function actualTarget(value) {
  return Number(value) === 1 ? 'Positive event (1)' : 'Negative event (0)';
}

function thresholdMetrics(threshold) {
  const records = state.artifact.test_predictions;
  let tp = 0; let tn = 0; let fp = 0; let fn = 0;
  records.forEach(record => {
    const predicted = record.probability >= threshold ? 1 : 0;
    if (record.actual === 1 && predicted === 1) tp += 1;
    else if (record.actual === 0 && predicted === 0) tn += 1;
    else if (record.actual === 0 && predicted === 1) fp += 1;
    else fn += 1;
  });
  const precision = tp / (tp + fp) || 0;
  const recall = tp / (tp + fn) || 0;
  const specificity = tn / (tn + fp) || 0;
  const accuracy = (tp + tn) / records.length;
  const f1 = (2 * precision * recall) / (precision + recall) || 0;
  return {
    accuracy, precision, recall, f1, specificity,
    balanced_accuracy: (recall + specificity) / 2,
    tn, fp, fn, tp, threshold,
    predicted_positive: tp + fp,
    predicted_negative: tn + fn,
    roc_auc: modelResult(state.artifact.selected_model).metrics_at_050.roc_auc,
    pr_auc: modelResult(state.artifact.selected_model).metrics_at_050.pr_auc
  };
}

function destroyChart(name) {
  if (state.charts[name]) {
    state.charts[name].destroy();
    state.charts[name] = null;
  }
}

const thresholdLinesPlugin = {
  id: 'thresholdLines',
  afterDatasetsDraw(chart, args, options) {
    const xScale = chart.scales.x;
    if (!xScale) return;
    const context = chart.ctx;
    const drawLine = (value, color, dash, label) => {
      const x = xScale.getPixelForValue(value);
      if (!Number.isFinite(x)) return;
      context.save();
      context.beginPath();
      context.setLineDash(dash);
      context.lineWidth = value === options.selected ? 2 : 1;
      context.strokeStyle = color;
      context.moveTo(x, chart.chartArea.top);
      context.lineTo(x, chart.chartArea.bottom);
      context.stroke();
      context.setLineDash([]);
      context.fillStyle = color;
      context.font = '700 10px system-ui';
      context.fillText(label, x + 5, chart.chartArea.top + 13);
      context.restore();
    };
    drawLine(0.5, '#9ba8bd', [4, 4], '0.50 reference');
    drawLine(options.selected, '#3867ee', [], `threshold ${Number(options.selected).toFixed(2)}`);
  }
};

function renderStatus() {
  const {dataset, thresholds} = state.artifact;
  $('artifactStatus').textContent = 'Saved analysis artifact loaded';
  $('artifactMeta').textContent = `Generated by ${state.artifact.generated_by} · probabilities and metrics are from the existing Python run.`;
  $('statusTarget').textContent = dataset.target;
  $('statusThreshold').textContent = thresholds.analysis.toFixed(2);
  $('statusSplit').textContent = `${dataset.test_rows} / ${dataset.rows}`;
  $('stageModelMeta').textContent = `${state.artifact.model.estimator_count} trees`;
  $('stageThresholdMeta').textContent = `${thresholds.analysis.toFixed(2)} / ${thresholds.default.toFixed(2)}`;
  $('heroRows').textContent = dataset.rows;
  $('heroTestRows').textContent = dataset.test_rows;
  $('limitRows').textContent = dataset.rows;
  $('limitTestRows').textContent = dataset.test_rows;
}

function configureThresholdControls() {
  const thresholds = state.artifact.thresholds;
  $('thresholdSlider').min = thresholds.slider_min;
  $('thresholdSlider').max = thresholds.slider_max;
  const referenceButtons = document.querySelectorAll('.reference-button');
  if (referenceButtons[0]) {
    referenceButtons[0].dataset.threshold = thresholds.analysis;
    referenceButtons[0].innerHTML = `Current analysis · <b>${thresholds.analysis.toFixed(2)}</b>`;
  }
  if (referenceButtons[1]) {
    referenceButtons[1].dataset.threshold = thresholds.default;
    referenceButtons[1].innerHTML = `Default reference · <b>${thresholds.default.toFixed(2)}</b>`;
  }
}

function renderDatasetStats() {
  const dataset = state.artifact.dataset;
  const target = dataset.target;
  const values = state.rows.map(row => row[target]).filter(value => value !== '');
  const positive = values.filter(value => Number(value) === 1).length;
  const headers = Object.keys(state.rows[0] || {});
  const missing = state.rows.reduce((total, row) => total + headers.filter(header => row[header] === '').length, 0);
  const usable = state.rows.filter(row => headers.every(header => row[header] !== '')).length;
  const duplicateKeys = new Set(state.rows.map(row => headers.map(header => row[header]).join('\u0001'))).size;
  const duplicates = state.rows.length - duplicateKeys;
  const cards = [
    ['Total records', state.rows.length, 'actual rows in source CSV'],
    ['Usable records', usable, 'complete rows in this view'],
    ['Features', dataset.feature_count, 'predictors before target'],
    ['Positive events', positive, 'target value = 1'],
    ['Negative events', values.length - positive, 'target value = 0'],
    ['Missing values', missing, 'checked in this view'],
    ['Held-out test', dataset.test_rows, 'stratified test rows'],
    ['Duplicates', duplicates, 'checked in this view']
  ];
  $('datasetCards').innerHTML = cards.map(([label, value, note]) => `<div class="stat-card"><b>${esc(value)}</b><span>${esc(label)}</span><small>${esc(note)}</small></div>`).join('');
  $('targetName').textContent = target;
  $('targetDescription').textContent = dataset.target_description;
  $('numericFeatures').innerHTML = dataset.numerical_features.map(name => `<span class="feature-chip numeric-chip">${esc(featureInfo(name).label)}</span>`).join('');
  $('categoricalFeatures').innerHTML = dataset.categorical_features.map(name => `<span class="feature-chip binary-chip">${esc(featureInfo(name).label)} <i>0/1</i></span>`).join('');
  $('schemaNote').innerHTML = `<b>Technical schema:</b> all 12 predictors are numeric columns in the CSV. The six binary indicators are shown separately because their meaning is categorical, but the existing tree model receives the raw numeric values without encoding.`;
  $('featureSelect').innerHTML = dataset.features.map(feature => `<option value="${esc(feature.name)}">${esc(feature.label)} · ${feature.kind}</option>`).join('');
  state.selectedFeature = state.selectedFeature || dataset.features[0].name;
  $('featureSelect').value = state.selectedFeature;
  renderTargetChart(positive, values.length - positive);
  renderFeatureExplorer();
}

function renderTargetChart(positive, negative) {
  if (typeof Chart === 'undefined') return;
  destroyChart('target');
  state.charts.target = new Chart($('targetChart'), {
    type: 'doughnut',
    data: {labels: ['No event (0)', 'Event (1)'], datasets: [{data: [negative, positive], backgroundColor: ['#3867ee', '#ee6b75'], borderWidth: 0}]},
    options: {responsive: true, maintainAspectRatio: false, cutout: '70%', plugins: {legend: {display: false}, tooltip: {callbacks: {label: context => `${context.label}: ${context.raw}`}}}}
  });
  $('targetLegend').innerHTML = `<div><i class="blue-dot"></i><b>${negative}</b><span>No recorded event</span></div><div><i class="red-dot"></i><b>${positive}</b><span>Recorded event</span></div>`;
}

function renderFeatureExplorer() {
  const name = $('featureSelect').value || state.selectedFeature;
  state.selectedFeature = name;
  const info = featureInfo(name);
  const values = state.rows.map(row => Number(row[name])).filter(Number.isFinite);
  $('featureDescription').textContent = `${info.description} This view is calculated from the loaded source CSV, not simulated values.`;
  const sortedValues = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sortedValues.length / 2);
  const median = sortedValues.length ? (sortedValues.length % 2 ? sortedValues[middle] : (sortedValues[middle - 1] + sortedValues[middle]) / 2) : null;
  const categories = [...new Set(values)].sort((a, b) => a - b).map(value => ({value, count: values.filter(item => item === value).length}));
  $('featureStats').innerHTML = info.kind === 'binary'
    ? categories.map(category => `<div><span>Value ${category.value}</span><b>${category.count}</b><small>${pct(category.count / state.rows.length, 0)} of rows</small></div>`).join('')
    : [['Min', Math.min(...values)], ['Mean', values.reduce((sum, value) => sum + value, 0) / values.length], ['Median', median], ['Max', Math.max(...values)]].map(([label, value]) => `<div><span>${label}</span><b>${esc(fmt(value, 2))}</b><small>${esc(info.label)}</small></div>`).join('');
  if (typeof Chart === 'undefined') return;
  destroyChart('feature');
  if (info.kind === 'binary') {
    state.charts.feature = new Chart($('featureChart'), {type: 'bar', data: {labels: categories.map(category => `Value ${category.value}`), datasets: [{data: categories.map(category => category.count), backgroundColor: ['#3867ee', '#12b7a5'], borderRadius: 7, barThickness: 36}]}, options: chartOptions('rows', false)});
    return;
  }
  const bins = 8;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const width = (max - min || 1) / bins;
  const counts = Array(bins).fill(0);
  values.forEach(value => counts[Math.min(bins - 1, Math.floor((value - min) / width))] += 1);
  const labels = counts.map((_, index) => `${fmt(min + index * width, 1)}–${fmt(min + (index + 1) * width, 1)}`);
  state.charts.feature = new Chart($('featureChart'), {type: 'bar', data: {labels, datasets: [{data: counts, backgroundColor: '#3867ee', borderRadius: 7, barPercentage: .9}]}, options: chartOptions('rows', false)});
}

function chartOptions(yTitle, hideLegend = true) {
  return {responsive: true, maintainAspectRatio: false, plugins: {legend: {display: !hideLegend}}, scales: {x: {grid: {display: false}, ticks: {color: '#8794a9', maxRotation: 0, autoSkip: true, maxTicksLimit: 7}}, y: {beginAtZero: true, title: {display: Boolean(yTitle), text: yTitle, color: '#8794a9', font: {size: 10}}, grid: {color: '#edf0f5'}, ticks: {color: '#8794a9', precision: 0}}}};
}

const pipelineCopy = {
  dataset: {title: 'Dataset · actual source records', tag: 'ACTUAL', body: () => `The app uses the supplied UCI Heart Failure Clinical Records CSV: <b>${state.artifact.dataset.rows}</b> records, <b>${state.artifact.dataset.feature_count}</b> predictor columns, and target <code>${state.artifact.dataset.target}</code>. The CSV is bundled at <code>web/data/heart_failure_clinical_records_dataset.csv</code>.`},
  validation: {title: 'Data validation · checks before modeling', tag: 'ACTUAL', body: () => `The Python run checks missing values and duplicate rows before fitting. The supplied data has <b>${state.artifact.dataset.missing_values}</b> missing values and <b>${state.artifact.dataset.duplicate_rows}</b> duplicate rows. Target counts are <b>${state.artifact.dataset.positive_events}</b> positive and <b>${state.artifact.dataset.negative_events}</b> negative.`},
  preprocessing: {title: 'Preprocessing · split without leaking the test set', tag: 'ACTUAL', body: () => `The target <code>${state.artifact.dataset.target}</code> is separated, then the existing code runs <code>train_test_split(test_size=0.20, stratify=y, random_state=42)</code>. That creates <b>${state.artifact.dataset.train_rows}</b> training rows and <b>${state.artifact.dataset.test_rows}</b> held-out test rows. Gradient Boosting receives raw numeric predictors; no scaling or one-hot encoding is performed for this tree model.`},
  features: {title: 'Features · the 12 predictors', tag: 'ACTUAL', body: () => `The feature matrix contains: <b>${state.artifact.model.feature_names.map(titleize).join(', ')}</b>. The project keeps <code>time</code> in the primary experiment, while the report flags follow-up duration as a possible early-warning limitation.`},
  model: {title: 'Gradient Boosting · the fitted ensemble', tag: 'ACTUAL', body: () => `The selected estimator is <code>${state.artifact.model.class}</code>. It combines <b>${state.artifact.model.estimator_count}</b> shallow trees. Tuning used 5-fold stratified RandomizedSearchCV on the training data only, with recall as the search score. Click the model section below to inspect every selected hyperparameter.`},
  probability: {title: 'Probability · model ranking output', tag: 'ACTUAL', body: () => `For each held-out row, the fitted estimator's <code>predict_proba(X_test)[:, 1]</code> becomes the positive-class probability. For the selected record, the saved probability is <b>${pct(selectedRecord().probability)}</b>. ROC-AUC and PR-AUC summarize ranking across thresholds.`},
  threshold: {title: 'Threshold · turn probability into a class', tag: 'ACTUAL', body: () => `The project can apply a cutoff after probability generation. The analysis reference is <b>${state.artifact.thresholds.analysis.toFixed(2)}</b>; the default reference is <b>${state.artifact.thresholds.default.toFixed(2)}</b>. The Threshold Lab below recalculates the confusion matrix and threshold-dependent metrics directly from all 60 saved test probabilities.`},
  prediction: {title: 'Prediction · the final class label', tag: 'MODEL OUTPUT', body: () => { const record = selectedRecord(); const predicted = record.probability >= state.threshold ? 1 : 0; return `For record <b>#${record.record_number}</b>, probability <b>${pct(record.probability)}</b> is ${predicted ? 'above' : 'below'} the current <b>${state.threshold.toFixed(2)}</b> threshold, so the prediction is <b>${actualTarget(predicted)}</b>. The actual held-out target is <b>${actualTarget(record.actual)}</b>.`; }},
  evaluation: {title: 'Evaluation · compare predictions with labels', tag: 'ACTUAL TEST SET', body: () => `At the current threshold, the held-out predictions are compared with the real labels to calculate TP, TN, FP, FN, accuracy, precision, recall, F1, and balanced accuracy. ROC-AUC and PR-AUC use the saved probabilities rather than one cutoff.`}
};

function renderPipeline(stage = state.currentStage) {
  state.currentStage = stage;
  document.querySelectorAll('.pipeline-stage').forEach(button => button.classList.toggle('active', button.dataset.stage === stage));
  const copy = pipelineCopy[stage];
  $('pipelineDetail').innerHTML = `<div class="detail-icon">${String(Object.keys(pipelineCopy).indexOf(stage) + 1).padStart(2, '0')}</div><div><span class="source-label ${copy.tag === 'ACTUAL' || copy.tag.includes('ACTUAL') ? 'actual-label' : 'model-label'}">${copy.tag}</span><h3>${copy.title}</h3><p>${copy.body()}</p></div>`;
}

function renderModel() {
  const {model, validation} = state.artifact;
  $('modelSummary').innerHTML = [['Model type', 'Gradient Boosting'], ['Trees combined', model.estimator_count], ['Best CV recall', pct(validation.best_cv_recall)], ['Tuning score', validation.scoring]]
    .map(([label, value]) => `<div class="model-stat"><span>${label}</span><b>${value}</b></div>`).join('');
  const parameterDescriptions = {learning_rate: 'Controls how strongly each new tree contributes to the ensemble.', n_estimators: 'Controls how many boosting stages are combined.', max_depth: 'Limits the depth of each individual decision tree.', min_samples_leaf: 'Minimum training rows allowed in each leaf.', subsample: 'Fraction of training rows sampled for each boosting stage.'};
  const ordered = ['n_estimators', 'learning_rate', 'max_depth', 'min_samples_leaf', 'subsample'];
  $('parameterList').innerHTML = ordered.map(key => `<div class="parameter-row"><div><b>${key}</b><p>${parameterDescriptions[key]}</p></div><strong>${esc(model.parameters[key])}</strong></div>`).join('');
  $('treeSlider').max = model.estimator_count;
  $('treeSlider').value = 1;
  renderTree(1);
  if (typeof Chart === 'undefined') return;
  const importance = Object.entries(model.feature_importances).sort((a, b) => b[1] - a[1]);
  destroyChart('importance');
  state.charts.importance = new Chart($('importanceChart'), {type: 'bar', data: {labels: importance.map(([name]) => featureInfo(name).label).reverse(), datasets: [{data: importance.map(([, value]) => value).reverse(), backgroundColor: '#3867ee', borderRadius: 6, barThickness: 15}]}, options: {indexAxis: 'y', responsive: true, maintainAspectRatio: false, plugins: {legend: {display: false}, tooltip: {callbacks: {label: context => ` ${fmt(context.raw, 3)}`}}}, scales: {x: {beginAtZero: true, grid: {color: '#edf0f5'}, ticks: {color: '#8794a9'}}, y: {grid: {display: false}, ticks: {color: '#41516d', font: {size: 10}}}}}});
}

function renderTree(index) {
  const tree = state.artifact.model.tree_summaries[index - 1];
  if (!tree) return;
  $('treeSliderValue').textContent = `${index} / ${state.artifact.model.estimator_count}`;
  $('treeSequence').innerHTML = `<b>Tree ${tree.index}</b><small>depth ${tree.depth} · ${tree.leaves} leaves</small>`;
  $('treeDetails').innerHTML = `<div><span>Root split feature</span><b>${tree.root_feature ? esc(featureInfo(tree.root_feature).label) : 'leaf'}</b></div><div><span>Root split threshold</span><b>${tree.root_threshold === null ? '—' : fmt(tree.root_threshold, 2)}</b></div><div><span>Fitted stage</span><b>${tree.index} of ${state.artifact.model.estimator_count}</b></div>`;
}

function renderRecordSelect() {
  $('recordSelect').innerHTML = state.artifact.test_predictions.map((record, index) => `<option value="${index}">Record #${record.record_number} · actual ${record.actual}</option>`).join('');
  $('recordSelect').value = String(state.selectedRecord);
  renderPrediction();
}

function renderPrediction() {
  const record = selectedRecord();
  if (!record) return;
  const predicted = record.probability >= state.threshold ? 1 : 0;
  const correct = predicted === record.actual;
  $('recordNumber').textContent = `#${record.record_number}`;
  $('recordActual').textContent = actualTarget(record.actual);
  $('recordProbability').textContent = pct(record.probability);
  $('recordThreshold').textContent = state.threshold.toFixed(2);
  $('recordDecisionScore').textContent = fmt(record.decision_score);
  $('recordBaseScore').textContent = fmt(record.base_score);
  $('recordMeter').style.width = `${record.probability * 100}%`;
  $('recordThresholdMarker').style.left = `${state.threshold * 100}%`;
  $('recordPrediction').className = `prediction-pill ${predicted ? 'positive-pill' : 'negative-pill'}`;
  $('recordPrediction').textContent = `Prediction: ${actualTarget(predicted)} at threshold ${state.threshold.toFixed(2)}`;
  $('recordCorrect').className = `record-result-chip ${correct ? 'correct-chip' : 'incorrect-chip'}`;
  $('recordCorrect').textContent = correct ? '✓ Correct at this threshold' : '× Incorrect at this threshold';
  $('recordFeatureTable').innerHTML = state.artifact.model.feature_names.map(name => `<div><span>${esc(featureInfo(name).label)}</span><b>${esc(record.features[name])}</b></div>`).join('');
  renderContributions(record);
  $('heroProbability').textContent = pct(record.probability);
  $('heroMeter').style.width = `${record.probability * 100}%`;
  $('heroThreshold').style.left = `${state.threshold * 100}%`;
}

function renderContributions(record) {
  const sorted = Object.entries(record.contributions).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
  const maximum = Math.max(...sorted.map(([, value]) => Math.abs(value)), .001);
  $('contributionList').innerHTML = sorted.map(([name, value]) => `<div class="contribution-row"><div class="contribution-name"><b>${esc(featureInfo(name).label)}</b><small>value ${esc(record.features[name])}</small></div><div class="contribution-track"><i class="${value >= 0 ? 'positive-contribution' : 'negative-contribution'}" style="width:${Math.max(3, Math.abs(value) / maximum * 50)}%"></i></div><strong class="contribution-value ${value >= 0 ? 'positive-text' : 'negative-text'}">${value >= 0 ? '+' : ''}${fmt(value)}</strong></div>`).join('');
}

function renderThreshold() {
  const metrics = thresholdMetrics(state.threshold);
  $('thresholdValue').textContent = state.threshold.toFixed(2);
  $('thresholdSlider').value = String(state.threshold);
  document.querySelectorAll('.reference-button').forEach(button => button.classList.toggle('active', Number(button.dataset.threshold) === Number(state.threshold)));
  const metricCards = [['accuracy', 'Accuracy', pct(metrics.accuracy)], ['precision', 'Precision', pct(metrics.precision)], ['recall', 'Recall', pct(metrics.recall)], ['f1', 'F1 score', pct(metrics.f1)], ['specificity', 'Specificity', pct(metrics.specificity)], ['balanced_accuracy', 'Balanced accuracy', pct(metrics.balanced_accuracy)], ['tp', 'True positives', metrics.tp], ['tn', 'True negatives', metrics.tn], ['fp', 'False positives', metrics.fp], ['fn', 'False negatives', metrics.fn], ['predicted_positive', 'Predicted positive', metrics.predicted_positive], ['predicted_negative', 'Predicted negative', metrics.predicted_negative]];
  $('thresholdMetrics').innerHTML = metricCards.map(([key, label, value]) => `<button class="metric-card ${state.selectedMetric === key ? 'selected-metric' : ''}" data-metric="${key}"><span>${label}</span><b>${value}</b><small>click to explain</small></button>`).join('');
  $('matrixTn').textContent = metrics.tn;
  $('matrixFp').textContent = metrics.fp;
  $('matrixFn').textContent = metrics.fn;
  $('matrixTp').textContent = metrics.tp;
  renderProbabilityChart();
  renderMetricExplanation();
  renderPrediction();
}

function renderProbabilityChart() {
  if (typeof Chart === 'undefined') return;
  destroyChart('probability');
  const noEvent = [];
  const event = [];
  state.artifact.test_predictions.forEach((record, index) => {
    const point = {x: record.probability, y: record.actual ? .72 + ((index * 17) % 20) / 100 : .28 + ((index * 13) % 20) / 100};
    (record.actual ? event : noEvent).push(point);
  });
  state.charts.probability = new Chart($('probabilityChart'), {type: 'scatter', data: {datasets: [{label: 'Actual event (1)', data: event, backgroundColor: '#ee6b75', pointRadius: 5}, {label: 'No event (0)', data: noEvent, backgroundColor: '#3867ee', pointRadius: 5}]}, plugins: [thresholdLinesPlugin], options: {responsive: true, maintainAspectRatio: false, plugins: {legend: {position: 'bottom', labels: {color: '#66758f', boxWidth: 9, font: {size: 10}}}, thresholdLines: {selected: state.threshold}, tooltip: {callbacks: {label: context => ` probability ${Number(context.parsed.x).toFixed(3)}`}}}, scales: {x: {min: 0, max: 1, title: {display: true, text: 'positive-class probability', color: '#8794a9', font: {size: 10}}, grid: {color: '#edf0f5'}, ticks: {color: '#8794a9', stepSize: .1}}, y: {min: 0, max: 1, display: false, grid: {display: false}}}}});
}

const metricExplanations = {
  accuracy: ['Accuracy', 'The share of all held-out records classified correctly. It can look strong even when a model misses many minority-class events.', 'Accuracy = (TP + TN) / all records'],
  precision: ['Precision', 'Among records the model flagged positive, the share that were actually positive. Higher precision means fewer false alarms.', 'Precision = TP / (TP + FP)'],
  recall: ['Recall / sensitivity', 'Among the real positive events, the share the model detected. Recall falls when false negatives increase.', 'Recall = TP / (TP + FN)'],
  f1: ['F1 score', 'A single score balancing precision and recall. It is useful when both missed events and false alarms matter.', 'F1 = 2 × precision × recall / (precision + recall)'],
  balanced_accuracy: ['Balanced accuracy', 'The average of recall and specificity, giving equal weight to the positive and negative classes.', 'Balanced accuracy = (recall + specificity) / 2'],
  specificity: ['Specificity', 'Among the records with no recorded event, the share the model correctly kept negative.', 'Specificity = TN / (TN + FP)'],
  roc_auc: ['ROC-AUC', 'How well the model ranks positive records above negative records across all possible thresholds. It is not a per-person certainty.', 'Area under the receiver operating characteristic curve'],
  pr_auc: ['PR-AUC', 'Ranking quality summarized through the precision–recall curve. It is particularly useful when the positive class is less common.', 'Area under the precision–recall curve'],
  tp: ['True positive', 'A positive event was present and the model flagged this record positive.', 'Actual 1 · predicted 1'],
  tn: ['True negative', 'No event was recorded and the model correctly kept this record negative.', 'Actual 0 · predicted 0'],
  fp: ['False positive', 'The model flagged the positive class, but the actual held-out label was negative. This is a false alarm.', 'Actual 0 · predicted 1'],
  fn: ['False negative', 'The model predicted the negative class while the actual class was positive. Recall depends on reducing these misses.', 'Actual 1 · predicted 0'],
  predicted_positive: ['Predicted positive', 'How many held-out records are currently above the selected probability threshold.', 'TP + FP'],
  predicted_negative: ['Predicted negative', 'How many held-out records are currently below the selected probability threshold.', 'TN + FN']
};

function renderMetricExplanation() {
  const explanation = metricExplanations[state.selectedMetric] || metricExplanations.recall;
  $('metricExplanationTitle').textContent = explanation[0];
  $('metricExplanationText').textContent = explanation[1];
  $('metricExplanationFormula').textContent = explanation[2];
  if (state.selectedCell) $('matrixExplanation').textContent = explanation[1];
}

function renderModels() {
  const names = state.artifact.models.map(model => model.name);
  $('modelComparisonNote').textContent = `These cards use the actual saved evaluation results. The Gradient Boosting card uses the project’s recall-focused ${state.artifact.thresholds.analysis.toFixed(2)} operating point; the baseline cards show their saved ${state.artifact.thresholds.default.toFixed(2)} results.`;
  $('modelCards').innerHTML = names.map(name => {
    const model = modelResult(name);
    const metrics = name === state.artifact.selected_model ? model.metrics_at_035 : model.metrics_at_050;
    const active = name === state.artifact.selected_model;
    return `<button class="model-card ${active ? 'selected-model' : ''}" data-model="${esc(name)}"><div class="model-card-top"><span>${active ? 'SELECTED MODEL' : 'COMPARISON'}</span><i>${active ? '✓' : '→'}</i></div><h3>${esc(name)}</h3><div class="model-card-metrics"><div><b>${pct(metrics.recall)}</b><span>Recall</span></div><div><b>${pct(metrics.pr_auc)}</b><span>PR-AUC</span></div><div><b>${pct(metrics.accuracy)}</b><span>Accuracy</span></div></div><small>${active ? `Gradient Boosting at the project’s ${state.artifact.thresholds.analysis.toFixed(2)} analysis threshold.` : `Saved comparison result at threshold ${state.artifact.thresholds.default.toFixed(2)}.`}</small></button>`;
  }).join('');
  $('comparisonTable').querySelector('tbody').innerHTML = state.artifact.models.map(model => {
    const metrics = model.name === state.artifact.selected_model ? model.metrics_at_035 : model.metrics_at_050;
    return `<tr class="${model.name === state.artifact.selected_model ? 'selected-row' : ''}"><td><b>${esc(model.name)}</b></td><td>${pct(metrics.accuracy)}</td><td>${pct(metrics.precision)}</td><td>${pct(metrics.recall)}</td><td>${pct(metrics.f1)}</td><td>${pct(metrics.balanced_accuracy)}</td><td>${fmt(metrics.roc_auc)}</td><td>${fmt(metrics.pr_auc)}</td></tr>`;
  }).join('');
  renderValidation();
}

function renderValidation() {
  const {dataset, validation} = state.artifact;
  $('validationFacts').innerHTML = `<div><span>Split</span><b>80% train / 20% test</b></div><div><span>Training rows</span><b>${dataset.train_rows}</b></div><div><span>Held-out rows</span><b>${dataset.test_rows}</b></div><div><span>Cross-validation</span><b>${validation.folds}-fold stratified</b></div><div><span>Tuning score</span><b>${validation.scoring}</b></div><div><span>Test set during tuning</span><b>${validation.test_set_used_during_tuning ? 'Yes' : 'No'}</b></div>`;
  if (typeof Chart === 'undefined') return;
  destroyChart('cv');
  state.charts.cv = new Chart($('cvChart'), {type: 'bar', data: {labels: validation.top_cv_results.map(result => `#${result.rank}`), datasets: [{label: 'mean CV recall', data: validation.top_cv_results.map(result => result.mean_recall), backgroundColor: '#12b7a5', borderRadius: 5, barThickness: 13}]}, options: {responsive: true, maintainAspectRatio: false, plugins: {legend: {display: false}, tooltip: {callbacks: {label: context => ` mean recall ${pct(context.raw)}`}}}, scales: {x: {title: {display: true, text: 'ranked tuning candidate', color: '#8794a9', font: {size: 9}}, grid: {display: false}, ticks: {color: '#8794a9'}}, y: {min: 0, max: 1, title: {display: true, text: 'mean recall', color: '#8794a9', font: {size: 9}}, grid: {color: '#edf0f5'}, ticks: {color: '#8794a9', callback: value => `${Math.round(value * 100)}%`}}}}});
}

function setMode(mode) {
  state.mode = mode;
  document.body.dataset.mode = mode;
  document.querySelectorAll('.mode-button').forEach(button => button.classList.toggle('active', button.dataset.mode === mode));
}

const tourSteps = [
  {anchor: '#dataset', title: '1 / 12 · Start with the actual dataset', text: 'The lab begins with the bundled UCI Heart Failure Clinical Records CSV. It contains 299 records, 12 predictors, and the binary target DEATH_EVENT.', label: 'ACTUAL DATA'},
  {anchor: '#dataset', title: '2 / 12 · Identify the target', text: 'The target is DEATH_EVENT: 1 means a recorded death event during follow-up and 0 means no recorded event in this dataset.', label: 'ACTUAL TARGET'},
  {anchor: '#pipeline', stage: 'validation', title: '3 / 12 · Validate before modeling', text: 'The existing Python run checks missing values and duplicates. Here, both are zero for the supplied data.', label: 'ACTUAL CHECK'},
  {anchor: '#pipeline', stage: 'preprocessing', title: '4 / 12 · Prepare without leakage', text: 'The project uses an 80/20 stratified split with random_state=42. The 60-row test set stays held out until evaluation.', label: 'ACTUAL SPLIT'},
  {anchor: '#dataset', title: '5 / 12 · Prepare the features', text: 'All 12 predictor columns are passed to tree models as raw numeric values. The binary indicators are not one-hot encoded for Gradient Boosting.', label: 'ACTUAL FEATURES'},
  {anchor: '#model', title: '6 / 12 · Combine shallow trees', text: 'The tuned Gradient Boosting estimator combines 100 shallow trees. Each stage adds a small correction to the ensemble.', label: 'ACTUAL MODEL'},
  {anchor: '#prediction', title: '7 / 12 · Produce a probability', text: 'The fitted estimator produces a positive-class probability with predict_proba. This is a ranking output before any class cutoff is applied.', label: 'ACTUAL OUTPUT'},
  {anchor: '#threshold', title: '8 / 12 · Choose a threshold', text: 'A threshold turns the probability into a class. The project’s analysis point is 0.35, with 0.50 shown as a reference.', label: 'ACTUAL THRESHOLD'},
  {anchor: '#prediction', title: '9 / 12 · Get a prediction', text: 'Select a held-out record to see its actual feature values, probability, thresholded prediction, and actual label.', label: 'MODEL RESULT'},
  {anchor: '#threshold', title: '10 / 12 · Read the confusion matrix', text: 'TP, TN, FP, and FN count the four outcomes at the selected threshold. False negatives are the missed positive events that reduce recall.', label: 'ACTUAL EVALUATION'},
  {anchor: '#evaluation', title: '11 / 12 · Compare model performance', text: 'The comparison cards use the saved results from the existing pipeline. No metric is treated as sufficient on its own.', label: 'ACTUAL METRICS'},
  {anchor: '#evaluation', title: '12 / 12 · Interpret with limitations', text: 'The dataset is small, the test set is one split, follow-up time may limit early-warning interpretation, and external validation is still needed.', label: 'LIMITATIONS'}
];

function renderTour() {
  const step = tourSteps[state.tourStep];
  $('tourProgressBar').style.width = `${((state.tourStep + 1) / tourSteps.length) * 100}%`;
  $('tourContent').innerHTML = `<span class="source-label ${step.label.includes('ACTUAL') ? 'actual-label' : 'conceptual-label'}">${step.label}</span><h2 id="tourTitle">${step.title}</h2><p>${step.text}</p><small>Use Next to continue through the existing model pipeline.</small>`;
  $('tourBack').disabled = state.tourStep === 0;
  $('tourNext').textContent = state.tourStep === tourSteps.length - 1 ? 'Finish' : 'Next →';
  if (step.stage) renderPipeline(step.stage);
  document.querySelector(step.anchor)?.scrollIntoView({behavior: 'smooth', block: 'start'});
}

function openTour() {
  state.tourStep = 0;
  $('tourModal').classList.remove('hidden');
  renderTour();
}

function closeTour() {
  $('tourModal').classList.add('hidden');
}

function loadPreview(file) {
  const reader = new FileReader();
  reader.onload = () => {
    try {
      const rows = parseCsv(reader.result);
      const expected = state.artifact.model.feature_names.concat(state.artifact.dataset.target);
      const missing = expected.filter(name => !Object.prototype.hasOwnProperty.call(rows[0], name));
      if (missing.length) throw new Error(`Missing expected columns: ${missing.join(', ')}`);
      state.rows = rows;
      $('artifactStatus').textContent = 'CSV preview loaded — saved model artifact unchanged';
      $('artifactMeta').textContent = 'Dataset explorer is showing this uploaded file. Prediction and evaluation sections remain tied to the existing Python run.';
      renderDatasetStats();
      document.querySelector('#dataset').scrollIntoView({behavior: 'smooth'});
    } catch (error) {
      window.alert(`Could not load CSV preview: ${error.message}`);
    }
  };
  reader.readAsText(file);
}

function bindEvents() {
  document.querySelectorAll('.mode-button').forEach(button => button.addEventListener('click', () => setMode(button.dataset.mode)));
  $('tourButton').addEventListener('click', openTour);
  $('closeTour').addEventListener('click', closeTour);
  document.querySelector('[data-close-tour]').addEventListener('click', closeTour);
  $('tourRestart').addEventListener('click', () => { state.tourStep = 0; renderTour(); });
  $('tourBack').addEventListener('click', () => { state.tourStep = Math.max(0, state.tourStep - 1); renderTour(); });
  $('tourNext').addEventListener('click', () => { if (state.tourStep === tourSteps.length - 1) closeTour(); else { state.tourStep += 1; renderTour(); } });
  document.addEventListener('keydown', event => { if (event.key === 'Escape') closeTour(); });
  document.querySelectorAll('.pipeline-stage').forEach(button => button.addEventListener('click', () => renderPipeline(button.dataset.stage)));
  $('featureSelect').addEventListener('change', renderFeatureExplorer);
  $('treeSlider').addEventListener('input', event => renderTree(Number(event.target.value)));
  $('recordSelect').addEventListener('change', event => { state.selectedRecord = Number(event.target.value); renderPrediction(); renderPipeline(state.currentStage); });
  $('thresholdSlider').addEventListener('input', event => { state.threshold = Number(event.target.value); renderThreshold(); renderPipeline(state.currentStage); });
  document.querySelectorAll('.reference-button').forEach(button => button.addEventListener('click', () => { state.threshold = Number(button.dataset.threshold); renderThreshold(); renderPipeline(state.currentStage); }));
  document.querySelectorAll('.matrix button').forEach(button => button.addEventListener('click', () => { state.selectedCell = button.dataset.cell; state.selectedMetric = button.dataset.cell; renderMetricExplanation(); document.querySelectorAll('.matrix button').forEach(item => item.classList.toggle('active-cell', item === button)); }));
  document.addEventListener('click', event => {
    const metric = event.target.closest('[data-metric]');
    if (metric) { state.selectedMetric = metric.dataset.metric; state.selectedCell = null; renderThreshold(); }
    const model = event.target.closest('[data-model]');
    if (model) { document.querySelectorAll('[data-model]').forEach(item => item.classList.toggle('selected-model', item.dataset.model === model.dataset.model)); }
  });
  $('csvFile').addEventListener('change', event => { if (event.target.files[0]) loadPreview(event.target.files[0]); });
}

async function boot() {
  try {
    const [artifactResponse, csvResponse] = await Promise.all([fetch('data/dashboard_data.json'), fetch('data/heart_failure_clinical_records_dataset.csv')]);
    if (!artifactResponse.ok) throw new Error('The saved dashboard artifact is unavailable. Run python src/run_analysis.py first.');
    state.artifact = await artifactResponse.json();
    state.sourceRows = csvResponse.ok ? parseCsv(await csvResponse.text()) : [];
    state.rows = state.sourceRows;
    state.threshold = state.artifact.thresholds.analysis;
    renderStatus();
    configureThresholdControls();
    renderDatasetStats();
    renderModel();
    renderRecordSelect();
    renderThreshold();
    renderModels();
    renderPipeline('dataset');
    bindEvents();
  } catch (error) {
    $('artifactStatus').textContent = 'Saved analysis artifact unavailable';
    $('artifactMeta').textContent = error.message;
    $('pipelineDetail').innerHTML = `<div class="error-state"><b>Run the existing project pipeline to generate the explainability artifact.</b><code>python src/run_analysis.py</code></div>`;
  }
}

boot();
