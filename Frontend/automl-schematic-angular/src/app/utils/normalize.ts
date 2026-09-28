// ============================================================================
// These extractors are written against the exact response shape returned by
// your backend's GET /api/dataset/{id}, i.e.:
//   { success, data: { df, target, analysis, recommendation,
//                       preprocessed_data, training_result, preprocessed_info } }
// If your backend's field names change, this is the only file to edit.
// ============================================================================

export type ColType = 'numeric' | 'categorical' | 'boolean' | 'datetime';

export interface ColumnDef {
  name: string;
  type: ColType;
  selected: boolean;
}

export interface MetricRow {
  label: string;
  val: string;
}

export interface ModelComparisonRow {
  model: string;
  r2: string;
  rmse: string;
  mae: string;
  mse: string;
}

export interface PredictionSample {
  actual: string;
  predicted: string;
  error: string;
}

/** Every endpoint on this backend wraps its payload as { success, data }. */
function unwrap(res: any): any {
  return res && typeof res === 'object' && 'data' in res ? res.data : res;
}

function fmtNum(v: any, decimals = 3): string {
  if (typeof v !== 'number' || isNaN(v)) return v === null || v === undefined ? '—' : String(v);
  if (Math.abs(v) >= 1000) return v.toLocaleString(undefined, { maximumFractionDigits: 0 });
  return v.toFixed(decimals);
}

export function extractDatasetId(res: any): string | undefined {
  const data = unwrap(res);
  return data?.dataset_id ?? data?.datasetId ?? data?.id;
}

/** Columns come from df.column_names; numeric/categorical split comes from analysis.feature_info when available. */
// export function extractColumns(res: any): ColumnDef[] {
//   const data = unwrap(res);
//   const df = data?.df;
//   const rawNames = df?.column_names ?? data?.column_names ?? data?.fields;
//   const names: string[] = Array.isArray(rawNames) ? rawNames : [];
//   const preview: any[] = df?.preview ?? data?.preview ?? [];
//   const numerical: string[] = data?.analysis?.feature_info?.numerical_features ?? [];
//   const categorical: string[] = data?.analysis?.feature_info?.categorical_features ?? [];

//   return names.map((name) => {
//     let type: ColType;
//     if (numerical.includes(name)) type = 'numeric';
//     else if (categorical.includes(name)) type = 'categorical';
//     else type = inferTypeFromPreview(preview, name);
//     return { name, type, selected: true };
//   });
// }
export function extractColumns(res: any): ColumnDef[] {

  const data = unwrap(res);

  const details = data?.column_details ?? [];

  const columns: ColumnDef[] = details.map((col: any) => {

    const dtype = (col.type || '').toLowerCase();

    let type: ColType = 'categorical';

    if (
      dtype.includes('int') ||
      dtype.includes('float') ||
      dtype.includes('double') ||
      dtype.includes('decimal')
    ) {
      type = 'numeric';
    }

    return {
      name: col.name,
      type,
      selected: true
    };
  });

  // Add None option for Unsupervised Learning
  columns.unshift({
    name: 'None',
    type: 'categorical',
    selected: true
  });

  return columns;
}
function inferTypeFromPreview(preview: any[], colName: string): ColType {
  if (!Array.isArray(preview) || !preview.length) return 'categorical';
  const sample = preview.map((row) => row?.[colName]).filter((v) => v !== undefined && v !== null && v !== '');
  if (!sample.length) return 'categorical';
  const numericCount = sample.filter((v) => v !== '' && !isNaN(Number(v))).length;
  return numericCount / sample.length > 0.8 ? 'numeric' : 'categorical';
}

/** e.g. "4,746 rows · 12 columns · supervised learning · target: Rent (int64)" */
export function extractDatasetSummary(res: any): string {
  const data = unwrap(res);
  const rows = data?.df?.rows ?? data?.analysis?.dataset_info?.num_rows;
  const cols = data?.df?.columns ?? data?.analysis?.dataset_info?.num_columns;
  const problemType = data?.analysis?.problem_info?.problem_type;
  const targetType = data?.analysis?.problem_info?.target_type;
  const target = data?.target ?? data?.analysis?.problem_info?.target_column;

  const parts: string[] = [];
  if (rows != null && cols != null) parts.push(`${Number(rows).toLocaleString()} rows · ${cols} columns`);
  if (problemType) parts.push(String(problemType).replace(/_/g, ' '));
  if (target) parts.push(`target: ${target}${targetType ? ' (' + targetType + ')' : ''}`);
  return parts.join(' · ');
}

export function extractBestModel(res: any): string | undefined {
  const data = unwrap(res);
  return (
    data?.training_result?.result_evaluation?.best_model_name ??
    data?.training_result?.hyper_parameter_info?.hyper_parameter_evaluation?.best_model ??
    data?.analysis?.baseline_model?.recommended_model
  );
}

/** Prefers the post-tuning metrics (hyper_parameter_evaluation); falls back to the top row of final_evaluation_matrics. */
export function extractMetrics(res: any): MetricRow[] {
  const data = unwrap(res);
  const tuned = data?.training_result?.hyper_parameter_info?.hyper_parameter_evaluation;

  if (tuned) {
    const rows: MetricRow[] = [];
    if (tuned['R² Score'] != null) rows.push({ label: 'R² Score', val: fmtNum(tuned['R² Score']) });
    if (tuned.RMSE != null) rows.push({ label: 'RMSE', val: fmtNum(tuned.RMSE) });
    if (tuned.MAE != null) rows.push({ label: 'MAE', val: fmtNum(tuned.MAE) });
    if (tuned.MSE != null) rows.push({ label: 'MSE', val: fmtNum(tuned.MSE) });
    if (tuned.score != null) rows.push({ label: 'Tuned CV Score', val: fmtNum(tuned.score) });
    return rows;
  }

  const best = data?.training_result?.result_evaluation?.final_evaluation_matrics?.[0];
  if (best) {
    return [
      { label: 'R² Score', val: fmtNum(best['R² Score']) },
      { label: 'RMSE', val: fmtNum(best.RMSE) },
      { label: 'MAE', val: fmtNum(best.MAE) },
      { label: 'MSE', val: fmtNum(best.MSE) },
    ];
  }
  return [];
}

/** All candidate models side by side, best first — from result_evaluation.final_evaluation_matrics. */
export function extractModelComparison(res: any): ModelComparisonRow[] {
  const data = unwrap(res);
  const rows = data?.training_result?.result_evaluation?.final_evaluation_matrics ??
    data?.training_result?.evaluating_base_model ?? [];
  return rows.map((r: any) => ({
    model: r.model_name ?? r.model ?? '—',
    r2: fmtNum(r['R² Score']),
    rmse: fmtNum(r.RMSE),
    mae: fmtNum(r.MAE),
    mse: fmtNum(r.MSE),
  }));
}

/** Note: your backend's field is spelled "imporrtant_features" (typo preserved to match the API). */
export function extractImportantFeatures(res: any): { names: string[]; count: number } {
  const data = unwrap(res);
  const fs = data?.training_result?.feature_selection;
  const names: string[] = fs?.imporrtant_features ?? fs?.important_features ?? [];
  const count: number = fs?.important_features_count ?? names.length;
  return { names, count };
}

export function extractCrossValidation(res: any): { scores: string[]; mean: string } | null {
  const data = unwrap(res);
  const cv = data?.training_result?.cross_validation_score;
  if (!cv) return null;
  const scores = (cv.cv_scores ?? []).map((s: number) => fmtNum(s));
  const mean = fmtNum(cv['mean cv_scores'] ?? cv.mean_cv_scores);
  return { scores, mean };
}

/**
 * cluster_data is a large array of {Actual, Prediction, Error, "Absolute Error"} sorted worst-error-first.
 * We sample evenly across it so the table shows a representative spread rather than only the worst misses.
 */
export function extractPredictionSample(res: any, count = 10): PredictionSample[] {
  const data = unwrap(res);
  const cluster: any[] = data?.training_result?.hyper_parameter_info?.cluster_data ?? [];
  if (!cluster.length) return [];

  const step = Math.max(1, Math.floor(cluster.length / count));
  const sample: PredictionSample[] = [];
  for (let i = 0; i < cluster.length && sample.length < count; i += step) {
    const row = cluster[i];
    sample.push({
      actual: fmtNum(row.Actual, 0),
      predicted: fmtNum(row.Prediction, 0),
      error: fmtNum(row['Absolute Error'] ?? row.Error, 0),
    });
  }
  return sample;
}

export function extractReportText(res: any): string {
  const data = unwrap(res);
  const text = data?.report ?? data?.text ?? data?.summary ?? data?.content ?? data?.markdown;
  if (typeof text === 'string') return text;
  return JSON.stringify(data ?? res, null, 2);
}

export function pretty(res: any): string {
  try {
    return JSON.stringify(res, null, 2);
  } catch {
    return String(res);
  }
}
