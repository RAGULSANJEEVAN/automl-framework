import { CommonModule } from '@angular/common';
import { Component } from '@angular/core';
import { FormsModule } from '@angular/forms';
import {
  Observable,
  switchMap,
  filter,
  take,
  timer,
  firstValueFrom,
  mergeMap,
  throwError,
  tap
} from 'rxjs';

import { AutomlApiService } from './services/automl-api.service';
import {
  ColumnDef,
  MetricRow,
  ModelComparisonRow,
  PredictionSample,
  extractBestModel,
  extractColumns,
  extractCrossValidation,
  extractDatasetId,
  extractDatasetSummary,
  extractImportantFeatures,
  extractMetrics,
  extractModelComparison,
  extractPredictionSample,
  extractReportText,
  pretty,
} from './utils/normalize';

type StepKey = 'analyze' | 'recommend' | 'preprocess' | 'train' | 'dataset' | 'report';
type StepStatus = 'pending' | 'running' | 'done' | 'error';

interface PipelineStep {
  key: StepKey;
  label: string;
  status: StepStatus;
}

const STEP_DEFS: { key: StepKey; label: string }[] = [
  { key: 'analyze', label: 'Analyzing dataset' },
  { key: 'recommend', label: 'Recommending models' },
  { key: 'preprocess', label: 'Preprocessing data' },
  { key: 'train', label: 'Training models' },
  { key: 'dataset', label: 'Compiling results' },
  { key: 'report', label: 'Generating AI report' },
];

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './app.component.html',
})
export class AppComponent {
  constructor(private api: AutomlApiService) { }

  // ---- upload / dataset state ----
  fileLoaded = false;
  fileName = '';
  fileMeta = '';
  dragActive = false;
  uploading = false;
  uploadError: string | null = null;

  reportId: string | null = null;
  reportResponse: any = null;

  datasetId: string | null = null;
  columns: ColumnDef[] = [];

  // ---- target state ----
  targetIndex: number | null = null;
  targetStatus: 'idle' | 'saving' | 'saved' | 'error' = 'idle';
  targetError: string | null = null;

  // ---- pipeline run state ----
  running = false;
  showProgress = false;
  progress = 0;
  steps: PipelineStep[] = STEP_DEFS.map((s) => ({ ...s, status: 'pending' as StepStatus }));
  pipelineError: string | null = null;

  // ---- results state (matches /api/dataset/{id} shape) ----
  resultsReady = false;
  datasetSummary = '';
  bestModel: string | null = null;
  metrics: MetricRow[] = [];
  modelComparison: ModelComparisonRow[] = [];
  importantFeatureNames: string[] = [];
  importantFeatureCount = 0;
  crossValidation: { scores: string[]; mean: string } | null = null;
  predictions: PredictionSample[] = [];
  reportText = '';


  // ---- debug: raw responses, keyed by step, viewable via a toggle in the UI ----
  rawResponses: Record<string, any> = {};
  showRaw: Record<string, boolean> = {};

  reportStatus: any = null;

  // ================= derived state =================

  get isReady(): boolean {
    return !!this.datasetId && this.targetStatus === 'saved';
  }

  get runHintText(): string {
    if (!this.fileLoaded) return 'Upload a dataset to continue.';
    if (this.targetStatus === 'saving') return 'Saving target column…';
    if (this.targetStatus !== 'saved') return 'Pick a target column to continue.';
    return `Ready to run — predicting "${this.columns[this.targetIndex as number]?.name}"`;
  }

  get stage1Class(): string {
    return this.fileLoaded ? 'done' : 'active';
  }

  get stage2Class(): string {
    return this.fileLoaded ? 'done' : '';
  }

  get stage3Class(): string {
    if (this.resultsReady) return 'done';
    if (this.isReady) return 'active';
    return '';
  }

  get stage4Class(): string {

    if (this.resultsReady) return 'done';

    return '';
  }

  toggleRaw(key: string): void {
    this.showRaw[key] = !this.showRaw[key];
  }

  rawJson(key: string): string {
    return pretty(this.rawResponses[key]);
  }

  // ================= upload =================

  onDragOver(e: DragEvent): void {
    e.preventDefault();
    this.dragActive = true;
  }

  onDragLeave(e: DragEvent): void {
    e.preventDefault();
    this.dragActive = false;
  }

  onDrop(e: DragEvent): void {
    e.preventDefault();
    this.dragActive = false;
    const file = e.dataTransfer?.files?.[0];
    if (file) this.handleFile(file);
  }

  onFileSelected(e: Event): void {
    const input = e.target as HTMLInputElement;
    const file = input.files?.[0];
    if (file) this.handleFile(file);
  }

  private async handleFile(file: File): Promise<void> {
    this.uploadError = null;
    this.uploading = true;
    try {
      const res = await firstValueFrom(this.api.upload(file));
      this.rawResponses['upload'] = res;

      const datasetId = extractDatasetId(res);
      const columns = extractColumns(res);

      if (!datasetId) {
        this.uploadError = 'Upload succeeded but no dataset_id was found in the response — check "raw response" below.';
      }
      if (!columns.length) {
        this.uploadError = (this.uploadError ? this.uploadError + ' ' : '') +
          'Could not detect columns in the response — check "raw response" below.';
      }

      this.datasetId = datasetId ?? null;
      this.columns = columns;
      this.fileName = file.name;
      this.fileMeta = `${(file.size / 1024).toFixed(1)} KB${this.datasetId ? ' · dataset ' + this.datasetId.slice(0, 8) + '…' : ''}`;
      this.fileLoaded = true;
      this.targetIndex = null;
      this.targetStatus = 'idle';
      this.resetPipeline();
    } catch (err) {
      this.uploadError = this.formatError(err);
    } finally {
      this.uploading = false;
    }
  }

  // ================= target =================

  async selectTarget(idx: number): Promise<void> {
    if (!this.datasetId) return;
    this.targetIndex = idx;
    this.targetStatus = 'saving';
    this.targetError = null;
    try {
      const res = await firstValueFrom(this.api.setTarget(this.datasetId, this.columns[idx].name));
      this.rawResponses['target'] = res;
      this.targetStatus = 'saved';
      this.resetPipeline();
    } catch (err) {
      this.targetStatus = 'error';
      this.targetError = this.formatError(err);
    }
  }

  // ================= pipeline run =================

  private resetPipeline(): void {
    this.showProgress = false;
    this.resultsReady = false;
    this.progress = 0;
    this.pipelineError = null;
    this.steps = STEP_DEFS.map((s) => ({ ...s, status: 'pending' as StepStatus }));
    this.datasetSummary = '';
    this.bestModel = null;
    this.metrics = [];
    this.modelComparison = [];
    this.importantFeatureNames = [];
    this.importantFeatureCount = 0;
    this.crossValidation = null;
    this.predictions = [];
    this.reportText = '';
  }

  // async runPipeline(): Promise<void> {
  //   if (!this.datasetId || this.targetStatus !== 'saved') return;
  //   this.resetPipeline();
  //   this.running = true;
  //   this.showProgress = true;

  //   try {
  //     await this.runStep('analyze', () => this.api.analyze(this.datasetId as string));
  //     await this.runStep('recommend', () => this.api.recommend(this.datasetId as string));
  //     await this.runStep('preprocess', () => this.api.preprocess(this.datasetId as string));
  //     await this.runStep('train', () => this.api.train(this.datasetId as string));
  //     const combined = await this.runStep('dataset', () => this.api.getDataset(this.datasetId as string));
  //     this.applyResults(combined);
  //     const report = await this.runStep('report', () => this.api.generateReport(this.datasetId as string));
  //     this.reportText = extractReportText(report);
  //     this.resultsReady = true;
  //   } catch (err) {
  //     this.pipelineError = this.formatError(err);
  //   } finally {
  //     this.running = false;
  //   }
  // }

  async runPipeline(): Promise<void> {

    if (!this.datasetId || this.targetStatus !== 'saved') return;

    this.resetPipeline();
    this.running = true;
    this.showProgress = true;

    try {

      await this.runStep('analyze', () =>
        this.api.analyze(this.datasetId!)
      );

      await this.runStep('recommend', () =>
        this.api.recommend(this.datasetId!)
      );

      await this.runStep('preprocess', () =>
        this.api.preprocess(this.datasetId!)
      );

      await this.runStep('train', () =>
        this.api.train(this.datasetId!)
      );

      // Get Dataset
      const dataset = await this.runStep(
        'dataset',
        () => this.api.getDataset(this.datasetId!)
      );

      this.applyResults(dataset);

      // Generate Report using Dataset Response
      const reportGeneration = await firstValueFrom(

        this.api.generateReport(dataset).pipe(

          switchMap((response: any) => {

            this.reportId = response.report_id;

            return timer(0, 3000).pipe(

              switchMap(() =>
                this.api.getReportStatus(this.reportId!)
              ),

              tap((status: any) => {

                // Store latest status for UI
                this.reportStatus = status;

              }),

              mergeMap((status: any) => {

                if (status.status === 'failed') {
                  return throwError(() => new Error(
                    status.error || 'Report generation failed.'
                  ));
                }

                return [status];

              }),

              filter((status: any) => status.status === 'completed'),

              take(1),

              switchMap(() =>
                this.api.getReport(this.reportId!)
              )

            );

          })

        )

      );

      this.rawResponses['report'] = reportGeneration;
      this.reportResponse = reportGeneration;

      // this.reportResponse = reportGeneration;

      const reportStep = this.steps.find(s => s.key === 'report');
      if (reportStep) {
        reportStep.status = 'done';
      }

      this.progress = 100;
      this.resultsReady = true;

    }
    catch (err) {

      const reportStep = this.steps.find(s => s.key === 'report');

      if (reportStep) {
        reportStep.status = 'error';
      }

      this.pipelineError = this.formatError(err);

    }
    finally {

      this.running = false;

    }

  }

  get reportSections(): any[] {
    return this.reportResponse
      ? Object.values(this.reportResponse) as any[]
      : [];
  }

  private async runStep(key: StepKey, call: () => import('rxjs').Observable<any>): Promise<any> {
    const step = this.steps.find((s) => s.key === key)!;
    step.status = 'running';
    try {
      const res = await firstValueFrom(call());
      this.rawResponses[key] = res;
      step.status = 'done';
      const doneCount = this.steps.filter((s) => s.status === 'done').length;
      this.progress = (doneCount / this.steps.length) * 100;
      return res;
    } catch (err) {
      step.status = 'error';
      throw err;
    }
  }

  private applyResults(combined: any): void {
    this.datasetSummary = extractDatasetSummary(combined);
    this.bestModel = extractBestModel(combined) ?? null;
    this.metrics = extractMetrics(combined);
    this.modelComparison = extractModelComparison(combined);

    const feat = extractImportantFeatures(combined);
    this.importantFeatureNames = feat.names;
    this.importantFeatureCount = feat.count;

    this.crossValidation = extractCrossValidation(combined);
    this.predictions = extractPredictionSample(combined, 10);
  }

  private formatError(err: any): string {
    if (err?.error?.detail) return typeof err.error.detail === 'string' ? err.error.detail : pretty(err.error.detail);
    if (err?.error?.message) return err.error.message;
    if (err?.message) return err.message;
    if (err?.status === 0) return 'Could not reach the backend at 127.0.0.1:8000 — is it running, and does it allow CORS from this origin?';
    return 'Something went wrong. Check the browser console for details.';
  }
}
