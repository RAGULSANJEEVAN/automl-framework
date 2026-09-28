import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';

/**
 * Thin wrapper around the AutoML backend.
 * Change BASE_URL here if your API moves off localhost:8000.
 */
@Injectable({ providedIn: 'root' })
export class AutomlApiService {
  private readonly baseUrl = 'http://127.0.0.1:8000/api';

  constructor(private http: HttpClient) {}

  /** multipart/form-data upload. Assumes the backend expects the field name "file" — change if yours differs. */
  upload(file: File): Observable<any> {
    const form = new FormData();
    form.append('file', file);
    return this.http.post(`${this.baseUrl}/upload`, form);
  }

  setTarget(datasetId: string, target: string): Observable<any> {

    target == "None" ? target = "" : target = target;

    return this.http.post(`${this.baseUrl}/target`, { dataset_id: datasetId, target });
  }

  analyze(datasetId: string): Observable<any> {
    return this.http.post(`${this.baseUrl}/analyze`, { dataset_id: datasetId });
  }

  recommend(datasetId: string): Observable<any> {
    return this.http.post(`${this.baseUrl}/recommend`, { dataset_id: datasetId });
  }

  preprocess(datasetId: string): Observable<any> {
    return this.http.post(`${this.baseUrl}/preprocess`, { dataset_id: datasetId });
  }

  train(datasetId: string): Observable<any> {
    return this.http.post(`${this.baseUrl}/train`, { dataset_id: datasetId });
  }

  // getDataset(datasetId: string): Observable<any> {
  //   return this.http.get(`${this.baseUrl}/dataset/${datasetId}`);
  // }

  // generateReport(datasetId: string): Observable<any> {
  //   return this.http.post(`${this.baseUrl}/report/generate`, { dataset_id: datasetId });
  // }

  getDataset(datasetId: string): Observable<any> {
    return this.http.get<any>(`${this.baseUrl}/dataset/${datasetId}`);
  }

  generateReport(dataset: any): Observable<any> {
    return this.http.post<any>(
      `${this.baseUrl}/report/generate`,
      dataset
    );
  }

  getReportStatus(reportId: string): Observable<any> {
    return this.http.get<any>(
        `${this.baseUrl}/report/status/${reportId}`
    );
  }

  getReport(reportId: string): Observable<any> {
    return this.http.get<any>(
        `${this.baseUrl}/report/${reportId}`
    );
  }

  getDetailedReport(reportId: string): Observable<any> {
    return this.http.get<any>(
        `${this.baseUrl}/report/${reportId}/detailed`
    );
  }
}
