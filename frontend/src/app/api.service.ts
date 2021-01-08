import { Injectable } from '@angular/core';
import { HttpClient, HttpErrorResponse, HttpParams } from '@angular/common/http';
import { retry, catchError } from 'rxjs/operators';
import { Observable, throwError } from 'rxjs';
import { Summary } from './model/summary';

@Injectable({
  providedIn: 'root'
})
export class ApiService {

  private url = "http://127.0.0.1:5000/summary"; //Local dev
  
  constructor(private httpClient: HttpClient) { }

  public getSummary(url: string): Observable<Summary> {
    let params = new HttpParams().append('url', url);
    return this.httpClient.get<Summary>(this.url, {params: params}).pipe(retry(1),catchError(this.handleError));
  }

  handleError(error: HttpErrorResponse) {
    let errorMessage = '';
    if (error.error instanceof ErrorEvent) {
      errorMessage = `${error.error.message}`;
    } else {
      errorMessage = `Error Code: ${error.status}\n ${error.message}`;
    }
    return throwError(errorMessage);
  }
}
