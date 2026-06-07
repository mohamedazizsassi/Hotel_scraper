import { Injectable, inject } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Observable, map } from 'rxjs';
import { API_BASE } from '../config/api.config';
import {
  AnomalyDto, CalendarOptionsDto, CalendarQuery, CalendarRowDto,
  CompetitorDto, DataResponse, DateRangeQuery, RecommendationDto,
} from './dto';

function toParams<T extends object>(q: T): HttpParams {
  let p = new HttpParams();
  for (const [k, v] of Object.entries(q)) {
    // Send empty string ('' = the no-tier/no-view taxonomy value) but skip
    // null/undefined (= "do not constrain").
    if (v !== undefined && v !== null) p = p.set(k, String(v));
  }
  return p;
}

/** Typed access to the RevWay manager endpoints. Returns wire DTOs; mapping
 *  to domain models happens in adapters.ts at the call site. */
@Injectable({ providedIn: 'root' })
export class ApiService {
  private http = inject(HttpClient);

  getCalendarOptions(): Observable<CalendarOptionsDto> {
    return this.http.get<CalendarOptionsDto>(`${API_BASE}/manager/calendar/options`);
  }

  getCalendar(query: CalendarQuery): Observable<CalendarRowDto[]> {
    return this.http
      .get<DataResponse<CalendarRowDto>>(`${API_BASE}/manager/calendar`, { params: toParams(query) })
      .pipe(map(r => r.data));
  }

  getCompetitors(): Observable<CompetitorDto[]> {
    return this.http
      .get<DataResponse<CompetitorDto>>(`${API_BASE}/manager/competitors`)
      .pipe(map(r => r.data));
  }

  getRecommendations(query: DateRangeQuery): Observable<RecommendationDto[]> {
    return this.http
      .get<DataResponse<RecommendationDto>>(`${API_BASE}/manager/recommendations`, { params: toParams(query) })
      .pipe(map(r => r.data));
  }

  getAnomalies(query: DateRangeQuery): Observable<AnomalyDto[]> {
    return this.http
      .get<DataResponse<AnomalyDto>>(`${API_BASE}/manager/anomalies`, { params: toParams(query) })
      .pipe(map(r => r.data));
  }
}
