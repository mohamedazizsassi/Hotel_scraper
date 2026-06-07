import { Injectable, inject } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Observable, map } from 'rxjs';
import { API_BASE } from '../config/api.config';
import {
  AdminAlertDto, AdminCompetitorRowDto, AdminHotelDto, AdminManagerDto,
  AnomalyDto, AssignmentDto, CalendarOptionsDto, CalendarQuery, CalendarRowDto,
  CompetitorDto, DailyRollupDto, DataResponse, DateRangeQuery,
  DiscoverableHotelDto, HotelCreateBody, HotelUpdateBody, ManagerCreateBody,
  ManagerUpdateBody, MonitoringSummaryDto, RecommendationDto, ScrapeRunDto,
  SelectableHotelDto,
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

  // --- Admin: hotels ---
  getAdminHotels(): Observable<AdminHotelDto[]> {
    return this.http
      .get<DataResponse<AdminHotelDto>>(`${API_BASE}/admin/hotels`)
      .pipe(map(r => r.data));
  }

  getDiscoverableHotels(): Observable<DiscoverableHotelDto[]> {
    return this.http
      .get<DataResponse<DiscoverableHotelDto>>(`${API_BASE}/admin/hotels/discoverable`)
      .pipe(map(r => r.data));
  }

  // POST /admin/hotels returns the bare AdminHotelRow (201), NOT a DataResponse —
  // confirmed via `response_model=AdminHotelRow` in routers/admin/hotels.py.
  promoteHotel(body: HotelCreateBody): Observable<AdminHotelDto> {
    return this.http.post<AdminHotelDto>(`${API_BASE}/admin/hotels`, body);
  }

  updateHotel(id: number, body: HotelUpdateBody): Observable<AdminHotelDto> {
    return this.http.patch<AdminHotelDto>(`${API_BASE}/admin/hotels/${id}`, body);
  }

  // --- Admin: managers ---
  getAdminManagers(): Observable<AdminManagerDto[]> {
    return this.http
      .get<DataResponse<AdminManagerDto>>(`${API_BASE}/admin/managers`)
      .pipe(map(r => r.data));
  }

  // POST/PATCH return the bare AdminManagerRow (201/200) — response_model=AdminManagerRow.
  createManager(body: ManagerCreateBody): Observable<AdminManagerDto> {
    return this.http.post<AdminManagerDto>(`${API_BASE}/admin/managers`, body);
  }

  updateManager(id: string, body: ManagerUpdateBody): Observable<AdminManagerDto> {
    return this.http.patch<AdminManagerDto>(`${API_BASE}/admin/managers/${id}`, body);
  }

  // 204 No Content — admin SUPPLIES new_password (not server-generated).
  // Observable<void> is correct; Angular maps a 204 body to null.
  resetManagerPassword(id: string, new_password: string): Observable<void> {
    return this.http.post<void>(`${API_BASE}/admin/managers/${id}/reset-password`, { new_password });
  }

  // --- Admin: assignments ---
  getAssignments(): Observable<AssignmentDto[]> {
    return this.http
      .get<DataResponse<AssignmentDto>>(`${API_BASE}/admin/assignments`)
      .pipe(map(r => r.data));
  }

  // POST/PATCH return the bare AdminAssignmentRow — response_model=AdminAssignmentRow.
  createAssignment(body: { user_id: string; hotel_id: number; max_competitors?: number }): Observable<AssignmentDto> {
    return this.http.post<AssignmentDto>(`${API_BASE}/admin/assignments`, body);
  }

  // 204 No Content.
  deleteAssignment(id: number): Observable<void> {
    return this.http.delete<void>(`${API_BASE}/admin/assignments/${id}`);
  }

  // --- Admin: competitors (D11 — admin-only selection) ---
  getManagerCompetitors(managerId: string): Observable<AdminCompetitorRowDto[]> {
    return this.http
      .get<DataResponse<AdminCompetitorRowDto>>(`${API_BASE}/admin/managers/${managerId}/competitors`)
      .pipe(map(r => r.data));
  }

  getSelectableCompetitors(managerId: string): Observable<SelectableHotelDto[]> {
    return this.http
      .get<DataResponse<SelectableHotelDto>>(`${API_BASE}/admin/managers/${managerId}/selectable-competitors`)
      .pipe(map(r => r.data));
  }

  setManagerCompetitors(managerId: string, hotel_ids: number[]): Observable<AdminCompetitorRowDto[]> {
    return this.http
      .put<DataResponse<AdminCompetitorRowDto>>(`${API_BASE}/admin/managers/${managerId}/competitors`, { hotel_ids })
      .pipe(map(r => r.data));
  }

  // --- Admin: monitoring + alerts ---
  getMonitoringSummary(): Observable<MonitoringSummaryDto> {
    return this.http.get<MonitoringSummaryDto>(`${API_BASE}/admin/monitoring/summary`);
  }

  getMonitoringRuns(limit = 50): Observable<ScrapeRunDto[]> {
    return this.http
      .get<DataResponse<ScrapeRunDto>>(`${API_BASE}/admin/monitoring/runs`, { params: toParams({ limit }) })
      .pipe(map(r => r.data));
  }

  getMonitoringDaily(days = 30): Observable<DailyRollupDto[]> {
    return this.http
      .get<DataResponse<DailyRollupDto>>(`${API_BASE}/admin/monitoring/daily`, { params: toParams({ days }) })
      .pipe(map(r => r.data));
  }

  getAdminAlerts(): Observable<AdminAlertDto[]> {
    return this.http
      .get<DataResponse<AdminAlertDto>>(`${API_BASE}/admin/alerts`)
      .pipe(map(r => r.data));
  }
}
