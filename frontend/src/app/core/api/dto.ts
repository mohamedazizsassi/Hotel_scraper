/**
 * Wire-format DTOs — snake_case, exactly as the FastAPI backend serialises
 * them. Components consume domain models; adapters.ts maps DTO -> domain.
 */

export interface DataResponse<T> {
  data: T[];
  count: number;
}

/** GET /manager/calendar — enriched hotel_features_full row + recommendation. */
export interface CalendarRowDto {
  hotel_name_normalized: string;
  city_name: string;
  stars_int: number | null;
  check_in: string;
  scrape_date: string;
  scraped_at: string;
  nights: number;
  adults: number;
  children: number;
  boarding_canonical: string;
  room_base: string;
  room_view: string;
  room_tier: string;
  room_occupancy: string;
  is_supplement_variant: boolean;
  has_free_view_upgrade: boolean;
  price_per_night: number;
  peer_medium_median: number | null;
  peer_medium_count: number | null;
  best_peer_granularity_used: string | null;
  competitor_avg_per_night: number | null;
  recommended_price_per_night: number;
  forecaster_confidence: number;
  sur_demande_rate_city_stars_checkin: number | null;
  days_until_checkin: number;
  is_weekend_checkin: boolean;
  is_ramadan: boolean;
  is_tunisia_public_holiday: boolean;
  is_tunisia_school_holiday: boolean;
  is_school_holiday_france: boolean;
  is_school_holiday_germany: boolean;
  is_school_holiday_uk: boolean;
}

export interface CalendarConfigDto {
  room_base: string | null;
  room_view: string | null;
  room_tier: string | null;
  room_occupancy: string | null;
  boarding_canonical: string | null;
  nights: number | null;
  adults: number | null;
}

/** GET /manager/calendar/options */
export interface CalendarOptionsDto {
  room_base: string[];
  room_view: string[];
  room_tier: string[];
  room_occupancy: string[];
  boarding_canonical: string[];
  nights: number[];
  adults: number[];
  scrape_dates: string[];
  check_in_min: string | null;
  check_in_max: string | null;
  default: CalendarConfigDto;
}

/** GET /manager/competitors */
export interface CompetitorDto {
  hotel_name_normalized: string;
  city_name: string;
  stars_int: number | null;
  hotel_name_display: string;
  display_order: number;
  avg_price_per_night: number | null;
  latest_scraped_at: string | null;
}

/** GET /manager/recommendations */
export interface RecommendationDto {
  hotel_name_normalized: string;
  city_name: string;
  stars_int: number | null;
  scraped_at: string;
  check_in: string;
  nights: number;
  adults: number;
  boarding_canonical: string;
  current_price_tnd: number;
  q10_cal_tnd: number;
  q50_tnd: number;
  q90_cal_tnd: number;
  interval_status: string;
  direction: 'raise' | 'hold' | 'lower';
  recommended_price_tnd: number;
  delta_pct_vs_current: number;
  peer_medium_median: number | null;
  peer_medium_count: number | null;
  reasons: string[];
  decision_status: 'accepted' | 'dismissed' | null;
}

/** GET /manager/anomalies */
export interface AnomalyDto {
  hotel_name_normalized: string;
  city_name: string;
  stars_int: number | null;
  scraped_at: string;
  check_in: string;
  nights: number;
  adults: number;
  boarding_canonical: string;
  price: number;
  price_per_night: number;
  q10_cal_tnd: number;
  q90_cal_tnd: number;
  anomaly_score: number;
  interval_status: string; // "below_band" | "above_band"
}

/** Query params accepted by /manager/calendar (snake_case = column = filter). */
export interface CalendarQuery {
  check_in_from?: string;
  check_in_to?: string;
  nights?: number;
  adults?: number;
  room_base?: string;
  room_view?: string;
  room_tier?: string;
  room_occupancy?: string;
  boarding_canonical?: string;
  best_peer_granularity_used?: string;
  scrape_date?: string;
}

/** Recommendations / anomalies accept the same product-config scope as the
 *  calendar, so the manager sees one configuration's rows rather than every
 *  variant (which would be thousands of rows and slow to score). */
export interface DateRangeQuery {
  check_in_from?: string;
  check_in_to?: string;
  nights?: number;
  adults?: number;
  room_base?: string;
  room_view?: string;
  room_tier?: string;
  room_occupancy?: string;
  boarding_canonical?: string;
}

// --- Admin: hotels (backend/schemas/admin_hotel.py — AdminHotelRow / DiscoverableHotel / HotelCreate / HotelUpdate) ---
export interface AdminHotelDto {
  id: number;
  hotel_name_normalized: string;
  hotel_name_display: string;
  city_name: string;
  stars_int: number | null;
  is_active: boolean;
  region: string | null;
  contact_email: string | null;
  contact_phone: string | null;
  sources: string;                    // e.g. "promohotel,tunisiepromo" — comma-joined, NOT a list
  manager_id: string | null;           // null = unassigned
  manager_name: string | null;
  latest_scraped_at: string | null;
}

// NOTE: deliberately has NO hotel_name_display and NO source/sources field —
// these candidates come straight off distinct hotel_features identities,
// which only carry the normalized name. The display name gets chosen at
// promote time (see HotelCreate below).
export interface DiscoverableHotelDto {
  hotel_name_normalized: string;
  city_name: string;
  stars_int: number | null;
}

export interface HotelCreateBody {
  hotel_name_normalized: string;
  hotel_name_display: string;
  city_name: string;
  stars_int?: number | null;
  contact_email?: string | null;
  contact_phone?: string | null;
  sources?: string[];
}

export interface HotelUpdateBody {
  hotel_name_display?: string | null;
  stars_int?: number | null;
  is_active?: boolean | null;
  contact_email?: string | null;
  contact_phone?: string | null;
}

// --- Admin: managers (backend/schemas/admin_manager.py — AdminManagerRow / ManagerCreate / ManagerUpdate / PasswordReset) ---
export interface AdminManagerDto {
  id: string;
  email: string;
  full_name: string | null;
  is_active: boolean;
  last_login_at: string | null;
  assigned_hotel_id: number | null;    // null = unassigned
  assigned_hotel_name: string | null;
}

export interface ManagerCreateBody {
  email: string;
  full_name?: string | null;
  initial_password: string;            // admin sets the manager's first password — there is no auto-generated invite flow
}

export interface ManagerUpdateBody {
  email?: string | null;
  full_name?: string | null;
  is_active?: boolean | null;
}

// --- Admin: assignments (backend/schemas/admin_assignment.py — AdminAssignmentRow / AssignmentCreate / AssignmentUpdate) ---
export interface AssignmentDto {
  id: number;
  user_id: string;
  manager_email: string;
  manager_name: string | null;
  hotel_id: number;
  hotel_name: string;
  max_competitors: number;
  is_active: boolean;
}

// --- Admin: competitors (backend/schemas/admin_competitor.py) ---
export interface AdminCompetitorRowDto {
  hotel_id: number;
  hotel_name_display: string;
  city_name: string;
  stars_int: number | null;
  display_order: number;
}

export interface SelectableHotelDto {
  hotel_id: number;
  hotel_name_display: string;
  city_name: string;
  stars_int: number | null;
}

// --- Admin: monitoring (backend/schemas/admin_monitoring.py) ---
export interface MonitoringSummaryDto {
  total_rows: number | null;
  logged_window_items: number;
  runs_count: number;
  finished_runs: number;
  failed_runs: number;
  latest_scrape_at: string | null;
  last_run_status: string | null;
  last_run_items: number | null;
  hotels_scraped_distinct: number;
}

export interface ScrapeRunDto {
  run_ts: string;
  log_filename: string;
  source: string | null;
  items_total: number;
  errors_total: number;
  duration_s: number | null;
  status: string;
}

export interface DailyRollupDto {
  day: string;
  items_total: number;
  runs: number;
}

// --- Admin: alerts (backend/schemas/admin_alert.py) ---
export interface AdminAlertDto {
  type: string;
  severity: string;
  message: string;
  run_ts: string;
  log_filename: string;
}

/** GET /manager/me */
export interface HotelBriefDto {
  id: number;
  hotel_name_display: string;
  city_name: string;
  stars_int: number | null;
}
export interface ManagerProfileDto {
  full_name: string | null;
  email: string;
  role: string;
  preferences: Record<string, any>;
  hotel: HotelBriefDto | null;
  competitor_count: number;
  max_competitors: number;
  last_login_at: string | null;
}
export interface ProfileUpdateDto {
  full_name?: string;
  preferences?: Record<string, any>;
}

/** GET /manager/dashboard */
export interface DashboardKpisDto {
  avg_listed_rate_tnd: number | null;
  market_position_pct: number | null;
  vs_competitor_pct: number | null;
  opportunity_tnd: number | null;
  opportunity_pct: number | null;
  open_recommendations: number;
  active_alerts: number;
}
export interface PriceSeriesPointDto {
  check_in: string;
  price_per_night: number;
  peer_medium_median: number | null;
  recommended_price_per_night: number;
}
export interface DashboardDto {
  kpis: DashboardKpisDto;
  price_series: PriceSeriesPointDto[];
  top_recommendations: RecommendationDto[];
  competitors: CompetitorDto[];
  recent_alerts: AnomalyDto[];
}

/** POST /manager/recommendations/decision */
export interface DecisionDto {
  check_in: string;
  nights: number;
  adults: number;
  boarding_canonical: string;
  recommended_price_tnd?: number;
  status: 'accepted' | 'dismissed';
}
export interface DecisionBulkDto {
  status: 'accepted' | 'dismissed';
  items: Omit<DecisionDto, 'status'>[];
}
