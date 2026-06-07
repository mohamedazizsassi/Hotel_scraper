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
