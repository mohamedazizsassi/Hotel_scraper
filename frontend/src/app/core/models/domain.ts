export type Role = 'admin' | 'manager';

export interface Hotel {
  id: string;
  name: string;
  city: string;
  stars: number;
  source: 'promohotel' | 'tunisiepromo' | 'both';
  rooms: number;
  active: boolean;
  managerId?: string;
}

export interface ManagerUser {
  id: string;
  name: string;
  email: string;
  hotelId?: string;
  competitorIds: string[];
  lastSeen: string;
}

export interface ScraperRun {
  id: string;
  source: 'promohotel' | 'tunisiepromo';
  startedAt: string;
  finishedAt?: string;
  status: 'running' | 'success' | 'partial' | 'failed';
  itemsScraped: number;
  hotelsCovered: number;
  durationSec: number;
}

// ---------------------------------------------------------------------------
// Postgres-aligned taxonomy values (mirror ml/feature_engineering/config.py
// and segment_features). Frontend uses these literal strings so the eventual
// API binding is a straight column read.
// ---------------------------------------------------------------------------

export type BoardingCanonical =
  | 'BB' | 'LOG' | 'HDP' | 'HDP_PLUS' | 'PC' | 'PC_PLUS'
  | 'AI' | 'AI_SOFT' | 'AI_ULTRA' | 'UNKNOWN';

export type RoomBase =
  | 'chambre' | 'suite' | 'studio' | 'appartement' | 'bungalow' | 'villa';

export type RoomView =
  | 'none' | 'mer' | 'jardin' | 'piscine' | 'montagne';

export type RoomTier =
  | 'standard' | 'superieure' | 'deluxe' | 'junior' | 'senior'
  | 'executive' | 'presidentielle';

export type RoomOccupancy =
  | 'single' | 'double' | 'triple' | 'familiale';

export type PeerGranularity = 'tight' | 'medium' | 'loose';

/**
 * One row of the manager's calendar. Field names mirror columns from the
 * Postgres view `hotel_features_full` (= hotel_features ⋈ calendar_dim ⋈
 * segment_dim). The three TND values (`price_per_night`, `peer_*_median`,
 * `recommended_price_per_night`) come from the latest scrape_date snapshot.
 */
export interface PricePoint {
  // Identity / join keys (from hotel_features)
  check_in: string;                 // ISO date, join key for calendar_dim
  scrape_date: string;              // freshness of the snapshot
  hotel_name_normalized: string;

  // Room + stay configuration (from hotel_features)
  nights: 1 | 2 | 3 | 5 | 7;
  adults: number;
  children: number;
  boarding_canonical: BoardingCanonical;
  room_base: RoomBase;
  room_view: RoomView;
  room_tier: RoomTier;
  room_occupancy: RoomOccupancy;
  is_supplement_variant: boolean;
  has_free_view_upgrade: boolean;

  // Pricing — TND, per-night
  price_per_night: number;          // your observed price
  peer_medium_median: number;       // primary peer aggregate (medium granularity)
  peer_medium_count: number;        // sample size in the peer slice
  best_peer_granularity_used: PeerGranularity;

  // Recommendation overlay (from forecaster + recommender)
  recommended_price_per_night: number;
  forecaster_confidence: number;    // 0..1

  // Demand proxy
  sur_demande_rate_city_stars_checkin: number;  // 0..1
  days_until_checkin: number;

  // Calendar overlays (from calendar_dim — joined on check_in)
  is_weekend_checkin: boolean;
  is_ramadan: boolean;
  is_tunisia_public_holiday: boolean;
  is_tunisia_school_holiday: boolean;
  is_school_holiday_france: boolean;
  is_school_holiday_germany: boolean;
  is_school_holiday_uk: boolean;
}

export interface Recommendation {
  id: string;
  date: string;
  currentPrice: number;
  recommendedPrice: number;
  delta: number;          // signed TND
  deltaPct: number;
  confidence: number;     // 0..1
  rationale: string[];
  status: 'new' | 'accepted' | 'dismissed';
}

export interface Alert {
  id: string;
  date: string;
  severity: 'info' | 'warning' | 'critical';
  type: 'price_drop' | 'price_spike' | 'competitor_undercut' | 'anomaly' | 'data_gap';
  message: string;
  hotelId?: string;
  competitorId?: string;
}

export interface Competitor {
  id: string;
  name: string;
  city: string;
  stars: number;
  avgPrice7d: number;
  trend: number[]; // sparkline
  distanceKm: number;
}
