import {
  Alert, BoardingCanonical, Competitor, Hotel, ManagerUser,
  PeerGranularity, PricePoint, Recommendation, RoomBase, RoomOccupancy,
  RoomTier, RoomView, ScraperRun,
} from '../models/domain';

// ---------------------------------------------------------------------------
// Reference value lists — mirror ml/feature_engineering/config.py taxonomies.
// Used by the calendar filter bar (single source of truth).
// ---------------------------------------------------------------------------

export const BOARDING_VALUES: { value: BoardingCanonical; label: string }[] = [
  { value: 'BB',       label: 'Bed & breakfast (BB)' },
  { value: 'LOG',      label: 'Room only (LOG)' },
  { value: 'HDP',      label: 'Half board (HDP)' },
  { value: 'HDP_PLUS', label: 'Half board + (HDP_PLUS)' },
  { value: 'PC',       label: 'Full board (PC)' },
  { value: 'PC_PLUS',  label: 'Full board + (PC_PLUS)' },
  { value: 'AI',       label: 'All inclusive (AI)' },
  { value: 'AI_SOFT',  label: 'AI soft (AI_SOFT)' },
  { value: 'AI_ULTRA', label: 'AI ultra (AI_ULTRA)' },
];

export const ROOM_BASE_VALUES: RoomBase[] =
  ['chambre', 'suite', 'studio', 'appartement', 'bungalow', 'villa'];
export const ROOM_VIEW_VALUES: RoomView[] =
  ['none', 'mer', 'jardin', 'piscine', 'montagne'];
export const ROOM_TIER_VALUES: RoomTier[] =
  ['standard', 'superieure', 'deluxe', 'junior', 'senior', 'executive', 'presidentielle'];
export const ROOM_OCCUPANCY_VALUES: RoomOccupancy[] =
  ['single', 'double', 'triple', 'familiale'];

export const NIGHTS_VALUES = [1, 2, 3, 5, 7] as const;
export const ADULTS_VALUES = [1, 2, 3, 4] as const;
export const PEER_GRANULARITY_VALUES: PeerGranularity[] = ['tight', 'medium', 'loose'];

/**
 * Available `scrape_date` snapshots. Two daily runs at 10:00 and 15:00.
 * Latest is always today's 10:00 — what the manager opens to by default.
 */
export const SCRAPE_DATES: string[] = [
  '2026-05-23T10:00:00Z',
  '2026-05-22T15:00:00Z',
  '2026-05-22T10:00:00Z',
  '2026-05-21T15:00:00Z',
  '2026-05-21T10:00:00Z',
];

export const LATEST_SCRAPE_DATE = SCRAPE_DATES[0]!;

const today = new Date('2026-05-23T10:00:00Z');

function addDays(d: Date, n: number): Date {
  const c = new Date(d); c.setDate(c.getDate() + n); return c;
}
function iso(d: Date): string { return d.toISOString().slice(0, 10); }

export const HOTELS: Hotel[] = [
  { id: 'h-001', name: 'Hôtel El Mouradi Palace',     city: 'Sousse',      stars: 5, source: 'both',         rooms: 312, active: true,  managerId: 'u-201' },
  { id: 'h-002', name: 'Iberostar Selection Diar',    city: 'Hammamet',    stars: 5, source: 'promohotel',   rooms: 410, active: true,  managerId: 'u-202' },
  { id: 'h-003', name: 'Hôtel Marhaba Beach',         city: 'Sousse',      stars: 4, source: 'tunisiepromo', rooms: 232, active: true,  managerId: 'u-203' },
  { id: 'h-004', name: 'Royal Thalassa Monastir',     city: 'Monastir',    stars: 5, source: 'both',         rooms: 380, active: true,  managerId: 'u-204' },
  { id: 'h-005', name: 'Vincci Nozha Beach',          city: 'Hammamet',    stars: 4, source: 'promohotel',   rooms: 280, active: false },
  { id: 'h-006', name: 'Hôtel Riu Imperial Marhaba',  city: 'Sousse',      stars: 5, source: 'both',         rooms: 472, active: true,  managerId: 'u-205' },
  { id: 'h-007', name: 'Le Royal Hammamet',           city: 'Hammamet',    stars: 5, source: 'promohotel',   rooms: 340, active: true },
  { id: 'h-008', name: 'Hôtel Sahara Beach',          city: 'Monastir',    stars: 3, source: 'tunisiepromo', rooms: 160, active: true },
  { id: 'h-009', name: 'Bel Azur Thalasso',           city: 'Hammamet',    stars: 4, source: 'both',         rooms: 290, active: true },
  { id: 'h-010', name: 'Marhaba Royal Salem',         city: 'Sousse',      stars: 4, source: 'promohotel',   rooms: 250, active: true },
  { id: 'h-011', name: 'Concorde Marco Polo',         city: 'Hammamet',    stars: 4, source: 'both',         rooms: 305, active: true },
  { id: 'h-012', name: 'Magic Life Skanes',           city: 'Monastir',    stars: 4, source: 'tunisiepromo', rooms: 410, active: false },
];

export const MANAGERS: ManagerUser[] = [
  { id: 'u-201', name: 'Sami Bouazizi',     email: 'sami@elmouradi.tn',    hotelId: 'h-001', competitorIds: ['h-006','h-003','h-010'], lastSeen: '2026-05-23T08:42:00Z' },
  { id: 'u-202', name: 'Yasmine Trabelsi',  email: 'yasmine@iberostar.tn', hotelId: 'h-002', competitorIds: ['h-007','h-009','h-011'], lastSeen: '2026-05-23T09:11:00Z' },
  { id: 'u-203', name: 'Mehdi Karoui',      email: 'mehdi@marhaba.tn',     hotelId: 'h-003', competitorIds: ['h-001','h-006','h-010'], lastSeen: '2026-05-22T18:05:00Z' },
  { id: 'u-204', name: 'Leila Ben Salah',   email: 'leila@royalthalassa.tn', hotelId: 'h-004', competitorIds: ['h-008','h-012'], lastSeen: '2026-05-23T07:50:00Z' },
  { id: 'u-205', name: 'Oussama Hammami',   email: 'oussama@riu.tn',       hotelId: 'h-006', competitorIds: ['h-001','h-003','h-010'], lastSeen: '2026-05-23T09:34:00Z' },
  { id: 'u-206', name: 'Nour Jaziri',       email: 'nour@vincci.tn',                         competitorIds: [],                       lastSeen: '2026-05-15T10:00:00Z' },
];

export const COMPETITORS: Competitor[] = [
  { id: 'h-006', name: 'Riu Imperial Marhaba',   city: 'Sousse',  stars: 5, avgPrice7d: 412, distanceKm: 1.2, trend: [398,402,407,410,415,412,412] },
  { id: 'h-003', name: 'Marhaba Beach',          city: 'Sousse',  stars: 4, avgPrice7d: 268, distanceKm: 2.4, trend: [262,266,271,272,268,265,268] },
  { id: 'h-010', name: 'Marhaba Royal Salem',    city: 'Sousse',  stars: 4, avgPrice7d: 295, distanceKm: 3.1, trend: [288,289,294,298,300,296,295] },
];

export const SCRAPER_RUNS: ScraperRun[] = [
  { id: 'r-101', source: 'promohotel',   startedAt: '2026-05-23T10:00:00Z', finishedAt: '2026-05-23T10:46:12Z', status: 'success', itemsScraped: 248_312, hotelsCovered: 312, durationSec: 2772 },
  { id: 'r-102', source: 'tunisiepromo', startedAt: '2026-05-23T10:00:00Z', finishedAt: '2026-05-23T10:38:50Z', status: 'success', itemsScraped: 196_440, hotelsCovered: 264, durationSec: 2330 },
  { id: 'r-103', source: 'promohotel',   startedAt: '2026-05-22T15:00:00Z', finishedAt: '2026-05-22T15:51:08Z', status: 'success', itemsScraped: 251_018, hotelsCovered: 316, durationSec: 3068 },
  { id: 'r-104', source: 'tunisiepromo', startedAt: '2026-05-22T15:00:00Z', finishedAt: '2026-05-22T15:48:20Z', status: 'partial', itemsScraped: 142_801, hotelsCovered: 198, durationSec: 2900 },
  { id: 'r-105', source: 'promohotel',   startedAt: '2026-05-22T10:00:00Z', finishedAt: '2026-05-22T10:43:30Z', status: 'success', itemsScraped: 247_902, hotelsCovered: 311, durationSec: 2610 },
  { id: 'r-106', source: 'tunisiepromo', startedAt: '2026-05-22T10:00:00Z', finishedAt: '2026-05-22T10:36:11Z', status: 'success', itemsScraped: 194_500, hotelsCovered: 261, durationSec: 2171 },
  { id: 'r-107', source: 'promohotel',   startedAt: '2026-05-21T15:00:00Z', finishedAt: '2026-05-21T15:48:40Z', status: 'failed',  itemsScraped: 12_088,  hotelsCovered: 18,  durationSec: 2920 },
];

/**
 * 30-day calendar for the manager's hotel under the *default* product
 * configuration (Chambre · Double · Standard · no view · BB · 2 nights ·
 * 2 adults · 0 children). The filter bar shifts these values via deterministic
 * multipliers (see CalendarComponent.adjustForFilters) — in production the
 * Postgres view `hotel_features_full` already carries one row per
 * (room_base, room_view, room_tier, room_occupancy, boarding_canonical,
 * nights, adults) combination, so the API will return a pre-filtered slice
 * and the multipliers go away.
 */
export const CALENDAR: PricePoint[] = Array.from({ length: 30 }).map((_, i) => {
  const d = addDays(today, i);
  const dow = d.getUTCDay();
  const weekend = dow === 5 || dow === 6;
  const seasonal = 1 + Math.sin(i / 6) * 0.07;
  const own  = Math.round((weekend ? 380 : 320) * seasonal);
  const peer = Math.round((weekend ? 372 : 312) * (seasonal + (i % 5 === 0 ? 0.04 : 0)));
  const rec  = Math.round(own * (weekend ? 1.04 : (i % 7 === 0 ? 0.97 : 1.01)));
  const month = d.getUTCMonth() + 1;     // 1..12
  const day = d.getUTCDate();

  return {
    check_in: iso(d),
    scrape_date: LATEST_SCRAPE_DATE,
    hotel_name_normalized: 'hotel el mouradi palace',

    nights: 2,
    adults: 2,
    children: 0,
    boarding_canonical: 'BB',
    room_base: 'chambre',
    room_view: 'none',
    room_tier: 'standard',
    room_occupancy: 'double',
    is_supplement_variant: false,
    has_free_view_upgrade: false,

    price_per_night: own,
    peer_medium_median: peer,
    peer_medium_count: weekend ? 18 : 22,
    best_peer_granularity_used: weekend ? 'medium' : (i % 6 === 0 ? 'loose' : 'tight'),

    recommended_price_per_night: rec,
    forecaster_confidence: +(0.62 + (i % 5) * 0.06).toFixed(2),

    sur_demande_rate_city_stars_checkin: +(weekend ? 0.18 : 0.06 + (i % 7) * 0.01).toFixed(2),
    days_until_checkin: i,

    is_weekend_checkin: weekend,
    is_ramadan: false,                                    // 2026 Ramadan was Feb–Mar
    is_tunisia_public_holiday: (month === 7 && day === 25),
    is_tunisia_school_holiday: (month === 7 || month === 8),
    is_school_holiday_france:  (month === 7 || month === 8),
    is_school_holiday_germany: (month === 7 || month === 8),
    is_school_holiday_uk:      (month === 7 || month === 8) || (month === 5 && day >= 24 && day <= 31),
  };
});

export const RECOMMENDATIONS: Recommendation[] = CALENDAR.slice(0, 8).map((p, i) => {
  const delta = p.recommended_price_per_night - p.price_per_night;
  return {
    id: `rec-${i + 1}`,
    date: p.check_in,
    currentPrice: p.price_per_night,
    recommendedPrice: p.recommended_price_per_night,
    delta,
    deltaPct: +(delta / p.price_per_night * 100).toFixed(1),
    confidence: p.forecaster_confidence,
    rationale: [
      p.is_weekend_checkin ? 'Weekend demand window — competitors trending up' : 'Mid-week softness in market average',
      `Peer median ${p.peer_medium_median} TND vs your ${p.price_per_night} TND (n=${p.peer_medium_count}, granularity=${p.best_peer_granularity_used})`,
      i % 3 === 0 ? 'Booking-window pressure: 7-day forecast tightening' : 'Stable booking pace, no anomaly detected',
    ],
    status: i === 0 ? 'new' : (i === 1 ? 'accepted' : 'new'),
  };
});

export const ALERTS: Alert[] = [
  { id: 'a-1', date: '2026-05-23T09:12:00Z', checkIn: '2026-05-30', severity: 'critical', type: 'competitor_undercut', message: 'Riu Imperial Marhaba dropped 14% below your rate for 2026-05-30.',       hotelId: 'h-001', competitorId: 'h-006' },
  { id: 'a-2', date: '2026-05-23T08:40:00Z', checkIn: '2026-06-04', severity: 'warning',  type: 'price_spike',         message: 'Your rate is 11% above market average on 2026-06-04 (weekend).',         hotelId: 'h-001' },
  { id: 'a-3', date: '2026-05-22T17:55:00Z', checkIn: '2026-05-29', severity: 'info',     type: 'anomaly',             message: 'Unusual price volatility detected in Sousse market for next 7 days.' },
  { id: 'a-4', date: '2026-05-22T11:02:00Z', checkIn: '2026-05-29', severity: 'warning',  type: 'data_gap',            message: 'tunisiepromo run partial: 198/264 hotels covered.' },
  { id: 'a-5', date: '2026-05-21T14:31:00Z', checkIn: '2026-05-28', severity: 'info',     type: 'price_drop',          message: 'Market average down 3.2% week-over-week for 4-star Hammamet.' },
];
