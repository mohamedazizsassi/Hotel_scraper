import { Alert, Recommendation } from '../models/domain';
import { AnomalyDto, RecommendationDto } from './dto';

/**
 * UI confidence proxy from the calibrated 80% interval (same shape the backend
 * uses for the calendar): a narrower [q10_cal, q90_cal] relative to q50 reads
 * as higher confidence. Display only — not a calibrated probability.
 */
export function confidenceFromInterval(q10: number, q50: number, q90: number): number {
  const q50Safe = Math.abs(q50) < 1e-9 ? 1e-9 : q50;
  const conf = 1 - (q90 - q10) / q50Safe / 2;
  return Math.max(0.05, Math.min(0.99, +conf.toFixed(2)));
}

export function recommendationFromDto(d: RecommendationDto): Recommendation {
  const delta = Math.round(d.recommended_price_tnd - d.current_price_tnd);
  return {
    id: `${d.check_in}-${d.nights}n-${d.boarding_canonical}-${d.adults}a`,
    date: d.check_in,
    currentPrice: Math.round(d.current_price_tnd),
    recommendedPrice: Math.round(d.recommended_price_tnd),
    delta,
    deltaPct: +d.delta_pct_vs_current.toFixed(1),
    confidence: confidenceFromInterval(d.q10_cal_tnd, d.q50_tnd, d.q90_cal_tnd),
    rationale: d.reasons,
    status: 'new',
  };
}

export function alertFromDto(d: AnomalyDto, i: number): Alert {
  const below = d.interval_status === 'below_band';
  const mag = Math.abs(d.anomaly_score);
  const severity: Alert['severity'] = mag >= 1 ? 'critical' : mag >= 0.4 ? 'warning' : 'info';
  return {
    id: `anom-${d.check_in}-${d.nights}n-${i}`,
    date: d.scraped_at,
    severity,
    type: below ? 'price_drop' : 'price_spike',
    message:
      `Check-in ${d.check_in} (${d.nights}n, ${d.boarding_canonical}): your ` +
      `${Math.round(d.price)} TND is ${below ? 'below' : 'above'} the calibrated band ` +
      `[${Math.round(d.q10_cal_tnd)}–${Math.round(d.q90_cal_tnd)}] (score ${d.anomaly_score.toFixed(2)}).`,
  };
}
