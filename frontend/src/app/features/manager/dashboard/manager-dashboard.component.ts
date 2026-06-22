import { Component, OnInit, computed, inject, signal } from '@angular/core';
import { DatePipe, DecimalPipe } from '@angular/common';
import { RouterLink } from '@angular/router';
import { KpiCardComponent } from '../../../shared/components/kpi-card/kpi-card.component';
import { StatusPillComponent } from '../../../shared/components/status-pill/status-pill.component';
import { ApiService } from '../../../core/api/api.service';
import { recommendationFromDto, alertFromDto } from '../../../core/api/adapters';
import { DashboardDto, ManagerProfileDto } from '../../../core/api/dto';

@Component({
  selector: 'rw-manager-dashboard',
  standalone: true,
  imports: [DatePipe, DecimalPipe, RouterLink, KpiCardComponent, StatusPillComponent],
  template: `
    <div class="page-head">
      <div>
        <h1>Hello, {{ profile()?.full_name }}</h1>
        <div class="sub">
          {{ profile()?.hotel?.hotel_name_display }} · {{ profile()?.hotel?.city_name }} · {{ profile()?.hotel?.stars_int }}★
        </div>
      </div>
      <div class="row">
        <button class="btn">Export</button>
        <a class="btn primary" routerLink="/manager/recommendations">Review {{ kpis()?.open_recommendations ?? 0 }} recommendations</a>
      </div>
    </div>

    @if (loading()) {
      <div class="card card-pad muted">Loading dashboard…</div>
    }
    @if (error()) {
      <div class="card card-pad" style="color:var(--color-destructive)">{{ error() }}</div>
    }

    <div class="grid cols-4">
      <rw-kpi-card label="Your avg listed rate (30d)" [value]="fmt(kpis()?.avg_listed_rate_tnd)" unit="TND" />
      <rw-kpi-card label="Market position" [value]="fmt(kpis()?.market_position_pct, 1)" unit="%" />
      <rw-kpi-card label="Vs competitor set" [value]="fmt(kpis()?.vs_competitor_pct, 1)" unit="%" />
      <rw-kpi-card label="Pricing opportunity" [value]="fmt(kpis()?.opportunity_pct, 1)" unit="%"
                   [sub]="fmt(kpis()?.opportunity_tnd) + ' TND'" />
    </div>

    <div class="grid cols-2" style="margin-top:16px;">
      <section class="card">
        <div class="card-head">
          <h3>Next 14 days — price vs market</h3>
          <a class="small" routerLink="/manager/calendar">Full calendar →</a>
        </div>
        <div class="card-body">
          <div class="chart">
            <svg viewBox="0 0 600 200" preserveAspectRatio="none" role="img" aria-label="price trend">
              <polyline [attr.points]="line(ownPath())" fill="none" stroke="var(--color-primary)" stroke-width="2"/>
              <polyline [attr.points]="line(mktPath())" fill="none" stroke="var(--color-muted-2)" stroke-width="2" stroke-dasharray="4 4"/>
              <polyline [attr.points]="line(recPath())" fill="none" stroke="var(--color-accent)" stroke-width="2"/>
            </svg>
          </div>
          <div class="row" style="gap:14px; margin-top: 8px;">
            <span class="legend"><span class="sw" style="background:var(--color-primary)"></span> Your price</span>
            <span class="legend"><span class="sw" style="background:var(--color-muted-2); border:1px dashed var(--color-muted)"></span> Market avg</span>
            <span class="legend"><span class="sw" style="background:var(--color-accent)"></span> Recommended</span>
          </div>
        </div>
      </section>

      <section class="card">
        <div class="card-head">
          <h3>Today's top recommendations</h3>
          <a class="small" routerLink="/manager/recommendations">All →</a>
        </div>
        <div>
          @for (r of recs(); track r.id) {
            <div class="rec-row">
              <div>
                <div class="rec-date mono">{{ r.date | date:'EEE · MMM d' }}</div>
                <div class="muted small">Confidence {{ r.confidence }}</div>
              </div>
              <div class="prices">
                <div class="mono small muted"><s>{{ r.currentPrice }}</s></div>
                <div class="mono big">{{ r.recommendedPrice }}<span class="unit">TND</span></div>
              </div>
              <div>
                @if (r.delta > 0) { <span class="badge ok">+{{ r.deltaPct }}%</span> }
                @else if (r.delta < 0) { <span class="badge err">{{ r.deltaPct }}%</span> }
                @else { <span class="badge">flat</span> }
              </div>
              <div class="row" style="gap:6px;">
                <a class="btn sm" routerLink="/manager/recommendations">Review →</a>
              </div>
            </div>
          }
        </div>
      </section>
    </div>

    <div class="grid cols-2" style="margin-top:16px;">
      <section class="card">
        <div class="card-head">
          <h3>Your competitor set</h3>
          <a class="small" routerLink="/manager/competitors">Manage →</a>
        </div>
        <div>
          @for (c of competitors(); track c.hotel_name_normalized) {
            <div class="comp-row">
              <div>
                <div style="font-weight:500;">{{ c.hotel_name_display }}</div>
                <div class="small muted">{{ c.city_name }} · {{ c.stars_int }}★</div>
              </div>
              <div class="row" style="gap:14px;">
                <div class="mono" style="min-width:64px; text-align:right;">
                  @if (c.avg_price_per_night != null) {
                    {{ c.avg_price_per_night | number:'1.0-0' }}<span class="unit"> TND</span>
                  } @else {
                    —
                  }
                </div>
              </div>
            </div>
          }
        </div>
      </section>

      <section class="card">
        <div class="card-head">
          <h3>Recent alerts</h3>
          <a class="small" routerLink="/manager/alerts">All →</a>
        </div>
        <div>
          @for (a of alerts(); track a.id) {
            <div class="alert-row">
              <rw-status-pill [tone]="a.severity === 'critical' ? 'err' : a.severity === 'warning' ? 'warn' : 'info'">
                {{ a.severity }}
              </rw-status-pill>
              <div>
                <div class="small">{{ a.message }}</div>
                <div class="muted tiny mono">{{ a.date | date:'MMM d, HH:mm' }} · {{ a.type }}</div>
              </div>
            </div>
          }
        </div>
      </section>
    </div>
  `,
  styles: [`
    .chart { height: 200px; width: 100%; background: linear-gradient(180deg, rgba(37,99,235,0.04), transparent); border-radius: var(--radius-sm); }
    .chart svg { width: 100%; height: 100%; }
    .legend { display: inline-flex; align-items: center; gap: 6px; font-size: 12px; color: var(--color-muted); }
    .sw { display: inline-block; width: 14px; height: 3px; border-radius: 2px; }

    .rec-row, .comp-row, .alert-row {
      display: grid; align-items: center;
      padding: 12px 18px;
      border-bottom: 1px solid var(--color-border);
      gap: 12px;
    }
    .rec-row { grid-template-columns: 130px 1fr auto auto; }
    .comp-row { grid-template-columns: 1fr auto; }
    .alert-row { grid-template-columns: 80px 1fr; }
    .rec-row:last-child, .comp-row:last-child, .alert-row:last-child { border-bottom: 0; }
    .rec-date { font-size: 13px; font-weight: 500; }
    .prices { text-align: right; }
    .big { font-size: 18px; font-weight: 600; }
    .unit { font-size: 11px; color: var(--color-muted); margin-left: 3px; font-weight: 500; }
  `],
})
export class ManagerDashboardComponent implements OnInit {
  private api = inject(ApiService);

  profile = signal<ManagerProfileDto | null>(null);
  dash = signal<DashboardDto | null>(null);
  loading = signal(true);
  error = signal<string | null>(null);

  kpis = computed(() => this.dash()?.kpis ?? null);
  series = computed(() => this.dash()?.price_series ?? []);
  recs = computed(() => (this.dash()?.top_recommendations ?? []).slice(0, 4).map(recommendationFromDto));
  competitors = computed(() => this.dash()?.competitors ?? []);
  alerts = computed(() => (this.dash()?.recent_alerts ?? []).map((a, i) => alertFromDto(a, i)).slice(0, 4));

  ownPath = computed(() => this.series().map(p => p.price_per_night));
  mktPath = computed(() => this.series().map(p => p.peer_medium_median ?? 0));
  recPath = computed(() => this.series().map(p => p.recommended_price_per_night));

  ngOnInit(): void {
    this.api.getMe().subscribe({
      next: p => this.profile.set(p),
      error: () => {},
    });
    this.api.getDashboard(30).subscribe({
      next: d => {
        this.dash.set(d);
        this.loading.set(false);
      },
      error: () => {
        this.error.set('Failed to load dashboard.');
        this.loading.set(false);
      },
    });
  }

  fmt(v: number | null | undefined, digits = 0): string {
    return v === null || v === undefined ? '—' : v.toFixed(digits);
  }

  line(values: number[]): string {
    if (!values.length) return '';
    const W = 600, H = 200, pad = 10;
    const all = [...this.ownPath(), ...this.mktPath(), ...this.recPath()];
    const min = Math.min(...all), max = Math.max(...all);
    const span = max - min || 1;
    const step = (W - pad * 2) / (values.length - 1 || 1);
    return values.map((v, i) => {
      const x = +(pad + i * step).toFixed(1);
      const y = +(H - pad - ((v - min) / span) * (H - pad * 2)).toFixed(1);
      return `${x},${y}`;
    }).join(' ');
  }
}
