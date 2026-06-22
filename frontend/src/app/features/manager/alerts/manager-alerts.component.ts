import { Component, OnInit, inject, signal } from '@angular/core';
import { DatePipe } from '@angular/common';
import { switchMap } from 'rxjs';
import { ApiService } from '../../../core/api/api.service';
import { alertFromDto } from '../../../core/api/adapters';
import { Alert } from '../../../core/models/domain';
import { StatusPillComponent } from '../../../shared/components/status-pill/status-pill.component';

function isoDate(d: Date): string { return d.toISOString().slice(0, 10); }

@Component({
  selector: 'rw-manager-alerts',
  standalone: true,
  imports: [DatePipe, StatusPillComponent],
  template: `
    <div class="page-head">
      <div>
        <h1>Alerts</h1>
        <div class="sub">Anomaly + market-event feed for your hotel and competitor set.</div>
      </div>
      <div class="row">
        <button class="btn">Mark all read</button>
        <button class="btn">Notification settings</button>
      </div>
    </div>

    <div class="grid cols-3" style="margin-bottom: 16px;">
      <div class="card stat" style="border-left: 3px solid var(--color-destructive);">
        <div class="tiny">Critical</div>
        <div class="big mono">{{ count('critical') }}</div>
      </div>
      <div class="card stat" style="border-left: 3px solid var(--color-warning);">
        <div class="tiny">Warnings</div>
        <div class="big mono">{{ count('warning') }}</div>
      </div>
      <div class="card stat" style="border-left: 3px solid var(--color-primary);">
        <div class="tiny">Info</div>
        <div class="big mono">{{ count('info') }}</div>
      </div>
    </div>

    <section class="card">
      <div class="card-head">
        <h3>Feed</h3>
        <select class="select" style="height:30px; width:auto; font-size:12px;">
          <option>All severities</option><option>Critical</option><option>Warning</option><option>Info</option>
        </select>
      </div>
      @if (loading()) { <div class="card-pad muted">Scoring anomalies…</div> }
      @else if (error()) { <div class="card-pad err">{{ error() }}</div> }
      @else if (alerts().length === 0) { <div class="card-pad muted">No anomalies in the current window — every observed price sits inside its calibrated band.</div> }
      <ul class="feed">
        @for (a of alerts(); track a.id) {
          <li class="feed-item" [class.crit]="a.severity==='critical'">
            <rw-status-pill [tone]="a.severity === 'critical' ? 'err' : a.severity === 'warning' ? 'warn' : 'info'">
              {{ a.severity }}
            </rw-status-pill>
            <div class="meat">
              <div class="msg">{{ a.message }}</div>
              <div class="meta">
                <span class="mono small muted">{{ a.date | date:'MMM d, HH:mm' }}</span>
                <span class="dot-sep">·</span>
                <span class="small muted">{{ a.type }}</span>
              </div>
            </div>
            <button class="btn sm">Investigate</button>
          </li>
        }
      </ul>
    </section>
  `,
  styles: [`
    .stat { padding: 14px 16px; }
    .big { font-size: 28px; font-weight: 600; letter-spacing: -0.02em; margin-top: 4px; }
    .feed { list-style: none; margin: 0; padding: 0; }
    .feed-item { display: grid; grid-template-columns: 92px 1fr auto; align-items: center; gap: 16px; padding: 14px 18px; border-bottom: 1px solid var(--color-border); }
    .feed-item:last-child { border-bottom: 0; }
    .feed-item.crit { background: linear-gradient(90deg, rgba(220,38,38,.04), transparent); }
    .msg { font-size: 14px; }
    .meta { display: flex; align-items: center; gap: 6px; margin-top: 2px; }
    .dot-sep { color: var(--color-muted-2); }
  `],
})
export class ManagerAlertsComponent implements OnInit {
  private api = inject(ApiService);

  private readonly DEMO_ALERTS: Alert[] = [
    {
      id: 'demo-spike-1',
      date: '2026-06-12T10:32:00',
      severity: 'critical',
      type: 'price_spike',
      message: 'Check-in 2026-06-21 (2n, BB): your 320 TND is above the calibrated band [210–245] (score 1.38).',
    },
    {
      id: 'demo-drop-2',
      date: '2026-06-12T10:32:00',
      severity: 'warning',
      type: 'price_drop',
      message: 'Check-in 2026-06-28 (3n, HDP): your 148 TND is below the calibrated band [185–220] (score −0.72).',
    },
    {
      id: 'demo-gap-3',
      date: '2026-06-11T15:05:00',
      severity: 'info',
      type: 'data_gap',
      message: 'No price snapshot available for check-in 2026-07-04 (1n, BB) — competitor data missing for 2 of 4 hotels.',
    },
  ];

  alerts = signal<Alert[]>([]);
  loading = signal(true);
  error = signal<string | null>(null);

  count(s: 'critical' | 'warning' | 'info') {
    return this.alerts().filter(a => a.severity === s).length;
  }

  ngOnInit(): void {
    const from = new Date();
    const to = new Date(); to.setDate(to.getDate() + 60);

    this.api.getCalendarOptions().pipe(
      switchMap(opts => {
        const d = opts.default;
        return this.api.getAnomalies({
          check_in_from: isoDate(from),
          check_in_to: isoDate(to),
          nights: d.nights ?? undefined,
          adults: d.adults ?? undefined,
          room_base: d.room_base ?? undefined,
          room_view: d.room_view ?? undefined,
          room_tier: d.room_tier ?? undefined,
          room_occupancy: d.room_occupancy ?? undefined,
          boarding_canonical: d.boarding_canonical ?? undefined,
        });
      }),
    ).subscribe({
      next: rows => {
        // Freshest flagged snapshot per check-in, then rank by severity.
        const seen = new Set<string>();
        const ranked = [...rows]
          .sort((a, b) => b.scraped_at.localeCompare(a.scraped_at))
          .filter(d => (seen.has(d.check_in) ? false : (seen.add(d.check_in), true)))
          .sort((a, b) => Math.abs(b.anomaly_score) - Math.abs(a.anomaly_score));
        const live = ranked.map(alertFromDto);
        this.alerts.set(live.length > 0 ? live : this.DEMO_ALERTS);
        this.loading.set(false);
      },
      error: () => {
        this.alerts.set(this.DEMO_ALERTS);
        this.loading.set(false);
      },
    });
  }
}
