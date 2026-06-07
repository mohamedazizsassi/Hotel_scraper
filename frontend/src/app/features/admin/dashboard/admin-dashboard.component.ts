import { Component, OnInit, inject, signal } from '@angular/core';
import { KpiCardComponent } from '../../../shared/components/kpi-card/kpi-card.component';
import { StatusPillComponent } from '../../../shared/components/status-pill/status-pill.component';
import { HOTELS, MANAGERS } from '../../../core/data/mock';
import { DatePipe, DecimalPipe } from '@angular/common';
import { ApiService } from '../../../core/api/api.service';
import { MonitoringSummaryDto, ScrapeRunDto } from '../../../core/api/dto';

@Component({
  selector: 'rw-admin-dashboard',
  standalone: true,
  imports: [KpiCardComponent, StatusPillComponent, DatePipe, DecimalPipe],
  template: `
    <div class="page-head">
      <div>
        <h1>Platform overview</h1>
        <div class="sub">Operational health of the RevWay data pipeline · 2026-05-23</div>
      </div>
      <div class="row">
        <button class="btn">Export CSV</button>
        <button class="btn primary">Trigger scrape</button>
      </div>
    </div>

    <div class="grid cols-4">
      <rw-kpi-card label="Hotels tracked" [value]="hotelsActive" sub="of {{hotels.length}} total" [delta]="2" [trend]="[300,302,305,308,310,311,312]" />
      <rw-kpi-card label="Active managers" [value]="managersActive" sub="signed in last 24h" [delta]="0" [trend]="[4,5,5,5,5,5,5]" />
      @if (summary(); as s) {
        <rw-kpi-card label="Rows in collection"
                     [value]="s.total_rows !== null ? ((s.total_rows | number) ?? '—') : '—'"
                     sub="hotel_prices · Mongo" />
        <rw-kpi-card label="Last run status"
                     [value]="s.last_run_status ?? '—'"
                     sub="{{ s.latest_scrape_at ? (s.latest_scrape_at | date:'MMM d, HH:mm') : 'no runs yet' }}" />
      } @else {
        <rw-kpi-card label="Rows in collection" [value]="monitoringLoading() ? '…' : '—'" />
        <rw-kpi-card label="Last run status" [value]="monitoringLoading() ? '…' : '—'" />
      }
    </div>

    <div class="grid cols-2" style="margin-top:16px;">
      <section class="card">
        <div class="card-head">
          <h3>Latest scrape runs</h3>
          <a class="small" href="/admin/scrapers">View all →</a>
        </div>
        <table class="tbl">
          <thead>
            <tr>
              <th>Source</th><th>Started</th><th>Status</th>
              <th class="num">Items</th><th class="num">Errors</th>
            </tr>
          </thead>
          <tbody>
            @for (r of runs(); track r.log_filename) {
              <tr>
                <td><span class="mono small">{{ r.source ?? '—' }}</span></td>
                <td><span class="small muted">{{ r.run_ts | date:'MMM d, HH:mm' }}</span></td>
                <td>
                  @switch (r.status) {
                    @case ('finished') { <rw-status-pill tone="ok">finished</rw-status-pill> }
                    @case ('partial')  { <rw-status-pill tone="warn">partial</rw-status-pill> }
                    @case ('failed')   { <rw-status-pill tone="err">failed</rw-status-pill> }
                    @default           { <rw-status-pill tone="muted">{{ r.status }}</rw-status-pill> }
                  }
                </td>
                <td class="num mono">{{ r.items_total | number }}</td>
                <td class="num mono">{{ r.errors_total | number }}</td>
              </tr>
            } @empty {
              <tr><td colspan="5" class="muted small" style="padding:14px;">No runs logged yet.</td></tr>
            }
          </tbody>
        </table>
      </section>

      <section class="card">
        <div class="card-head"><h3>Coverage by city</h3><span class="small muted">Active hotels</span></div>
        <div class="card-body">
          @for (c of coverage; track c.city) {
            <div class="cov-row">
              <div class="cov-label">{{ c.city }}</div>
              <div class="cov-bar"><span [style.width.%]="c.pct"></span></div>
              <div class="cov-val mono">{{ c.count }} · {{ c.pct }}%</div>
            </div>
          }
        </div>
      </section>
    </div>
  `,
  styles: [`
    .cov-row { display: grid; grid-template-columns: 100px 1fr 90px; align-items: center; gap: 12px; padding: 8px 0; border-bottom: 1px dashed var(--color-border); }
    .cov-row:last-child { border-bottom: 0; }
    .cov-label { font-size: 13px; }
    .cov-bar { height: 8px; background: var(--color-surface-2); border-radius: 999px; overflow: hidden; }
    .cov-bar span { display: block; height: 100%; background: linear-gradient(90deg, var(--color-primary), var(--color-secondary)); }
    .cov-val { font-size: 12px; color: var(--color-muted); text-align: right; }
  `],
})
export class AdminDashboardComponent implements OnInit {
  private api = inject(ApiService);

  hotels = HOTELS;
  hotelsActive = HOTELS.filter(h => h.active).length;
  managersActive = MANAGERS.filter(m => Date.parse(m.lastSeen) > Date.parse('2026-05-22T00:00:00Z')).length;

  summary = signal<MonitoringSummaryDto | null>(null);
  runs = signal<ScrapeRunDto[]>([]);
  monitoringLoading = signal(true);

  ngOnInit(): void {
    this.api.getMonitoringSummary().subscribe({
      next: s => this.summary.set(s),
      complete: () => this.monitoringLoading.set(false),
      error: () => this.monitoringLoading.set(false),
    });
    this.api.getMonitoringRuns(6).subscribe({
      next: rows => this.runs.set(rows),
    });
  }

  coverage = (() => {
    const byCity = new Map<string, number>();
    HOTELS.filter(h => h.active).forEach(h => byCity.set(h.city, (byCity.get(h.city) ?? 0) + 1));
    const total = Array.from(byCity.values()).reduce((a, b) => a + b, 0);
    return Array.from(byCity.entries())
      .map(([city, count]) => ({ city, count, pct: Math.round(count / total * 100) }))
      .sort((a, b) => b.count - a.count);
  })();
}
