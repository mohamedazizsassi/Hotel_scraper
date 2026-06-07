import { Component, OnInit, inject, signal } from '@angular/core';
import { DatePipe, DecimalPipe } from '@angular/common';
import { ApiService } from '../../../core/api/api.service';
import { AdminAlertDto, MonitoringSummaryDto, ScrapeRunDto } from '../../../core/api/dto';
import { StatusPillComponent } from '../../../shared/components/status-pill/status-pill.component';
import { KpiCardComponent } from '../../../shared/components/kpi-card/kpi-card.component';

@Component({
  selector: 'rw-admin-scrapers',
  standalone: true,
  imports: [DatePipe, DecimalPipe, StatusPillComponent, KpiCardComponent],
  template: `
    <div class="page-head">
      <div>
        <h1>Monitoring</h1>
        <div class="sub">Collection health — Mongo row counts, scrape-run history, and pipeline alerts.</div>
      </div>
    </div>

    @if (loading()) { <div class="card card-pad muted">Loading monitoring data…</div> }
    @else if (error()) { <div class="card card-pad err">{{ error() }}</div> }
    @else {
      <div class="grid cols-4">
        <rw-kpi-card label="Rows in collection"
                     [value]="summary()!.total_rows !== null ? ((summary()!.total_rows | number) ?? '—') : '—'"
                     sub="hotel_prices · Mongo" />
        <rw-kpi-card label="Runs (logged window)"
                     [value]="summary()!.runs_count"
                     sub="{{ summary()!.finished_runs }} finished · {{ summary()!.failed_runs }} failed" />
        <rw-kpi-card label="Last run status"
                     [value]="summary()!.last_run_status ?? '—'"
                     sub="{{ summary()!.latest_scrape_at ? (summary()!.latest_scrape_at | date:'MMM d, HH:mm') : 'no runs yet' }}" />
        <rw-kpi-card label="Hotels scraped"
                     [value]="summary()!.hotels_scraped_distinct"
                     sub="distinct · logged window" />
      </div>

      <div class="grid cols-2" style="margin-top:16px; align-items: start;">
        <section class="card">
          <div class="card-head">
            <h3>Run history</h3>
            <span class="small muted">{{ runs().length }} runs</span>
          </div>
          <table class="tbl">
            <thead>
              <tr>
                <th>Run</th><th>Source</th><th>Started</th><th>Status</th>
                <th class="num">Items</th><th class="num">Errors</th><th class="num">Duration</th>
              </tr>
            </thead>
            <tbody>
              @for (r of runs(); track r.log_filename) {
                <tr>
                  <td class="mono small">{{ r.log_filename }}</td>
                  <td><span class="mono small">{{ r.source ?? '—' }}</span></td>
                  <td class="small muted">{{ r.run_ts | date:'MMM d, HH:mm' }}</td>
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
                  <td class="num mono">{{ r.duration_s !== null ? ((r.duration_s / 60) | number:'1.0-0') + ' min' : '—' }}</td>
                </tr>
              } @empty {
                <tr><td colspan="7" class="muted small" style="padding:14px;">No runs logged yet.</td></tr>
              }
            </tbody>
          </table>
        </section>

        <section class="card">
          <div class="card-head">
            <h3>Alerts</h3>
            <span class="small muted">{{ alerts().length }}</span>
          </div>
          <div class="list">
            @for (a of alerts(); track a.run_ts + a.type) {
              <div class="row between item">
                <div>
                  <div style="font-weight:500;">{{ a.message }}</div>
                  <div class="small muted">
                    <span class="mono">{{ a.run_ts | date:'MMM d, HH:mm' }}</span>
                    <span class="dot-sep">·</span>
                    <span>{{ a.type }}</span>
                    <span class="dot-sep">·</span>
                    <span class="mono">{{ a.log_filename }}</span>
                  </div>
                </div>
                <rw-status-pill [tone]="a.severity === 'error' ? 'err' : a.severity === 'warning' ? 'warn' : 'info'">
                  {{ a.severity }}
                </rw-status-pill>
              </div>
            } @empty {
              <div class="muted small" style="padding:14px;">No alerts — pipeline looks healthy.</div>
            }
          </div>
        </section>
      </div>
    }
  `,
  styles: [`
    .list { padding: 8px; display: flex; flex-direction: column; }
    .item { padding: 10px 12px; border-radius: var(--radius-sm); gap: 16px; }
    .item + .item { border-top: 1px solid var(--color-border); }
    .dot-sep { color: var(--color-muted-2); margin: 0 4px; }
    .err { background: var(--color-destructive-soft); color: var(--color-destructive); }
  `],
})
export class AdminScrapersComponent implements OnInit {
  private api = inject(ApiService);

  summary = signal<MonitoringSummaryDto | null>(null);
  runs = signal<ScrapeRunDto[]>([]);
  alerts = signal<AdminAlertDto[]>([]);
  loading = signal(true);
  error = signal<string | null>(null);

  ngOnInit(): void {
    let pending = 3;
    const settle = () => { if (--pending === 0) this.loading.set(false); };
    const fail = (msg: string) => { this.error.set(msg); this.loading.set(false); };

    this.api.getMonitoringSummary().subscribe({
      next: s => { this.summary.set(s); settle(); },
      error: () => fail('Could not load the monitoring summary.'),
    });
    this.api.getMonitoringRuns(50).subscribe({
      next: rows => { this.runs.set(rows); settle(); },
      error: () => fail('Could not load the run history.'),
    });
    this.api.getAdminAlerts().subscribe({
      next: rows => { this.alerts.set(rows); settle(); },
      error: () => fail('Could not load alerts.'),
    });
  }
}
