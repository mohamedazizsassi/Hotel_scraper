import { Component } from '@angular/core';
import { KpiCardComponent } from '../../../shared/components/kpi-card/kpi-card.component';
import { StatusPillComponent } from '../../../shared/components/status-pill/status-pill.component';
import { HOTELS, MANAGERS, SCRAPER_RUNS } from '../../../core/data/mock';
import { DatePipe, DecimalPipe } from '@angular/common';

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
      <rw-kpi-card label="Rows ingested today" value="444,752" unit="" [delta]="-1.2" [trend]="[420,430,445,450,442,448,445]" />
      <rw-kpi-card label="Scrape success" value="83" unit="%" [delta]="-12" [trend]="[100,100,100,100,86,83,83]" />
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
              <th class="num">Items</th><th class="num">Hotels</th>
            </tr>
          </thead>
          <tbody>
            @for (r of runs; track r.id) {
              <tr>
                <td><span class="mono small">{{ r.source }}</span></td>
                <td><span class="small muted">{{ r.startedAt | date:'MMM d, HH:mm' }}</span></td>
                <td>
                  @switch (r.status) {
                    @case ('success') { <rw-status-pill tone="ok">success</rw-status-pill> }
                    @case ('partial') { <rw-status-pill tone="warn">partial</rw-status-pill> }
                    @case ('failed')  { <rw-status-pill tone="err">failed</rw-status-pill> }
                    @case ('running') { <rw-status-pill tone="info">running</rw-status-pill> }
                  }
                </td>
                <td class="num">{{ r.itemsScraped | number }}</td>
                <td class="num">{{ r.hotelsCovered }}</td>
              </tr>
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
export class AdminDashboardComponent {
  hotels = HOTELS;
  hotelsActive = HOTELS.filter(h => h.active).length;
  managersActive = MANAGERS.filter(m => Date.parse(m.lastSeen) > Date.parse('2026-05-22T00:00:00Z')).length;
  runs = SCRAPER_RUNS.slice(0, 6);

  coverage = (() => {
    const byCity = new Map<string, number>();
    HOTELS.filter(h => h.active).forEach(h => byCity.set(h.city, (byCity.get(h.city) ?? 0) + 1));
    const total = Array.from(byCity.values()).reduce((a, b) => a + b, 0);
    return Array.from(byCity.entries())
      .map(([city, count]) => ({ city, count, pct: Math.round(count / total * 100) }))
      .sort((a, b) => b.count - a.count);
  })();
}
