import { Component } from '@angular/core';
import { DatePipe, DecimalPipe } from '@angular/common';
import { SCRAPER_RUNS } from '../../../core/data/mock';
import { StatusPillComponent } from '../../../shared/components/status-pill/status-pill.component';
import { KpiCardComponent } from '../../../shared/components/kpi-card/kpi-card.component';

@Component({
  selector: 'rw-admin-scrapers',
  standalone: true,
  imports: [DatePipe, DecimalPipe, StatusPillComponent, KpiCardComponent],
  template: `
    <div class="page-head">
      <div>
        <h1>Scrapers</h1>
        <div class="sub">promohotel · tunisiepromo · runs at 10:00 and 15:00 daily</div>
      </div>
      <div class="row">
        <button class="btn">Pause schedule</button>
        <button class="btn primary">Run now</button>
      </div>
    </div>

    <div class="grid cols-4">
      <rw-kpi-card label="Total runs (7d)" [value]="runs.length" [trend]="[10,12,12,14,14,13,14]" />
      <rw-kpi-card label="Success rate" value="83" unit="%" [delta]="-12" [trend]="[100,100,100,100,86,83,83]" />
      <rw-kpi-card label="Avg duration" value="45" unit="min" [delta]="-3" />
      <rw-kpi-card label="Items / day" value="444,752" [delta]="-1.2" />
    </div>

    <section class="card" style="margin-top:16px;">
      <div class="card-head">
        <h3>Run history</h3>
        <div class="row" style="gap:6px;">
          <select class="select sm" style="height:30px; font-size:12px; width:auto;">
            <option>All sources</option>
            <option>promohotel</option>
            <option>tunisiepromo</option>
          </select>
        </div>
      </div>
      <table class="tbl">
        <thead>
          <tr>
            <th>Run</th><th>Source</th><th>Started</th><th>Finished</th><th>Status</th>
            <th class="num">Items</th><th class="num">Hotels</th><th class="num">Duration</th>
          </tr>
        </thead>
        <tbody>
          @for (r of runs; track r.id) {
            <tr>
              <td class="mono small">{{ r.id }}</td>
              <td><span class="mono small">{{ r.source }}</span></td>
              <td class="small muted">{{ r.startedAt | date:'MMM d, HH:mm:ss' }}</td>
              <td class="small muted">{{ r.finishedAt ? (r.finishedAt | date:'HH:mm:ss') : '—' }}</td>
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
              <td class="num mono">{{ (r.durationSec/60) | number:'1.0-0' }} min</td>
            </tr>
          }
        </tbody>
      </table>
    </section>
  `,
})
export class AdminScrapersComponent {
  runs = SCRAPER_RUNS;
}
