import { Component, OnInit, inject, signal } from '@angular/core';
import { DatePipe, DecimalPipe } from '@angular/common';
import { ApiService } from '../../../core/api/api.service';
import { CompetitorDto } from '../../../core/api/dto';

@Component({
  selector: 'rw-manager-competitors',
  standalone: true,
  imports: [DatePipe, DecimalPipe],
  template: `
    <div class="page-head">
      <div>
        <h1>Your competitor set</h1>
        <div class="sub">
          {{ selected().length }} of 4 picks · scoped to your view only — the competitor average uses these hotels.
        </div>
      </div>
    </div>

    @if (loading()) { <div class="card card-pad muted">Loading competitors…</div> }
    @else if (error()) { <div class="card card-pad err">{{ error() }}</div> }

    <div class="grid cols-3">
      @for (c of selected(); track c.hotel_name_normalized) {
        <article class="card comp">
          <div class="card-head">
            <div>
              <div style="font-weight: 600;">{{ c.hotel_name_display }}</div>
              <div class="small muted">{{ c.city_name }} · {{ c.stars_int }}★ · pick #{{ c.display_order }}</div>
            </div>
          </div>
          <div class="card-body">
            <div class="row between">
              <div>
                <div class="tiny">avg price_per_night</div>
                <div class="big mono">
                  @if (c.avg_price_per_night !== null) { {{ c.avg_price_per_night | number:'1.0-0' }}<span class="unit">TND</span> }
                  @else { <span class="muted">—</span> }
                </div>
              </div>
              @if (c.latest_scraped_at) {
                <div class="tiny muted" style="text-align:right;">
                  latest scrape<br><span class="mono">{{ c.latest_scraped_at | date:'MMM d, HH:mm' }}</span>
                </div>
              }
            </div>
          </div>
        </article>
      }
    </div>
  `,
  styles: [`
    .comp { padding: 0; }
    .big { font-size: 24px; font-weight: 600; letter-spacing: -0.02em; }
    .unit { font-size: 11px; color: var(--color-muted); margin-left: 4px; font-weight: 500; }
    .err { background: var(--color-destructive-soft); color: var(--color-destructive); }
  `],
})
export class ManagerCompetitorsComponent implements OnInit {
  private api = inject(ApiService);

  selected = signal<CompetitorDto[]>([]);
  loading = signal(true);
  error = signal<string | null>(null);

  ngOnInit(): void {
    this.api.getCompetitors().subscribe({
      next: rows => { this.selected.set(rows); this.loading.set(false); },
      error: () => { this.error.set('Could not load competitors.'); this.loading.set(false); },
    });
  }
}
