import { Component, OnInit, computed, inject, signal } from '@angular/core';
import { DatePipe } from '@angular/common';
import { switchMap } from 'rxjs';
import { ApiService } from '../../../core/api/api.service';
import { recommendationFromDto } from '../../../core/api/adapters';
import { Recommendation } from '../../../core/models/domain';
import { RecommendationDto, DecisionDto } from '../../../core/api/dto';

type Filter = 'all' | 'new' | 'accepted' | 'dismissed';

function isoDate(d: Date): string { return d.toISOString().slice(0, 10); }

@Component({
  selector: 'rw-manager-recommendations',
  standalone: true,
  imports: [DatePipe],
  template: `
    <div class="page-head">
      <div>
        <h1>Recommendations</h1>
        <div class="sub">Rule-based suggestions with rationale you can defend to the GM.</div>
      </div>
      <div class="row">
        <button class="btn" (click)="bulk('dismissed')">Dismiss all</button>
        <button class="btn primary" (click)="bulk('accepted')">Accept all new</button>
      </div>
    </div>

    <div class="filters">
      @for (f of filters; track f) {
        <button class="chip" [class.active]="filter() === f" (click)="filter.set(f)">
          {{ f }} <span class="count">{{ countFor(f) }}</span>
        </button>
      }
    </div>

    @if (loading()) { <div class="card card-pad muted">Loading recommendations…</div> }
    @else if (error()) { <div class="card card-pad err">{{ error() }}</div> }

    <div class="rec-list">
      @for (r of visible(); track r.id) {
        <article class="card rec">
          <div class="rec-left">
            <div class="rec-date">{{ r.date | date:'EEE, MMM d' }}</div>
            <div class="muted tiny">conf {{ r.confidence }}</div>
            <div class="conf-bar" [style.--w.%]="r.confidence * 100"></div>
          </div>

          <div class="rec-mid">
            <div class="prices">
              <div class="p">
                <span class="tiny">Current</span>
                <span class="mono num">{{ r.currentPrice }}<span class="unit">TND</span></span>
              </div>
              <span class="arrow">→</span>
              <div class="p hi">
                <span class="tiny">Recommended</span>
                <span class="mono num">{{ r.recommendedPrice }}<span class="unit">TND</span></span>
              </div>
              <div class="delta" [class.up]="r.delta > 0" [class.down]="r.delta < 0">
                {{ r.delta > 0 ? '+' : '' }}{{ r.delta }} TND · {{ r.deltaPct > 0 ? '+' : '' }}{{ r.deltaPct }}%
              </div>
            </div>

            <ul class="rationale">
              @for (line of r.rationale; track line) {
                <li>
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="m9 12 2 2 4-4"/><circle cx="12" cy="12" r="9"/></svg>
                  <span>{{ line }}</span>
                </li>
              }
            </ul>
          </div>

          <div class="rec-right">
            @if (r.status === 'accepted') { <span class="badge ok">accepted</span> }
            @else if (r.status === 'dismissed') { <span class="badge">dismissed</span> }
            @else { <span class="badge info">new</span> }
            <div class="actions">
              <button class="btn sm" [disabled]="r.status !== 'new'" (click)="decide(r, 'dismissed')">Dismiss</button>
              <button class="btn primary sm" [disabled]="r.status !== 'new'" (click)="decide(r, 'accepted')">Accept</button>
            </div>
          </div>
        </article>
      } @empty {
        <div class="card card-pad muted">No recommendations match this filter.</div>
      }
    </div>
  `,
  styles: [`
    .filters { display: flex; gap: 8px; margin-bottom: 14px; }
    .chip { padding: 6px 12px; background: var(--color-surface); border: 1px solid var(--color-border); border-radius: 999px; font-size: 12px; font-weight: 500; color: var(--color-muted); text-transform: capitalize; cursor: pointer; transition: all .15s ease; }
    .chip:hover { color: var(--color-foreground); border-color: var(--color-border-strong); }
    .chip.active { background: var(--color-primary); border-color: var(--color-primary); color: #fff; }
    .chip .count { font-family: var(--font-mono); font-size: 11px; opacity: .8; margin-left: 2px; }

    .rec-list { display: flex; flex-direction: column; gap: 10px; }
    .rec { display: grid; grid-template-columns: 120px 1fr 180px; gap: 18px; padding: 16px 18px; align-items: start; }

    .rec-left { border-right: 1px solid var(--color-border); padding-right: 14px; }
    .rec-date { font-weight: 600; font-size: 14px; }
    .conf-bar { margin-top: 8px; height: 4px; border-radius: 999px; background: var(--color-surface-2); position: relative; }
    .conf-bar::after { content: ""; position: absolute; inset: 0; width: var(--w); background: linear-gradient(90deg, var(--color-primary), var(--color-accent)); border-radius: 999px; }

    .prices { display: flex; align-items: center; gap: 14px; margin-bottom: 10px; flex-wrap: wrap; }
    .p { display: flex; flex-direction: column; }
    .p .tiny { color: var(--color-muted); margin-bottom: 2px; }
    .num { font-size: 20px; font-weight: 600; letter-spacing: -0.01em; }
    .p.hi .num { color: var(--color-accent); }
    .unit { font-size: 11px; color: var(--color-muted); margin-left: 4px; font-weight: 500; }
    .arrow { color: var(--color-muted); font-size: 18px; }
    .delta { font-size: 12px; font-weight: 600; padding: 4px 10px; border-radius: 999px; background: var(--color-surface-2); color: var(--color-muted); }
    .delta.up { background: var(--color-success-soft); color: var(--color-success); }
    .delta.down { background: var(--color-destructive-soft); color: var(--color-destructive); }

    .rationale { margin: 0; padding: 0; list-style: none; display: flex; flex-direction: column; gap: 6px; }
    .rationale li { display: flex; gap: 8px; align-items: flex-start; font-size: 13px; color: var(--color-foreground); }
    .rationale svg { color: var(--color-accent); flex-shrink: 0; margin-top: 2px; }

    .rec-right { display: flex; flex-direction: column; gap: 12px; align-items: flex-end; }
    .actions { display: flex; gap: 6px; }

    @media (max-width: 900px) {
      .rec { grid-template-columns: 1fr; }
      .rec-left { border-right: 0; border-bottom: 1px solid var(--color-border); padding: 0 0 10px; }
      .rec-right { align-items: flex-start; }
    }
  `],
})
export class ManagerRecommendationsComponent implements OnInit {
  private api = inject(ApiService);
  private dtoById = new Map<string, RecommendationDto>();

  all = signal<Recommendation[]>([]);
  loading = signal(true);
  error = signal<string | null>(null);

  filters: Filter[] = ['all', 'new', 'accepted', 'dismissed'];
  filter = signal<Filter>('all');

  visible = computed(() => {
    const f = this.filter();
    const rows = this.all();
    return f === 'all' ? rows : rows.filter(r => r.status === f);
  });
  countFor(f: Filter) {
    const rows = this.all();
    return f === 'all' ? rows.length : rows.filter(r => r.status === f).length;
  }

  ngOnInit(): void {
    // Scope to the hotel's default product config (so we score ~one row per
    // date, not every variant) over a forward window, keeping the freshest
    // snapshot per check-in. Holds are shown too — "price within band" is
    // itself a defensible recommendation.
    const from = new Date();
    const to = new Date(); to.setDate(to.getDate() + 45);

    this.api.getCalendarOptions().pipe(
      switchMap(opts => {
        const d = opts.default;
        return this.api.getRecommendations({
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
        const seen = new Set<string>();
        const latestPerDateDtos = [...rows]
          .sort((a, b) => b.scraped_at.localeCompare(a.scraped_at))
          .filter(d => (seen.has(d.check_in) ? false : (seen.add(d.check_in), true)))
          .sort((a, b) => a.check_in.localeCompare(b.check_in));
        this.all.set(latestPerDateDtos.map(recommendationFromDto));
        this.dtoById.clear();
        latestPerDateDtos.forEach(d => this.dtoById.set(
          `${d.check_in}-${d.nights}n-${d.boarding_canonical}-${d.adults}a`, d));
        this.loading.set(false);
      },
      error: () => { this.error.set('Could not load recommendations.'); this.loading.set(false); },
    });
  }

  decide(r: Recommendation, status: 'accepted' | 'dismissed') {
    const d = this.dtoById.get(r.id);
    if (!d) return;
    const body: DecisionDto = {
      check_in: d.check_in, nights: d.nights, adults: d.adults,
      boarding_canonical: d.boarding_canonical,
      recommended_price_tnd: d.recommended_price_tnd, status,
    };
    this.api.postDecision(body).subscribe({
      next: () => this.all.update(rows => rows.map(x => x.id === r.id ? { ...x, status } : x)),
      error: () => {},
    });
  }

  bulk(status: 'accepted' | 'dismissed') {
    const targets = this.all().filter(r => status === 'accepted' ? r.status === 'new' : true);
    const items = targets
      .map(r => this.dtoById.get(r.id))
      .filter((d): d is RecommendationDto => !!d)
      .map(d => ({ check_in: d.check_in, nights: d.nights, adults: d.adults,
                   boarding_canonical: d.boarding_canonical,
                   recommended_price_tnd: d.recommended_price_tnd }));
    if (!items.length) return;
    this.api.postDecisionBulk({ status, items }).subscribe({
      next: () => this.all.update(rows => rows.map(x =>
        targets.some(t => t.id === x.id) ? { ...x, status } : x)),
      error: () => {},
    });
  }
}
