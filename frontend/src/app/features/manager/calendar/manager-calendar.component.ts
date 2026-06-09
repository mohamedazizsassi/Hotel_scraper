import { Component, OnInit, computed, inject, signal } from '@angular/core';
import { DatePipe, DecimalPipe } from '@angular/common';
import { ApiService } from '../../../core/api/api.service';
import { CalendarOptionsDto, CalendarRowDto, CalendarQuery } from '../../../core/api/dto';

/**
 * Calendar filters. Keys mirror Postgres columns in `hotel_features_full` and
 * are sent verbatim as query params to GET /manager/calendar. The dropdown
 * values come from GET /manager/calendar/options, so every offered
 * configuration is one that actually returns rows for this hotel.
 */
interface CalendarFilters {
  scrape_date: string;                 // '' = latest snapshot per check_in
  room_base: string;
  room_view: string;
  room_tier: string;
  room_occupancy: string;
  boarding_canonical: string;
  nights: number | null;
  adults: number | null;
  best_peer_granularity_used: string;  // 'any' = no constraint
}

interface Overlays {
  is_ramadan: boolean;
  is_tunisia_public_holiday: boolean;
  is_tunisia_school_holiday: boolean;
  is_school_holiday_france: boolean;
  is_school_holiday_germany: boolean;
  is_school_holiday_uk: boolean;
}

function isoDate(d: Date): string { return d.toISOString().slice(0, 10); }
function label(v: string): string { return v === '' ? '(none)' : v; }

@Component({
  selector: 'rw-manager-calendar',
  standalone: true,
  imports: [DatePipe, DecimalPipe],
  template: `
    <div class="page-head">
      <div>
        <h1>Price calendar</h1>
        <div class="sub">
          TND/night · live <span class="mono small">hotel_features_full</span> + recommender ·
          snapshot <span class="mono small">{{ filters().scrape_date || 'latest' }}</span>
        </div>
      </div>
      <div class="row">
        <button class="btn" (click)="reset()">Reset filters</button>
      </div>
    </div>

    <!-- Filter bar -->
    <section class="card filter-bar">
      <div class="filters">
        <div class="field f-sm">
          <label for="f-scrape">scrape_date <span class="hint">freshness</span></label>
          <select id="f-scrape" class="select" [value]="filters().scrape_date"
                  (change)="set('scrape_date', $any($event.target).value)">
            <option value="">latest</option>
            @for (d of opt().scrape_dates; track d) { <option [value]="d">{{ d }}</option> }
          </select>
        </div>

        <div class="field f-sm">
          <label for="f-base">room_base</label>
          <select id="f-base" class="select" [value]="filters().room_base"
                  (change)="set('room_base', $any($event.target).value)">
            @for (v of opt().room_base; track v) { <option [value]="v">{{ label(v) }}</option> }
          </select>
        </div>

        <div class="field f-sm">
          <label for="f-view">room_view</label>
          <select id="f-view" class="select" [value]="filters().room_view"
                  (change)="set('room_view', $any($event.target).value)">
            @for (v of opt().room_view; track v) { <option [value]="v">{{ label(v) }}</option> }
          </select>
        </div>

        <div class="field f-sm">
          <label for="f-tier">room_tier</label>
          <select id="f-tier" class="select" [value]="filters().room_tier"
                  (change)="set('room_tier', $any($event.target).value)">
            @for (v of opt().room_tier; track v) { <option [value]="v">{{ label(v) }}</option> }
          </select>
        </div>

        <div class="field f-sm">
          <label for="f-occ">room_occupancy</label>
          <select id="f-occ" class="select" [value]="filters().room_occupancy"
                  (change)="set('room_occupancy', $any($event.target).value)">
            @for (v of opt().room_occupancy; track v) { <option [value]="v">{{ label(v) }}</option> }
          </select>
        </div>

        <div class="field f-md">
          <label for="f-board">boarding_canonical</label>
          <select id="f-board" class="select" [value]="filters().boarding_canonical"
                  (change)="set('boarding_canonical', $any($event.target).value)">
            @for (b of opt().boarding_canonical; track b) { <option [value]="b">{{ label(b) }}</option> }
          </select>
        </div>

        <div class="field f-xs">
          <label for="f-n">nights</label>
          <select id="f-n" class="select mono" [value]="filters().nights"
                  (change)="set('nights', +$any($event.target).value)">
            @for (n of opt().nights; track n) { <option [value]="n">{{ n }}</option> }
          </select>
        </div>

        <div class="field f-xs">
          <label for="f-a">adults</label>
          <select id="f-a" class="select mono" [value]="filters().adults"
                  (change)="set('adults', +$any($event.target).value)">
            @for (a of opt().adults; track a) { <option [value]="a">{{ a }}</option> }
          </select>
        </div>

        <div class="field f-sm">
          <label for="f-peer">best_peer_granularity_used</label>
          <select id="f-peer" class="select" [value]="filters().best_peer_granularity_used"
                  (change)="set('best_peer_granularity_used', $any($event.target).value)">
            <option value="any">any</option>
            <option value="tight">tight</option>
            <option value="medium">medium</option>
            <option value="loose">loose</option>
          </select>
        </div>
      </div>

      <div class="overlays">
        <span class="tiny">calendar_dim overlays</span>
        <label class="chip-toggle">
          <input type="checkbox" [checked]="overlays().is_ramadan" (change)="toggle('is_ramadan')">
          <span>is_ramadan</span>
        </label>
        <label class="chip-toggle">
          <input type="checkbox" [checked]="overlays().is_tunisia_public_holiday" (change)="toggle('is_tunisia_public_holiday')">
          <span>is_tunisia_public_holiday</span>
        </label>
        <label class="chip-toggle">
          <input type="checkbox" [checked]="overlays().is_tunisia_school_holiday" (change)="toggle('is_tunisia_school_holiday')">
          <span>is_tunisia_school_holiday</span>
        </label>
        <label class="chip-toggle">
          <input type="checkbox" [checked]="overlays().is_school_holiday_france" (change)="toggle('is_school_holiday_france')">
          <span>is_school_holiday_france</span>
        </label>
        <label class="chip-toggle">
          <input type="checkbox" [checked]="overlays().is_school_holiday_germany" (change)="toggle('is_school_holiday_germany')">
          <span>is_school_holiday_germany</span>
        </label>
        <label class="chip-toggle">
          <input type="checkbox" [checked]="overlays().is_school_holiday_uk" (change)="toggle('is_school_holiday_uk')">
          <span>is_school_holiday_uk</span>
        </label>
      </div>
    </section>

    <!-- Summary strip -->
    <div class="summary">
      <span class="lg"><span class="sw o"></span>price_per_night</span>
      <span class="lg"><span class="sw m"></span>peer_medium_median</span>
      <span class="lg"><span class="sw r"></span>recommended_price_per_night</span>
      <span class="spacer"></span>
      @if (loading()) { <span class="lg mono small">loading…</span> }
      @else { <span class="lg mono small">{{ visible().length }} rows · avg gap {{ avgGap() | number:'1.1-1' }}%</span> }
    </div>

    <!-- Calendar grid -->
    <section class="card">
      @if (error()) { <div class="cell empty wide muted">{{ error() }}</div> }
      <div class="cal-grid">
        <div class="cal-head">Mon</div>
        <div class="cal-head">Tue</div>
        <div class="cal-head">Wed</div>
        <div class="cal-head">Thu</div>
        <div class="cal-head">Fri</div>
        <div class="cal-head">Sat</div>
        <div class="cal-head">Sun</div>

        @for (s of spacers(); track $index) { <div class="cell empty"></div> }

        @for (p of visible(); track p.check_in) {
          <div class="cell" [class.weekend]="p.is_weekend_checkin"
               [title]="granularityTip(p)">
            <!-- Date line + context tags + action badge -->
            <div class="cell-top">
              <span class="cal-day mono">{{ p.check_in | date:'d MMM' }}</span>
              <span class="badge-action" [class]="badge(p).cls">{{ badge(p).label }}</span>
            </div>
            @if (tags(p).length > 0) {
              <div class="tag-row">
                @for (t of tags(p); track t) {
                  <span class="ctx-tag">{{ t }}</span>
                }
              </div>
            }

            <!-- Price rows -->
            <div class="prices">
              <div class="price-row">
                <span class="price-lbl">Your rate</span>
                <span class="mono fw">{{ p.price_per_night | number:'1.0-0' }} TND</span>
              </div>
              <div class="price-row muted">
                <span class="price-lbl">Market avg</span>
                <span class="mono">{{ p.peer_medium_median !== null ? (p.peer_medium_median | number:'1.0-0') + ' TND' : '—' }}</span>
              </div>
              <div class="price-row muted">
                <span class="price-lbl">Competitors</span>
                <span class="mono">{{ p.competitor_avg_per_night !== null ? (p.competitor_avg_per_night | number:'1.0-0') + ' TND' : '—' }}</span>
              </div>
              <div class="price-row suggested">
                <span class="price-lbl">Suggested</span>
                <span class="mono fw">{{ p.recommended_price_per_night | number:'1.0-0' }} TND</span>
              </div>
            </div>

            <div class="delta-bar" [class]="badge(p).cls">{{ deltaPct(p) }}</div>
          </div>
        }

        @if (!loading() && visible().length === 0) {
          <div class="cell empty wide muted">
            No rows for this configuration in the current window.
          </div>
        }
      </div>
    </section>
  `,
  styles: [`
    .filter-bar { padding: 14px 16px 12px; margin-bottom: 12px; }
    .filters { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px 12px; align-items: end; }
    .filters .field label { font-family: var(--font-mono); font-size: 11px; color: var(--color-muted); letter-spacing: 0; text-transform: none; font-weight: 500; }
    .filters .field label .hint { color: var(--color-muted-2); font-weight: 400; margin-left: 4px; }
    .filters .select { height: 32px; font-size: 12px; padding: 0 8px; }
    .filters .select.mono { font-family: var(--font-mono); }
    .f-xs { max-width: 100px; }
    .f-sm { max-width: 200px; }
    .f-md { max-width: 240px; }

    .overlays {
      margin-top: 12px; padding-top: 10px;
      border-top: 1px dashed var(--color-border);
      display: flex; flex-wrap: wrap; gap: 8px; align-items: center;
    }
    .overlays .tiny { margin-right: 4px; }
    .chip-toggle {
      display: inline-flex; align-items: center; gap: 6px;
      padding: 4px 10px; border-radius: 999px;
      border: 1px solid var(--color-border);
      background: var(--color-surface-2); cursor: pointer;
      font-family: var(--font-mono); font-size: 11px; color: var(--color-muted);
      transition: all .15s ease;
    }
    .chip-toggle:hover { color: var(--color-foreground); border-color: var(--color-border-strong); }
    .chip-toggle input { accent-color: var(--color-primary); }
    .chip-toggle:has(input:checked) {
      background: rgba(37,99,235,.10); border-color: var(--color-primary); color: var(--color-primary);
    }

    .summary { display: flex; align-items: center; gap: 16px; margin-bottom: 12px; padding: 10px 14px; background: var(--color-surface); border: 1px solid var(--color-border); border-radius: var(--radius-sm); }
    .lg { display: inline-flex; align-items: center; gap: 6px; font-size: 12px; color: var(--color-muted); }
    .spacer { flex: 1; }
    .sw { width: 14px; height: 3px; border-radius: 2px; display: inline-block; }
    .sw.o { background: var(--color-primary); }
    .sw.m { background: var(--color-muted-2); }
    .sw.r { background: var(--color-accent); }

    .cal-grid { display: grid; grid-template-columns: repeat(7, 1fr); gap: 1px; background: var(--color-border); padding: 1px; }
    .cal-head { background: var(--color-surface-2); padding: 8px 10px; font-size: 11px; text-transform: uppercase; letter-spacing: .08em; color: var(--color-muted); font-weight: 600; }

    .cell { background: var(--color-surface); padding: 10px; min-height: 150px; display: flex; flex-direction: column; gap: 6px; transition: background-color .12s ease; }
    .cell:hover { background: var(--color-surface-2); }
    .cell.empty { background: transparent; min-height: 80px; }
    .cell.empty.wide { grid-column: 1 / -1; min-height: 60px; display: flex; align-items: center; justify-content: center; font-size: 13px; padding: 18px; }
    .cell.weekend { background: linear-gradient(180deg, rgba(37,99,235,.04), var(--color-surface)); }

    .cell-top { display: flex; justify-content: space-between; align-items: center; gap: 4px; }
    .cal-day { font-size: 13px; font-weight: 700; color: var(--color-foreground); }

    .badge-action { font-size: 10px; font-weight: 700; padding: 2px 8px; border-radius: 6px; white-space: nowrap; }
    .badge-action.hold  { background: var(--color-success-soft);      color: var(--color-success); }
    .badge-action.raise { background: var(--color-warning-soft);       color: var(--color-warning); }
    .badge-action.lower { background: var(--color-destructive-soft);   color: var(--color-destructive); }

    .tag-row { display: flex; flex-wrap: wrap; gap: 3px; }
    .ctx-tag { font-size: 9px; padding: 1px 5px; border-radius: 3px; background: var(--color-surface-2); color: var(--color-muted); border: 1px solid var(--color-border); }

    .prices { display: flex; flex-direction: column; gap: 3px; flex: 1; }
    .price-row { display: flex; justify-content: space-between; align-items: baseline; font-size: 12px; }
    .price-row.muted .price-lbl, .price-row.muted .mono { color: var(--color-muted-2); }
    .price-row.suggested { border-top: 1px solid var(--color-border); padding-top: 4px; margin-top: 2px; }
    .price-row.suggested .price-lbl { color: var(--color-primary); font-weight: 600; }
    .price-row.suggested .mono { color: var(--color-primary); }
    .price-lbl { color: var(--color-muted); }
    .fw { font-weight: 700; color: var(--color-foreground); }

    .delta-bar { font-size: 11px; font-weight: 700; padding: 2px 8px; border-radius: 4px; text-align: center; margin-top: auto; }
    .delta-bar.hold  { background: var(--color-surface-2); color: var(--color-muted); }
    .delta-bar.raise { background: var(--color-warning-soft); color: var(--color-warning); }
    .delta-bar.lower { background: var(--color-destructive-soft); color: var(--color-destructive); }

    @media (max-width: 900px) { .cell { min-height: 120px; } }
  `],
})
export class ManagerCalendarComponent implements OnInit {
  private api = inject(ApiService);

  readonly label = label;

  private static readonly EMPTY_OPTIONS: CalendarOptionsDto = {
    room_base: [], room_view: [], room_tier: [], room_occupancy: [],
    boarding_canonical: [], nights: [], adults: [], scrape_dates: [],
    check_in_min: null, check_in_max: null,
    default: { room_base: null, room_view: null, room_tier: null, room_occupancy: null, boarding_canonical: null, nights: null, adults: null },
  };

  opt = signal<CalendarOptionsDto>(ManagerCalendarComponent.EMPTY_OPTIONS);
  rows = signal<CalendarRowDto[]>([]);
  loading = signal(true);
  error = signal<string | null>(null);

  filters = signal<CalendarFilters>({
    scrape_date: '', room_base: '', room_view: '', room_tier: '',
    room_occupancy: '', boarding_canonical: '', nights: null, adults: null,
    best_peer_granularity_used: 'any',
  });

  overlays = signal<Overlays>({
    is_ramadan: true,
    is_tunisia_public_holiday: true,
    is_tunisia_school_holiday: true,
    is_school_holiday_france: true,
    is_school_holiday_germany: false,
    is_school_holiday_uk: false,
  });

  private windowFrom = '';
  private windowTo = '';

  visible = computed(() => this.rows());

  avgGap = computed(() => {
    const rows = this.rows();
    if (!rows.length) return 0;
    const sum = rows.reduce((acc, p) =>
      acc + ((p.recommended_price_per_night - p.price_per_night) / p.price_per_night * 100), 0);
    return sum / rows.length;
  });

  spacers = computed(() => {
    const rows = this.rows();
    if (!rows.length) return [] as number[];
    const first = new Date(rows[0].check_in).getUTCDay();
    return Array.from({ length: (first + 6) % 7 }); // Monday-first
  });

  ngOnInit(): void {
    this.api.getCalendarOptions().subscribe({
      next: opts => {
        this.opt.set(opts);
        const d = opts.default;
        this.filters.set({
          scrape_date: '',
          room_base: d.room_base ?? '',
          room_view: d.room_view ?? '',
          room_tier: d.room_tier ?? '',
          room_occupancy: d.room_occupancy ?? '',
          boarding_canonical: d.boarding_canonical ?? '',
          nights: d.nights,
          adults: d.adults,
          best_peer_granularity_used: 'any',
        });
        this.computeWindow(opts);
        this.load();
      },
      error: () => { this.error.set('Could not load calendar options.'); this.loading.set(false); },
    });
  }

  /** A ~5-week window starting at the later of today / earliest available date. */
  private computeWindow(opts: CalendarOptionsDto): void {
    const today = isoDate(new Date());
    let from = opts.check_in_min && opts.check_in_min > today ? opts.check_in_min : today;
    if (opts.check_in_max && from > opts.check_in_max) from = opts.check_in_min ?? from;
    const toDate = new Date(from); toDate.setDate(toDate.getDate() + 34);
    let to = isoDate(toDate);
    if (opts.check_in_max && to > opts.check_in_max) to = opts.check_in_max;
    this.windowFrom = from;
    this.windowTo = to;
  }

  private load(): void {
    this.loading.set(true);
    this.error.set(null);
    const f = this.filters();
    const q: CalendarQuery = {
      check_in_from: this.windowFrom || undefined,
      check_in_to: this.windowTo || undefined,
      nights: f.nights ?? undefined,
      adults: f.adults ?? undefined,
      room_base: f.room_base,
      room_view: f.room_view,
      room_tier: f.room_tier,
      room_occupancy: f.room_occupancy,
      boarding_canonical: f.boarding_canonical,
      best_peer_granularity_used: f.best_peer_granularity_used === 'any' ? undefined : f.best_peer_granularity_used,
      scrape_date: f.scrape_date || undefined,
    };
    this.api.getCalendar(q).subscribe({
      next: rows => { this.rows.set(rows); this.loading.set(false); },
      error: () => { this.error.set('Could not load calendar data.'); this.rows.set([]); this.loading.set(false); },
    });
  }

  set<K extends keyof CalendarFilters>(key: K, value: CalendarFilters[K]) {
    this.filters.update(f => ({ ...f, [key]: value }));
    this.load();
  }
  toggle(key: keyof Overlays) {
    this.overlays.update(o => ({ ...o, [key]: !o[key] }));
  }
  reset() {
    const d = this.opt().default;
    this.filters.set({
      scrape_date: '',
      room_base: d.room_base ?? '',
      room_view: d.room_view ?? '',
      room_tier: d.room_tier ?? '',
      room_occupancy: d.room_occupancy ?? '',
      boarding_canonical: d.boarding_canonical ?? '',
      nights: d.nights,
      adults: d.adults,
      best_peer_granularity_used: 'any',
    });
    this.load();
  }

  /** Action badge derived from recommended vs current price. */
  badge(p: CalendarRowDto): { label: string; cls: string } {
    const delta = (p.recommended_price_per_night - p.price_per_night) / p.price_per_night;
    if (delta > 0.02)  return { label: '↑ Raise', cls: 'raise' };
    if (delta < -0.02) return { label: '↓ Lower', cls: 'lower' };
    return { label: '✓ Hold', cls: 'hold' };
  }

  /** Context tags shown inline in the date line. */
  tags(p: CalendarRowDto): string[] {
    const o = this.overlays();
    const t: string[] = [];
    if (p.is_weekend_checkin)                                     t.push('Weekend');
    if (o.is_tunisia_public_holiday && p.is_tunisia_public_holiday) t.push('TN Holiday');
    if (o.is_ramadan && p.is_ramadan)                             t.push('Ramadan');
    if (o.is_school_holiday_france && p.is_school_holiday_france) t.push('FR school');
    if (o.is_tunisia_school_holiday && p.is_tunisia_school_holiday) t.push('TN school');
    if (o.is_school_holiday_germany && p.is_school_holiday_germany) t.push('DE school');
    if (o.is_school_holiday_uk && p.is_school_holiday_uk)         t.push('UK school');
    return t;
  }

  /** Human-readable delta percentage string (e.g. "+5.2%" or "−3.1%"). */
  deltaPct(p: CalendarRowDto): string {
    const d = (p.recommended_price_per_night - p.price_per_night) / p.price_per_night * 100;
    return (d >= 0 ? '+' : '') + d.toFixed(1) + '%';
  }

  /** Tooltip for the granularity info (moved off the visible cell). */
  granularityTip(p: CalendarRowDto): string {
    if (!p.best_peer_granularity_used) return '';
    return `Market avg based on ${p.peer_medium_count ?? '?'} hotels (${p.best_peer_granularity_used} neighbourhood)`;
  }

  /** Combined school-holiday toggle state (FR + TN). */
  schoolActive = computed(() =>
    this.overlays().is_school_holiday_france || this.overlays().is_tunisia_school_holiday
  );

  toggleSchool() {
    const next = !this.schoolActive();
    this.overlays.update(o => ({
      ...o,
      is_school_holiday_france: next,
      is_tunisia_school_holiday: next,
    }));
  }
}
