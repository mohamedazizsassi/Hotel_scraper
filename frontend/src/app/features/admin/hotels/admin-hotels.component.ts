import { Component, OnInit, computed, inject, signal } from '@angular/core';
import { DatePipe } from '@angular/common';
import { StatusPillComponent } from '../../../shared/components/status-pill/status-pill.component';
import { ApiService } from '../../../core/api/api.service';
import { AdminHotelDto, DiscoverableHotelDto } from '../../../core/api/dto';

const KNOWN_SOURCES = ['promohotel', 'tunisiepromo'] as const;

function titleCase(s: string): string {
  return s
    .split(/[\s_-]+/)
    .filter(Boolean)
    .map(w => w.charAt(0).toUpperCase() + w.slice(1))
    .join(' ');
}

@Component({
  selector: 'rw-admin-hotels',
  standalone: true,
  imports: [StatusPillComponent, DatePipe],
  template: `
    <div class="page-head">
      <div>
        <h1>Hotels</h1>
        <div class="sub">{{ filtered().length }} of {{ hotels().length }} hotels in the pool</div>
      </div>
      <div class="row">
        <input class="input" style="width:240px;" type="search" placeholder="Search by name or city…" (input)="onSearch($any($event.target).value)" />
        <select class="select" style="width:140px;" (change)="onCity($any($event.target).value)">
          <option value="">All cities</option>
          @for (c of cities(); track c) { <option [value]="c">{{ c }}</option> }
        </select>
        <button class="btn primary" (click)="togglePanel()">+ Add hotel</button>
      </div>
    </div>

    @if (loading()) { <div class="card card-pad muted">Loading hotels…</div> }
    @else if (error()) { <div class="card card-pad err">{{ error() }}</div> }

    @if (panelOpen()) {
      <section class="card card-pad">
        <div class="row between" style="margin-bottom: 8px;">
          <h2 style="margin:0;">Discover &amp; promote a hotel</h2>
          <button class="btn ghost" (click)="togglePanel()">Close</button>
        </div>
        <div class="sub" style="margin-bottom: 12px;">
          Candidates are distinct hotels seen in <span class="mono">hotel_features</span> that are not yet
          in the platform pool. They carry no display name or source — supply both before promoting.
        </div>

        @if (discoverLoading()) { <div class="muted small">Loading discoverable candidates…</div> }
        @else if (discoverError()) { <div class="small err">{{ discoverError() }}</div> }
        @else if (discoverable().length === 0) {
          <div class="muted small">No undiscovered candidates — every known hotel is already in the pool.</div>
        } @else {
          <table class="tbl">
            <thead>
              <tr>
                <th>Candidate</th><th>City</th><th>Stars</th><th>Display name</th><th>Sources</th><th></th>
              </tr>
            </thead>
            <tbody>
              @for (d of discoverable(); track d.hotel_name_normalized + d.city_name) {
                <tr>
                  <td><span class="mono small">{{ d.hotel_name_normalized }}</span></td>
                  <td>{{ d.city_name }}</td>
                  <td><span class="mono">{{ d.stars_int ?? '—' }}{{ d.stars_int ? '★' : '' }}</span></td>
                  <td>
                    <input class="input" style="width:200px;"
                           [value]="displayName(d)"
                           (input)="setDisplayName(d, $any($event.target).value)" />
                  </td>
                  <td>
                    <div class="row" style="gap: 8px;">
                      @for (src of knownSources; track src) {
                        <label class="small row" style="gap: 4px; align-items:center;">
                          <input type="checkbox"
                                 [checked]="isSourceChecked(d, src)"
                                 (change)="toggleSource(d, src, $any($event.target).checked)" />
                          {{ src }}
                        </label>
                      }
                    </div>
                  </td>
                  <td>
                    <button class="btn primary" [disabled]="promoting() === key(d)" (click)="promote(d)">
                      @if (promoting() === key(d)) { Promoting… } @else { Promote }
                    </button>
                  </td>
                </tr>
              }
            </tbody>
          </table>
        }
        @if (promoteError()) { <div class="small err" style="margin-top: 8px;">{{ promoteError() }}</div> }
      </section>
    }

    <section class="card">
      <table class="tbl">
        <thead>
          <tr>
            <th>Hotel</th><th>City</th><th>Region</th><th>Stars</th><th>Source</th>
            <th>Manager</th><th>Latest scrape</th><th>Status</th>
          </tr>
        </thead>
        <tbody>
          @for (h of filtered(); track h.id) {
            <tr>
              <td>
                <div class="hname">{{ h.hotel_name_display }}</div>
                <div class="muted small mono">{{ h.hotel_name_normalized }}</div>
              </td>
              <td>{{ h.city_name }}</td>
              <td>{{ h.region ?? '—' }}</td>
              <td><span class="mono">{{ h.stars_int ?? '—' }}{{ h.stars_int ? '★' : '' }}</span></td>
              <td>
                @if (h.sources) {
                  @for (s of h.sources.split(','); track s) { <span class="badge info">{{ s }}</span> }
                } @else {
                  <span class="muted small">—</span>
                }
              </td>
              <td>
                @if (h.manager_id !== null) {
                  <span class="small">{{ h.manager_name }}</span>
                } @else {
                  <span class="muted small">— unassigned —</span>
                }
              </td>
              <td>
                @if (h.latest_scraped_at) {
                  <span class="small mono">{{ h.latest_scraped_at | date:'MMM d, HH:mm' }}</span>
                } @else {
                  <span class="muted small">—</span>
                }
              </td>
              <td>
                @if (h.is_active) { <rw-status-pill tone="ok">active</rw-status-pill> }
                @else { <rw-status-pill tone="muted">paused</rw-status-pill> }
              </td>
            </tr>
          }
        </tbody>
      </table>
    </section>
  `,
  styles: [`
    .hname { font-weight: 500; }
    .err { background: var(--color-destructive-soft); color: var(--color-destructive); }
  `],
})
export class AdminHotelsComponent implements OnInit {
  private api = inject(ApiService);

  hotels = signal<AdminHotelDto[]>([]);
  loading = signal(true);
  error = signal<string | null>(null);

  q = signal('');
  city = signal('');
  cities = computed(() => Array.from(new Set(this.hotels().map(h => h.city_name))).sort());

  filtered = computed(() => {
    const q = this.q().toLowerCase();
    const city = this.city();
    return this.hotels().filter(h =>
      (!q || h.hotel_name_display.toLowerCase().includes(q) || h.city_name.toLowerCase().includes(q)) &&
      (!city || h.city_name === city),
    );
  });

  // --- Discovery → promote panel ---
  knownSources = KNOWN_SOURCES;
  panelOpen = signal(false);
  discoverable = signal<DiscoverableHotelDto[]>([]);
  discoverLoading = signal(false);
  discoverError = signal<string | null>(null);
  promoting = signal<string | null>(null);
  promoteError = signal<string | null>(null);

  /** Per-candidate edit state, keyed by `${hotel_name_normalized}::${city_name}`. */
  private displayNames = signal<Record<string, string>>({});
  private selectedSources = signal<Record<string, Set<string>>>({});

  ngOnInit(): void {
    this.loadHotels();
  }

  private loadHotels(): void {
    this.loading.set(true);
    this.error.set(null);
    this.api.getAdminHotels().subscribe({
      next: rows => { this.hotels.set(rows); this.loading.set(false); },
      error: () => { this.error.set('Could not load hotels.'); this.loading.set(false); },
    });
  }

  onSearch(v: string) { this.q.set(v); }
  onCity(v: string) { this.city.set(v); }

  key(d: DiscoverableHotelDto): string {
    return `${d.hotel_name_normalized}::${d.city_name}`;
  }

  togglePanel(): void {
    const next = !this.panelOpen();
    this.panelOpen.set(next);
    if (next && this.discoverable().length === 0 && !this.discoverLoading()) {
      this.loadDiscoverable();
    }
  }

  private loadDiscoverable(): void {
    this.discoverLoading.set(true);
    this.discoverError.set(null);
    this.api.getDiscoverableHotels().subscribe({
      next: rows => { this.discoverable.set(rows); this.discoverLoading.set(false); },
      error: () => { this.discoverError.set('Could not load discoverable hotels.'); this.discoverLoading.set(false); },
    });
  }

  displayName(d: DiscoverableHotelDto): string {
    const k = this.key(d);
    return this.displayNames()[k] ?? titleCase(d.hotel_name_normalized);
  }

  setDisplayName(d: DiscoverableHotelDto, value: string): void {
    this.displayNames.update(m => ({ ...m, [this.key(d)]: value }));
  }

  isSourceChecked(d: DiscoverableHotelDto, src: string): boolean {
    return this.selectedSources()[this.key(d)]?.has(src) ?? false;
  }

  toggleSource(d: DiscoverableHotelDto, src: string, checked: boolean): void {
    const k = this.key(d);
    this.selectedSources.update(m => {
      const set = new Set(m[k] ?? []);
      if (checked) { set.add(src); } else { set.delete(src); }
      return { ...m, [k]: set };
    });
  }

  promote(d: DiscoverableHotelDto): void {
    const k = this.key(d);
    this.promoteError.set(null);
    this.promoting.set(k);
    const sources = Array.from(this.selectedSources()[k] ?? []);
    this.api.promoteHotel({
      hotel_name_normalized: d.hotel_name_normalized,
      hotel_name_display: this.displayName(d),
      city_name: d.city_name,
      stars_int: d.stars_int,
      sources,
    }).subscribe({
      next: () => {
        this.promoting.set(null);
        this.discoverable.update(rows => rows.filter(r => this.key(r) !== k));
        this.loadHotels();
      },
      error: () => {
        this.promoting.set(null);
        this.promoteError.set(`Could not promote "${d.hotel_name_normalized}" — it may already be registered.`);
      },
    });
  }
}
