import { Component, OnInit, computed, inject, signal } from '@angular/core';
import { ActivatedRoute, RouterLink } from '@angular/router';
import { ApiService } from '../../../core/api/api.service';
import { AdminCompetitorRowDto, AdminManagerDto, SelectableHotelDto } from '../../../core/api/dto';

@Component({
  selector: 'rw-admin-manager-competitors',
  standalone: true,
  imports: [RouterLink],
  template: `
    <div class="page-head">
      <div>
        <h1>Competitor selection</h1>
        <div class="sub">
          @if (manager(); as m) {
            For <strong>{{ m.full_name ?? m.email }}</strong>
            @if (cap() !== null) { · up to {{ cap() }} competitors (set on the manager's hotel assignment) }
          } @else {
            Admin-only — the manager's own competitors view stays read-only.
          }
        </div>
      </div>
      <a class="btn ghost" routerLink="/admin/managers">&larr; Back to managers</a>
    </div>

    @if (loading()) { <div class="card card-pad muted">Loading selection…</div> }
    @else if (error()) { <div class="card card-pad err">{{ error() }}</div> }
    @else {
      <div class="grid cols-2">
        <section class="card">
          <div class="card-head">
            <h3>Current working selection</h3>
            <span class="small muted">{{ workingRows().length }} @if (cap() !== null) { / {{ cap() }} }</span>
          </div>
          <div class="list">
            @for (r of workingRows(); track r.hotel_id; let i = $index) {
              <div class="row between item">
                <div>
                  <span class="mono small muted" style="margin-right:8px;">#{{ i + 1 }}</span>
                  <span style="font-weight:500;">{{ r.hotel_name_display }}</span>
                  <span class="small muted"> · {{ r.city_name }} · {{ r.stars_int ?? '—' }}★</span>
                </div>
                <button class="btn ghost sm danger" (click)="remove(r.hotel_id)">Remove</button>
              </div>
            } @empty {
              <div class="muted small" style="padding:14px;">No competitors selected yet — add some from the pool on the right.</div>
            }
          </div>
        </section>

        <section class="card">
          <div class="card-head"><h3>Add from pool</h3><span class="small muted">{{ addable().length }} available</span></div>
          <div class="card-pad">
            <div class="row" style="gap: 8px;">
              <select class="select" style="flex:1;" [value]="addHotelId()" (change)="addHotelId.set($any($event.target).value)">
                <option value="">— choose a hotel —</option>
                @for (s of addable(); track s.hotel_id) {
                  <option [value]="s.hotel_id">{{ s.hotel_name_display }} · {{ s.city_name }} · {{ s.stars_int ?? '—' }}★</option>
                }
              </select>
              <button class="btn" [disabled]="!addHotelId()" (click)="add()">+ Add</button>
            </div>
            @if (addable().length === 0) {
              <div class="muted small" style="margin-top: 10px;">Every active hotel is already in the working selection (or it's the manager's own hotel).</div>
            }
          </div>
        </section>
      </div>

      <section class="card card-pad" style="margin-top: 14px;">
        @if (saveError()) { <div class="small err" style="margin-bottom: 8px;">{{ saveError() }}</div> }
        @if (saveDone()) { <div class="muted small" style="margin-bottom: 8px;">Selection saved — {{ selection().length }} competitors persisted to user_competitor_selections.</div> }
        <div class="row" style="gap: 8px;">
          <button class="btn primary" [disabled]="saving() || !dirty()" (click)="save()">
            @if (saving()) { Saving… } @else { Save selection }
          </button>
          @if (dirty()) { <span class="small muted">Unsaved changes — order matters (display_order = position in this list).</span> }
        </div>
      </section>
    }
  `,
  styles: [`
    .list { padding: 8px; display: flex; flex-direction: column; }
    .item { padding: 10px 12px; border-radius: var(--radius-sm); }
    .item + .item { border-top: 1px solid var(--color-border); }
    .err { background: var(--color-destructive-soft); color: var(--color-destructive); }
  `],
})
export class AdminManagerCompetitorsComponent implements OnInit {
  private api = inject(ApiService);
  private route = inject(ActivatedRoute);

  managerId = signal('');
  manager = signal<AdminManagerDto | null>(null);
  cap = signal<number | null>(null);

  selection = signal<AdminCompetitorRowDto[]>([]);
  selectable = signal<SelectableHotelDto[]>([]);
  // Ordered hotel_ids the admin is building — initialized from the saved
  // selection, then mutated locally until "Save selection" round-trips it.
  working = signal<number[]>([]);

  loading = signal(true);
  error = signal<string | null>(null);

  saving = signal(false);
  saveError = signal<string | null>(null);
  saveDone = signal(false);

  addHotelId = signal('');

  // Resolve display fields for each working hotel_id from whichever source
  // (saved selection or selectable pool) currently has them — both carry the
  // same hotel_name_display/city_name/stars_int.
  workingRows = computed<AdminCompetitorRowDto[]>(() => {
    const sel = this.selection();
    const pool = this.selectable();
    return this.working().map((id, i) => {
      const found = sel.find(r => r.hotel_id === id) ?? pool.find(r => r.hotel_id === id);
      return {
        hotel_id: id,
        hotel_name_display: found?.hotel_name_display ?? `Hotel #${id}`,
        city_name: found?.city_name ?? '—',
        stars_int: found?.stars_int ?? null,
        display_order: i + 1,
      };
    });
  });

  // Selectable hotels not already in the working list (the API excludes only
  // the manager's own hotel, not current picks — dedupe client-side).
  addable = computed(() => {
    const inWorking = new Set(this.working());
    return this.selectable().filter(s => !inWorking.has(s.hotel_id));
  });

  dirty = computed(() => {
    const saved = this.selection().map(r => r.hotel_id);
    const cur = this.working();
    return saved.length !== cur.length || saved.some((id, i) => id !== cur[i]);
  });

  ngOnInit(): void {
    const id = this.route.snapshot.paramMap.get('id') ?? '';
    this.managerId.set(id);
    this.loadAll(id);
  }

  private loadAll(id: string): void {
    this.loading.set(true);
    this.error.set(null);
    this.api.getAdminManagers().subscribe({
      next: managers => {
        this.manager.set(managers.find(m => m.id === id) ?? null);
        this.api.getAssignments().subscribe({
          next: assignments => {
            const own = assignments.find(a => a.user_id === id);
            this.cap.set(own?.max_competitors ?? null);
            this.api.getManagerCompetitors(id).subscribe({
              next: selection => {
                this.selection.set(selection);
                this.working.set(selection.map(r => r.hotel_id));
                this.api.getSelectableCompetitors(id).subscribe({
                  next: selectable => { this.selectable.set(selectable); this.loading.set(false); },
                  error: () => { this.error.set('Could not load selectable hotels.'); this.loading.set(false); },
                });
              },
              error: () => { this.error.set('Could not load the current selection.'); this.loading.set(false); },
            });
          },
          error: () => { this.error.set('Could not load the assignment.'); this.loading.set(false); },
        });
      },
      error: () => { this.error.set('Could not load the manager.'); this.loading.set(false); },
    });
  }

  add(): void {
    const idStr = this.addHotelId();
    if (!idStr) { return; }
    const hotelId = +idStr;
    if (this.working().includes(hotelId)) { return; }
    this.working.update(w => [...w, hotelId]);
    this.addHotelId.set('');
    this.saveDone.set(false);
    this.saveError.set(null);
  }

  remove(hotelId: number): void {
    this.working.update(w => w.filter(id => id !== hotelId));
    this.saveDone.set(false);
    this.saveError.set(null);
  }

  save(): void {
    this.saving.set(true);
    this.saveError.set(null);
    this.saveDone.set(false);
    this.api.setManagerCompetitors(this.managerId(), this.working()).subscribe({
      next: rows => {
        this.saving.set(false);
        this.saveDone.set(true);
        this.selection.set(rows);
        this.working.set(rows.map(r => r.hotel_id));
      },
      // Surface the backend's exact validation message — e.g. "At most {cap}
      // competitors allowed" / "A manager cannot select their own hotel as a
      // competitor" / "Manager has no active hotel assignment; assign a hotel
      // first" — these tell the admin precisely what to fix.
      error: err => {
        this.saving.set(false);
        this.saveError.set(err?.error?.detail ?? 'Could not save the selection.');
      },
    });
  }
}
