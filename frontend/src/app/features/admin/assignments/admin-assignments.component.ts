import { Component, OnInit, computed, inject, signal } from '@angular/core';
import { ApiService } from '../../../core/api/api.service';
import { AdminHotelDto, AdminManagerDto, AssignmentDto } from '../../../core/api/dto';

@Component({
  selector: 'rw-admin-assignments',
  standalone: true,
  template: `
    <div class="page-head">
      <div>
        <h1>Assignments</h1>
        <div class="sub">Map managers to the hotels they own. Manager authorization is enforced on every API call. Each assign/unassign saves immediately — there is no separate "save" step.</div>
      </div>
    </div>

    @if (loading()) { <div class="card card-pad muted">Loading assignments…</div> }
    @else if (error()) { <div class="card card-pad err">{{ error() }}</div> }

    <div class="grid cols-2">
      <section class="card">
        <div class="card-head"><h3>Unassigned hotels</h3><span class="small muted">{{ unassigned().length }}</span></div>
        <div class="list">
          @for (h of unassigned(); track h.id) {
            <div class="item">
              <div class="row between">
                <div>
                  <div style="font-weight:500;">{{ h.hotel_name_display }}</div>
                  <div class="small muted">{{ h.city_name }} · {{ h.stars_int ?? '—' }}★</div>
                </div>
                <button class="btn sm" (click)="toggleAssign(h)">
                  @if (assignHotelId() === h.id) { Cancel } @else { Assign… }
                </button>
              </div>
              @if (assignHotelId() === h.id) {
                <div class="card card-pad" style="margin-top: 10px;">
                  <div class="grid cols-2" style="gap: 12px;">
                    <label class="field">
                      manager
                      <select class="select" [value]="assignManagerId()" (change)="assignManagerId.set($any($event.target).value)">
                        <option value="">— choose a manager —</option>
                        @for (m of availableManagers(); track m.id) {
                          <option [value]="m.id">{{ m.full_name ?? m.email }}</option>
                        }
                      </select>
                    </label>
                    <label class="field">
                      max competitors <span class="hint">(default 4)</span>
                      <input class="input" type="number" min="1"
                             [value]="assignMaxCompetitors() ?? ''"
                             (input)="assignMaxCompetitors.set($any($event.target).value ? +$any($event.target).value : null)" />
                    </label>
                  </div>
                  @if (availableManagers().length === 0) {
                    <div class="small muted" style="margin-top: 8px;">No unassigned managers available — every active manager already owns a hotel.</div>
                  }
                  @if (assignError()) { <div class="small err" style="margin-top: 8px;">{{ assignError() }}</div> }
                  <div class="row" style="margin-top: 12px; gap: 8px;">
                    <button class="btn primary" [disabled]="assigning() || !assignManagerId()" (click)="submitAssign(h)">
                      @if (assigning()) { Assigning… } @else { Confirm assignment }
                    </button>
                  </div>
                </div>
              }
            </div>
          } @empty {
            <div class="muted small" style="padding:14px;">All active hotels are assigned.</div>
          }
        </div>
      </section>

      <section class="card">
        <div class="card-head"><h3>Current assignments</h3><span class="small muted">{{ assigned().length }}</span></div>
        <table class="tbl">
          <thead><tr><th>Manager</th><th>Hotel</th><th>Max competitors</th><th></th></tr></thead>
          <tbody>
            @for (a of assigned(); track a.id) {
              <tr>
                <td>{{ a.manager_name ?? a.manager_email }}</td>
                <td>{{ a.hotel_name }}</td>
                <td class="mono">{{ a.max_competitors }}</td>
                <td class="num">
                  <button class="btn ghost sm danger" [disabled]="unassigningId() === a.id" (click)="unassign(a)">
                    @if (unassigningId() === a.id) { Removing… } @else { Unassign }
                  </button>
                </td>
              </tr>
            } @empty {
              <tr><td colspan="4" class="muted small" style="padding:14px;">No assignments yet.</td></tr>
            }
          </tbody>
        </table>
        @if (unassignError()) { <div class="small err" style="padding: 8px 14px;">{{ unassignError() }}</div> }
      </section>
    </div>
  `,
  styles: [`
    .list { padding: 8px; display: flex; flex-direction: column; gap: 4px; }
    .item { padding: 10px 12px; border-radius: var(--radius-sm); }
    .item + .item { border-top: 1px solid var(--color-border); }
    .err { background: var(--color-destructive-soft); color: var(--color-destructive); }
  `],
})
export class AdminAssignmentsComponent implements OnInit {
  private api = inject(ApiService);

  hotels = signal<AdminHotelDto[]>([]);
  assignments = signal<AssignmentDto[]>([]);
  managers = signal<AdminManagerDto[]>([]);
  loading = signal(true);
  error = signal<string | null>(null);

  // A hotel counts as unassigned by manager_id, the field the API computes
  // server-side from user_hotel_assignments — no client-side cross-referencing.
  unassigned = computed(() => this.hotels().filter(h => h.is_active && h.manager_id === null));
  assigned = computed(() => this.assignments());
  // A manager owns at most one hotel — only offer ones without one.
  availableManagers = computed(() => this.managers().filter(m => m.assigned_hotel_id === null && m.is_active));

  // --- Assign ---
  assignHotelId = signal<number | null>(null);
  assignManagerId = signal<string>('');
  assignMaxCompetitors = signal<number | null>(null);
  assigning = signal(false);
  assignError = signal<string | null>(null);

  // --- Unassign ---
  unassigningId = signal<number | null>(null);
  unassignError = signal<string | null>(null);

  ngOnInit(): void {
    this.loadAll();
  }

  private loadAll(): void {
    this.loading.set(true);
    this.error.set(null);
    this.api.getAdminHotels().subscribe({
      next: hotels => {
        this.hotels.set(hotels);
        this.api.getAssignments().subscribe({
          next: assignments => {
            this.assignments.set(assignments);
            this.api.getAdminManagers().subscribe({
              next: managers => { this.managers.set(managers); this.loading.set(false); },
              error: () => { this.error.set('Could not load managers.'); this.loading.set(false); },
            });
          },
          error: () => { this.error.set('Could not load assignments.'); this.loading.set(false); },
        });
      },
      error: () => { this.error.set('Could not load hotels.'); this.loading.set(false); },
    });
  }

  // --- Assign ---
  toggleAssign(h: AdminHotelDto): void {
    if (this.assignHotelId() === h.id) {
      this.assignHotelId.set(null);
      return;
    }
    this.assignHotelId.set(h.id);
    this.assignManagerId.set('');
    this.assignMaxCompetitors.set(null);
    this.assignError.set(null);
  }

  submitAssign(h: AdminHotelDto): void {
    const userId = this.assignManagerId();
    if (!userId) {
      this.assignError.set('Choose a manager first.');
      return;
    }
    this.assigning.set(true);
    this.assignError.set(null);
    const body: { user_id: string; hotel_id: number; max_competitors?: number } = {
      user_id: userId,
      hotel_id: h.id,
    };
    const cap = this.assignMaxCompetitors();
    if (cap !== null) { body.max_competitors = cap; }
    this.api.createAssignment(body).subscribe({
      next: () => {
        this.assigning.set(false);
        this.assignHotelId.set(null);
        this.loadAll();
      },
      error: err => {
        this.assigning.set(false);
        this.assignError.set(err?.error?.detail ?? 'Could not create the assignment.');
      },
    });
  }

  // --- Unassign ---
  unassign(a: AssignmentDto): void {
    this.unassigningId.set(a.id);
    this.unassignError.set(null);
    this.api.deleteAssignment(a.id).subscribe({
      next: () => {
        this.unassigningId.set(null);
        this.loadAll();
      },
      error: err => {
        this.unassigningId.set(null);
        this.unassignError.set(err?.error?.detail ?? 'Could not remove the assignment.');
      },
    });
  }
}
