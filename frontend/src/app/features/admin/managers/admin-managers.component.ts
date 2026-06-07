import { Component, OnInit, computed, inject, signal } from '@angular/core';
import { DatePipe } from '@angular/common';
import { StatusPillComponent } from '../../../shared/components/status-pill/status-pill.component';
import { ApiService } from '../../../core/api/api.service';
import { AdminManagerDto, ManagerCreateBody, ManagerUpdateBody } from '../../../core/api/dto';

@Component({
  selector: 'rw-admin-managers',
  standalone: true,
  imports: [DatePipe, StatusPillComponent],
  template: `
    <div class="page-head">
      <div>
        <h1>Managers</h1>
        <div class="sub">{{ managers().length }} accounts · {{ assigned() }} assigned to a hotel</div>
      </div>
      <button class="btn primary" (click)="toggleCreate()">+ New manager</button>
    </div>

    @if (loading()) { <div class="card card-pad muted">Loading managers…</div> }
    @else if (error()) { <div class="card card-pad err">{{ error() }}</div> }

    @if (createOpen()) {
      <section class="card card-pad">
        <div class="row between" style="margin-bottom: 8px;">
          <h2 style="margin:0;">New manager</h2>
          <button class="btn ghost" (click)="toggleCreate()">Close</button>
        </div>
        <div class="sub" style="margin-bottom: 12px;">
          The admin sets the manager's first password directly — there is no email-invite flow.
        </div>
        <div class="grid cols-3" style="gap: 12px;">
          <label class="field">
            email
            <input class="input" type="email" [value]="createEmail()" (input)="createEmail.set($any($event.target).value)" />
          </label>
          <label class="field">
            full name
            <input class="input" type="text" [value]="createFullName()" (input)="createFullName.set($any($event.target).value)" />
          </label>
          <label class="field">
            initial password
            <input class="input" type="text" [value]="createPassword()" (input)="createPassword.set($any($event.target).value)" />
          </label>
        </div>
        @if (createError()) { <div class="small err" style="margin-top: 8px;">{{ createError() }}</div> }
        <div class="row" style="margin-top: 12px; gap: 8px;">
          <button class="btn primary" [disabled]="creating()" (click)="submitCreate()">
            @if (creating()) { Creating… } @else { Create manager }
          </button>
        </div>
      </section>
    }

    <section class="card">
      <table class="tbl">
        <thead>
          <tr>
            <th>Name</th><th>Email</th><th>Hotel</th><th>Last seen</th><th></th>
          </tr>
        </thead>
        <tbody>
          @for (m of managers(); track m.id) {
            <tr>
              <td>
                <div class="row" style="gap:10px;">
                  <span class="avatar mono">{{ initials(m.full_name ?? m.email) }}</span>
                  <span style="font-weight:500;">{{ m.full_name ?? '—' }}</span>
                  @if (!m.is_active) { <rw-status-pill tone="muted">inactive</rw-status-pill> }
                </div>
              </td>
              <td class="muted small">{{ m.email }}</td>
              <td>
                @if (m.assigned_hotel_id !== null) { <span>{{ m.assigned_hotel_name }}</span> }
                @else { <rw-status-pill tone="warn">unassigned</rw-status-pill> }
              </td>
              <td class="small muted">
                @if (m.last_login_at) { {{ m.last_login_at | date:'MMM d, HH:mm' }} }
                @else { <span>never</span> }
              </td>
              <td class="num">
                <div class="row" style="gap: 6px; justify-content:flex-end;">
                  <button class="btn ghost sm" (click)="toggleEdit(m)">Edit</button>
                  <button class="btn ghost sm" (click)="toggleReset(m)">Reset password</button>
                </div>
              </td>
            </tr>
            @if (editId() === m.id) {
              <tr>
                <td colspan="5">
                  <div class="card card-pad" style="margin: 4px 0;">
                    <div class="grid cols-3" style="gap: 12px;">
                      <label class="field">
                        email
                        <input class="input" type="email" [value]="editEmail()" (input)="editEmail.set($any($event.target).value)" />
                      </label>
                      <label class="field">
                        full name
                        <input class="input" type="text" [value]="editFullName()" (input)="editFullName.set($any($event.target).value)" />
                      </label>
                      <label class="small row" style="gap: 6px; align-items:center; margin-top: 18px;">
                        <input type="checkbox" [checked]="editActive()" (change)="editActive.set($any($event.target).checked)" />
                        active
                      </label>
                    </div>
                    @if (editError()) { <div class="small err" style="margin-top: 8px;">{{ editError() }}</div> }
                    <div class="row" style="margin-top: 12px; gap: 8px;">
                      <button class="btn primary" [disabled]="saving()" (click)="submitEdit(m)">
                        @if (saving()) { Saving… } @else { Save changes }
                      </button>
                      <button class="btn ghost" (click)="toggleEdit(m)">Cancel</button>
                    </div>
                  </div>
                </td>
              </tr>
            }
            @if (resetId() === m.id) {
              <tr>
                <td colspan="5">
                  <div class="card card-pad" style="margin: 4px 0;">
                    <div class="grid cols-3" style="gap: 12px;">
                      <label class="field">
                        new password
                        <input class="input" type="text" [value]="resetPassword()" (input)="resetPassword.set($any($event.target).value)" />
                      </label>
                    </div>
                    @if (resetError()) { <div class="small err" style="margin-top: 8px;">{{ resetError() }}</div> }
                    @if (resetDone()) { <div class="muted small" style="margin-top: 8px;">Password updated.</div> }
                    <div class="row" style="margin-top: 12px; gap: 8px;">
                      <button class="btn primary" [disabled]="resetting()" (click)="submitReset(m)">
                        @if (resetting()) { Resetting… } @else { Set new password }
                      </button>
                      <button class="btn ghost" (click)="toggleReset(m)">Cancel</button>
                    </div>
                  </div>
                </td>
              </tr>
            }
          }
        </tbody>
      </table>
    </section>
  `,
  styles: [`
    .avatar { width: 28px; height: 28px; border-radius: 50%; background: var(--color-surface-2); color: var(--color-muted); font-size: 11px; font-weight: 600; display: inline-flex; align-items: center; justify-content: center; }
    .err { background: var(--color-destructive-soft); color: var(--color-destructive); }
  `],
})
export class AdminManagersComponent implements OnInit {
  private api = inject(ApiService);

  managers = signal<AdminManagerDto[]>([]);
  loading = signal(true);
  error = signal<string | null>(null);

  assigned = computed(() => this.managers().filter(m => m.assigned_hotel_id !== null).length);

  // --- Create ---
  createOpen = signal(false);
  createEmail = signal('');
  createFullName = signal('');
  createPassword = signal('');
  creating = signal(false);
  createError = signal<string | null>(null);

  // --- Edit ---
  editId = signal<string | null>(null);
  editEmail = signal('');
  editFullName = signal('');
  editActive = signal(true);
  saving = signal(false);
  editError = signal<string | null>(null);

  // --- Reset password ---
  resetId = signal<string | null>(null);
  resetPassword = signal('');
  resetting = signal(false);
  resetError = signal<string | null>(null);
  resetDone = signal(false);
  private resetDoneTimer: ReturnType<typeof setTimeout> | null = null;

  ngOnInit(): void {
    this.loadManagers();
  }

  private loadManagers(): void {
    this.loading.set(true);
    this.error.set(null);
    this.api.getAdminManagers().subscribe({
      next: rows => { this.managers.set(rows); this.loading.set(false); },
      error: () => { this.error.set('Could not load managers.'); this.loading.set(false); },
    });
  }

  initials(n: string): string {
    return n.split(/[\s.@]+/).filter(Boolean).map(p => p[0]).slice(0, 2).join('').toUpperCase();
  }

  // --- Create ---
  toggleCreate(): void {
    const next = !this.createOpen();
    this.createOpen.set(next);
    if (next) {
      this.createEmail.set('');
      this.createFullName.set('');
      this.createPassword.set('');
      this.createError.set(null);
    }
  }

  submitCreate(): void {
    const body: ManagerCreateBody = {
      email: this.createEmail().trim(),
      full_name: this.createFullName().trim() || null,
      initial_password: this.createPassword(),
    };
    if (!body.email || !body.initial_password) {
      this.createError.set('Email and initial password are required.');
      return;
    }
    this.creating.set(true);
    this.createError.set(null);
    this.api.createManager(body).subscribe({
      next: () => {
        this.creating.set(false);
        this.createOpen.set(false);
        this.loadManagers();
      },
      error: () => {
        this.creating.set(false);
        this.createError.set('Could not create manager — the email may already be in use.');
      },
    });
  }

  // --- Edit ---
  toggleEdit(m: AdminManagerDto): void {
    if (this.editId() === m.id) {
      this.editId.set(null);
      return;
    }
    this.editId.set(m.id);
    this.editEmail.set(m.email);
    this.editFullName.set(m.full_name ?? '');
    this.editActive.set(m.is_active);
    this.editError.set(null);
    this.resetId.set(null);
  }

  submitEdit(m: AdminManagerDto): void {
    const body: ManagerUpdateBody = {
      email: this.editEmail().trim(),
      full_name: this.editFullName().trim() || null,
      is_active: this.editActive(),
    };
    if (!body.email) {
      this.editError.set('Email is required.');
      return;
    }
    this.saving.set(true);
    this.editError.set(null);
    this.api.updateManager(m.id, body).subscribe({
      next: () => {
        this.saving.set(false);
        this.editId.set(null);
        this.loadManagers();
      },
      error: () => {
        this.saving.set(false);
        this.editError.set('Could not save changes — the email may already be in use.');
      },
    });
  }

  // --- Reset password ---
  toggleReset(m: AdminManagerDto): void {
    if (this.resetId() === m.id) {
      this.resetId.set(null);
      return;
    }
    this.resetId.set(m.id);
    this.resetPassword.set('');
    this.resetError.set(null);
    this.resetDone.set(false);
    this.editId.set(null);
    if (this.resetDoneTimer) { clearTimeout(this.resetDoneTimer); this.resetDoneTimer = null; }
  }

  submitReset(m: AdminManagerDto): void {
    const pwd = this.resetPassword();
    if (!pwd) {
      this.resetError.set('Enter a new password.');
      return;
    }
    this.resetting.set(true);
    this.resetError.set(null);
    this.resetDone.set(false);
    this.api.resetManagerPassword(m.id, pwd).subscribe({
      next: () => {
        this.resetting.set(false);
        this.resetPassword.set('');
        this.resetDone.set(true);
        if (this.resetDoneTimer) { clearTimeout(this.resetDoneTimer); }
        this.resetDoneTimer = setTimeout(() => { this.resetDone.set(false); this.resetId.set(null); }, 2000);
      },
      error: () => {
        this.resetting.set(false);
        this.resetError.set('Could not reset the password.');
      },
    });
  }
}
