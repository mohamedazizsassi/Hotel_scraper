import { Component, OnInit, inject, signal } from '@angular/core';
import { Router } from '@angular/router';
import { ApiService } from '../../../core/api/api.service';
import { AuthService } from '../../../core/auth/auth.service';
import { ManagerProfileDto } from '../../../core/api/dto';

interface AlertPrefs {
  competitor_undercut: boolean;
  price_spike: boolean;
  anomaly_digest: boolean;
  data_quality: boolean;
}

@Component({
  selector: 'rw-manager-settings',
  standalone: true,
  template: `
    <div class="page-head">
      <div>
        <h1>Settings</h1>
        <div class="sub">Account, hotel profile, and alert preferences.</div>
      </div>
      <button class="btn primary" (click)="save()" [disabled]="saving()">
        {{ saving() ? 'Saving…' : 'Save changes' }}
      </button>
    </div>

    <div class="grid cols-2">
      <section class="card">
        <div class="card-head"><h3>Profile</h3></div>
        <div class="card-body col">
          <div class="field">
            <label>Full name</label>
            <input class="input" [value]="fullName()" (input)="fullName.set($any($event.target).value)" />
          </div>
          <div class="field">
            <label>Email</label>
            <input class="input" [value]="profile()?.email" disabled />
          </div>
          <div class="field">
            <label>Language</label>
            <select class="select" [value]="language()" (change)="language.set($any($event.target).value)">
              <option value="en">English</option>
              <option value="fr">Français</option>
            </select>
          </div>
          <div class="row" style="gap:8px; align-items:center;">
            @if (saved()) { <span class="muted small">Saved.</span> }
            @if (error()) { <span class="muted small" style="color:var(--color-destructive)">{{ error() }}</span> }
          </div>
        </div>
      </section>

      <section class="card">
        <div class="card-head"><h3>Hotel</h3></div>
        <div class="card-body col">
          <div class="field"><label>Hotel</label><input class="input" [value]="profile()?.hotel?.hotel_name_display" disabled /></div>
          <div class="field"><label>City</label><input class="input" [value]="profile()?.hotel?.city_name" disabled /></div>
          <div class="row" style="gap:12px;">
            <div class="field" style="flex:1"><label>Stars</label><input class="input mono" [value]="profile()?.hotel?.stars_int" disabled /></div>
          </div>
          <p class="muted small">Hotel profile is managed by your admin.</p>
        </div>
      </section>

      <section class="card">
        <div class="card-head"><h3>Alert preferences</h3></div>
        <div class="card-body col">
          <label class="toggle">
            <input type="checkbox" [checked]="alerts().competitor_undercut" (change)="toggleAlert('competitor_undercut')" />
            <span class="track"><span class="thumb"></span></span>
            <span class="t-text">
              <span class="t-title">Competitor undercut</span>
              <span class="muted small">Notify when a competitor drops &gt; 8% below your price</span>
            </span>
          </label>
          <label class="toggle">
            <input type="checkbox" [checked]="alerts().price_spike" (change)="toggleAlert('price_spike')" />
            <span class="track"><span class="thumb"></span></span>
            <span class="t-text">
              <span class="t-title">Price spike</span>
              <span class="muted small">Notify when your price is &gt; 10% above market average</span>
            </span>
          </label>
          <label class="toggle">
            <input type="checkbox" [checked]="alerts().anomaly_digest" (change)="toggleAlert('anomaly_digest')" />
            <span class="track"><span class="thumb"></span></span>
            <span class="t-text">
              <span class="t-title">Anomaly detection</span>
              <span class="muted small">Daily anomaly digest from the forecaster</span>
            </span>
          </label>
          <label class="toggle">
            <input type="checkbox" [checked]="alerts().data_quality" (change)="toggleAlert('data_quality')" />
            <span class="track"><span class="thumb"></span></span>
            <span class="t-text">
              <span class="t-title">Data quality</span>
              <span class="muted small">Scraper coverage drops or partial runs</span>
            </span>
          </label>
        </div>
      </section>

      <section class="card">
        <div class="card-head"><h3>Danger zone</h3></div>
        <div class="card-body col">
          <p class="muted small">Disconnect your account from this hotel. Your admin must reassign you.</p>
          <button class="btn danger" style="align-self:flex-start;" (click)="signOut()">Sign out</button>
        </div>
      </section>
    </div>
  `,
  styles: [`
    .toggle { display: grid; grid-template-columns: 36px 1fr; align-items: center; gap: 12px; cursor: pointer; padding: 6px 0; }
    .toggle input { position: absolute; opacity: 0; pointer-events: none; }
    .track { position: relative; width: 36px; height: 20px; border-radius: 999px; background: var(--color-border-strong); transition: background-color .15s ease; }
    .thumb { position: absolute; top: 2px; left: 2px; width: 16px; height: 16px; border-radius: 50%; background: #fff; box-shadow: 0 1px 2px rgba(0,0,0,.2); transition: transform .15s ease; }
    .toggle input:checked + .track { background: var(--color-primary); }
    .toggle input:checked + .track .thumb { transform: translateX(16px); }
    .t-text { display: flex; flex-direction: column; gap: 2px; }
    .t-title { font-size: 13px; font-weight: 500; }
  `],
})
export class ManagerSettingsComponent implements OnInit {
  private api = inject(ApiService);
  private auth = inject(AuthService);
  private router = inject(Router);

  profile = signal<ManagerProfileDto | null>(null);
  fullName = signal('');
  language = signal('en');
  alerts = signal<AlertPrefs>({ competitor_undercut: true, price_spike: true, anomaly_digest: false, data_quality: true });
  saving = signal(false);
  saved = signal(false);
  error = signal<string | null>(null);

  ngOnInit(): void {
    this.api.getMe().subscribe({
      next: p => {
        this.profile.set(p);
        this.fullName.set(p.full_name ?? '');
        this.language.set(p.preferences?.['language'] ?? 'en');
        const a = p.preferences?.['alerts'] ?? {};
        this.alerts.set({
          competitor_undercut: a.competitor_undercut ?? true,
          price_spike: a.price_spike ?? true,
          anomaly_digest: a.anomaly_digest ?? false,
          data_quality: a.data_quality ?? true,
        });
      },
      error: () => this.error.set('Could not load your profile.'),
    });
  }

  toggleAlert(key: keyof AlertPrefs): void {
    this.alerts.update(a => ({ ...a, [key]: !a[key] }));
    this.saved.set(false);
  }

  save(): void {
    this.saving.set(true);
    this.saved.set(false);
    this.error.set(null);
    this.api.updateMe({
      full_name: this.fullName(),
      preferences: { language: this.language(), alerts: this.alerts() },
    }).subscribe({
      next: p => { this.profile.set(p); this.saving.set(false); this.saved.set(true); },
      error: err => { this.saving.set(false); this.error.set(err?.error?.detail ?? 'Save failed.'); },
    });
  }

  signOut(): void {
    this.auth.logout();
    this.router.navigateByUrl('/login');
  }
}
