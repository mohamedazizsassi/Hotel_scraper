import { Component } from '@angular/core';

@Component({
  selector: 'rw-manager-settings',
  standalone: true,
  template: `
    <div class="page-head">
      <div>
        <h1>Settings</h1>
        <div class="sub">Account, hotel profile, and alert preferences.</div>
      </div>
      <button class="btn primary">Save changes</button>
    </div>

    <div class="grid cols-2">
      <section class="card">
        <div class="card-head"><h3>Profile</h3></div>
        <div class="card-body col">
          <div class="field"><label>Full name</label><input class="input" value="Sami Bouazizi" /></div>
          <div class="field"><label>Email</label><input class="input" value="sami@elmouradi.tn" /></div>
          <div class="field">
            <label>Language</label>
            <select class="select"><option>English</option><option>Français</option></select>
          </div>
        </div>
      </section>

      <section class="card">
        <div class="card-head"><h3>Hotel</h3></div>
        <div class="card-body col">
          <div class="field"><label>Hotel</label><input class="input" value="Hôtel El Mouradi Palace" disabled /></div>
          <div class="field"><label>City</label><input class="input" value="Sousse" disabled /></div>
          <div class="row" style="gap:12px;">
            <div class="field" style="flex:1"><label>Stars</label><input class="input mono" value="5" disabled /></div>
            <div class="field" style="flex:1"><label>Rooms</label><input class="input mono" value="312" disabled /></div>
          </div>
          <p class="muted small">Hotel profile is managed by your admin.</p>
        </div>
      </section>

      <section class="card">
        <div class="card-head"><h3>Alert preferences</h3></div>
        <div class="card-body col">
          @for (p of prefs; track p.id) {
            <label class="toggle">
              <input type="checkbox" [checked]="p.enabled" />
              <span class="track"><span class="thumb"></span></span>
              <span class="t-text">
                <span class="t-title">{{ p.title }}</span>
                <span class="muted small">{{ p.help }}</span>
              </span>
            </label>
          }
        </div>
      </section>

      <section class="card">
        <div class="card-head"><h3>Danger zone</h3></div>
        <div class="card-body col">
          <p class="muted small">Disconnect your account from this hotel. Your admin must reassign you.</p>
          <button class="btn danger" style="align-self:flex-start;">Sign out of all devices</button>
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
export class ManagerSettingsComponent {
  prefs = [
    { id: 1, title: 'Competitor undercut',  help: 'Notify when a competitor drops > 8% below your price', enabled: true },
    { id: 2, title: 'Price spike',          help: 'Notify when your price is > 10% above market average', enabled: true },
    { id: 3, title: 'Anomaly detection',    help: 'Daily anomaly digest from the forecaster',             enabled: false },
    { id: 4, title: 'Data quality',         help: 'Scraper coverage drops or partial runs',               enabled: true },
  ];
}
