import { Component, computed, signal } from '@angular/core';
import { StatusPillComponent } from '../../../shared/components/status-pill/status-pill.component';
import { HOTELS, MANAGERS } from '../../../core/data/mock';

@Component({
  selector: 'rw-admin-hotels',
  standalone: true,
  imports: [StatusPillComponent],
  template: `
    <div class="page-head">
      <div>
        <h1>Hotels</h1>
        <div class="sub">{{ filtered().length }} of {{ all.length }} hotels in the pool</div>
      </div>
      <div class="row">
        <input class="input" style="width:240px;" type="search" placeholder="Search by name or city…" (input)="onSearch($any($event.target).value)" />
        <select class="select" style="width:140px;" (change)="onCity($any($event.target).value)">
          <option value="">All cities</option>
          @for (c of cities; track c) { <option [value]="c">{{ c }}</option> }
        </select>
        <button class="btn primary">+ Add hotel</button>
      </div>
    </div>

    <section class="card">
      <table class="tbl">
        <thead>
          <tr>
            <th>Hotel</th><th>City</th><th>Stars</th><th>Source</th>
            <th class="num">Rooms</th><th>Manager</th><th>Status</th>
          </tr>
        </thead>
        <tbody>
          @for (h of filtered(); track h.id) {
            <tr>
              <td>
                <div class="hname">{{ h.name }}</div>
                <div class="muted small mono">{{ h.id }}</div>
              </td>
              <td>{{ h.city }}</td>
              <td><span class="mono">{{ h.stars }}★</span></td>
              <td><span class="badge info">{{ h.source }}</span></td>
              <td class="num">{{ h.rooms }}</td>
              <td>
                @if (h.managerId) {
                  <span class="small">{{ managerName(h.managerId) }}</span>
                } @else {
                  <span class="muted small">— unassigned —</span>
                }
              </td>
              <td>
                @if (h.active) { <rw-status-pill tone="ok">active</rw-status-pill> }
                @else { <rw-status-pill tone="muted">paused</rw-status-pill> }
              </td>
            </tr>
          }
        </tbody>
      </table>
    </section>
  `,
  styles: [`.hname { font-weight: 500; }`],
})
export class AdminHotelsComponent {
  all = HOTELS;
  q = signal('');
  city = signal('');
  cities = Array.from(new Set(HOTELS.map(h => h.city))).sort();
  filtered = computed(() => {
    const q = this.q().toLowerCase();
    const city = this.city();
    return this.all.filter(h =>
      (!q || h.name.toLowerCase().includes(q) || h.city.toLowerCase().includes(q)) &&
      (!city || h.city === city),
    );
  });
  onSearch(v: string) { this.q.set(v); }
  onCity(v: string) { this.city.set(v); }
  managerName(id: string) { return MANAGERS.find(m => m.id === id)?.name ?? '—'; }
}
