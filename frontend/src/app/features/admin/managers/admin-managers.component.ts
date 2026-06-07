import { Component } from '@angular/core';
import { DatePipe } from '@angular/common';
import { HOTELS, MANAGERS } from '../../../core/data/mock';
import { StatusPillComponent } from '../../../shared/components/status-pill/status-pill.component';

@Component({
  selector: 'rw-admin-managers',
  standalone: true,
  imports: [DatePipe, StatusPillComponent],
  template: `
    <div class="page-head">
      <div>
        <h1>Managers</h1>
        <div class="sub">{{ managers.length }} accounts · {{ assigned }} assigned to a hotel</div>
      </div>
      <button class="btn primary">+ Invite manager</button>
    </div>

    <section class="card">
      <table class="tbl">
        <thead>
          <tr>
            <th>Name</th><th>Email</th><th>Hotel</th>
            <th class="num">Competitors</th><th>Last seen</th><th></th>
          </tr>
        </thead>
        <tbody>
          @for (m of managers; track m.id) {
            <tr>
              <td>
                <div class="row" style="gap:10px;">
                  <span class="avatar mono">{{ initials(m.name) }}</span>
                  <span style="font-weight:500;">{{ m.name }}</span>
                </div>
              </td>
              <td class="muted small">{{ m.email }}</td>
              <td>
                @if (m.hotelId) { <span>{{ hotelName(m.hotelId) }}</span> }
                @else { <rw-status-pill tone="warn">unassigned</rw-status-pill> }
              </td>
              <td class="num">{{ m.competitorIds.length }}</td>
              <td class="small muted">{{ m.lastSeen | date:'MMM d, HH:mm' }}</td>
              <td class="num">
                <button class="btn ghost sm">Edit</button>
              </td>
            </tr>
          }
        </tbody>
      </table>
    </section>
  `,
  styles: [`
    .avatar { width: 28px; height: 28px; border-radius: 50%; background: var(--color-surface-2); color: var(--color-muted); font-size: 11px; font-weight: 600; display: inline-flex; align-items: center; justify-content: center; }
  `],
})
export class AdminManagersComponent {
  managers = MANAGERS;
  assigned = MANAGERS.filter(m => m.hotelId).length;
  hotelName(id: string) { return HOTELS.find(h => h.id === id)?.name ?? '—'; }
  initials(n: string) { return n.split(' ').map(p => p[0]).slice(0, 2).join(''); }
}
