import { Component } from '@angular/core';
import { HOTELS, MANAGERS } from '../../../core/data/mock';

@Component({
  selector: 'rw-admin-assignments',
  standalone: true,
  template: `
    <div class="page-head">
      <div>
        <h1>Assignments</h1>
        <div class="sub">Map managers to the hotels they own. Manager authorization is enforced on every API call.</div>
      </div>
      <button class="btn primary">Save changes</button>
    </div>

    <div class="grid cols-2">
      <section class="card">
        <div class="card-head"><h3>Unassigned hotels</h3><span class="small muted">{{ unassigned.length }}</span></div>
        <div class="list">
          @for (h of unassigned; track h.id) {
            <div class="row between item">
              <div>
                <div style="font-weight:500;">{{ h.name }}</div>
                <div class="small muted">{{ h.city }} · {{ h.stars }}★</div>
              </div>
              <button class="btn sm">Assign…</button>
            </div>
          } @empty {
            <div class="muted small" style="padding:14px;">All hotels are assigned.</div>
          }
        </div>
      </section>

      <section class="card">
        <div class="card-head"><h3>Current assignments</h3><span class="small muted">{{ assigned.length }}</span></div>
        <table class="tbl">
          <thead><tr><th>Manager</th><th>Hotel</th><th></th></tr></thead>
          <tbody>
            @for (a of assigned; track a.hotelId) {
              <tr>
                <td>{{ a.managerName }}</td>
                <td>{{ a.hotelName }} <span class="muted small">· {{ a.city }}</span></td>
                <td class="num"><button class="btn ghost sm danger">Unassign</button></td>
              </tr>
            }
          </tbody>
        </table>
      </section>
    </div>
  `,
  styles: [`
    .list { padding: 8px; display: flex; flex-direction: column; }
    .item { padding: 10px 12px; border-radius: var(--radius-sm); }
    .item + .item { border-top: 1px solid var(--color-border); }
  `],
})
export class AdminAssignmentsComponent {
  unassigned = HOTELS.filter(h => h.active && !h.managerId);
  assigned = HOTELS
    .filter(h => h.managerId)
    .map(h => {
      const m = MANAGERS.find(x => x.id === h.managerId);
      return { hotelId: h.id, hotelName: h.name, city: h.city, managerName: m?.name ?? '—' };
    });
}
