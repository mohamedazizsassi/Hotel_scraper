import { Component, Input } from '@angular/core';
import { RouterLink, RouterLinkActive } from '@angular/router';

export interface NavItem { label: string; icon: string; route: string; }

@Component({
  selector: 'rw-sidebar',
  standalone: true,
  imports: [RouterLink, RouterLinkActive],
  template: `
    <aside class="sidebar">
      <a class="brand" routerLink="/">
        <span class="brand-mark" aria-hidden="true">
          <svg width="22" height="22" viewBox="0 0 32 32"><rect width="32" height="32" rx="7" fill="currentColor"/><path d="M8 22V10h6a4 4 0 0 1 1.4 7.75L19 22h-3.4l-3.2-4.5H11V22H8zm3-7h2.8a1.7 1.7 0 0 0 0-3.4H11v3.4z" fill="#fff"/></svg>
        </span>
        <span class="brand-name">RevWay</span>
        <span class="brand-role">{{ role }}</span>
      </a>

      <nav>
        @for (item of items; track item.route) {
          <a [routerLink]="item.route" routerLinkActive="active" class="nav-item">
            <span class="nav-icon" [innerHTML]="item.icon"></span>
            <span>{{ item.label }}</span>
          </a>
        }
      </nav>

      <div class="sidebar-foot">
        <div class="foot-card">
          <div class="tiny">Data scale</div>
          <div class="mono small">24.4M rows · ~625K/day</div>
          <div class="muted tiny">Last refresh 10:46</div>
        </div>
      </div>
    </aside>
  `,
  styles: [`
    .sidebar {
      background: var(--color-surface);
      border-right: 1px solid var(--color-border);
      display: flex; flex-direction: column;
      height: 100vh; position: sticky; top: 0;
    }
    .brand {
      display: flex; align-items: center; gap: 10px;
      padding: 14px 16px;
      border-bottom: 1px solid var(--color-border);
      color: var(--color-foreground);
      text-decoration: none;
    }
    .brand-mark { color: var(--color-primary); display: inline-flex; }
    .brand-name { font-weight: 700; letter-spacing: -0.01em; font-size: 15px; }
    .brand-role { margin-left: auto; font-size: 10px; letter-spacing: .12em; text-transform: uppercase; color: var(--color-muted); padding: 2px 6px; border: 1px solid var(--color-border); border-radius: 4px; }
    nav { padding: 10px 8px; display: flex; flex-direction: column; gap: 2px; flex: 1; }
    .nav-item {
      display: flex; align-items: center; gap: 10px;
      padding: 8px 12px;
      border-radius: var(--radius-sm);
      color: var(--color-muted);
      font-size: 13px; font-weight: 500;
      text-decoration: none;
      transition: background-color .15s ease, color .15s ease;
    }
    .nav-item:hover { background: var(--color-surface-2); color: var(--color-foreground); }
    .nav-item.active { background: rgba(37,99,235,.10); color: var(--color-primary); }
    .nav-item.active .nav-icon { color: var(--color-primary); }
    .nav-icon { display: inline-flex; width: 18px; height: 18px; color: var(--color-muted-2); }
    .nav-icon ::ng-deep svg { width: 18px; height: 18px; }
    .sidebar-foot { padding: 12px; }
    .foot-card { background: var(--color-surface-2); border-radius: var(--radius-sm); padding: 10px 12px; }
  `],
})
export class SidebarComponent {
  @Input({ required: true }) role!: string;
  @Input({ required: true }) items!: NavItem[];
}
