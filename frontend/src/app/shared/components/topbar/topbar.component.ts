import { Component, Input } from '@angular/core';
import { RouterLink } from '@angular/router';

@Component({
  selector: 'rw-topbar',
  standalone: true,
  imports: [RouterLink],
  template: `
    <header class="topbar">
      <div class="search">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/></svg>
        <input class="input-bare" type="text" placeholder="Search hotels, dates, alerts…" aria-label="Search" />
        <kbd>⌘K</kbd>
      </div>

      <div class="spacer"></div>

      <button class="icon-btn" aria-label="Notifications" title="Notifications">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M6 8a6 6 0 1 1 12 0c0 7 3 9 3 9H3s3-2 3-9"/><path d="M10 21a2 2 0 0 0 4 0"/></svg>
        <span class="pip"></span>
      </button>

      <div class="user">
        <div class="avatar mono">{{ initials }}</div>
        <div class="user-meta">
          <div class="user-name">{{ name }}</div>
          <div class="user-sub muted small">{{ subtitle }}</div>
        </div>
        <a class="btn ghost sm" routerLink="/">Sign out</a>
      </div>
    </header>
  `,
  styles: [`
    .topbar {
      display: flex; align-items: center; gap: 12px;
      padding: 0 18px;
      height: var(--topbar-h);
      background: var(--color-surface);
      border-bottom: 1px solid var(--color-border);
      position: sticky; top: 0; z-index: 20;
    }
    .search {
      display: flex; align-items: center; gap: 8px;
      background: var(--color-surface-2);
      border: 1px solid var(--color-border);
      border-radius: var(--radius-sm);
      padding: 0 10px;
      height: 34px;
      min-width: 260px;
      color: var(--color-muted);
    }
    .input-bare {
      flex: 1; background: transparent; border: 0; outline: 0;
      font: inherit; color: var(--color-foreground);
    }
    kbd { font-family: var(--font-mono); font-size: 10px; padding: 2px 5px; background: var(--color-surface); border: 1px solid var(--color-border); border-radius: 4px; color: var(--color-muted); }
    .icon-btn {
      position: relative;
      height: 34px; width: 34px;
      display: inline-flex; align-items: center; justify-content: center;
      background: transparent; border: 1px solid var(--color-border);
      border-radius: var(--radius-sm); color: var(--color-muted);
      transition: background-color .15s ease, color .15s ease;
    }
    .icon-btn:hover { background: var(--color-surface-2); color: var(--color-foreground); }
    .pip { position: absolute; top: 7px; right: 7px; width: 7px; height: 7px; border-radius: 50%; background: var(--color-destructive); box-shadow: 0 0 0 2px var(--color-surface); }
    .user { display: flex; align-items: center; gap: 10px; padding-left: 8px; border-left: 1px solid var(--color-border); }
    .avatar { width: 30px; height: 30px; border-radius: 50%; display: inline-flex; align-items: center; justify-content: center; background: var(--color-primary); color: #fff; font-size: 11px; font-weight: 600; letter-spacing: .04em; }
    .user-meta { line-height: 1.15; }
    .user-name { font-size: 13px; font-weight: 600; }
    .user-sub { font-size: 11px; }
    @media (max-width: 640px) {
      .search { display: none; }
      .user-meta { display: none; }
    }
  `],
})
export class TopbarComponent {
  @Input({ required: true }) name!: string;
  @Input({ required: true }) initials!: string;
  @Input({ required: true }) subtitle!: string;
}
