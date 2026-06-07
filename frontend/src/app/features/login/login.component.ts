import { Component, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { Router, RouterLink } from '@angular/router';
import { AuthService } from '../../core/auth/auth.service';

@Component({
  selector: 'rw-login',
  standalone: true,
  imports: [RouterLink, FormsModule],
  template: `
    <div class="auth">
      <a class="brand" routerLink="/">
        <svg width="26" height="26" viewBox="0 0 32 32"><rect width="32" height="32" rx="7" fill="#2563EB"/><path d="M8 22V10h6a4 4 0 0 1 1.4 7.75L19 22h-3.4l-3.2-4.5H11V22H8zm3-7h2.8a1.7 1.7 0 0 0 0-3.4H11v3.4z" fill="#fff"/></svg>
          <span>RevWay</span>
      </a>

      <form class="card" (submit)="$event.preventDefault(); submit()">
        <h1>Sign in</h1>
        <p class="muted small">Use your manager or admin account.</p>

        <div class="field">
          <label for="e">Email</label>
          <input id="e" class="input" type="email" autocomplete="email"
                 placeholder="you@hotel.tn" [(ngModel)]="email" name="email" />
        </div>
        <div class="field">
          <label for="p">Password</label>
          <input id="p" class="input" type="password" autocomplete="current-password"
                 placeholder="••••••••" [(ngModel)]="password" name="password" />
        </div>

        @if (error()) {
          <div class="err" role="alert">{{ error() }}</div>
        }

        <div class="row between">
          <label class="row" style="gap:6px; font-size:12px;">
            <input type="checkbox" /> Remember me
          </label>
          <a href="#" class="small">Forgot password?</a>
        </div>

        <button type="submit" class="btn primary" style="width:100%; margin-top: 6px;"
                [disabled]="loading()">
          {{ loading() ? 'Signing in…' : 'Continue' }}
        </button>

        <p class="hint-creds tiny muted">
          Dev seed: <span class="mono">manager&#64;revway.tn</span> / <span class="mono">REDACTED_DEV_PASSWORD</span>
        </p>
      </form>

      <p class="foot-note muted small">© 2026 RevWay · Satisfy Insight</p>
    </div>
  `,
  styles: [`
    .auth { min-height: 100vh; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 18px; padding: 24px; }
    .brand { display:flex; align-items:center; gap:10px; font-weight:700; font-size:16px; color: var(--color-foreground); text-decoration:none; }
    form.card { width: 100%; max-width: 380px; padding: 24px; display: flex; flex-direction: column; gap: 14px; box-shadow: var(--shadow-md); }
    h1 { margin: 0; font-size: 20px; letter-spacing: -0.02em; }
    .err { background: var(--color-destructive-soft); color: var(--color-destructive); border-radius: var(--radius-sm); padding: 8px 12px; font-size: 13px; }
    .hint-creds { margin: 4px 0 0; text-align: center; }
    .foot-note { margin: 0; }
  `],
})
export class LoginComponent {
  private auth = inject(AuthService);
  private router = inject(Router);

  email = 'manager@revway.tn';
  password = 'REDACTED_DEV_PASSWORD';
  error = signal<string | null>(null);
  loading = signal(false);

  submit(): void {
    if (this.loading()) return;
    this.error.set(null);
    this.loading.set(true);
    this.auth.login(this.email.trim(), this.password).subscribe({
      next: () => {
        this.loading.set(false);
        const home = this.auth.role() === 'admin' ? '/admin/dashboard' : '/manager/dashboard';
        this.router.navigateByUrl(home);
      },
      error: err => {
        this.loading.set(false);
        this.error.set(err?.status === 401 ? 'Invalid email or password.' : 'Could not reach the server.');
      },
    });
  }
}
