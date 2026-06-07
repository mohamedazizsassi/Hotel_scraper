import { Injectable, computed, inject, signal } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, tap } from 'rxjs';
import { API_BASE } from '../config/api.config';
import { Role } from '../models/domain';

interface TokenResponse {
  access_token: string;
  token_type: string;
}

interface JwtClaims {
  sub: string;
  role: Role;
  hotel_id: number | null;
  exp: number; // seconds since epoch
}

const TOKEN_KEY = 'revway_token';

function decodeJwt(token: string): JwtClaims | null {
  try {
    const payload = token.split('.')[1];
    const json = atob(payload.replace(/-/g, '+').replace(/_/g, '/'));
    return JSON.parse(json) as JwtClaims;
  } catch {
    return null;
  }
}

/**
 * Holds the JWT issued by POST /auth/login and exposes the decoded role/hotel.
 * Route guards use this for UX redirects only — real authorization is enforced
 * server-side on every request (see backend get_current_manager).
 */
@Injectable({ providedIn: 'root' })
export class AuthService {
  private http = inject(HttpClient);

  private _token = signal<string | null>(localStorage.getItem(TOKEN_KEY));
  private _claims = computed(() => {
    const t = this._token();
    return t ? decodeJwt(t) : null;
  });

  readonly role = computed<Role | null>(() => this._claims()?.role ?? null);
  readonly hotelId = computed<number | null>(() => this._claims()?.hotel_id ?? null);

  token(): string | null {
    return this._token();
  }

  isAuthenticated(): boolean {
    const c = this._claims();
    if (!c) return false;
    return c.exp * 1000 > Date.now();
  }

  login(email: string, password: string): Observable<TokenResponse> {
    return this.http
      .post<TokenResponse>(`${API_BASE}/auth/login`, { email, password })
      .pipe(tap(res => this.setToken(res.access_token)));
  }

  logout(): void {
    this._token.set(null);
    localStorage.removeItem(TOKEN_KEY);
  }

  private setToken(token: string): void {
    this._token.set(token);
    localStorage.setItem(TOKEN_KEY, token);
  }
}
