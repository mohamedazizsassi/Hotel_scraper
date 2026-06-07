import { inject } from '@angular/core';
import { CanActivateFn, Router } from '@angular/router';
import { AuthService } from './auth.service';
import { Role } from '../models/domain';

/**
 * UX-only route guard: keeps unauthenticated users out of the shells and
 * sends each role to its own area. Server-side checks remain authoritative.
 */
export function roleGuard(required: Role): CanActivateFn {
  return () => {
    const auth = inject(AuthService);
    const router = inject(Router);

    if (!auth.isAuthenticated()) {
      return router.parseUrl('/login');
    }
    if (auth.role() !== required) {
      const home = auth.role() === 'admin' ? '/admin/dashboard' : '/manager/dashboard';
      return router.parseUrl(home);
    }
    return true;
  };
}
