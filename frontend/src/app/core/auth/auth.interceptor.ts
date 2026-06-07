import { HttpInterceptorFn, HttpErrorResponse } from '@angular/common/http';
import { inject } from '@angular/core';
import { Router } from '@angular/router';
import { catchError, throwError } from 'rxjs';
import { AuthService } from './auth.service';
import { API_BASE } from '../config/api.config';

/**
 * Attaches the JWT to outgoing RevWay API calls and, on a 401, clears the
 * token and bounces to /login.
 */
export const authInterceptor: HttpInterceptorFn = (req, next) => {
  const auth = inject(AuthService);
  const router = inject(Router);

  let request = req;
  const token = auth.token();
  if (token && req.url.startsWith(API_BASE)) {
    request = req.clone({ setHeaders: { Authorization: `Bearer ${token}` } });
  }

  return next(request).pipe(
    catchError((err: HttpErrorResponse) => {
      if (err.status === 401 && req.url.startsWith(API_BASE)) {
        auth.logout();
        router.navigateByUrl('/login');
      }
      return throwError(() => err);
    }),
  );
};
