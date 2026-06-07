import { Routes } from '@angular/router';
import { roleGuard } from './core/auth/auth.guard';

export const routes: Routes = [
  { path: '', loadComponent: () => import('./features/landing/landing.component').then(m => m.LandingComponent) },
  { path: 'login', loadComponent: () => import('./features/login/login.component').then(m => m.LoginComponent) },
  {
    path: 'admin',
    canActivate: [roleGuard('admin')],
    loadComponent: () => import('./features/admin/shell/admin-shell.component').then(m => m.AdminShellComponent),
    children: [
      { path: '', pathMatch: 'full', redirectTo: 'dashboard' },
      { path: 'dashboard', loadComponent: () => import('./features/admin/dashboard/admin-dashboard.component').then(m => m.AdminDashboardComponent) },
      { path: 'hotels',    loadComponent: () => import('./features/admin/hotels/admin-hotels.component').then(m => m.AdminHotelsComponent) },
      { path: 'managers',  loadComponent: () => import('./features/admin/managers/admin-managers.component').then(m => m.AdminManagersComponent) },
      { path: 'assignments', loadComponent: () => import('./features/admin/assignments/admin-assignments.component').then(m => m.AdminAssignmentsComponent) },
      { path: 'scrapers',  loadComponent: () => import('./features/admin/scrapers/admin-scrapers.component').then(m => m.AdminScrapersComponent) },
    ],
  },
  {
    path: 'manager',
    canActivate: [roleGuard('manager')],
    loadComponent: () => import('./features/manager/shell/manager-shell.component').then(m => m.ManagerShellComponent),
    children: [
      { path: '', pathMatch: 'full', redirectTo: 'dashboard' },
      { path: 'dashboard',       loadComponent: () => import('./features/manager/dashboard/manager-dashboard.component').then(m => m.ManagerDashboardComponent) },
      { path: 'calendar',        loadComponent: () => import('./features/manager/calendar/manager-calendar.component').then(m => m.ManagerCalendarComponent) },
      { path: 'competitors',     loadComponent: () => import('./features/manager/competitors/manager-competitors.component').then(m => m.ManagerCompetitorsComponent) },
      { path: 'recommendations', loadComponent: () => import('./features/manager/recommendations/manager-recommendations.component').then(m => m.ManagerRecommendationsComponent) },
      { path: 'alerts',          loadComponent: () => import('./features/manager/alerts/manager-alerts.component').then(m => m.ManagerAlertsComponent) },
      { path: 'settings',        loadComponent: () => import('./features/manager/settings/manager-settings.component').then(m => m.ManagerSettingsComponent) },
    ],
  },
  { path: '**', redirectTo: '' },
];
