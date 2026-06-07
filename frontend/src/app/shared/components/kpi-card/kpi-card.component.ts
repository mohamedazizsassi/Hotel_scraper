import { Component, Input } from '@angular/core';
import { SparklineComponent } from '../sparkline/sparkline.component';

@Component({
  selector: 'rw-kpi-card',
  standalone: true,
  imports: [SparklineComponent],
  template: `
    <div class="card kpi">
      <div class="kpi-head">
        <span class="tiny">{{ label }}</span>
        @if (delta != null) {
          <span class="delta" [class.up]="delta > 0" [class.down]="delta < 0">
            {{ delta > 0 ? '+' : '' }}{{ delta }}{{ deltaUnit }}
          </span>
        }
      </div>
      <div class="kpi-value mono">{{ value }}<span class="unit">{{ unit }}</span></div>
      @if (trend?.length) {
        <rw-sparkline [values]="trend!" [ariaLabel]="label + ' trend'" />
      }
      @if (sub) { <div class="sub muted small">{{ sub }}</div> }
    </div>
  `,
  styles: [`
    .kpi { padding: 14px 16px; display: flex; flex-direction: column; gap: 8px; min-height: 116px; }
    .kpi-head { display: flex; justify-content: space-between; align-items: center; }
    .kpi-value { font-size: 24px; font-weight: 600; letter-spacing: -0.02em; line-height: 1.1; }
    .kpi-value .unit { font-size: 12px; font-weight: 500; color: var(--color-muted); margin-left: 4px; }
    .delta { font-size: 11px; font-weight: 600; padding: 2px 6px; border-radius: 999px; background: var(--color-surface-2); color: var(--color-muted); }
    .delta.up { background: var(--color-success-soft); color: var(--color-success); }
    .delta.down { background: var(--color-destructive-soft); color: var(--color-destructive); }
    .sub { margin-top: 2px; }
  `],
})
export class KpiCardComponent {
  @Input({ required: true }) label!: string;
  @Input({ required: true }) value!: string | number;
  @Input() unit = '';
  @Input() sub?: string;
  @Input() delta?: number | null;
  @Input() deltaUnit = '%';
  @Input() trend?: number[];
}
