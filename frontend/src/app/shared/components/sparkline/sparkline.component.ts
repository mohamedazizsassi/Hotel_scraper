import { Component, Input, computed, signal } from '@angular/core';

@Component({
  selector: 'rw-sparkline',
  standalone: true,
  template: `
    <svg [attr.width]="width" [attr.height]="height" [attr.viewBox]="'0 0 ' + width + ' ' + height" role="img"
         [attr.aria-label]="ariaLabel">
      <polyline [attr.points]="poly()" fill="none" [attr.stroke]="color" stroke-width="1.75" stroke-linejoin="round" stroke-linecap="round" />
      @if (showArea) {
        <polyline [attr.points]="area()" [attr.fill]="color" opacity="0.10" stroke="none" />
      }
    </svg>
  `,
})
export class SparklineComponent {
  @Input({ required: true }) set values(v: number[]) { this._values.set(v); }
  @Input() width = 120;
  @Input() height = 32;
  @Input() color = 'var(--color-primary)';
  @Input() showArea = true;
  @Input() ariaLabel = 'trend';

  private _values = signal<number[]>([]);

  poly = computed(() => this.build().join(' '));
  area = computed(() => {
    const pts = this.build();
    if (!pts.length) return '';
    return [`0,${this.height}`, ...pts, `${this.width},${this.height}`].join(' ');
  });

  private build(): string[] {
    const v = this._values();
    if (v.length < 2) return [];
    const min = Math.min(...v), max = Math.max(...v);
    const span = max - min || 1;
    const stepX = this.width / (v.length - 1);
    return v.map((y, i) => {
      const x = +(i * stepX).toFixed(2);
      const ny = +(this.height - ((y - min) / span) * (this.height - 4) - 2).toFixed(2);
      return `${x},${ny}`;
    });
  }
}
