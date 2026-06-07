import { Component, Input } from '@angular/core';

type Tone = 'ok' | 'warn' | 'err' | 'info' | 'muted';

@Component({
  selector: 'rw-status-pill',
  standalone: true,
  template: `
    <span class="badge" [class.ok]="tone==='ok'" [class.warn]="tone==='warn'" [class.err]="tone==='err'" [class.info]="tone==='info'">
      <span class="dot" [class.ok]="tone==='ok'" [class.warn]="tone==='warn'" [class.err]="tone==='err'"></span>
      <ng-content></ng-content>
    </span>
  `,
})
export class StatusPillComponent {
  @Input() tone: Tone = 'muted';
}
