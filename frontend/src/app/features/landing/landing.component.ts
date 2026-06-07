import { Component } from '@angular/core';
import { RouterLink } from '@angular/router';

@Component({
  selector: 'rw-landing',
  standalone: true,
  imports: [RouterLink],
  template: `
    <div class="page">
      <header class="top">
        <a class="brand" routerLink="/">
          <svg width="26" height="26" viewBox="0 0 32 32" aria-hidden="true"><rect width="32" height="32" rx="7" fill="#2563EB"/><path d="M8 22V10h6a4 4 0 0 1 1.4 7.75L19 22h-3.4l-3.2-4.5H11V22H8zm3-7h2.8a1.7 1.7 0 0 0 0-3.4H11v3.4z" fill="#fff"/><circle cx="24" cy="11" r="2" fill="#059669"/></svg>
          <span>RevWay</span>
        </a>
        <nav class="top-nav">
          <a href="#how">How it works</a>
          <a href="#metrics">Metrics</a>
          <a href="#data">Data</a>
        </nav>
        <div class="top-actions">
          <a class="btn ghost sm" routerLink="/login">Sign in</a>
          <a class="btn primary sm" routerLink="/manager/dashboard">Open demo</a>
        </div>
      </header>

      <section class="hero">
        <div class="hero-copy">
          <span class="eyebrow">Satisfy Insight · for Tunisian hoteliers</span>
          <h1>Price with the market, not against it.</h1>
          <p>RevWay turns the Tunisian OTA landscape into a live signal — competitor rates,
            anomaly alerts, and AI-driven price recommendations, refreshed twice a day.</p>
          <div class="cta-row">
            <a class="btn primary" routerLink="/manager/dashboard">Manager view →</a>
            <a class="btn" routerLink="/admin/dashboard">Admin console</a>
          </div>
          <div class="trust">
            <span class="badge info">Tunisia · TND</span>
            <span class="badge">~24.4M observations</span>
            <span class="badge">2× / day scrape</span>
          </div>
        </div>

        <div class="hero-preview">
          <div class="preview-card">
            <div class="preview-head">
              <span class="dot ok"></span><span class="mono small muted">live · 10:46</span>
              <span class="spacer"></span>
              <span class="tiny">Sousse · 5★</span>
            </div>
            <div class="metric-row">
              <div>
                <div class="tiny">Your ADR</div>
                <div class="big mono">342<span class="unit">TND</span></div>
              </div>
              <div>
                <div class="tiny">Market avg</div>
                <div class="big mono">328<span class="unit">TND</span></div>
              </div>
              <div>
                <div class="tiny">Suggested</div>
                <div class="big mono" style="color:var(--color-accent)">335<span class="unit">TND</span></div>
              </div>
            </div>
            <div class="bars" aria-hidden="true">
              @for (b of bars; track $index) {
                <span class="bar" [style.height.%]="b"></span>
              }
            </div>
            <div class="row between tiny muted">
              <span>Next 30 days</span>
              <span>Confidence 0.78</span>
            </div>
          </div>
        </div>
      </section>

      <section id="metrics" class="metrics">
        <div class="m">
          <div class="m-value mono">312</div>
          <div class="m-label">hotels tracked</div>
        </div>
        <div class="m">
          <div class="m-value mono">2× <span class="muted">/ day</span></div>
          <div class="m-label">scrape cadence</div>
        </div>
        <div class="m">
          <div class="m-value mono">≤45<span class="muted small">s</span></div>
          <div class="m-label">recommendation latency</div>
        </div>
        <div class="m">
          <div class="m-value mono">4</div>
          <div class="m-label">competitors per manager</div>
        </div>
      </section>

      <section id="how" class="how">
        <h2>How it works</h2>
        <ol>
          <li><strong>Collect.</strong> Scrapy spiders harvest promohotel and tunisiepromo into MongoDB twice daily.</li>
          <li><strong>Engineer.</strong> Features land in Postgres — calendar, segment, booking-window signals.</li>
          <li><strong>Forecast.</strong> A LightGBM quantile model produces market-aligned price intervals.</li>
          <li><strong>Recommend.</strong> Rule-based logic surfaces actionable nudges with rationale you can defend.</li>
        </ol>
      </section>

      <section id="data" class="data">
        <h2>Built for the data we actually have</h2>
        <p class="muted">Public OTA snapshots, not internal occupancy. Recommendations reflect the market price you should match — never a guarantee of bookings.</p>
        <div class="data-grid">
          <div class="card card-pad"><div class="tiny">Source A</div><div class="mono big">promohotel</div><div class="muted small">supplements expanded as variants</div></div>
          <div class="card card-pad"><div class="tiny">Source B</div><div class="mono big">tunisiepromo</div><div class="muted small">view encoded in room name</div></div>
          <div class="card card-pad"><div class="tiny">Identity</div><div class="mono big">name + city</div><div class="muted small">city_id varies per source</div></div>
        </div>
      </section>

      <footer class="foot">
        <span class="mono small muted">© 2026 RevWay · PFE — Satisfy Insight</span>
        <span class="spacer"></span>
        <a routerLink="/login" class="small">Sign in</a>
      </footer>
    </div>
  `,
  styles: [`
    .page { max-width: 1180px; margin: 0 auto; padding: 20px 24px 60px; }
    .top { display: flex; align-items: center; gap: 24px; padding: 8px 0 24px; }
    .brand { display: flex; align-items: center; gap: 10px; font-weight: 700; color: var(--color-foreground); text-decoration: none; font-size: 16px; }
    .top-nav { display: flex; gap: 22px; }
    .top-nav a { color: var(--color-muted); font-size: 13px; }
    .top-nav a:hover { color: var(--color-foreground); text-decoration: none; }
    .top-actions { margin-left: auto; display: flex; gap: 8px; }

    .hero { display: grid; grid-template-columns: 1.05fr 1fr; gap: 36px; align-items: center; padding: 40px 0 56px; }
    .eyebrow { display: inline-block; font-size: 11px; letter-spacing: .12em; text-transform: uppercase; color: var(--color-primary); font-weight: 600; margin-bottom: 14px; }
    h1 { font-size: 44px; line-height: 1.05; letter-spacing: -0.03em; margin: 0 0 14px; font-weight: 600; }
    .hero-copy p { color: var(--color-muted); font-size: 16px; max-width: 480px; }
    .cta-row { display: flex; gap: 10px; margin: 22px 0 18px; }
    .trust { display: flex; gap: 8px; flex-wrap: wrap; }

    .preview-card { background: var(--color-surface); border: 1px solid var(--color-border); border-radius: var(--radius-lg); padding: 18px; box-shadow: var(--shadow-lg); }
    .preview-head { display: flex; align-items: center; gap: 8px; margin-bottom: 14px; }
    .metric-row { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; padding: 6px 0 14px; }
    .big { font-size: 26px; font-weight: 600; letter-spacing: -0.02em; }
    .unit { font-size: 11px; color: var(--color-muted); margin-left: 4px; font-weight: 500; }
    .bars { display: flex; align-items: flex-end; gap: 3px; height: 70px; padding: 8px 0 12px; }
    .bar { flex: 1; background: linear-gradient(180deg, var(--color-primary), var(--color-secondary)); border-radius: 2px; min-height: 6px; }

    .metrics { display: grid; grid-template-columns: repeat(4, 1fr); gap: 18px; padding: 28px 0; border-top: 1px solid var(--color-border); border-bottom: 1px solid var(--color-border); }
    .m-value { font-size: 28px; font-weight: 600; letter-spacing: -0.02em; }
    .m-label { color: var(--color-muted); font-size: 12px; text-transform: uppercase; letter-spacing: .06em; font-weight: 600; margin-top: 4px; }

    .how, .data { padding: 48px 0 0; }
    h2 { font-size: 24px; letter-spacing: -0.02em; margin: 0 0 16px; font-weight: 600; }
    .how ol { padding-left: 0; list-style: none; counter-reset: step; display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
    .how li { background: var(--color-surface); border: 1px solid var(--color-border); border-radius: var(--radius-md); padding: 16px; position: relative; padding-left: 48px; }
    .how li::before { counter-increment: step; content: counter(step); position: absolute; left: 14px; top: 14px; width: 24px; height: 24px; border-radius: 50%; background: var(--color-primary); color: #fff; font-weight: 600; font-size: 12px; display: flex; align-items: center; justify-content: center; }
    .data-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin-top: 14px; }

    .foot { display: flex; align-items: center; gap: 16px; padding: 48px 0 0; color: var(--color-muted); }

    @media (max-width: 900px) {
      .hero { grid-template-columns: 1fr; }
      .metrics, .how ol, .data-grid { grid-template-columns: 1fr 1fr; }
      h1 { font-size: 34px; }
    }
    @media (max-width: 560px) {
      .metrics, .how ol, .data-grid { grid-template-columns: 1fr; }
      .top-nav { display: none; }
    }
  `],
})
export class LandingComponent {
  bars = Array.from({ length: 30 }).map((_, i) => 32 + Math.round(Math.sin(i / 3) * 22 + (i % 5) * 4 + 28));
}
