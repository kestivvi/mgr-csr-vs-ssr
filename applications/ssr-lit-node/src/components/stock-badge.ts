import { LitElement, html } from 'lit';
import { customElement, property } from 'lit/decorators.js';

@customElement('stock-badge')
export class StockBadge extends LitElement {
  @property({ type: Boolean })
  inStock = false;

  override createRenderRoot() {
    return this;
  }

  override render() {
    return this.inStock
      ? html`<span class="badge badge-in">IN</span>`
      : html`<span class="badge badge-out">OUT</span>`;
  }
}
