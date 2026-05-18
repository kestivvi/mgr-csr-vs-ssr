import { LitElement, html } from 'lit';
import { customElement, state } from 'lit/decorators.js';

@customElement('counter-element')
export class CounterElement extends LitElement {
  @state()
  private count = 0;

  // Use Light DOM to inherit global styles
  protected override createRenderRoot() {
    return this;
  }

  render() {
    return html`
      <div>
        <p>Count: ${this.count}</p>
        <button @click=${() => this.count++}>Increment</button>
      </div>
    `;
  }
}
