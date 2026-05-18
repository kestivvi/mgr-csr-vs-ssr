import { LitElement, html } from 'lit';
import { customElement, property } from 'lit/decorators.js';
import { globalStyles } from '../styles.js';

@customElement('counter-element')
export class CounterElement extends LitElement {
  static override styles = globalStyles;

  @property({ type: Number })
  count = 0;

  render() {
    return html`
      <div>
        <p>Count: ${this.count}</p>
        <button @click=${() => this.count++}>Increment</button>
      </div>
    `;
  }
}
