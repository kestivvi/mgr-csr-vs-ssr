import { LitElement, html } from 'lit';
import { customElement } from 'lit/decorators.js';
import '../components/counter-element';

@customElement('home-page')
export class HomePage extends LitElement {
  protected override createRenderRoot() {
    return this;
  }

  render() {
    return html`
      <main>
        <h1>Hello World</h1>
        <counter-element></counter-element>
      </main>
    `;
  }
}
