import { LitElement, html } from 'lit';
import { customElement, property } from 'lit/decorators.js';
import type { RouterLocation } from '@vaadin/router';

@customElement('dynamic-page')
export class DynamicPage extends LitElement {
  @property({ type: Object })
  location?: RouterLocation;

  protected override createRenderRoot() {
    return this;
  }

  render() {
    const id = this.location?.params?.id || 'unknown';
    return html`
      <main>
        <h1>Hello World</h1>
        <p>Dynamic ID: ${id}</p>
      </main>
    `;
  }
}
