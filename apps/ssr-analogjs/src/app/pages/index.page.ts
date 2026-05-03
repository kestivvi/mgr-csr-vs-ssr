import { Component } from '@angular/core';

@Component({
  selector: 'app-home',
  standalone: true,
  template: `
    <main>
      <h1>Hello World</h1>
      <div>
        <p>Count: {{ count }}</p>
        <button (click)="count = count + 1">Increment</button>
      </div>
    </main>
  `
})
export default class HomeComponent {
  count = 0;
}
