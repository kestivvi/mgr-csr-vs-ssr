import { Component, inject } from '@angular/core';
import { ActivatedRoute } from '@angular/router';
import { map } from 'rxjs';
import { AsyncPipe } from '@angular/common';

@Component({
  selector: 'app-dynamic',
  standalone: true,
  imports: [AsyncPipe],
  template: `
    <main>
      <h1>Hello World</h1>
      <p>Dynamic ID: {{ name$ | async }}</p>
      <div>
        <p>Count: {{ count }}</p>
        <button (click)="count = count + 1">Increment</button>
      </div>
    </main>
  `
})
export default class DynamicComponent {
  private route = inject(ActivatedRoute);
  name$ = this.route.params.pipe(map(params => params['name']));
  count = 0;
}
