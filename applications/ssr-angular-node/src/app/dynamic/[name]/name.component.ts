import { Component, inject } from '@angular/core';
import { ActivatedRoute } from '@angular/router';

@Component({
  selector: 'app-name',
  standalone: true,
  template: `
    <div>
      <h1>Hello World</h1>
      <p>Dynamic ID: {{ name }}</p>
    </div>
  `
})
export class NameComponent {
  private route = inject(ActivatedRoute);
  name = this.route.snapshot.paramMap.get('name') || '';
}

