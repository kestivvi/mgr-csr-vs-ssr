import { Component, inject } from '@angular/core';
import { ActivatedRoute } from '@angular/router';

@Component({
  selector: 'app-name',
  standalone: true,
  template: `
    <div>
      <h1>Hello, {{ name }}</h1>
    </div>
  `
})
export class NameComponent {
  private route = inject(ActivatedRoute);
  name = this.route.snapshot.paramMap.get('name') || '';
}

