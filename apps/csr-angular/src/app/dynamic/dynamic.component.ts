import { Component } from '@angular/core';
import { ActivatedRoute } from '@angular/router';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-dynamic',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div>
      <h1>Hello World</h1>
      <p>Dynamic ID: {{ name }}</p>
    </div>
  `
})
export class DynamicComponent {
  name: string = '';

  constructor(private route: ActivatedRoute) {
    this.route.params.subscribe(params => {
      this.name = params['name'];
    });
  }
}
