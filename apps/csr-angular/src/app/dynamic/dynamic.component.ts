import { Component } from '@angular/core';
import { ActivatedRoute } from '@angular/router';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-dynamic',
  standalone: true,
  imports: [CommonModule],
  template: '<h1>Hello, {{ name }}</h1>'
})
export class DynamicComponent {
  name: string = '';

  constructor(private route: ActivatedRoute) {
    this.route.params.subscribe(params => {
      this.name = params['name'];
    });
  }
}
