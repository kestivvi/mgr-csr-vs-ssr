import { Component, inject } from '@angular/core';
import { ActivatedRoute } from '@angular/router';

@Component({
  selector: 'app-dynamic',
  template: '<h1>Hello, {{ name }}</h1>',
})
export default class DynamicComponent {
  private route = inject(ActivatedRoute);
  name: string = '';

  ngOnInit() {
    this.route.params.subscribe(params => {
      this.name = params['name'];
    });
  }
}
