import { Routes } from '@angular/router';
import { NameComponent } from './dynamic/[name]/name.component';
import { HomeComponent } from './home.component';

export const routes: Routes = [
  {
    path: '',
    component: HomeComponent
  },
  {
    path: 'dynamic/:name',
    component: NameComponent
  }
];
