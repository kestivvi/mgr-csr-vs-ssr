import { Routes } from '@angular/router';
import { HomeComponent } from './home/home.component';
import { DynamicComponent } from './dynamic/dynamic.component';
import { DynamicAppComponent } from './dynamic-app/dynamic-app.component';

export const routes: Routes = [
  { path: '', component: HomeComponent },
  { path: 'dynamic/:name', component: DynamicComponent },
  { path: 'dynamic-app/:id', component: DynamicAppComponent }
];
