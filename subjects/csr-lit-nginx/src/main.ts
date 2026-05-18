import { Router } from '@vaadin/router';
import './global.css';

const root = document.getElementById('root');
const router = new Router(root);

router.setRoutes([
  {
    path: '/',
    component: 'home-page',
    action: async () => {
      await import('./pages/home-page');
    },
  },
  {
    path: '/dynamic/:id',
    component: 'dynamic-page',
    action: async () => {
      await import('./pages/dynamic-page');
    },
  },
  {
    path: '/dynamic-app/:id',
    component: 'dynamic-app-page',
    action: async () => {
      await import('./pages/dynamic-app-page');
    },
  },
]);
