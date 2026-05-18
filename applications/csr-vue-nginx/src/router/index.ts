import { createRouter, createWebHistory } from 'vue-router'
import HomeView from '../views/HomeView.vue'
import DynamicView from '../views/DynamicView.vue'
import DynamicAppView from '../views/DynamicAppView.vue'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      name: 'home',
      component: HomeView
    },
    {
      path: '/dynamic/:name',
      name: 'dynamic',
      component: DynamicView
    },
    {
      path: '/dynamic-app/:id',
      name: 'dynamic-app',
      component: DynamicAppView
    }
  ],
})

export default router
