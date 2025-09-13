import { createRouter, createWebHistory } from 'vue-router'
import HomeView from '../views/HomeView.vue'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      name: 'home',
      component: HomeView,
    },
    {
      path: '/about',
      name: 'about',
      component: () => import('../views/AboutView.vue'),
    },
    {
      path: '/manageScan',
      name: 'manageScan',
      component: () => import('../views/ManageScanView.vue'),
    },
    {
      path: '/newScan',
      name: 'newScan',
      component: () => import('../views/newScan.vue'),
    },
    {
      path: '/checkScan',
      name: 'checkScan',
      component: () => import('../views/checkScan.vue'),
    },
    {
      path: '/addUser',
      name: 'addUser',
      component: () => import('../views/UserManagment/addUser.vue'),
    },
  ],
})

export default router
