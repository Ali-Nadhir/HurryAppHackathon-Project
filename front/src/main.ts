import { createApp } from 'vue'
import App from './App.vue'
import router from './router'
import { createPinia } from 'pinia'
import { i18n } from '@/i18n'

import './assets/tailwind.css'
import 'bootstrap-icons/font/bootstrap-icons.css'

createApp(App).use(i18n).use(createPinia()).use(router).mount('#app')
