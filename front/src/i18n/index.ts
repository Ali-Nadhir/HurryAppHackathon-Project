import { createI18n } from 'vue-i18n'
import en from '@/locales/en.json'
import ar from '@/locales/ar.json'

type AppLocale = 'en' | 'ar'
const STORAGE_KEY = 'locale'

function detectInitialLocale(): AppLocale {
  const saved = (localStorage.getItem(STORAGE_KEY) as AppLocale) || null
  if (saved === 'en' || saved === 'ar') return saved
  return navigator.language?.toLowerCase().startsWith('ar') ? 'ar' : 'en'
}

export const i18n = createI18n({
  legacy: false,
  locale: detectInitialLocale(),
  fallbackLocale: 'en',
  messages: { en, ar },
})

export function setAppLocale(locale: AppLocale) {
  i18n.global.locale.value = locale
  document.documentElement.lang = locale
  document.documentElement.dir = locale === 'ar' ? 'rtl' : 'ltr'
  localStorage.setItem(STORAGE_KEY, locale)
}

// initialize <html lang/dir> once
setAppLocale(i18n.global.locale.value as AppLocale)
