import { computed } from 'vue'
import { i18n, setAppLocale } from '@/i18n'

export function useLocale() {
  const locale = computed({
    get: () => i18n.global.locale.value as 'en' | 'ar',
    set: (val: 'en' | 'ar') => setAppLocale(val),
  })
  const t = i18n.global.t
  const toggleLan = () => setAppLocale(locale.value === 'ar' ? 'en' : 'ar')
  return { locale, t, toggleLan, setLocale: setAppLocale }
}
