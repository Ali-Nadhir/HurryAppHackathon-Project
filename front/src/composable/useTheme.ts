import { useToggle, useDark } from '@vueuse/core'

export function useTheme() {
  const isDark = useDark({
    selector: 'html',
    attribute: 'class',
    valueDark: 'dark',
    valueLight: '',
    storageKey: 'vueuse-color-scheme',
  })

  const toggle = useToggle(isDark)

  return { isDark, toggle }
}
