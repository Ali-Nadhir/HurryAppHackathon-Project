import { useColorMode } from '@vueuse/core'
import { computed } from 'vue'
import Swal from '@/plugins/swal-theme'

export function useTheme() {
  // key is what powers the no-FOUC script
  const mode = useColorMode({
    emitAuto: true, // 'auto' follows system
    storageKey: 'vueuse-color-scheme',
    initialValue: 'auto', // 'auto' | 'dark' | 'light'
  })
  const isDark = computed(() => mode.value === 'dark')
  function toggle() {
    mode.value = isDark.value ? 'light' : 'dark'

    Swal.fire({
      title: ` if you can imagin it you can believe it, its ${mode.value} now 🧙‍♂️`,
    })
  }

  return { mode, isDark, toggle }
}
