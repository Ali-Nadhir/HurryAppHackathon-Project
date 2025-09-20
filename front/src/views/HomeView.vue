<template>
  <main
    class="text-slate-900 bg-slate-50 dark:text-white dark:bg-gradient-to-b dark:from-slate-900 dark:via-slate-950 dark:to-black"
  >
    <finger-print-hero />

    <section class="relative py-16 md:py-24">
      <div class="relative z-10 container mx-auto px-6">
        <header class="text-center max-w-2xl mx-auto">
          <h2 class="text-2xl md:text-3xl font-extrabold text-slate-900 dark:text-white">
            {{ $t('common.Do') }}
          </h2>
          <p class="text-slate-600 dark:text-white/60 mt-2">
            {{ $t('common.fast') }}
          </p>
        </header>

        <!-- cards -->
        <div class="mt-10 grid sm:grid-cols-2 lg:grid-cols-3 gap-5">
          <!-- Card: Check Fingerprint -->
          <RouterLink
            to="/checkScan"
            class="group relative overflow-hidden rounded-3xl border border-slate-900/10 dark:bg-white/5 dark:shadow-2xl dark:border-white/10 bg-white/5 backdrop-blur-xl shadow-2xl transition-transform hover:-translate-y-0.5 dark:hover:shadow-[0_30px_60px_rgba(0,0,0,0.35)];"
          >
            <div class="feature-glow"></div>
            <div class="relative p-6">
              <div
                class="size-11 rounded-2xl grid place-items-center bg-slate-900/10 dark:bg-white/10 border-slate900/10 border dark:border-white/10"
              >
                <i class="bi bi-shield-check text-xl"></i>
              </div>
              <h3 class="mt-4 text-lg font-semibold text-slate-900 dark:text-white">
                {{ $t('common.CheckFingerprint') }}
              </h3>
              <p class="mt-1 text-sm text-slate-600 dark:text-white/60">
                {{ $t('common.card3') }}
              </p>
              <div
                class="mt-5 inline-flex items-center gap-1 text-sm text-indigo-600 dark:text-indigo-300 group-hover:gap-2 transition-all"
              >
                {{ $t('common.checker') }} <i class="bi bi-arrow-right-short text-xl"></i>
              </div>
            </div>
          </RouterLink>

          <!-- Card: Add New Scan -->
          <RouterLink
            to="/newScan"
            class="group relative overflow-hidden rounded-3xl border border-slate-900/10 dark:bg-white/5 dark:shadow-2xl dark:border-white/10 bg-white/5 backdrop-blur-xl shadow-2xl transition-transform hover:-translate-y-0.5 dark:hover:shadow-[0_30px_60px_rgba(0,0,0,0.35)];"
          >
            <div class="feature-glow"></div>
            <div class="relative p-6">
              <div
                class="size-11 rounded-2xl grid place-items-center bg-slate-900/10 dark:bg-white/10 border-slate900/10 border dark:border-white/10"
              >
                <i class="bi bi-fingerprint text-xl"></i>
              </div>
              <h3 class="mt-4 text-lg font-semibold text-slate-900 dark:text-white">
                {{ $t('common.newScan') }}
              </h3>
              <p class="mt-1 text-sm text-slate-600 dark:text-white/60">
                {{ $t('common.card2') }}
              </p>
              <div
                class="mt-5 inline-flex items-center gap-1 text-sm text-indigo-600 dark:text-indigo-300 group-hover:gap-2 transition-all"
              >
                {{ $t('common.start') }} <i class="bi bi-arrow-right-short text-xl"></i>
              </div>
            </div>
          </RouterLink>

          <!-- Card: Manage Scans -->
          <RouterLink
            to="/manageScan"
            class="group relative overflow-hidden rounded-3xl border border-slate-900/10 dark:bg-white/5 dark:shadow-2xl dark:border-white/10 bg-white/5 backdrop-blur-xl shadow-2xl transition-transform hover:-translate-y-0.5 dark:hover:shadow-[0_30px_60px_rgba(0,0,0,0.35)];"
          >
            <div class="feature-glow"></div>
            <div class="relative p-6">
              <div
                class="size-11 rounded-2xl grid place-items-center bg-slate-900/10 dark:bg-white/10 border-slate900/10 border dark:border-white/10"
              >
                <i class="bi bi-collection text-xl"></i>
              </div>
              <h3 class="mt-4 text-lg font-semibold text-slate-900 dark:text-white">
                {{ $t('common.ManageUsers') }}
              </h3>
              <p class="mt-1 text-sm text-slate-900 dark:text-white/60">
                {{ $t('common.card1') }}
              </p>
              <div
                class="mt-5 inline-flex items-center gap-1 text-sm text-indigo-600 dark:text-indigo-300 group-hover:gap-2 transition-all"
              >
                {{ $t('common.ManageUsers') }} <i class="bi bi-arrow-right-short text-xl"></i>
              </div>
            </div>
          </RouterLink>
        </div>

        <!-- stats / highlights -->
        <div class="mt-14 grid sm:grid-cols-3 gap-5">
          <div
            class="rounded-2xl border bg-white border-slate-900/10 dark:border-white/10 dark:bg-white/5 dark:backdrop-blur/10 p-5 shadow"
          >
            <p class="text-xs text-slate-600 dark:text-white/60">Latency</p>
            <p class="text-2xl text-slate-900 dark:text-white font-semibold">~120ms</p>
          </div>
          <div
            class="rounded-2xl border bg-white border-slate-900/10 dark:border-white/10 dark:bg-white/5 dark:backdrop-blur/10 p-5 shadow"
          >
            <p class="text-xs text-slate-600 dark:text-white/60">Matches/sec</p>
            <p class="text-2xl text-slate-900 dark:text-white font-semibold">10k+</p>
          </div>
          <div
            class="rounded-2xl border bg-white border-slate-900/10 dark:border-white/10 dark:bg-white/5 dark:backdrop-blur/10 p-5 shadow"
          >
            <p class="text-xs text-slate-600 dark:text-white/60">Security</p>
            <p class="text-2xl text-slate-900 dark:text-white font-semibold">AES‑256</p>
          </div>
        </div>
      </div>
    </section>

    <!-- FOOTER -->
    <footer class="relative border-t dark:border-white/10 border-slate-900/10">
      <div
        class="absolute -top-10 h-20 bg-gradient-to-t from-indigo-500/10 to-transparent blur-2xl"
      ></div>
      <div class="relative container mx-auto px-6 py-10">
        <div class="grid md:grid-cols-3 gap-8 items-start">
          <div>
            <div class="flex items-center gap-3">
              <div
                class="h-10 w-10 rounded-xl bg-slate-900/5 dark:bg-white/10 grid place-items-center backdrop-blur"
              >
                <i class="bi bi-fingerprint text-xl"></i>
              </div>
              <span class="font-semibold tracking-wide text-slate-900 dark:text-white">{{
                $t('common.appName')
              }}</span>
            </div>
            <p class="dark:text-white/60 mt-3 max-w-sm text-slate-600">
              {{ $t('common.desc') }}
            </p>
          </div>

          <div class="text-sm">
            <h4 class="font-semibold mb-3 text-slate-900 dark:text-white">
              {{ $t('common.quick') }}
            </h4>
            <ul class="space-y-2 dark:text-white/70 text-slate-600">
              <li>
                <RouterLink class="hover:text-slate-900 dark:hover:text-white" to="/checkScan">{{
                  $t('common.CheckFingerprint')
                }}</RouterLink>
              </li>
              <li>
                <RouterLink class="hover:text-slate-900 dark:hover:text-white" to="/newScan">{{
                  $t('common.newScan')
                }}</RouterLink>
              </li>
              <li>
                <RouterLink class="hover:text-slate-900 dark:hover:text-white" to="/manageScan">{{
                  $t('common.ManageUsers')
                }}</RouterLink>
              </li>
            </ul>
          </div>

          <div class="text-sm">
            <h4 class="font-semibold mb-3 text-slate-900 dark:text-white">Tech</h4>
            <ul class="space-y-2 text-slate-600 dark:text-white/70">
              <li>Vue 3 + Vite</li>
              <li>Pinia + Router</li>
              <li>Tailwind + SweetAlert2</li>
            </ul>
          </div>
        </div>

        <div
          class="mt-10 flex flex-col sm:flex-row items-center justify-between gap-4 text-slate-500 dark:text-white/50 text-xs"
        >
          <p>© {{ year }} Fingerprint System. All rights reserved.</p>
          <div class="flex items-center gap-4">
            <a class="hover:text-slate-900 dark:hover:text-white" href="#">Privacy</a>
            <a class="hover:text-slate-900 dark:hover:text-white" href="#">Terms</a>
            <a class="hover:text-slate-900 dark:hover:text-white" href="#">Status</a>
          </div>
        </div>
      </div>
    </footer>
  </main>
</template>

<script setup lang="ts">
import fingerPrintHero from '@/components/fingerPrintHero.vue'
import { computed } from 'vue'
import { useLocale } from '@/composable/useLocale'

const { locale, t } = useLocale()

const year = computed(() => new Date().getFullYear())
</script>

<style scoped>
.feature-glow {
  position: absolute;
  inset: -2px;
  border-radius: 1.5rem;
  filter: blur(28px);
  background: linear-gradient(90deg, rgba(129, 140, 248, 0.2), rgba(232, 121, 249, 0.2));
}

/***** blob anim reuse *****/
@keyframes blob {
  0%,
  100% {
    transform: translate(0, 0) scale(1);
  }
  50% {
    transform: translate(20px, -20px) scale(1.05);
  }
}
.animate-blob {
  animation: blob 12s ease-in-out infinite;
}
.animation-delay-2000 {
  animation-delay: 2s;
}
</style>
