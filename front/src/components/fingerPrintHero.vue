<template>
  <section
    class="relative min-h-[90vh] overflow-hidden bg-gradient-to-b from-slate-900 via-slate-950 to-black text-white"
  >
    <!-- floating blobs -->
    <div
      class="pointer-events-none absolute -top-20 -left-20 h-96 w-96 rounded-full blur-3xl opacity-30 animate-blob mix-blend-soft-light bg-indigo-600"
    ></div>
    <div
      class="pointer-events-none absolute top-40 -right-10 h-[28rem] w-[28rem] rounded-full blur-3xl opacity-30 animate-blob animation-delay-2000 mix-blend-soft-light bg-fuchsia-600"
    ></div>
    <div
      class="pointer-events-none absolute -bottom-24 left-1/2 -translate-x-1/2 h-[26rem] w-[26rem] rounded-full blur-3xl opacity-25 animate-blob animation-delay-4000 mix-blend-soft-light bg-cyan-500"
    ></div>

    <!-- content -->
    <div class="relative z-10 container mx-auto px-6">
      <nav class="flex items-center justify-between py-6">
        <div class="flex items-center gap-3">
          <div class="h-10 w-10 rounded-xl bg-white/10 grid place-items-center backdrop-blur">
            <i class="i bi-fingerprint text-xl"></i>
          </div>
          <span class="font-semibold tracking-wide">Fingerprint System</span>
        </div>
        <!-- <RouterLink to="/docs" class="text-sm text-white/70 hover:text-white transition"
          >Docs</RouterLink
        > -->
        <div class="flex gap-5">
          <button
            @click="toggle"
            class="inline-flex items-center gap-2 px-3 py-2 rounded-2xl border backdrop-blur text-sm transition border-black/10 bg-white/70 hover:bg-white dark:border-white/10 dark:bg-white/10 dark:hover:bg-white/15"
            aria-label="Toggle color mode"
          >
            <span v-if="isDark" class="i bi-moon-stars"></span>
            <span v-else class="i bi-brightness-high"></span>
            <span class="font-medium">{{ label }}</span>
          </button>
          <button
            @click="changeLang"
            class="inline-flex items-center gap-2 px-3 py-2 rounded-2xl border backdrop-blur text-sm transition border-black/10 bg-white/70 hover:bg-white dark:border-white/10 dark:bg-white/10 dark:hover:bg-white/15"
          >
            En/Ar
          </button>
        </div>
      </nav>

      <div class="grid lg:grid-cols-2 gap-10 items-center pt-8">
        <!-- left: copy -->
        <div class="text-center lg:text-right order-2 lg:order-1">
          <h1 class="text-4xl md:text-5xl font-extrabold leading-tight">
            Secure, Fast, and
            <span
              class="text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 to-fuchsia-400"
              >Beautiful</span
            >
          </h1>
          <p class="mt-4 text-white/70 max-w-xl lg:ml-auto lg:mr-0 mx-auto">
            Modren fingerprint system built with Vue&nbsp;3, Pinia, Tailwind, and SweetAlert2.
            Smooth animations, glass UI, and production-ready structure.
          </p>

          <div class="mt-8 flex flex-col sm:flex-row gap-4 sm:justify-center lg:justify-end">
            <button
              class="group inline-flex items-center justify-center gap-2 rounded-2xl px-6 py-3 bg-white text-slate-900 font-semibold shadow-lg hover:shadow-2xl transition active:scale-[.99]"
              @click="goScan"
            >
              Start Scan
              <i class="bi bi-arrow-left-short text-xl group-hover:translate-x-1 transition"></i>
            </button>

            <RouterLink
              to="/manageScan"
              class="inline-flex items-center justify-center gap-2 rounded-2xl px-6 py-3 bg-white/10 hover:bg-white/15 backdrop-blur border border-white/10 text-white font-semibold transition"
            >
              Manage Users
            </RouterLink>
          </div>

          <!-- small trust bar -->
          <div class="mt-8 flex items-center justify-center lg:justify-end gap-6 text-white/50">
            <span class="text-sm">Vue 3</span>
            <span class="text-sm">Pinia</span>
            <span class="text-sm">Tailwind</span>
            <span class="text-sm">SweetAlert2</span>
            <span class="text-sm">Axios</span>
          </div>
        </div>

        <!-- right: animated fingerprint card -->
        <div class="order-1 lg:order-2">
          <div
            class="relative mx-auto w-full max-w-xl rounded-3xl border border-white/10 bg-white/5 p-8 backdrop-blur-xl shadow-2xl"
          >
            <!-- glow ring -->
            <div
              class="absolute -inset-0.5 rounded-3xl bg-gradient-to-r from-indigo-500/30 to-fuchsia-500/30 blur-2xl"
            ></div>

            <div class="mx-auto size-72 md:size-80 grid place-items-center">
              <svg
                viewBox="0 0 200 200"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
                class="w-full h-full"
              >
                <g
                  stroke="url(#grad)"
                  stroke-width="3"
                  stroke-linecap="round"
                  stroke-linejoin="round"
                >
                  <!-- Outer arcs -->
                  <path d="M100,30 a70,70 0 1,1 0,140 a70,70 0 1,1 0,-140" class="trace trace0" />
                  <path d="M100,42 a58,58 0 1,1 0,116 a58,58 0 1,1 0,-116" class="trace trace1" />
                  <!-- Middle arcs -->
                  <path d="M100,56 a44,44 0 1,1 0,88 a44,44 0 1,1 0,-88" class="trace trace2" />
                  <path d="M100,68 a32,32 0 1,1 0,64 a32,32 0 1,1 0,-64" class="trace trace3" />
                  <!-- Core -->
                  <path d="M100,80 a20,20 0 1,1 0,40 a20,20 0 1,1 0,-40" class="trace trace4" />
                </g>

                <defs>
                  <linearGradient id="grad" x1="0" y1="0" x2="1" y2="1">
                    <stop offset="0%" stop-color="#818cf8" />
                    <!-- indigo-400 -->
                    <stop offset="100%" stop-color="#e879f9" />
                    <!-- fuchsia-400 -->
                  </linearGradient>
                </defs>
              </svg>
            </div>
          </div>
        </div>
      </div>

      <!-- scroll hint -->
      <div class="mt-12 flex justify-center">
        <div class="flex items-center gap-2 text-white/60 text-sm animate-bounce">
          <i class="bi bi-mouse"></i><span>Scroll</span>
        </div>
      </div>
    </div>
  </section>
</template>

<script setup lang="ts">
import { useRouter } from 'vue-router'
const router = useRouter()

const dash = (i: number) => {
  const len = [480, 390, 300, 220, 140][i] || 200
  return {
    strokeDasharray: `${len}`,
    animation: `trace 2.2s ease-in-out ${i * 0.18}s infinite alternate`,
  } as Record<string, string>
}

const goScan = () => {
  router.push('/newScan') // you can add this route later
}

import { computed } from 'vue'
import { useTheme } from '../composable/useTheme'
import Swal from '@/plugins/swal-theme'

const { isDark, toggle, mode } = useTheme()
const label = computed(() => (isDark.value ? 'Dark' : 'Light'))

const changeLang = () => {
  Swal.fire({
    title: 'easy take english course',
  })
}
</script>

<style scoped>
@keyframes trace {
  0% {
    stroke-dashoffset: 0;
    opacity: 0.55;
  }
  100% {
    stroke-dashoffset: -140;
    opacity: 1;
  }
}
@keyframes blob {
  0%,
  100% {
    transform: translate(0px, 0px) scale(1);
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
.animation-delay-4000 {
  animation-delay: 4s;
}

.trace {
  stroke-dasharray: 500;
  stroke-dashoffset: 500;
  animation: draw 3s ease-in-out infinite alternate;
}

.trace0 {
  animation-delay: 0s;
}
.trace1 {
  animation-delay: 0.3s;
}
.trace2 {
  animation-delay: 0.6s;
}
.trace3 {
  animation-delay: 0.9s;
}
.trace4 {
  animation-delay: 1.2s;
}

@keyframes draw {
  to {
    stroke-dashoffset: 0;
    opacity: 1;
  }
}
</style>
