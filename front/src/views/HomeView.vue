<template>
  <main class="bg-gradient-to-b from-slate-900 via-slate-950 to-black text-white overflow-hidden">
    <finger-print-hero />

    <section class="relative py-16 md:py-24">
      <!-- floating blobs echo -->
      <div
        class="pointer-events-none absolute -top-24 -left-20 h-[22rem] w-[22rem] rounded-full blur-3xl opacity-25 animate-blob mix-blend-soft-light bg-fuchsia-600"
      ></div>
      <div
        class="pointer-events-none absolute -bottom-24 right-0 h-[22rem] w-[22rem] rounded-full blur-3xl opacity-20 animate-blob animation-delay-2000 mix-blend-soft-light bg-cyan-500"
      ></div>

      <div class="relative z-10 container mx-auto px-6">
        <header class="text-center max-w-2xl mx-auto">
          <h2 class="text-2xl md:text-3xl font-extrabold">Do more with your fingerprints</h2>
          <p class="text-white/60 mt-2">
            Fast actions to test, add and verify scans. Everything is mock-ready and API-friendly.
          </p>
        </header>

        <!-- cards -->
        <div class="mt-10 grid sm:grid-cols-2 lg:grid-cols-3 gap-5">
          <!-- Card: Check Fingerprint -->
          <RouterLink
            to="/checkScan"
            class="group relative overflow-hidden rounded-3xl border border-white/10 bg-white/5 backdrop-blur-xl shadow-2xl transition-transform hover:-translate-y-0.5 hover:shadow-[0_30px_60px_rgba(0,0,0,0.35)];"
          >
            <div class="feature-glow"></div>
            <div class="relative p-6">
              <div
                class="size-11 rounded-2xl grid place-items-center bg-white/10 border border-white/10"
              >
                <i class="bi bi-shield-check text-xl"></i>
              </div>
              <h3 class="mt-4 text-lg font-semibold">Check Fingerprint</h3>
              <p class="mt-1 text-sm text-white/60">
                Activate device, scan and preview a captured image (mock now, API later).
              </p>
              <div
                class="mt-5 inline-flex items-center gap-1 text-sm text-indigo-300 group-hover:gap-2 transition-all"
              >
                Go to checker <i class="bi bi-arrow-right-short text-xl"></i>
              </div>
            </div>
          </RouterLink>

          <!-- Card: Add New Scan -->
          <RouterLink
            to="/newScan"
            class="group relative overflow-hidden rounded-3xl border border-white/10 bg-white/5 backdrop-blur-xl shadow-2xl transition-transform hover:-translate-y-0.5 hover:shadow-[0_30px_60px_rgba(0,0,0,0.35)];"
          >
            <div class="feature-glow"></div>
            <div class="relative p-6">
              <div
                class="size-11 rounded-2xl grid place-items-center bg-white/10 border border-white/10"
              >
                <i class="bi bi-fingerprint text-xl"></i>
              </div>
              <h3 class="mt-4 text-lg font-semibold">Add New Scan</h3>
              <p class="mt-1 text-sm text-white/60">
                Pick a user and capture a new fingerprint. Guided SweetAlert2 stages.
              </p>
              <div
                class="mt-5 inline-flex items-center gap-1 text-sm text-indigo-300 group-hover:gap-2 transition-all"
              >
                Start a scan <i class="bi bi-arrow-right-short text-xl"></i>
              </div>
            </div>
          </RouterLink>

          <!-- Card: Manage Scans -->
          <RouterLink
            to="/manageScan"
            class="group relative overflow-hidden rounded-3xl border border-white/10 bg-white/5 backdrop-blur-xl shadow-2xl transition-transform hover:-translate-y-0.5 hover:shadow-[0_30px_60px_rgba(0,0,0,0.35)];"
          >
            <div class="feature-glow"></div>
            <div class="relative p-6">
              <div
                class="size-11 rounded-2xl grid place-items-center bg-white/10 border border-white/10"
              >
                <i class="bi bi-collection text-xl"></i>
              </div>
              <h3 class="mt-4 text-lg font-semibold">Manage Scans</h3>
              <p class="mt-1 text-sm text-white/60">
                Browse users, trigger scans, delete, and filter—all in one place.
              </p>
              <div
                class="mt-5 inline-flex items-center gap-1 text-sm text-indigo-300 group-hover:gap-2 transition-all"
              >
                Open dashboard <i class="bi bi-arrow-right-short text-xl"></i>
              </div>
            </div>
          </RouterLink>
        </div>

        <!-- stats / highlights -->
        <div class="mt-14 grid sm:grid-cols-3 gap-5">
          <div class="rounded-2xl border border-white/10 bg-white/5 backdrop-blur p-5 shadow">
            <p class="text-xs text-white/60">Latency</p>
            <p class="text-2xl font-semibold">~120ms</p>
          </div>
          <div class="rounded-2xl border border-white/10 bg-white/5 backdrop-blur p-5 shadow">
            <p class="text-xs text-white/60">Matches/sec</p>
            <p class="text-2xl font-semibold">10k+</p>
          </div>
          <div class="rounded-2xl border border-white/10 bg-white/5 backdrop-blur p-5 shadow">
            <p class="text-xs text-white/60">Security</p>
            <p class="text-2xl font-semibold">AES‑256</p>
          </div>
        </div>
      </div>
    </section>

    <!-- FOOTER -->
    <footer class="relative border-t border-white/10">
      <div
        class="absolute -inset-x-10 -top-10 h-20 bg-gradient-to-t from-indigo-500/10 to-transparent blur-2xl"
      ></div>
      <div class="relative container mx-auto px-6 py-10">
        <div class="grid md:grid-cols-3 gap-8 items-start">
          <div>
            <div class="flex items-center gap-3">
              <div class="h-10 w-10 rounded-xl bg-white/10 grid place-items-center backdrop-blur">
                <i class="bi bi-fingerprint text-xl"></i>
              </div>
              <span class="font-semibold tracking-wide">Fingerprint System</span>
            </div>
            <p class="text-white/60 mt-3 max-w-sm">
              A modern frontend built with Vue 3, Tailwind, Pinia, and SweetAlert2. Smooth
              animations, glass UI, mock-ready.
            </p>
          </div>

          <div class="text-sm">
            <h4 class="font-semibold mb-3">Quick Links</h4>
            <ul class="space-y-2 text-white/70">
              <li>
                <RouterLink class="hover:text-white" to="/check">Check Fingerprint</RouterLink>
              </li>
              <li><RouterLink class="hover:text-white" to="/scan">Add New Scan</RouterLink></li>
              <li><RouterLink class="hover:text-white" to="/devices">Manage Scans</RouterLink></li>
            </ul>
          </div>

          <div class="text-sm">
            <h4 class="font-semibold mb-3">Tech</h4>
            <ul class="space-y-2 text-white/70">
              <li>Vue 3 + Vite</li>
              <li>Pinia + Router</li>
              <li>Tailwind + SweetAlert2</li>
            </ul>
          </div>
        </div>

        <div
          class="mt-10 flex flex-col sm:flex-row items-center justify-between gap-4 text-white/50 text-xs"
        >
          <p>© {{ year }} Fingerprint System. All rights reserved.</p>
          <div class="flex items-center gap-4">
            <a class="hover:text-white" href="#">Privacy</a>
            <a class="hover:text-white" href="#">Terms</a>
            <a class="hover:text-white" href="#">Status</a>
          </div>
        </div>
      </div>
    </footer>
  </main>
</template>

<script setup lang="ts">
import fingerPrintHero from '@/components/fingerPrintHero.vue'
import { computed } from 'vue'

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
