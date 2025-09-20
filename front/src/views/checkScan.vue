<template>
  <section
    class="relative min-h-screen overflow-hidden bg-gradient-to-b from-slate-50 via-slate-100 to-white text-slate-900 dark:from-slate-900 dark:via-slate-950 dark:to-black dark:text-white"
  >
    <!-- floating gradient blobs -->
    <div
      class="pointer-events-none absolute -top-24 -left-20 h-[28rem] w-[28rem] rounded-full blur-3xl opacity-40 mix-blend-multiply bg-indigo-500/50 animate-blob dark:opacity-30 dark:mix-blend-soft-light dark:bg-indigo-600"
    ></div>
    <div
      class="pointer-events-none absolute top-40 -right-10 h-[30rem] w-[30rem] rounded-full blur-3xl opacity-40 mix-blend-multiply bg-fuchsia-500/50 animate-blob animation-delay-2000 dark:opacity-30 dark:mix-blend-soft-light dark:bg-fuchsia-600"
    ></div>
    <div
      class="pointer-events-none absolute -bottom-24 left-1/2 -translate-x-1/2 h-[26rem] w-[26rem] rounded-full blur-3xl opacity-30 mix-blend-multiply bg-cyan-400/50 animate-blob animation-delay-4000 dark:opacity-25 dark:mix-blend-soft-light dark:bg-cyan-500"
    ></div>

    <div class="relative z-10 container mx-auto px-6 py-10">
      <!-- Header -->
      <div class="flex flex-col lg:flex-row items-start lg:items-center justify-between gap-6">
        <div>
          <h1 class="text-3xl md:text-4xl font-extrabold leading-tight">
            Check
            <span
              class="text-transparent bg-clip-text bg-gradient-to-r from-indigo-600 to-fuchsia-600 dark:from-indigo-400 dark:to-fuchsia-400"
              >Fingerprint</span
            >
          </h1>
          <p class="mt-1 text-slate-600 dark:text-white/60">
            Activate the device, scan, and preview the captured image (mock flow).
          </p>
        </div>
        <RouterLink
          to="/manageScan"
          class="px-4 py-2 rounded-2xl bg-white hover:bg-slate-50 border border-slate-900/10 backdrop-blur text-sm text-slate-900 dark:bg-white/10 dark:hover:bg-white/15 dark:border-white/10 dark:text-white"
          >← Back to Manage Scans</RouterLink
        >
      </div>

      <div class="mt-8 grid lg:grid-cols-2 gap-6">
        <!-- Left: Scan Controls Card -->
        <div
          class="relative rounded-3xl border bg-white backdrop-blur-xl shadow-lg overflow-hidden p-6 border-slate-200 dark:border-white/10 dark:bg-white/5 dark:shadow-2xl"
        >
          <div
            class="absolute -inset-0.5 rounded-3xl bg-gradient-to-r from-indigo-500/10 to-fuchsia-500/10 blur-2xl dark:from-indigo-500/20 dark:to-fuchsia-500/20"
          ></div>
          <div class="relative">
            <h3 class="text-lg font-semibold mb-4 text-slate-900 dark:text-white">Device</h3>

            <div class="grid sm:grid-cols-1 gap-4">
              <div class="relative">
                <label class="block text-sm text-slate-700 dark:text-white/70 mb-1"
                  >Select device</label
                >
                <select
                  v-model="deviceId"
                  class="appearance-none w-full rounded-2xl bg-white border border-slate-300 shadow-sm px-4 py-2.5 pr-10 outline-none focus:ring-2 focus:ring-indigo-400/30 text-slate-900 dark:bg-white/5 dark:border-white/10 dark:focus:ring-indigo-400/40 dark:text-white"
                >
                  <option
                    disabled
                    value=""
                    class="bg-white text-slate-900 dark:bg-slate-900 dark:text-white"
                  >
                    Select device…
                  </option>
                  <option
                    v-for="d in devices"
                    :key="d.id"
                    :value="d.id"
                    class="bg-white text-slate-900 dark:bg-slate-900 dark:text-white"
                  >
                    {{ d.name }} • {{ d.location }}
                  </option>
                </select>
                <i
                  class="bi bi-chevron-down absolute right-3 top-9 pointer-events-none text-slate-500 dark:text-white/60"
                ></i>
              </div>

              <!-- <div>
                <label class="block text-sm text-slate-700 dark:text-white/70 mb-1">Mode</label>
                <select
                  v-model="mode"
                  class="appearance-none w-full rounded-2xl bg-white border border-slate-300 shadow-sm px-4 py-2.5 pr-10 outline-none focus:ring-2 focus:ring-fuchsia-400/30 text-slate-900 dark:bg-white/5 dark:border-white/10 dark:focus:ring-fuchsia-400/40 dark:text-white"
                >
                  <option class="bg-white text-slate-900 dark:bg-slate-900 dark:text-white" value="quick">Quick</option>
                  <option class="bg-white text-slate-900 dark:bg-slate-900 dark:text-white" value="detailed">Detailed</option>
                </select>
              </div> -->
            </div>

            <div class="mt-6 flex items-center gap-3">
              <button
                :disabled="loading"
                @click="scan"
                class="px-5 py-2.5 rounded-2xl bg-slate-900 text-white font-semibold shadow-lg hover:shadow-2xl disabled:opacity-60 disabled:cursor-not-allowed dark:bg-white dark:text-slate-900"
              >
                <i class="bi bi-fingerprint mr-1"></i> Scan
              </button>
              <button
                :disabled="loading || !capturedImage"
                @click="download"
                class="px-5 py-2.5 rounded-2xl bg-white hover:bg-slate-50 border border-slate-900/10 backdrop-blur text-sm text-slate-900 disabled:opacity-60 disabled:cursor-not-allowed dark:bg-white/10 dark:hover:bg-white/15 dark:border-white/10 dark:text-white"
              >
                <i class="bi bi-download mr-1"></i> Download PNG
              </button>
              <button
                :disabled="loading || !capturedImage"
                @click="reset"
                class="px-5 py-2.5 rounded-2xl bg-white hover:bg-slate-50 border border-slate-900/10 backdrop-blur text-sm text-slate-900 disabled:opacity-60 disabled:cursor-not-allowed dark:bg-white/10 dark:hover:bg-white/15 dark:border-white/10 dark:text-white"
              >
                <i class="bi bi-arrow-clockwise mr-1"></i> Reset
              </button>
            </div>

            <!-- Meta info -->
            <div class="mt-6 grid sm:grid-cols-2 gap-4">
              <div
                class="rounded-2xl border bg-white p-4 border-slate-200 dark:border-white/10 dark:bg-white/5"
              >
                <p class="text-xs text-slate-600 dark:text-white/60">State</p>
                <p class="text-lg font-semibold text-slate-900 dark:text-white">{{ stateLabel }}</p>
              </div>
              <div
                class="rounded-2xl border bg-white p-4 border-slate-200 dark:border-white/10 dark:bg-white/5"
              >
                <p class="text-xs text-slate-600 dark:text-white/60">Device</p>
                <p class="text-lg font-semibold truncate text-slate-900 dark:text-white">
                  {{ deviceLabel }}
                </p>
              </div>
              <!-- <div class="rounded-2xl border bg-white p-4 border-slate-200 dark:border-white/10 dark:bg-white/5">
                <p class="text-xs text-slate-600 dark:text-white/60">Mode</p>
                <p class="text-lg font-semibold capitalize text-slate-900 dark:text-white">{{ mode }}</p>
              </div> -->
            </div>
          </div>
        </div>

        <!-- Right: Live/Result Card -->
        <div
          class="relative rounded-3xl border bg-white backdrop-blur-xl shadow-lg overflow-hidden p-6 border-slate-200 dark:border-white/10 dark:bg-white/5 dark:shadow-2xl"
        >
          <div
            class="absolute -inset-0.5 rounded-3xl bg-gradient-to-r from-indigo-500/10 to-fuchsia-500/10 blur-2xl dark:from-indigo-500/20 dark:to-fuchsia-500/20"
          ></div>
          <div class="relative">
            <h3 class="text-lg font-semibold mb-4 text-slate-900 dark:text-white">Live Preview</h3>

            <div v-if="!capturedImage" class="mx-auto size-80 grid place-items-center">
              <!-- animated fingerprint while scanning or idle -->
              <svg
                viewBox="0 0 200 200"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
                class="w-full h-full drop-shadow-[0_0_18px_rgba(99,102,241,0.35)]"
              >
                <g
                  stroke="url(#grad)"
                  stroke-width="3"
                  stroke-linecap="round"
                  stroke-linejoin="round"
                >
                  <path d="M100,30 a70,70 0 1,1 0,140 a70,70 0 1,1 0,-140" :class="scanClass(0)" />
                  <path d="M100,42 a58,58 0 1,1 0,116 a58,58 0 1,1 0,-116" :class="scanClass(1)" />
                  <path d="M100,56 a44,44 0 1,1 0,88 a44,44 0 1,1 0,-88" :class="scanClass(2)" />
                  <path d="M100,68 a32,32 0 1,1 0,64 a32,32 0 1,1 0,-64" :class="scanClass(3)" />
                  <path d="M100,80 a20,20 0 1,1 0,40 a20,20 0 1,1 0,-40" :class="scanClass(4)" />
                </g>
                <defs>
                  <linearGradient id="grad" x1="0" y1="0" x2="1" y2="1">
                    <stop offset="0%" stop-color="#818cf8" />
                    <stop offset="100%" stop-color="#e879f9" />
                  </linearGradient>
                </defs>
              </svg>
            </div>

            <div v-else class="mx-auto max-w-md">
              <div
                class="rounded-3xl overflow-hidden border border-slate-200 shadow-xl bg-white dark:border-white/10 dark:bg-black/40"
              >
                <img :src="capturedImage" alt="Captured fingerprint" class="w-full h-auto" />
              </div>
              <div class="mt-4 grid grid-cols-2 gap-4">
                <div
                  class="rounded-2xl border bg-white p-4 border-slate-200 dark:border-white/10 dark:bg-white/5"
                >
                  <p class="text-xs text-slate-600 dark:text-white/60">Captured</p>
                  <p class="text-lg font-semibold text-slate-900 dark:text-white">
                    {{ capturedAtLabel }}
                  </p>
                </div>
                <div
                  class="rounded-2xl border bg-white p-4 border-slate-200 dark:border-white/10 dark:bg_white/5 dark:bg-white/5"
                >
                  <p class="text-xs text-slate-600 dark:text-white/60">Confidence</p>
                  <p class="text-lg font-semibold text-slate-900 dark:text-white">
                    {{ confidence }}%
                  </p>
                </div>
                <div
                  class="rounded-2xl border bg-white p-4 border-slate-200 dark:border-white/10 dark:bg-white/5"
                >
                  <p class="text-xs text-slate-600 dark:text-white/60">Person</p>
                  <p class="text-lg font-semibold text-slate-900 dark:text-white">
                    {{ name || '—' }}
                  </p>
                </div>
                <div
                  class="rounded-2xl border bg-white p-4 border-slate-200 dark:border-white/10 dark:bg-white/5"
                >
                  <p class="text-xs text-slate-600 dark:text-white/60">Latency</p>
                  <p class="text-lg font-semibold text-slate-900 dark:text-white">
                    {{ latencyLabel }}
                  </p>
                </div>
              </div>
              <div
                class="mt-5 text-xs text-center text-slate-600 dark:text-white/60"
                :class="confidenceClass(confidence)"
              >
                {{ confidenceLabel(confidence) }}
              </div>
            </div>
          </div>
        </div>
      </div>
      <!-- /grid -->
    </div>
  </section>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import Swal from '@/plugins/swal-theme'
import axios from 'axios'

const apiURL = import.meta.env.VITE_API_BASE

function confidenceLabel(val: number) {
  if (val > 80) return 'Bank level security'
  if (val > 50) return 'Gym level security'
  return '😂 Just for fun'
}

function confidenceClass(val: number) {
  if (val > 80) return 'text-green-400'
  if (val > 50) return 'text-yellow-400'
  return 'text-rose-400'
}

interface DeviceDto {
  id: string
  name: string
  location: string
}
const devices: DeviceDto[] = [
  { id: 'usb-3', name: 'R307', location: 'Front Desk' },
  { id: 'usb-1', name: 'ZK-U450', location: 'Front Desk' },
  { id: 'usb-2', name: 'SecuScan-2', location: 'Server Room' },
  { id: 'ip-10.0.0.12', name: 'BioNet-Edge', location: 'HR Office' },
]

const deviceId = ref<string>('usb-3')
const mode = ref<'quick' | 'detailed'>('quick')
const loading = ref(false)
const scanning = ref(false)
const capturedImage = ref<string | null>(null)
const capturedAt = ref<number | null>(null)
const confidence = ref<number>(0)
const name = ref('')
const latency = ref<number>(0)

const stateLabel = computed(() =>
  scanning.value ? 'Scanning' : capturedImage.value ? 'Captured' : 'Idle',
)
const deviceLabel = computed(() => devices.find((d) => d.id === deviceId.value)?.name || '—')
const capturedAtLabel = computed(() =>
  capturedAt.value ? new Date(capturedAt.value).toLocaleString() : '—',
)
const latencyLabel = computed(() => {
  const v = latency.value ?? 0
  if (v < 1 && v >= 0) return `${Math.round(v * 1000)} ms` // seconds -> ms
  return `${v.toFixed(2)} s`
})

function scanClass(i: number) {
  return scanning.value ? `trace trace${i} animate` : `trace idle`
}

// normalize API match_percent which may be in [-1..1] or [0..100], clamp 0..100
function normalizePercent(raw: any): number {
  const n = Number(raw)
  if (!isFinite(n)) return 0
  const maybeFraction = Math.abs(n) <= 1.0001
  const pct = maybeFraction ? n * 100 : n
  return Math.max(0, Math.min(100, Math.round(pct)))
}

async function scan() {
  if (!deviceId.value) {
    await Swal.fire({ title: 'Select a device', icon: 'info' })
    return
  }

  // reset + start scanning visuals early
  loading.value = true
  scanning.value = true
  capturedImage.value = null
  capturedAt.value = null
  confidence.value = 0
  name.value = ''
  latency.value = 0

  try {
    // Kick off API call
    const req = axios.get(`${apiURL}match`)

    // Nice staged UI while waiting
    await Swal.fire({
      title: 'Activating device…',
      allowOutsideClick: false,
      didOpen: () => Swal.showLoading(),
      timer: 700,
    })
    await Swal.fire({
      title: 'Scanning…',
      allowOutsideClick: false,
      didOpen: () => Swal.showLoading(),
      timer: 1100,
    })

    // Await response
    const { data, status } = await req
    if (status === 200) {
      const result = data?.results?.[0] || {}
      confidence.value = normalizePercent(result.match_percent)
      name.value = result.name || ''
      latency.value = Number(data?.latency ?? 0)
    } else {
      throw new Error(`Unexpected status ${status}`)
    }

    // Generate the mock image after we have the result
    capturedImage.value = await generateFingerprintPng(mode.value)
    capturedAt.value = Date.now()

    await Swal.fire({
      title: 'Capture complete',
      icon: 'success',
      timer: 900,
      showConfirmButton: false,
    })
  } catch (err: any) {
    console.error(err)
    await Swal.fire({
      title: 'Scan failed',
      html: err?.response?.data?.message || err?.message || 'Unknown error',
      icon: 'error',
    })
  } finally {
    scanning.value = false
    loading.value = false
  }
}

function reset() {
  capturedImage.value = null
  capturedAt.value = null
  confidence.value = 0
  name.value = ''
  latency.value = 0
}

function download() {
  if (!capturedImage.value) return
  const a = document.createElement('a')
  a.href = capturedImage.value
  a.download = `fingerprint-${new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-')}.png`
  document.body.appendChild(a)
  a.click()
  a.remove()
}

async function generateFingerprintPng(detail: 'quick' | 'detailed'): Promise<string> {
  const size = 512
  const canvas = document.createElement('canvas')
  canvas.width = size
  canvas.height = size
  const ctx = canvas.getContext('2d')!

  // bg
  const g = ctx.createLinearGradient(0, 0, size, size)
  g.addColorStop(0, '#0f172a')
  g.addColorStop(1, '#020617')
  ctx.fillStyle = g
  ctx.fillRect(0, 0, size, size)

  // subtle vignette
  const vg = ctx.createRadialGradient(size / 2, size / 2, size / 6, size / 2, size / 2, size / 1.1)
  vg.addColorStop(0, 'rgba(0,0,0,0)')
  vg.addColorStop(1, 'rgba(0,0,0,0.5)')
  ctx.fillStyle = vg
  ctx.fillRect(0, 0, size, size)

  // fingerprint rings
  const grad = ctx.createLinearGradient(0, 0, size, size)
  grad.addColorStop(0, '#818cf8')
  grad.addColorStop(1, '#e879f9')
  ctx.strokeStyle = grad
  ctx.lineWidth = 3
  ctx.lineCap = 'round'

  const rings = detail === 'detailed' ? 24 : 16
  for (let i = 0; i < rings; i++) {
    const r = 40 + i * (detail === 'detailed' ? 9 : 12)
    ctx.beginPath()
    ctx.ellipse(size / 2, size / 2, r, r * 0.85, 0, 0, Math.PI * (1.8 - (i % 3 ? 0.0 : 0.2)))
    ctx.stroke()
  }

  // random micro-noise lines
  ctx.globalAlpha = 0.12
  for (let i = 0; i < 120; i++) {
    const x = Math.random() * size,
      y = Math.random() * size,
      w = 8 + Math.random() * 28
    ctx.beginPath()
    ctx.moveTo(x, y)
    ctx.lineTo(x + w, y + 1)
    ctx.stroke()
  }
  ctx.globalAlpha = 1

  return canvas.toDataURL('image/png')
}
</script>

<style scoped>
/* fingerprint animation */
.trace {
  stroke-dasharray: 500;
  stroke-dashoffset: 500;
  opacity: 0.7;
}
.trace.idle {
  stroke-dashoffset: 0;
  opacity: 0.35;
  transition: opacity 0.5s ease;
}
.trace.animate {
  animation: draw 1.4s ease-in-out infinite alternate;
}
.trace:nth-child(1).animate {
  animation-delay: 0s;
}
.trace:nth-child(2).animate {
  animation-delay: 0.18s;
}
.trace:nth-child(3).animate {
  animation-delay: 0.36s;
}
.trace:nth-child(4).animate {
  animation-delay: 0.54s;
}
.trace:nth-child(5).animate {
  animation-delay: 0.72s;
}
@keyframes draw {
  to {
    stroke-dashoffset: 0;
    opacity: 1;
  }
}

/* blobs */
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
.animation-delay-4000 {
  animation-delay: 4s;
}
</style>
