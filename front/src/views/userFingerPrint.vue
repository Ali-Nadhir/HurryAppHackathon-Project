<template>
  <section
    class="relative min-h-screen overflow-hidden bg-gradient-to-b from-slate-50 via-slate-100 to-white text-slate-900 dark:from-slate-900 dark:via-slate-950 dark:to-black dark:text-white"
  >
    <!-- floating gradient blobs -->
    <div
      class="pointer-events-none absolute -top-24 -left-24 h-[28rem] w-[28rem] rounded-full blur-3xl opacity-40 mix-blend-multiply bg-indigo-500/50 animate-blob dark:opacity-30 dark:mix-blend-soft-light dark:bg-indigo-600"
    ></div>
    <div
      class="pointer-events-none absolute top-40 -right-14 h-[30rem] w-[30rem] rounded-full blur-3xl opacity-40 mix-blend-multiply bg-fuchsia-500/50 animate-blob animation-delay-2000 dark:opacity-30 dark:mix-blend-soft-light dark:bg-fuchsia-600"
    ></div>
    <div
      class="pointer-events-none absolute -bottom-24 left-1/2 -translate-x-1/2 h-[26rem] w-[26rem] rounded-full blur-3xl opacity-30 mix-blend-multiply bg-cyan-400/50 animate-blob animation-delay-4000 dark:opacity-25 dark:mix-blend-soft-light dark:bg-cyan-500"
    ></div>

    <div class="relative z-10 container mx-auto px-6 py-10">
      <!-- Header -->
      <div class="flex flex-col lg:flex-row items-start lg:items-center justify-between gap-6">
        <div class="flex items-center gap-4">
          <div
            class="h-12 w-12 rounded-xl grid place-items-center border text-sm font-bold bg-slate-900/5 border-slate-900/10 dark:bg-white/10 dark:border-white/10"
          >
            {{ initials(user?.name || 'User') }}
          </div>
          <div>
            <h1 class="text-2xl md:text-3xl font-extrabold leading-tight">
              {{ user?.name || 'User' }}
              <span class="ml-2 text-sm font-semibold text-slate-500 dark:text-white/60">
                (EMP{{ String(user?.id).padStart(4, '0') }})</span
              >
            </h1>
            <p class="mt-1 text-slate-600 text-sm dark:text-white/60">
              All captured fingerprints for {{ user?.name }}
            </p>
          </div>
        </div>
        <div class="flex flex-wrap items-center gap-3">
          <RouterLink
            to="/manageScan"
            class="px-3 py-2 rounded-2xl bg-white hover:bg-slate-50 border border-slate-900/10 backdrop-blur text-sm text-slate-900 dark:bg-white/10 dark:hover:bg-white/15 dark:border-white/10 dark:text-white"
            >← Back</RouterLink
          >
          <button
            @click="refresh"
            class="px-3 py-2 rounded-2xl bg-white hover:bg-slate-50 border border-slate-900/10 backdrop-blur text-sm text-slate-900 dark:bg-white/10 dark:hover:bg-white/15 dark:border-white/10 dark:text-white"
          >
            <i class="bi bi-arrow-clockwise mr-1"></i>Refresh
          </button>
        </div>
      </div>

      <!-- Filters -->
      <div class="mt-6 grid md:grid-cols-3 gap-4">
        <div>
          <label class="block text-xs text-slate-600 mb-1 dark:text-white/60">Search</label>
          <input
            v-model.trim="q"
            placeholder="Search by device or location…"
            class="w-full rounded-2xl bg-slate-200 border border-slate-300 px-4 py-2.5 shadow-sm backdrop-blur outline-none focus:ring-2 focus:ring-fuchsia-400/30 text-slate-900 placeholder-slate-400 dark:bg-white/5 dark:border-white/10 dark:focus:ring-fuchsia-400/40 dark:text-white dark:placeholder-white/60"
          />
        </div>
        <div>
          <label class="block text-xs text-slate-600 mb-1 dark:text-white/60">Finger</label>
          <div class="relative">
            <select
              v-model="finger"
              class="appearance-none w-full rounded-2xl bg-slate-200 border border-slate-300 px-4 py-2.5 pr-10 outline-none focus:ring-2 focus:ring-indigo-400/30 text-slate-900 backdrop-blur dark:bg-white/5 dark:border-white/10 dark:focus:ring-indigo-400/40 dark:text-white"
            >
              <option value="" class="bg-white text-slate-900 dark:bg-slate-900 dark:text-white">
                All fingers
              </option>
              <option
                v-for="f in FINGERS"
                :key="f.key"
                :value="f.id"
                class="bg-white text-slate-900 dark:bg-slate-900 dark:text-white"
              >
                {{ f.label }}
              </option>
            </select>
            <i
              class="bi bi-chevron-down absolute right-3 top-1/2 -translate-y-1/2 pointer-events-none text-slate-500 dark:text-white/60"
            ></i>
          </div>
        </div>
        <div>
          <label class="block text-xs text-slate-600 mb-1 dark:text-white/60">Sort</label>
          <div class="relative">
            <select
              v-model="sort"
              class="appearance-none w-full rounded-2xl bg-slate-200 border border-slate-300 px-4 py-2.5 pr-10 outline-none focus:ring-2 focus:ring-indigo-400/30 text-slate-900 backdrop-blur dark:bg-white/5 dark:border-white/10 dark:focus:ring-indigo-400/40 dark:text-white"
            >
              <option value="new" class="bg-white text-slate-900 dark:bg-slate-900 dark:text-white">
                Newest first
              </option>
              <option value="old" class="bg-white text-slate-900 dark:bg-slate-900 dark:text-white">
                Oldest first
              </option>
              <option
                value="conf"
                class="bg-white text-slate-900 dark:bg-slate-900 dark:text-white"
              >
                Highest confidence
              </option>
            </select>
            <i
              class="bi bi-chevron-down absolute right-3 top-1/2 -translate-y-1/2 pointer-events-none text-slate-500 dark:text-white/60"
            ></i>
          </div>
        </div>
      </div>

      <!-- Grid of captures -->
      <div class="mt-8 grid sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-5" v-if="paged.length">
        <div
          v-for="cap in paged"
          :key="cap.id"
          class="relative overflow-hidden rounded-3xl border bg-white backdrop-blur-xl shadow-lg border-slate-200 dark:border-white/10 dark:bg-white/5 dark:shadow-2xl"
        >
          <div
            class="absolute -inset-0.5 rounded-3xl bg-gradient-to-r from-indigo-500/10 to-fuchsia-500/10 blur-2xl dark:from-indigo-500/10 dark:to-fuchsia-500/10"
          ></div>
          <div class="relative p-4">
            <div
              class="rounded-2xl overflow-hidden border bg-white border-slate-200 dark:border-white/10 dark:bg-black/40"
            >
              <div class="aspect-square grid place-items-center">
                <img
                  v-if="imgMap[cap.id] || cap.thumb"
                  :src="imgMap[cap.id] || cap.thumb"
                  class="w-full h-full object-cover"
                  :alt="`Fingerprint ${cap.id}`"
                  @error="onImgError(cap.id)"
                />

                <div
                  v-else
                  class="h-36 w-full grid place-items-center text-slate-400 text-xs dark:text-white/50"
                >
                  Generating…
                </div>
              </div>
            </div>
            <div class="mt-3 flex items-center justify-between text-sm">
              <span
                class="px-2 py-1 rounded-lg border bg-white border-slate-200 dark:border-white/10 dark:bg-white/5"
                >{{ labelFor(cap.finger) }}</span
              >
              <!-- <span class="text-white/70">{{ cap.confidence }}%</span> -->
            </div>
            <!-- <div class="mt-1 text-xs" :class="confidenceClass(cap.confidence)">
              {{ confidenceLabel(cap.confidence) }}
            </div> -->
            <div class="mt-2 text-xs text-slate-600 dark:text-white/60">
              {{ new Date(cap.created_at).toLocaleString() }} • {{ cap.device_name }} ({{
                cap.device_loc
              }})
            </div>
            <div class="mt-3 flex items-center gap-2">
              <button
                @click="view(cap)"
                class="px-3 py-2 rounded-xl bg-white hover:bg-slate-50 border border-slate-200 backdrop-blur text-sm text-slate-900 dark:bg-white/10 dark:hover:bg-white/15 dark:border-white/10 dark:text-white"
              >
                <i class="bi bi-eye mr-1"></i>View
              </button>
              <button
                :disabled="!cap.thumb"
                @click="download(cap)"
                class="px-3 py-2 rounded-xl bg-white hover:bg-slate-50 border border-slate-200 backdrop-blur text-sm disabled:opacity-50 disabled:cursor-not-allowed text-slate-900 dark:bg-white/10 dark:hover:bg-white/15 dark:border-white/10 dark:text-white"
              >
                <i class="bi bi-download mr-1"></i>Download
              </button>
              <button
                @click="remove(cap.id)"
                class="ml-auto px-3 py-2 rounded-xl bg-white hover:bg-rose-50 border border-rose-200 backdrop-blur text-sm text-rose-600 dark:bg-white/10 dark:hover:bg-white/15 dark:border-white/10 dark:text-rose-300"
              >
                <i class="bi bi-trash mr-1"></i>Delete
              </button>
            </div>
          </div>
        </div>
      </div>

      <!-- Empty state (when paged.length === 0) -->
      <div v-else class="mt-8">
        <div
          class="relative overflow-hidden rounded-3xl border bg-white backdrop-blur-xl shadow-lg border-slate-200 dark:border-white/10 dark:bg-white/5 dark:shadow-2xl"
        >
          <div
            class="absolute -inset-0.5 rounded-3xl bg-gradient-to-r from-indigo-500/10 to-fuchsia-500/10 blur-2xl dark:from-indigo-500/10 dark:to-fuchsia-500/10"
          ></div>

          <div class="relative p-8 flex flex-col md:flex-row items-center gap-6">
            <!-- Icon -->
            <div
              class="grid place-items-center size-16 rounded-2xl border bg-white border-slate-200 dark:border-white/10 dark:bg-white/10"
            >
              <i class="bi bi-fingerprint text-2xl opacity-80"></i>
            </div>

            <!-- Text -->
            <div class="text-center md:text-left md:flex-1">
              <h3 class="text-lg font-semibold text-slate-900 dark:text-white">
                No fingerprints yet
              </h3>
              <p class="text-sm text-slate-600 dark:text-white/60 mt-1">
                You haven’t added any fingerprint scans. Want to add your first one?
              </p>
            </div>

            <RouterLink
              :to="{ name: 'newScan' }"
              class="px-4 py-2 rounded-xl bg-white text-slate-900 text-sm hover:bg-slate-200/50 dark:bg-white/10 dark:hover:bg-white/15 border border-slate-900/10 dark:border-white/10 transition focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-500"
            >
              <i class="bi bi-plus-lg mr-1"></i>Add fingerprint
            </RouterLink>
          </div>
        </div>
      </div>

      <!-- Pagination -->
      <div
        class="flex flex-col sm:flex-row items-center justify-between gap-4 px-4 sm:px-6 py-4 border-t border-slate-200 dark:border-white/10"
      >
        <div class="text-xs text-slate-500 dark:text-white/50">
          Page {{ page }} of {{ totalPages }}
        </div>
        <div class="flex items-center gap-2">
          <button
            class="px-3 py-2 rounded-xl bg-white hover:bg-slate-50 border border-slate-200 backdrop-blur disabled:opacity-50 disabled:cursor-not-allowed text-slate-900 dark:bg-white/10 dark:hover:bg-white/15 dark:border-white/10 dark:text-white"
            :disabled="page === 1"
            @click="goto(1)"
          >
            <i class="bi bi-chevron-double-left"></i>
          </button>
          <button
            class="px-3 py-2 rounded-xl bg-white hover:bg-slate-50 border border-slate-200 backdrop-blur disabled:opacity-50 disabled:cursor-not-allowed text-slate-900 dark:bg-white/10 dark:hover:bg-white/15 dark:border-white/10 dark:text-white"
            :disabled="page === 1"
            @click="goto(page - 1)"
          >
            <i class="bi bi-chevron-left"></i>
          </button>
          <button
            class="px-3 py-2 rounded-xl bg-white hover:bg-slate-50 border border-slate-200 backdrop-blur disabled:opacity-50 disabled:cursor-not-allowed text-slate-900 dark:bg-white/10 dark:hover:bg-white/15 dark:border-white/10 dark:text-white"
            :disabled="page === totalPages"
            @click="goto(page + 1)"
          >
            <i class="bi bi-chevron-right"></i>
          </button>
          <button
            class="px-3 py-2 rounded-xl bg-white hover:bg-slate-50 border border-slate-200 backdrop-blur disabled:opacity-50 disabled:cursor-not-allowed text-slate-900 dark:bg-white/10 dark:hover:bg-white/15 dark:border-white/10 dark:text-white"
            :disabled="page === totalPages"
            @click="goto(totalPages)"
          >
            <i class="bi bi-chevron-double-right"></i>
          </button>
        </div>
      </div>
    </div>
  </section>
</template>

<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import { useRoute, RouterLink } from 'vue-router'
import Swal from '@/plugins/swal-theme'
import axios from 'axios'

const apiURL = import.meta.env.VITE_API_BASE

// --- route / user mock ---
interface UserDto {
  id: number
  name: string
  email: string
  phone: string
  role: string
  address: string
  age: number
  status: string
}
const route = useRoute()
const userId = Number(route.params.id || 1)
let user: UserDto | null = {
  name: 'abbas almahrus',
  email: 'test2@gmail.com',
  phone: '780456498484',
  role: 'Admin',
  address: 'baghdad\nbaghdad',
  age: 20,
  id: 3,
  status: 'active',
}

const fetchOneUser = async () => {
  try {
    const response = await axios.get(apiURL + 'people/' + userId)
    if (response.status == 200) {
      user = response.data
    }
  } catch (error) {
    Swal.fire({
      title: 'user not found',
    })
  }
}

function goto(p: number) {
  const target = Math.min(Math.max(1, p), totalPages.value)
  if (target !== page.value) {
    page.value = target
  }
}

// --- fingerprints mock ---

type FingerKey =
  | 'L_THUMB'
  | 'L_INDEX'
  | 'L_MIDDLE'
  | 'L_RING'
  | 'L_PINKY'
  | 'R_THUMB'
  | 'R_INDEX'
  | 'R_MIDDLE'
  | 'R_RING'
  | 'R_PINKY'

const FINGERS = [
  { id: 1, key: 'L_THUMB', label: 'Left Thumb' },
  { id: 2, key: 'L_INDEX', label: 'Left Index' },
  { id: 3, key: 'L_MIDDLE', label: 'Left Middle' },
  { id: 4, key: 'L_RING', label: 'Left Ring' },
  { id: 5, key: 'L_PINKY', label: 'Left Pinky' },
  { id: 6, key: 'R_THUMB', label: 'Right Thumb' },
  { id: 7, key: 'R_INDEX', label: 'Right Index' },
  { id: 8, key: 'R_MIDDLE', label: 'Right Middle' },
  { id: 9, key: 'R_RING', label: 'Right Ring' },
  { id: 10, key: 'R_PINKY', label: 'Right Pinky' },
] as const

const captures = ref<Capture[]>([])

interface Capture {
  id: number
  finger: number
  created_at: number
  device_name: string
  device_loc: string
  confidence: number
  thumb: string | null
}

const fingerPrints = ref<Capture[]>([])

async function fetchPrints() {
  try {
    const { data } = await axios.get(`${apiURL}people/${userId}/scans`)
    // map once and keep types aligned
    const mapped: Capture[] = data.map((scan: any) => {
      const d = randomDevice()
      return {
        id: scan.id,
        finger: Number(scan.finger), // ensure number
        created_at: randomDate(),
        device_name: d.name, // one call
        device_loc: d.loc,
        confidence: 50 + Math.floor(Math.random() * 50),
        thumb: null,
      }
    })
    captures.value = mapped

    // lazily generate thumbnails (optional)
    for (const cap of captures.value) {
      // if you want finger-specific bias, map id->key for the generator:
      const key = idToKey(cap.finger)
      cap.thumb = await generateFingerprintPng(key)
    }
  } catch (error) {
    console.error(error)
  }
}

const fetchImage = async (id: number) => {
  try {
    const { data, headers } = await axios.get(apiURL + 'scans/' + id, {
      responseType: 'blob',
    })

    if (data instanceof Blob) {
      // sanity: ensure it's an image (best-effort)
      const mime = data.type || headers['content-type'] || 'image/png'
      const blob = data.type ? data : new Blob([data], { type: mime })
      return URL.createObjectURL(blob)
    }
  } catch (error) {
    console.log(error)
  }
}

const imgMap = ref({})

function idToKey(id: number): FingerKey {
  return (FINGERS.find((f) => f.id === id)?.key || 'L_INDEX') as FingerKey
}

function randomDate(): number {
  const now = Date.now()
  const past = now - 1000 * 60 * 60 * 24 * 30 // last 30 days
  return past + Math.random() * (now - past)
}

function randomDevice() {
  const devices = [
    { name: 'ZK-U450', loc: 'Front Desk' },
    { name: 'SecuScan-2', loc: 'Server Room' },
    { name: 'BioNet-Edge', loc: 'HR Office' },
  ]
  return devices[Math.floor(Math.random() * devices.length)]
}
function labelFor(id: number) {
  return FINGERS.find((f) => f.id === id)?.label || `Finger ${id}`
}

const q = ref('')
const finger = ref<number | ''>('')
const sort = ref<'new' | 'old' | 'conf'>('new')
const page = ref(1)
const pageSize = ref(12)

onMounted(async () => {
  fetchOneUser()
  fetchPrints()
  // create mock list
  //   const base: Capture[] = Array.from({ length: 28 }).map((_, i) => ({
  //     id: i + 1,
  //     finger: FINGERS[i % FINGERS.length].key as FingerKey,
  //     confidence: 70 + Math.floor(Math.random() * 30),
  //     created_at: Date.now() - i * 3600 * 1000 * 4,
  //     device_name: ['ZK-U450', 'SecuScan-2', 'BioNet-Edge'][i % 3],
  //     device_loc: ['Front Desk', 'Server Room', 'HR Office'][i % 3],
  //     thumb: null,
  //   }))
  //   captures.value = base
  //   // lazily generate thumbnails
  //   for (const cap of captures.value) {
  //     cap.thumb = await generateFingerprintPng(cap.finger)
  //   }
})

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

const filtered = computed(() => {
  let list = captures.value
  if (q.value) {
    const s = q.value.toLowerCase()
    list = list.filter(
      (c) => c.device_name.toLowerCase().includes(s) || c.device_loc.toLowerCase().includes(s),
    )
  }
  if (finger.value) list = list.filter((c) => c.finger === finger.value)
  if (sort.value === 'new') list = [...list].sort((a, b) => b.created_at - a.created_at)
  else if (sort.value === 'old') list = [...list].sort((a, b) => a.created_at - b.created_at)
  else if (sort.value === 'conf') list = [...list].sort((a, b) => b.confidence - a.confidence)
  return list
})

const totalPages = computed(() => Math.max(1, Math.ceil(filtered.value.length / pageSize.value)))
const paged = computed(() =>
  filtered.value.slice((page.value - 1) * pageSize.value, page.value * pageSize.value),
)

async function loadPageImages() {
  const list = paged.value
  for (const cap of list) {
    if (imgMap.value[cap.id]) continue
    const url = await fetchImage(cap.id)
    if (url) {
      imgMap.value[cap.id] = url
    }
  }
}

watch(
  () => paged.value.map((c) => c.id).join(','),
  () => {
    loadPageImages()
  },
  { immediate: true },
)

function refresh() {
  page.value = 1
}

function initials(name: string) {
  return name
    .split(' ')
    .filter(Boolean)
    .slice(0, 2)
    .map((s) => s[0]?.toUpperCase())
    .join('')
}

async function ensureImageUrl(cap: Capture): Promise<string | null> {
  let url = imgMap.value[cap.id]
  if (!url) {
    url = await fetchImage(cap.id).catch(() => null)
    if (url) imgMap.value[cap.id] = url // cache
  }
  return url
}

async function view(cap: Capture) {
  // Try to get the full image (fetch if needed)
  const full = await ensureImageUrl(cap)
  const src = full || cap.thumb

  await Swal.fire({
    title: labelFor(cap.finger),
    imageUrl: src,
    imageAlt: `Fingerprint ${cap.id}`,
    imageWidth: 360,
    showConfirmButton: false,
    footer: `<small>${new Date(cap.created_at).toLocaleString()} • ${cap.device_name} (${cap.device_loc})${cap.confidence != null ? ` • ${cap.confidence}%` : ''}</small>`,
  })
}

function guessExt(src: string) {
  if (src.startsWith('data:')) {
    const m = /^data:image\/(\w+)/.exec(src)
    return (m?.[1] || 'png').replace('jpeg', 'jpg')
  }
  const m = src.match(/\.(png|jpe?g|webp|gif|bmp|tiff?)(?=$|\?)/i)
  return (m?.[1] || 'png').toLowerCase().replace('jpeg', 'jpg')
}

async function download(cap: Capture) {
  const src = imgMap.value[cap.id] || cap.thumb
  if (!src) return

  const a = document.createElement('a')
  a.href = src
  a.download = `fp-${userId}-${cap.id}.${guessExt(src)}`
  document.body.appendChild(a)
  a.click()
  a.remove()
}

function remove(id: number) {
  captures.value = captures.value.filter((c) => c.id !== id)
}

async function generateFingerprintPng(finger: FingerKey): Promise<string> {
  const size = 384
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

  // vignette
  const vg = ctx.createRadialGradient(size / 2, size / 2, size / 6, size / 2, size / 2, size / 1.05)
  vg.addColorStop(0, 'rgba(0,0,0,0)')
  vg.addColorStop(1, 'rgba(0,0,0,0.55)')
  ctx.fillStyle = vg
  ctx.fillRect(0, 0, size, size)

  // rings
  const grad = ctx.createLinearGradient(0, 0, size, size)
  grad.addColorStop(0, '#818cf8')
  grad.addColorStop(1, '#e879f9')
  ctx.strokeStyle = grad
  ctx.lineWidth = 3
  ctx.lineCap = 'round'

  const map = {
    L_THUMB: 0,
    L_INDEX: 1,
    L_MIDDLE: 2,
    L_RING: 3,
    L_PINKY: 4,
    R_THUMB: 0,
    R_INDEX: 1,
    R_MIDDLE: 2,
    R_RING: 3,
    R_PINKY: 4,
  } as Record<FingerKey, number>
  const bias = map[finger] * 0.08
  const rings = 18
  for (let i = 0; i < rings; i++) {
    const r = 42 + i * 12
    ctx.beginPath()
    ctx.ellipse(
      size / 2 + (i % 2 ? 6 : -6),
      size / 2 + (i % 3 ? 4 : -4),
      r * (0.98 - bias),
      r * 0.85,
      0,
      0,
      Math.PI * (1.8 - (i % 4 ? 0.2 : 0)),
    )
    ctx.stroke()
  }

  // tiny noise
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
