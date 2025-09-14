<template>
  <section
    class="relative min-h-screen overflow-hidden bg-gradient-to-b from-slate-900 via-slate-950 to-black text-white"
  >
    <!-- floating gradient blobs -->
    <div
      class="pointer-events-none absolute -top-24 -left-20 h-[28rem] w-[28rem] rounded-full blur-3xl opacity-30 animate-blob mix-blend-soft-light bg-indigo-600"
    ></div>
    <div
      class="pointer-events-none absolute top-40 -right-10 h-[30rem] w-[30rem] rounded-full blur-3xl opacity-30 animate-blob animation-delay-2000 mix-blend-soft-light bg-fuchsia-600"
    ></div>
    <div
      class="pointer-events-none absolute -bottom-24 left-1/2 -translate-x-1/2 h-[26rem] w-[26rem] rounded-full blur-3xl opacity-25 animate-blob animation-delay-4000 mix-blend-soft-light bg-cyan-500"
    ></div>

    <div class="relative z-10 container mx-auto px-6 py-10">
      <!-- Header -->
      <div class="flex flex-col lg:flex-row items-start lg:items-center justify-between gap-6">
        <div>
          <h1 class="text-3xl md:text-4xl font-extrabold leading-tight">
            New
            <span
              class="text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 to-fuchsia-400"
              >Scan</span
            >
          </h1>
          <p class="mt-1 text-white/60">
            Select a user, choose a finger, pick a device — then run a mock scan.
          </p>
        </div>
        <RouterLink
          to="/manageScan"
          class="px-4 py-2 rounded-2xl bg-white/10 hover:bg-white/15 border border-white/10 backdrop-blur text-sm"
        >
          ← Back to Manage Scans
        </RouterLink>
      </div>

      <div class="mt-8 grid lg:grid-cols-2 gap-6">
        <!-- Left: Form Card (user + device + finger selector + hand preview) -->
        <div
          class="relative rounded-3xl border border-white/10 bg-white/5 backdrop-blur-xl shadow-2xl overflow-hidden p-6"
        >
          <div
            class="absolute -inset-0.5 rounded-3xl bg-gradient-to-r from-indigo-500/20 to-fuchsia-500/20 blur-2xl"
          ></div>
          <div class="relative">
            <!-- User Autocomplete -->
            <label class="block text-sm text-white/70 mb-1">User</label>
            <div class="relative">
              <input
                v-model.trim="userQuery"
                @input="onUserInput"
                type="text"
                placeholder="Search user by name or ID…"
                class="w-full rounded-2xl bg-white/5 border border-white/10 backdrop-blur px-4 py-2.5 outline-none focus:ring-2 focus:ring-fuchsia-400/40"
              />
              <i class="bi bi-search absolute right-3 top-1/2 -translate-y-1/2 text-white/50"></i>
              <ul
                v-if="showUserList"
                class="absolute z-10 mt-2 w-full max-h-64 overflow-auto rounded-2xl border border-white/10 bg-slate-900/80 backdrop-blur"
              >
                <li
                  v-for="u in filteredUsers"
                  :key="u.id"
                  @click="selectUser(u)"
                  class="px-4 py-2 hover:bg-white/10 cursor-pointer"
                >
                  <div class="flex items-center gap-3">
                    <div
                      class="size-8 rounded-full bg-gradient-to-br from-indigo-400/60 to-fuchsia-400/60 grid place-items-center text-[11px] font-bold"
                    >
                      {{ initials(u.name) }}
                    </div>
                    <div>
                      <div class="text-sm font-semibold">{{ u.name }}</div>
                      <div class="text-white/50 text-xs">{{ u.id }} • {{ u.email }}</div>
                    </div>
                  </div>
                </li>
                <li v-if="filteredUsers.length === 0" class="px-4 py-3 text-white/50 text-sm">
                  No results
                </li>
              </ul>
            </div>

            <!-- Selected User Chip -->
            <div
              v-if="selectedUser"
              class="mt-3 inline-flex items-center gap-2 px-3 py-1.5 rounded-2xl bg-white/10 border border-white/10"
            >
              <span class="text-sm">{{ selectedUser.name }} ({{ selectedUser.id }})</span>
              <button class="text-white/60 hover:text-white" @click="clearUser">
                <i class="bi bi-x"></i>
              </button>
            </div>

            <!-- Device -->
            <div class="mt-6">
              <label class="block text-sm text-white/70 mb-1">Device</label>
              <div class="relative">
                <select
                  v-model="deviceId"
                  class="appearance-none w-full rounded-2xl bg-white/5 border border-white/10 backdrop-blur px-4 py-2.5 pr-10 outline-none focus:ring-2 focus:ring-indigo-400/40 text-white"
                >
                  <option disabled value="">Select device…</option>
                  <option
                    v-for="d in devices"
                    :key="d.id"
                    :value="d.id"
                    class="bg-slate-900 text-white"
                  >
                    {{ d.name }} • {{ d.location }}
                  </option>
                </select>
                <i
                  class="bi bi-chevron-down absolute right-3 top-1/2 -translate-y-1/2 pointer-events-none text-white/60"
                ></i>
              </div>
            </div>

            <!-- Finger selection -->
            <div class="mt-6">
              <label class="block text-sm text-white/70 mb-2">Finger</label>
              <div class="grid grid-cols-2 sm:grid-cols-5 gap-2">
                <button
                  v-for="f in FINGERS"
                  :key="f.key"
                  @click="selectFinger(f)"
                  :class="[
                    'px-3 py-2 rounded-xl border text-xs backdrop-blur transition',
                    selectedFinger.key === f.key
                      ? 'bg-white text-slate-900 border-white/30 shadow'
                      : 'bg-white/5 text-white/80 border-white/10 hover:bg-white/10',
                  ]"
                >
                  {{ f.short }}
                </button>
              </div>
              <p class="mt-2 text-xs text-white/60">
                Selected: <span class="font-semibold">{{ fingerLabel }}</span>
              </p>

              <!-- Hand preview (small SVG under buttons) -->
              <div class="mt-4 w-full max-w-[420px]">
                <svg
                  viewBox="0 0 640 200"
                  class="w-full h-auto"
                  xmlns="http://www.w3.org/2000/svg"
                  fill="none"
                >
                  <g stroke="#94a3b8" stroke-opacity=".35" stroke-width="2" fill="#0b1220">
                    <!-- left hand palm -->
                    <path
                      d="M118 120c-6-26 10-54 32-66 22-12 46-9 58 18 14 31 15 73-6 88-25 18-76 2-84-40z"
                    />
                    <!-- right hand palm -->
                    <path
                      d="M522 120c6-26-10-54-32-66-22-12-46-9-58 18-14 31-15 73 6 88 25 18 76 2 84-40z"
                    />
                    <!-- left fingers -->
                    <rect x="120" y="30" width="20" height="60" rx="10" />
                    <rect x="146" y="20" width="20" height="70" rx="10" />
                    <rect x="172" y="14" width="20" height="76" rx="10" />
                    <rect x="198" y="20" width="20" height="70" rx="10" />
                    <rect x="224" y="36" width="20" height="54" rx="10" />
                    <!-- right fingers -->
                    <rect x="498" y="30" width="20" height="60" rx="10" />
                    <rect x="472" y="20" width="20" height="70" rx="10" />
                    <rect x="446" y="14" width="20" height="76" rx="10" />
                    <rect x="420" y="20" width="20" height="70" rx="10" />
                    <rect x="394" y="36" width="20" height="54" rx="10" />
                  </g>
                  <g>
                    <!-- Left highlights -->
                    <rect x="224" y="36" width="20" height="54" rx="10" :class="hl('L_THUMB')" />
                    <rect x="198" y="20" width="20" height="70" rx="10" :class="hl('L_INDEX')" />
                    <rect x="172" y="14" width="20" height="76" rx="10" :class="hl('L_MIDDLE')" />
                    <rect x="146" y="20" width="20" height="70" rx="10" :class="hl('L_RING')" />
                    <rect x="120" y="30" width="20" height="60" rx="10" :class="hl('L_PINKY')" />
                    <!-- Right highlights -->
                    <rect x="498" y="30" width="20" height="60" rx="10" :class="hl('R_PINKY')" />
                    <rect x="472" y="20" width="20" height="70" rx="10" :class="hl('R_RING')" />
                    <rect x="446" y="14" width="20" height="76" rx="10" :class="hl('R_MIDDLE')" />
                    <rect x="420" y="20" width="20" height="70" rx="10" :class="hl('R_INDEX')" />
                    <rect x="394" y="36" width="20" height="54" rx="10" :class="hl('R_THUMB')" />
                  </g>
                  <defs>
                    <linearGradient id="fg" x1="0" y1="0" x2="1" y2="1">
                      <stop offset="0%" stop-color="#818cf8" />
                      <stop offset="100%" stop-color="#e879f9" />
                    </linearGradient>
                  </defs>
                </svg>
              </div>
            </div>

            <!-- Notes -->
            <div class="mt-6">
              <label class="block text-sm text-white/70 mb-1">Notes (optional)</label>
              <textarea
                v-model.trim="notes"
                rows="3"
                class="w-full rounded-2xl bg-white/5 border border-white/10 backdrop-blur px-4 py-2.5 outline-none focus:ring-2 focus:ring-fuchsia-400/40"
                placeholder="Anything relevant to this scan…"
              ></textarea>
            </div>

            <div class="mt-6 flex items-center justify-end gap-3">
              <button
                class="px-4 py-2 rounded-2xl bg-white/10 hover:bg-white/15 border border-white/10 backdrop-blur text-sm"
                @click="resetForm"
              >
                Reset
              </button>
              <button
                :disabled="!canStart || loading"
                class="px-4 py-2 rounded-2xl bg-white text-slate-900 font-semibold shadow-lg hover:shadow-2xl text-sm disabled:opacity-60 disabled:cursor-not-allowed"
                @click="startScan"
              >
                Start Scan
              </button>
            </div>
          </div>
        </div>

        <!-- Right: Live Card (animated fingerprint, % match, device, user) -->
        <div
          class="relative rounded-3xl border border-white/10 bg-white/5 backdrop-blur-xl shadow-2xl overflow-hidden p-6"
        >
          <div
            class="absolute -inset-0.5 rounded-3xl bg-gradient-to-r from-indigo-500/20 to-fuchsia-500/20 blur-2xl"
          ></div>
          <div class="relative">
            <h3 class="text-lg font-semibold mb-4">Live Preview</h3>
            <div class="mx-auto max-w-md">
              <div class="mx-auto size-72 md:size-80 grid place-items-center">
                <!-- animated fingerprint svg (loop while scanning) -->
                <svg
                  v-if="!imageUrl"
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
                    <path
                      d="M100,30 a70,70 0 1,1 0,140 a70,70 0 1,1 0,-140"
                      :class="scanClass(0)"
                    />
                    <path
                      d="M100,42 a58,58 0 1,1 0,116 a58,58 0 1,1 0,-116"
                      :class="scanClass(1)"
                    />
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

                <img v-if="imageUrl" :src="imageUrl" alt="finger print image" />
              </div>

              <div class="mt-6 grid grid-cols-2 gap-4">
                <div class="rounded-2xl border border-white/10 bg-white/5 p-4">
                  <p class="text-xs text-white/60">State</p>
                  <p class="text-lg font-semibold">{{ stateLabel }}</p>
                </div>
                <div class="rounded-2xl border border-white/10 bg-white/5 p-4">
                  <p class="text-xs text-white/60">Device</p>
                  <p class="text-lg font-semibold truncate">{{ deviceLabel }}</p>
                </div>
                <div class="rounded-2xl border border-white/10 bg-white/5 p-4">
                  <p class="text-xs text-white/60">User</p>
                  <p class="text-lg font-semibold truncate">{{ userLabel }}</p>
                </div>
                <div class="rounded-2xl border border-white/10 bg-white/5 p-4">
                  <p class="text-xs text-white/60">Match</p>
                  <p class="text-lg font-semibold">{{ matchPct }}%</p>
                </div>
              </div>

              <p class="mt-6 text-center text-xs text-white/50">Demo-only. No real devices yet.</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  </section>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import Swal from '@/plugins/swal-theme'
import { useUserStore } from '@/stores/usersStore'
import { storeToRefs } from 'pinia'
import axios from 'axios'
const apiURL = import.meta.env.VITE_API_BASE

// stores
const userStore = useUserStore()
const { allUsers } = storeToRefs(userStore)
// --- mock datasets ---

type Status = 'active' | 'inactive' | 'locked'
interface UserDto {
  id: number
  employee_id: string
  name: string
  email: string
  status: Status
  last_scan_at: string | null
}
interface DeviceDto {
  id: string
  name: string
  location: string
}

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
  { key: 'L_THUMB', short: 'L-Thumb', label: 'Left Thumb', id: 1 },
  { key: 'L_INDEX', short: 'L-Index', label: 'Left Index', id: 2 },
  { key: 'L_MIDDLE', short: 'L-Middle', label: 'Left Middle', id: 3 },
  { key: 'L_RING', short: 'L-Ring', label: 'Left Ring', id: 4 },
  { key: 'L_PINKY', short: 'L-Pinky', label: 'Left Pinky', id: 5 },
  { key: 'R_THUMB', short: 'R-Thumb', label: 'Right Thumb', id: 6 },
  { key: 'R_INDEX', short: 'R-Index', label: 'Right Index', id: 7 },
  { key: 'R_MIDDLE', short: 'R-Middle', label: 'Right Middle', id: 8 },
  { key: 'R_RING', short: 'R-Ring', label: 'Right Ring', id: 9 },
  { key: 'R_PINKY', short: 'R-Pinky', label: 'Right Pinky', id: 10 },
] as const

const FIRST = [
  'Ali',
  'Sara',
  'Zaid',
  'Noor',
  'Huda',
  'Omar',
  'Yasmin',
  'Kareem',
  'Maya',
  'Hassan',
  'Rana',
  'Zahra',
  'Mustafa',
  'Lina',
  'Basim',
  'Dalia',
]
const LAST = [
  'Hassan',
  'Ahmed',
  'Karim',
  'Alani',
  'Salman',
  'Tareq',
  'Farhan',
  'Sadeq',
  'Hakim',
  'Jabbar',
  'Sultan',
  'Abbas',
  'Nasser',
  'Husein',
]

const ALL_USERS: UserDto[] = Array.from({ length: 120 }).map((_, i) => {
  const f = FIRST[i % FIRST.length]
  const l = LAST[(i * 3) % LAST.length]
  return {
    id: i + 1,
    employee_id: `EMP-${(i + 1).toString().padStart(4, '0')}`,
    full_name: `${f} ${l}`,
    email: `${f.toLowerCase()}.${l.toLowerCase()}${i % 11 || ''}@example.com`,
    status: (['active', 'inactive', 'locked'] as Status[])[i % 3],
    last_scan_at: null,
  }
})

const devices: DeviceDto[] = [
  { id: 'usb-1', name: 'ZK-U450', location: 'Front Desk' },
  { id: 'usb-2', name: 'SecuScan-2', location: 'Server Room' },
  { id: 'ip-10.0.0.12', name: 'BioNet-Edge', location: 'HR Office' },
]

// --- form state ---
const userQuery = ref('')
const selectedUser = ref<UserDto | null>(null)
const selectedFinger = ref<FingerKey | {}>({})
const deviceId = ref<string>('')
const notes = ref('')
const loading = ref(false)
//const users = userStore.allUsers

const filteredUsers = computed(() => {
  const q = userQuery.value.toLowerCase().trim()
  if (!q) return allUsers.value.slice(0, 10)
  return allUsers.value.filter((u) => u.name.toLowerCase().includes(q)).slice(0, 10)
})

const showUserList = computed(() => !!userQuery.value && !selectedUser.value)
const canStart = computed(() => !!selectedUser.value && !!selectedFinger.value && !!deviceId.value)

// live preview state (right card)
const scanning = ref(false)
const matchPct = ref(0)
const stateLabel = computed(() => (scanning.value ? 'Scanning' : 'Idle'))
const userLabel = computed(() => (selectedUser.value ? selectedUser.value.name : '—'))
const deviceLabel = computed(() => devices.find((d) => d.id === deviceId.value)?.name || '—')
const fingerLabel = computed(
  () => FINGERS.find((f) => f.key === (selectedFinger.value as any))?.label || '—',
)

const selectFinger = (f) => {
  imageUrl = ''
  selectedFinger.value = f
}

function onUserInput() {
  /* reactive via computed */
}
function selectUser(u: UserDto) {
  selectedUser.value = u
  userQuery.value = ''
}
function clearUser() {
  selectedUser.value = null
  userQuery.value = ''
}

function initials(name: string) {
  return name
    .split(' ')
    .filter(Boolean)
    .slice(0, 2)
    .map((s) => s[0]?.toUpperCase())
    .join('')
}

function hl(key: FingerKey) {
  return selectedFinger.value.key === key
    ? 'fill-[url(#fg)] drop-shadow-[0_0_16px_rgba(99,102,241,0.35)]'
    : 'fill-transparent'
}

function scanClass(i: number) {
  // when scanning: loop; when idle: faint lines
  return scanning.value ? `trace trace${i} animate` : `trace idle`
}

let imageUrl = ''

async function startScan() {
  if (!canStart.value) return
  loading.value = true
  scanning.value = true
  matchPct.value = 0

  try {
    const payload = {
      person_id: selectedUser.value?.id,
      finger: selectedFinger.value.id.toString(),
    }

    const response = await axios.post(apiURL + 'scans', payload, {
      responseType: 'blob',
    })

    if (response.status == 200) {
      imageUrl = URL.createObjectURL(response.data)

      await Swal.fire({
        title: 'Connecting to device…',
        allowOutsideClick: false,
        didOpen: () => Swal.showLoading(),
        timer: 800,
      })
      await Swal.fire({
        title: 'Capturing fingerprint…',
        html: `<small>${fingerLabel.value}</small>`,
        allowOutsideClick: false,
        didOpen: () => Swal.showLoading(),
        timer: 1200,
      })

      // fake progress
      for (let i = 0; i <= 5; i++) {
        matchPct.value = Math.min(100, i * 18 + Math.floor(Math.random() * 10))
        await sleep(220)
      }

      scanning.value = false
      matchPct.value = Math.min(100, matchPct.value + 5 + Math.floor(Math.random() * 6))

      await Swal.fire({
        title: 'fingerprint saved successfuly',
        text: `${selectedUser.value!.name} • ${matchPct.value}% • ${fingerLabel.value}`,
        icon: 'success',
        timer: 1100,
        showConfirmButton: false,
      })
    }

    console.log(response)
  } catch (error) {
    console.log(error)
  } finally {
    loading.value = false
  }
}

function resetForm() {
  userQuery.value = ''
  selectedUser.value = null
  selectedFinger.value = ''
  deviceId.value = ''
  notes.value = ''
  scanning.value = false
  matchPct.value = 0
}

function sleep(ms: number) {
  return new Promise((r) => setTimeout(r, ms))
}

onMounted(() => {
  userStore.fetchUsers()
})
</script>

<style scoped>
/* fingerprint animation in right card */
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
  animation: draw 1.6s ease-in-out infinite alternate;
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

/* smooth highlight transition for finger preview */
svg rect {
  transition:
    fill 0.25s ease,
    filter 0.25s ease;
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
