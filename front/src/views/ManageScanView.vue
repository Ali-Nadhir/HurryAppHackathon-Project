<template>
  <section
    class="relative min-h-screen overflow-hidden bg-gradient-to-b from-slate-900 via-slate-950 to-black text-white"
  >
    <!-- floating gradient blobs (hero-style) -->
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
      <!-- Top bar -->
      <div class="flex flex-col lg:flex-row items-start lg:items-center justify-between gap-6">
        <div>
          <h1 class="text-3xl md:text-4xl font-extrabold leading-tight">
            Manage
            <span
              class="text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 to-fuchsia-400"
              >Scans</span
            >
          </h1>
          <p class="mt-1 text-white/60">List, search, and trigger scans for users in the system.</p>
        </div>
        <div class="flex items-center gap-3">
          <button
            class="px-4 py-2 rounded-2xl bg-white/10 hover:bg-white/15 border border-white/10 backdrop-blur text-sm"
            @click="refresh"
            :disabled="loading"
          >
            <i class="bi bi-arrow-clockwise mr-1"></i> Refresh
          </button>
          <RouterLink
            to="/newScan"
            class="px-4 py-2 rounded-2xl bg-white text-slate-900 font-semibold shadow-lg hover:shadow-2xl text-sm"
          >
            <i class="bi bi-fingerprint mr-1"></i> New Scan
          </RouterLink>
          <RouterLink
            to="/addUser"
            class="px-4 py-2 rounded-2xl bg-white text-slate-900 font-semibold shadow-lg hover:shadow-2xl text-sm"
          >
            <i class="bi bi-person mr-1"></i> Add User
          </RouterLink>
        </div>
      </div>

      <!-- Filters / Actions -->
      <div class="mt-8 grid md:grid-cols-2 gap-4">
        <div class="flex items-center gap-3">
          <div class="relative flex-1">
            <input
              v-model.trim="q"
              type="text"
              placeholder="Search users by name, email, or ID…"
              class="w-full rounded-2xl bg-white/5 border border-white/10 backdrop-blur px-4 py-2.5 outline-none focus:ring-2 focus:ring-fuchsia-400/40"
              @keyup.enter="applySearch"
            />
            <i class="bi bi-search absolute right-3 top-1/2 -translate-y-1/2 text-white/50"></i>
          </div>
          <button
            class="px-4 py-2 rounded-2xl bg-white/10 hover:bg-white/15 border border-white/10 backdrop-blur text-sm"
            @click="applySearch"
          >
            Search
          </button>
        </div>
        <div class="flex items-center gap-3 justify-start md:justify-end">
          <select
            v-model="status"
            class="rounded-2xl bg-white/5 border border-white/10 px-3 py-2 backdrop-blur text-sm"
          >
            <option value="" class="bg-slate-900 text-white">All statuses</option>
            <option value="active" class="bg-slate-900 text-white">Active</option>
            <option value="inactive" class="bg-slate-900 text-white">Inactive</option>
            <option value="locked" class="bg-slate-900 text-white">Locked</option>
          </select>
          <select
            v-model.number="perPage"
            class="rounded-2xl bg-white/5 border border-white/10 px-3 py-2 backdrop-blur text-sm"
          >
            <option :value="10" class="bg-slate-900 text-white">10 / page</option>
            <option :value="25" class="bg-slate-900 text-white">25 / page</option>
            <option :value="50" class="bg-slate-900 text-white">50 / page</option>
          </select>
        </div>
      </div>

      <!-- Glass table card -->
      <div
        class="mt-6 relative rounded-3xl border border-white/10 bg-white/5 backdrop-blur-xl shadow-2xl overflow-hidden"
      >
        <div
          class="absolute -inset-0.5 rounded-3xl bg-gradient-to-r from-indigo-500/20 to-fuchsia-500/20 blur-2xl"
        ></div>

        <div class="relative">
          <div class="flex items-center justify-between px-4 sm:px-6 py-4">
            <div class="text-white/70 text-sm">
              Showing {{ showingFrom }}–{{ showingTo }} of {{ total }} results
            </div>
            <div class="text-white/50 text-xs">
              Mock mode is <span class="font-semibold">ON</span>
            </div>
          </div>

          <div class="overflow-x-auto">
            <table class="min-w-full text-left">
              <thead class="sticky top-0 backdrop-blur bg-slate-900/40">
                <tr class="text-white/70 text-xs uppercase tracking-wide">
                  <th class="px-4 sm:px-6 py-3">User</th>
                  <th class="px-4 sm:px-6 py-3">Email</th>
                  <th class="px-4 sm:px-6 py-3">Last Scan</th>
                  <th class="px-4 sm:px-6 py-3">Status</th>
                  <th class="px-4 sm:px-6 py-3 text-right">Actions</th>
                </tr>
              </thead>
              <tbody>
                <!-- Loading skeleton -->
                <tr
                  v-if="loading"
                  v-for="n in 8"
                  :key="'skeleton-' + n"
                  class="border-t border-white/10"
                >
                  <td class="px-4 sm:px-6 py-4" colspan="5">
                    <div class="h-8 w-full rounded-xl bg-white/5 animate-pulse"></div>
                  </td>
                </tr>

                <tr
                  v-for="u in users"
                  :key="u.id"
                  class="border-t border-white/10 hover:bg-white/5/50 transition"
                >
                  <td class="px-4 sm:px-6 py-4">
                    <div class="flex items-center gap-3">
                      <div
                        class="size-9 rounded-full bg-gradient-to-br from-indigo-400/60 to-fuchsia-400/60 grid place-items-center text-xs font-bold"
                      >
                        {{ initials(u.full_name) }}
                      </div>
                      <div>
                        <div class="font-semibold">{{ u.full_name }}</div>
                        <div class="text-white/50 text-xs">ID: {{ u.employee_id }}</div>
                      </div>
                    </div>
                  </td>
                  <td class="px-4 sm:px-6 py-4">{{ u.email }}</td>
                  <td class="px-4 sm:px-6 py-4">{{ formatDate(u.last_scan_at) }}</td>
                  <td class="px-4 sm:px-6 py-4">
                    <span
                      :class="[
                        'px-2.5 py-1 rounded-xl text-xs border backdrop-blur',
                        u.status === 'active' &&
                          'bg-emerald-500/10 text-emerald-300 border-emerald-400/20',
                        u.status === 'inactive' &&
                          'bg-amber-500/10 text-amber-300 border-amber-400/20',
                        u.status === 'locked' && 'bg-rose-500/10 text-rose-300 border-rose-400/20',
                      ]"
                    >
                      {{ u.status }}
                    </span>
                  </td>
                  <td class="px-4 sm:px-6 py-4">
                    <div class="flex items-center gap-2 justify-end">
                      <button
                        class="px-3 py-2 rounded-xl bg-white/10 hover:bg-white/15 border border-white/10 backdrop-blur text-sm transition cursor-pointer"
                        @click="viewUser(u)"
                      >
                        <i class="bi bi-person"></i><span class="sr-only">View</span>
                      </button>
                      <RouterLink
                        to="/newScan"
                        class="px-3 py-2 rounded-xl bg-white/10 hover:bg-white/15 border border-white/10 backdrop-blur text-sm transition cursor-pointer"
                      >
                        <i class="bi bi-fingerprint"></i><span class="sr-only">Scan</span>
                      </RouterLink>
                      <button
                        class="px-3 py-2 rounded-xl bg-white/10 hover:bg-white/15 border border-white/10 backdrop-blur text-sm transition text-rose-300 hover:bg-rose-500/10 border-rose-400/20 cursor-pointer"
                        @click="removeUser(u)"
                      >
                        <i class="bi bi-trash"></i><span class="sr-only">Delete</span>
                      </button>
                    </div>
                  </td>
                </tr>

                <tr v-if="!loading && users.length === 0" class="border-t border-white/10">
                  <td colspan="5" class="px-4 sm:px-6 py-16 text-center text-white/50">
                    No users found.
                  </td>
                </tr>
              </tbody>
            </table>
          </div>

          <!-- Pagination -->
          <div
            class="flex flex-col sm:flex-row items-center justify-between gap-4 px-4 sm:px-6 py-4 border-t border-white/10"
          >
            <div class="text-xs text-white/50">Page {{ page }} of {{ totalPages }}</div>
            <div class="flex items-center gap-2">
              <button
                class="px-3 py-2 rounded-xl bg-white/10 hover:bg-white/15 border border-white/10 backdrop-blur disabled:opacity-50 disabled:cursor-not-allowed"
                :disabled="page === 1 || loading"
                @click="goto(1)"
              >
                <i class="bi bi-chevron-double-left"></i>
              </button>
              <button
                class="px-3 py-2 rounded-xl bg-white/10 hover:bg-white/15 border border-white/10 backdrop-blur disabled:opacity-50 disabled:cursor-not-allowed"
                :disabled="page === 1 || loading"
                @click="goto(page - 1)"
              >
                <i class="bi bi-chevron-left"></i>
              </button>
              <button
                class="px-3 py-2 rounded-xl bg-white/10 hover:bg-white/15 border border-white/10 backdrop-blur disabled:opacity-50 disabled:cursor-not-allowed"
                :disabled="page === totalPages || loading"
                @click="goto(page + 1)"
              >
                <i class="bi bi-chevron-right"></i>
              </button>
              <button
                class="px-3 py-2 rounded-xl bg-white/10 hover:bg-white/15 border border-white/10 backdrop-blur disabled:opacity-50 disabled:cursor-not-allowed"
                :disabled="page === totalPages || loading"
                @click="goto(totalPages)"
              >
                <i class="bi bi-chevron-double-right"></i>
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  </section>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, watch } from 'vue'
import { useRouter } from 'vue-router'
import Swal from '@/plugins/swal-theme'
// import api from '@/plugins/axios'; // keep for later when API is ready

// Mock mode settings
const USE_MOCK = true
const MOCK_DELAY = 450 // ms

type Status = 'active' | 'inactive' | 'locked'
interface UserDto {
  id: number
  employee_id: string
  full_name: string
  email: string
  status: Status
  last_scan_at: string | null // ISO
}

const router = useRouter()

// UI state
const loading = ref(false)
const q = ref('')
const status = ref<Status | ''>('')
const perPage = ref<number>(10)
const page = ref<number>(1)

// Data
const users = ref<UserDto[]>([])
const total = ref<number>(0)

const totalPages = computed(() => Math.max(1, Math.ceil(total.value / perPage.value)))
const showingFrom = computed(() => (total.value === 0 ? 0 : (page.value - 1) * perPage.value + 1))
const showingTo = computed(() => Math.min(page.value * perPage.value, total.value))

onMounted(() => {
  fetchUsers()
})

watch([status, perPage], () => {
  page.value = 1
  fetchUsers()
})

function applySearch() {
  page.value = 1
  fetchUsers()
}

async function fetchUsers() {
  loading.value = true
  try {
    if (USE_MOCK) {
      await sleep(MOCK_DELAY)
      const { items, total: t } = queryMock(q.value, status.value, page.value, perPage.value)
      users.value = items
      total.value = t
    } else {
      // const { data } = await api.get('/users', { params: { page: page.value, per_page: perPage.value, q: q.value || undefined, status: status.value || undefined } });
      // users.value = data?.items ?? [];
      // total.value = data?.total ?? 0;
    }
  } finally {
    loading.value = false
  }
}

function goto(p: number) {
  const target = Math.min(Math.max(1, p), totalPages.value)
  if (target !== page.value) {
    page.value = target
    fetchUsers()
  }
}

function initials(name: string) {
  return name
    .split(' ')
    .filter(Boolean)
    .slice(0, 2)
    .map((s) => s[0]?.toUpperCase())
    .join('')
}

function formatDate(iso: string | null) {
  if (!iso) return '—'
  const d = new Date(iso)
  return d.toLocaleString()
}

function viewUser(u: UserDto) {
  //router.push({ name: 'user-details', params: { id: u.id } })
  console.log('user page')
}

async function triggerScan(u: UserDto) {
  const res = await Swal.fire({
    title: 'Start Scan?',
    text: `Trigger scan for ${u.full_name}`,
    icon: 'question',
    showCancelButton: true,
    confirmButtonText: 'Start',
  })
  if (!res.isConfirmed) return

  await Swal.fire({
    title: 'Connecting to device…',
    allowOutsideClick: false,
    didOpen: () => Swal.showLoading(),
    timer: 800,
  })
  await Swal.fire({
    title: 'Scanning…',
    allowOutsideClick: false,
    didOpen: () => Swal.showLoading(),
    timer: 1400,
  })
  await Swal.fire({
    title: 'Matching…',
    allowOutsideClick: false,
    didOpen: () => Swal.showLoading(),
    timer: 800,
  })
  await Swal.fire({ title: 'Scan complete', icon: 'success', timer: 900, showConfirmButton: false })

  // update mock last_scan_at
  const idx = ALL_USERS.findIndex((x) => x.id === u.id)
  if (idx > -1) ALL_USERS[idx].last_scan_at = new Date().toISOString()
  fetchUsers()
}

async function removeUser(u: UserDto) {
  const ask = await Swal.fire({
    title: 'Delete user?',
    text: `This will remove ${u.full_name}`,
    icon: 'warning',
    showCancelButton: true,
    confirmButtonText: 'Delete',
    confirmButtonColor: '#ef4444',
  })
  if (!ask.isConfirmed) return

  // remove from mock dataset
  const i = ALL_USERS.findIndex((x) => x.id === u.id)
  if (i > -1) ALL_USERS.splice(i, 1)

  await Swal.fire({ title: 'Deleted', icon: 'success', timer: 900, showConfirmButton: false })
  // refresh current page; adjust if past the end
  const maxPages = Math.max(1, Math.ceil((total.value - 1) / perPage.value))
  page.value = Math.min(page.value, maxPages)
  fetchUsers()
}

function refresh() {
  fetchUsers()
}

function sleep(ms: number) {
  return new Promise((r) => setTimeout(r, ms))
}

// --- Mock data (remove when API is ready) ---
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
  'Fares',
  'Hani',
  'Ibrahim',
  'Jana',
  'Kamal',
  'Leen',
  'Mahdi',
  'Nadia',
  'Ola',
  'Qasim',
  'Rami',
  'Salma',
  'Tariq',
  'Wijdan',
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
  'Sadiq',
  'Shakir',
  'Sabah',
  'Mansour',
  'Mahmood',
  'Azzam',
  'Younis',
  'Ilyas',
  'Sami',
  'Abdul',
  'Fadhil',
  'Latif',
  'Shamkhi',
  'Dhari',
  'Hameed',
  'Najeeb',
]
const STATUSES: Status[] = ['active', 'inactive', 'locked']

let ALL_USERS: UserDto[] = (() => {
  const arr: UserDto[] = []
  let id = 1
  for (let i = 0; i < 200; i++) {
    const f = FIRST[i % FIRST.length]
    const l = LAST[(i * 3) % LAST.length]
    const full = `${f} ${l}`
    const emp = `EMP-${(i + 1).toString().padStart(4, '0')}`
    const email = `${f.toLowerCase()}.${l.toLowerCase()}${i % 13 || ''}@example.com`
    const status = STATUSES[i % STATUSES.length]
    // last scan: some nulls, some recent, some days ago/hours
    const mod = i % 7
    const last =
      mod === 0 ? null : new Date(Date.now() - (mod * 6 + (i % 5)) * 60 * 60 * 1000).toISOString()
    arr.push({ id: id++, employee_id: emp, full_name: full, email, status, last_scan_at: last })
  }
  return arr
})()

function queryMock(qStr: string, statusFilter: Status | '', pageNum: number, size: number) {
  const q = (qStr || '').toLowerCase().trim()
  let list = ALL_USERS.slice()
  if (q) {
    list = list.filter(
      (u) =>
        u.full_name.toLowerCase().includes(q) ||
        u.email.toLowerCase().includes(q) ||
        u.employee_id.toLowerCase().includes(q),
    )
  }
  if (statusFilter) list = list.filter((u) => u.status === statusFilter)

  // sort: newest scan first, then name
  list.sort((a, b) => {
    const at = a.last_scan_at ? Date.parse(a.last_scan_at) : 0
    const bt = b.last_scan_at ? Date.parse(b.last_scan_at) : 0
    if (bt !== at) return bt - at
    return a.full_name.localeCompare(b.full_name)
  })

  const total = list.length
  const start = (pageNum - 1) * size
  const items = list.slice(start, start + size)
  return { items, total }
}
</script>

<style scoped>
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
</style>
