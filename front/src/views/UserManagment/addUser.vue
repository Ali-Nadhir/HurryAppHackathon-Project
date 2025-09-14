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
            Add
            <span
              class="text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 to-fuchsia-400"
              >New User</span
            >
          </h1>
          <p class="mt-1 text-white/60">
            Fill the details below to create a user. (Mock submit for now)
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
        <!-- Left: Form Card -->
        <div
          class="relative rounded-3xl border border-white/10 bg-white/5 backdrop-blur-xl shadow-2xl overflow-hidden p-6"
        >
          <div
            class="absolute -inset-0.5 rounded-3xl bg-gradient-to-r from-indigo-500/20 to-fuchsia-500/20 blur-2xl"
          ></div>
          <form class="relative space-y-5" @submit.prevent="submit">
            <!-- Name -->
            <div>
              <label class="block text-sm text-white/70 mb-1">Full Name</label>
              <input
                v-model.trim="form.name"
                type="text"
                placeholder="e.g. Ali Hassan"
                class="w-full rounded-2xl bg-white/5 border border-white/10 backdrop-blur px-4 py-2.5 outline-none focus:ring-2 focus:ring-fuchsia-400/40"
              />
              <p v-if="errs.name" class="mt-1 text-xs text-rose-300">{{ errs.name }}</p>
            </div>

            <!-- Phone -->
            <div>
              <label class="block text-sm text-white/70 mb-1">Phone Number</label>
              <input
                v-model.trim="form.phone"
                type="tel"
                placeholder="e.g. +964 7xx xxx xxxx"
                class="w-full rounded-2xl bg-white/5 border border-white/10 backdrop-blur px-4 py-2.5 outline-none focus:ring-2 focus:ring-fuchsia-400/40"
              />
              <p v-if="errs.phone" class="mt-1 text-xs text-rose-300">{{ errs.phone }}</p>
            </div>

            <!-- Email -->
            <div>
              <label class="block text-sm text-white/70 mb-1">Email</label>
              <input
                v-model.trim="form.email"
                type="email"
                placeholder="e.g. ali.hassan@example.com"
                class="w-full rounded-2xl bg-white/5 border border-white/10 backdrop-blur px-4 py-2.5 outline-none focus:ring-2 focus:ring-fuchsia-400/40"
              />
              <p v-if="errs.email" class="mt-1 text-xs text-rose-300">{{ errs.email }}</p>
            </div>

            <!-- Role -->
            <div>
              <label class="block text-sm text-white/70 mb-1">Role</label>
              <div class="relative">
                <select
                  v-model="form.role"
                  class="appearance-none w-full rounded-2xl bg-white/5 border border-white/10 backdrop-blur px-4 py-2.5 pr-10 outline-none focus:ring-2 focus:ring-indigo-400/40 text-white"
                >
                  <option disabled value="">Select role…</option>
                  <option v-for="r in roles" :key="r" :value="r" class="bg-slate-900 text-white">
                    {{ r }}
                  </option>
                </select>
                <i
                  class="bi bi-chevron-down absolute right-3 top-1/2 -translate-y-1/2 pointer-events-none text-white/60"
                ></i>
              </div>
              <p v-if="errs.role" class="mt-1 text-xs text-rose-300">{{ errs.role }}</p>
            </div>

            <!-- Address -->
            <div>
              <label class="block text-sm text-white/70 mb-1">Address</label>
              <textarea
                v-model.trim="form.address"
                rows="3"
                placeholder="Street, City, Governorate"
                class="w-full rounded-2xl bg-white/5 border border-white/10 backdrop-blur px-4 py-2.5 outline-none focus:ring-2 focus:ring-fuchsia-400/40"
              ></textarea>
              <p v-if="errs.address" class="mt-1 text-xs text-rose-300">{{ errs.address }}</p>
            </div>
            <!-- age -->
            <div>
              <label class="block text-sm text-white/70 mb-1">Age</label>
              <input
                v-model.trim="form.age"
                type="text"
                placeholder="e.x 20"
                class="w-full rounded-2xl bg-white/5 border border-white/10 backdrop-blur px-4 py-2.5 outline-none focus:ring-2 focus:ring-fuchsia-400/40"
              />
              <p v-if="errs.age" class="mt-1 text-xs text-rose-300">{{ errs.age }}</p>
            </div>

            <!-- Actions -->
            <div class="pt-2 flex items-center justify-end gap-3">
              <button
                type="button"
                @click="reset"
                class="px-4 py-2 rounded-2xl bg-white/10 hover:bg-white/15 border border-white/10 backdrop-blur text-sm"
              >
                Reset
              </button>
              <button
                type="submit"
                :disabled="submitting || !isValid"
                class="px-4 py-2 rounded-2xl bg-white text-slate-900 font-semibold shadow-lg hover:shadow-2xl text-sm disabled:opacity-60 disabled:cursor-not-allowed"
              >
                Save User
              </button>
            </div>
          </form>
        </div>

        <!-- Right: Live Card / Preview -->
        <div
          class="relative rounded-3xl border border-white/10 bg-white/5 backdrop-blur-xl shadow-2xl overflow-hidden p-6"
        >
          <div
            class="absolute -inset-0.5 rounded-3xl bg-gradient-to-r from-indigo-500/20 to-fuchsia-500/20 blur-2xl"
          ></div>
          <div class="relative">
            <h3 class="text-lg font-semibold mb-4">Live Preview</h3>

            <div class="grid gap-4">
              <div class="flex items-center gap-3">
                <div
                  class="size-11 rounded-2xl grid place-items-center bg-white/10 border border-white/10 text-sm font-bold"
                >
                  {{ initials(form.name) || '—' }}
                </div>
                <div>
                  <div class="font-semibold">{{ form.name || 'Full Name' }}</div>
                  <div class="text-white/50 text-xs">{{ form.role || 'Role' }}</div>
                </div>
              </div>

              <div class="rounded-2xl border border-white/10 bg-white/5 p-4">
                <p class="text-xs text-white/60">Phone</p>
                <p class="text-lg font-semibold">{{ form.phone || '—' }}</p>
              </div>
              <div class="rounded-2xl border border-white/10 bg-white/5 p-4">
                <p class="text-xs text-white/60">Email</p>
                <p class="text-lg font-semibold truncate">{{ form.email || '—' }}</p>
              </div>
              <div class="rounded-2xl border border-white/10 bg-white/5 p-4">
                <p class="text-xs text-white/60">Address</p>
                <p class="text-sm">{{ form.address || '—' }}</p>
              </div>
              <div class="rounded-2xl border border-white/10 bg-white/5 p-4">
                <p class="text-xs text-white/60">Age</p>
                <p class="text-sm">{{ form.age || '—' }}</p>
              </div>

              <!-- decorative fingerprint -->
              <div class="mt-2 mx-auto size-44 grid place-items-center">
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
                    <path d="M100,30 a70,70 0 1,1 0,140 a70,70 0 1,1 0,-140" class="trace idle" />
                    <path d="M100,42 a58,58 0 1,1 0,116 a58,58 0 1,1 0,-116" class="trace idle" />
                    <path d="M100,56 a44,44 0 1,1 0,88 a44,44 0 1,1 0,-88" class="trace idle" />
                    <path d="M100,68 a32,32 0 1,1 0,64 a32,32 0 1,1 0,-64" class="trace idle" />
                    <path d="M100,80 a20,20 0 1,1 0,40 a20,20 0 1,1 0,-40" class="trace idle" />
                  </g>
                  <defs>
                    <linearGradient id="grad" x1="0" y1="0" x2="1" y2="1">
                      <stop offset="0%" stop-color="#818cf8" />
                      <stop offset="100%" stop-color="#e879f9" />
                    </linearGradient>
                  </defs>
                </svg>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </section>
</template>

<script setup lang="ts">
const apiURL = import.meta.env.VITE_API_BASE
import { reactive, computed, ref } from 'vue'
import Swal from '@/plugins/swal-theme'
import axios from 'axios'

const roles = ['Admin', 'Operator', 'Employee', 'Guest'] as const

const form = reactive({
  name: '',
  phone: '',
  email: '',
  role: '' as '' | (typeof roles)[number],
  address: '',
  age: '',
})

const errs = reactive<{ [k: string]: string | null }>({
  name: null,
  phone: null,
  email: null,
  role: null,
  address: null,
  age: null,
})
const submitting = ref(false)

const isValid = computed(() => {
  validate(false)
  return !errs.name && !errs.phone && !errs.email && !errs.role && !errs.address && !errs.age
})

function validate(show = true) {
  errs.name = form.name.length >= 3 ? null : 'Please enter a full name (min 3 chars).'
  errs.phone = /\d{7,}/.test(form.phone) ? null : 'Please enter a valid phone number.'
  errs.email = /.+@.+\..+/.test(form.email) ? null : 'Please enter a valid email.'
  errs.role = form.role ? null : 'Please select a role.'
  errs.address = form.address.length >= 5 ? null : 'Please enter an address.'
  errs.age = form.age.length == 2 ? null : 'please enter valid age'
  return show && Object.values(errs).every((v) => v === null)
}

function initials(name: string) {
  return name
    .split(' ')
    .filter(Boolean)
    .slice(0, 2)
    .map((s) => s[0]?.toUpperCase())
    .join('')
}

async function submit() {
  if (!validate(true)) return
  submitting.value = true

  try {
    const response = await axios.post(apiURL + 'people', form)

    if (response.status == 201) {
      await Swal.fire({
        title: 'Creating user…',
        allowOutsideClick: false,
        didOpen: () => Swal.showLoading(),
        timer: 900,
      })
      await Swal.fire({
        title: 'Saving to database…',
        allowOutsideClick: false,
        didOpen: () => Swal.showLoading(),
        timer: 900,
      })
      await Swal.fire({
        title: 'User created',
        icon: 'success',
        timer: 1100,
        showConfirmButton: false,
      })

      reset()
    }
  } catch (error) {
    console.log(error)
  } finally {
    submitting.value = false
  }
}

function reset() {
  form.name = ''
  form.phone = ''
  form.email = ''
  form.role = ''
  form.address = ''
  form.age = ''
}
</script>

<style scoped>
/* decorative fingerprint */
.trace {
  stroke-dasharray: 500;
  stroke-dashoffset: 0;
  opacity: 0.35;
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
