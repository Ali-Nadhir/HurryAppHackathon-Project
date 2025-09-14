import { ref, computed } from 'vue'
import { defineStore } from 'pinia'
import axios from 'axios'

export const useUserStore = defineStore('userStore', () => {
  const apiURL = import.meta.env.VITE_API_BASE
  const allUsers = ref([])

  const fetchUsers = async () => {
    try {
      const response = await axios.get(apiURL + 'people')
      allUsers.value = response.data

      return response
    } catch (error) {
      console.log(error)
    }
  }

  return { allUsers, fetchUsers }
})
