import Swal, { type SweetAlertIcon } from 'sweetalert2'
import 'sweetalert2/dist/sweetalert2.min.css'

// A pre-styled Swal instance matching your UI
const swal = Swal.mixin({
  buttonsStyling: false,
  showClass: { popup: 'sw-anim-in' },
  hideClass: { popup: 'sw-anim-out' },
  backdrop: true, // keeps default dimmed bg (we'll tint it in CSS)
  timerProgressBar: true,
  customClass: {
    popup: 'sw-pop',
    title: 'sw-title',
    htmlContainer: 'sw-text',
    confirmButton: 'sw-btn sw-btn-primary',
    cancelButton: 'sw-btn sw-btn-ghost',
    denyButton: 'sw-btn sw-btn-danger',
    actions: 'sw-actions',
    loader: 'sw-loader',
    timerProgressBar: 'sw-timer',
  },
})

export default swal
export type { SweetAlertIcon }
