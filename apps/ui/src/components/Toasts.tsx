import { useStore } from '../state/store'

/** Bottom-right transient notifications (the store owns their lifetime). */
export function Toasts() {
  const { toasts, dismissToast } = useStore()
  if (toasts.length === 0) return null

  return (
    <div className="toasts">
      {toasts.map((toast) => (
        <div key={toast.id} className={`toast ${toast.kind}`} onClick={() => dismissToast(toast.id)}>
          <strong>{toast.message}</strong>
          {toast.detail ? <span>{toast.detail}</span> : null}
        </div>
      ))}
    </div>
  )
}
