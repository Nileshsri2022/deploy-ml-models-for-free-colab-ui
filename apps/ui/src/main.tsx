import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { HashRouter } from 'react-router-dom'
import App from './App'
import { StoreProvider } from './state/store'
import './styles.css'

/*
  Hash routing, deliberately. The built bundle is served by the gateway itself
  (see `_mount_spa` in apps/gateway/gateway/main.py) on free hosts where nobody
  controls the reverse proxy in front of the app, so `/models` would 404 on a
  hard reload while `/#/models` never does. Relative asset URLs + hash routes =
  the same `dist/` works from any prefix, including a preview proxy.
*/

const container = document.getElementById('root')
if (!container) throw new Error('#root missing from index.html')

createRoot(container).render(
  <StrictMode>
    <HashRouter>
      <StoreProvider>
        <App />
      </StoreProvider>
    </HashRouter>
  </StrictMode>,
)
