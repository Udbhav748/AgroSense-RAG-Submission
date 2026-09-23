import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './styles/index.css'
import App from './App.jsx'
import ErrorBoundary from './components/ui/ErrorBoundary'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    {/* Top-most error boundary: any uncaught render error in the app
        (route lazily-load failure, provider throw, page crash) surfaces
        as the recoverable "Something went wrong" panel instead of React's
        default blank-screen. resetError() remounts the whole tree. */}
    <ErrorBoundary>
      <App />
    </ErrorBoundary>
  </StrictMode>,
)
