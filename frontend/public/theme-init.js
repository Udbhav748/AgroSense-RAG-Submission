// Applies the saved theme before first paint so there's no flash of the
// wrong color scheme. Mirrors ThemeContext.jsx's storage key and default.
// Loaded as an external file (not inlined in index.html) so the CSP's
// script-src can stay 'self' only, with no 'unsafe-inline'.
;(function () {
  try {
    var stored = localStorage.getItem('insightai-theme')
    document.documentElement.classList.toggle('dark', stored === 'dark' || stored !== 'light')
  } catch (e) {
    document.documentElement.classList.add('dark')
  }
})()
