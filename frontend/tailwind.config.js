/** @type {import('tailwindcss').Config} */
export default {
  darkMode: 'class',
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        // A serif display face is a deliberate departure from the
        // geometric-sans-on-dark formula almost every AI-tool template
        // ships with (Plus Jakarta Sans/Space Grotesk/Sora, previously
        // used here too) — it's what makes headings read as designed for
        // this product rather than left at a starting-point default.
        display: ['Fraunces', 'Georgia', 'serif'],
      },
      colors: {
        surface: {
          dark: '#0b0b0d',
          'dark-secondary': '#111315',
          'dark-panel': '#17191c',
          light: '#f6f7fb',
          'light-raised': '#ffffff',
        },
        // Soft elevated surfaces (the default for most cards — no blur).
        card: {
          DEFAULT: 'rgba(255,255,255,0.03)',
          hover: 'rgba(255,255,255,0.05)',
          light: 'rgba(255,255,255,0.75)',
          'light-hover': 'rgba(255,255,255,0.92)',
        },
        border: {
          DEFAULT: 'rgba(255,255,255,0.08)',
          light: 'rgba(15,23,42,0.08)',
        },
        ink: {
          primary: '#f5f5f5',
          secondary: '#b4b8c2',
          muted: '#7c8391',
        },
        // Single restrained accent — a muted terracotta/copper, chosen
        // specifically because almost no AI-tool UI uses a warm accent
        // (the field clusters overwhelmingly around blue/indigo/emerald/
        // violet); this is what gives buttons, focus rings, active nav,
        // and the logo mark an actual identity instead of the default.
        // Deliberately separate from `success` below — that stays green
        // (the universally-understood color for "succeeded"), so success
        // states never got a free ride on whatever the brand color is.
        accent: {
          50: '#fbf2ed',
          100: '#f5ddd0',
          200: '#eab89f',
          300: '#dd8f6a',
          400: '#cc6f42',
          500: '#b8572d',
          600: '#984624',
          700: '#7a3820',
        },
        success: '#10b981',
        warning: '#f59e0b',
        danger: '#ef4444',
      },
      borderRadius: {
        // Panel-level radius (16–20px range); left the global xl/2xl/3xl
        // scale untouched so small controls (buttons, badges) stay compact.
        panel: '1.125rem',
      },
      boxShadow: {
        soft: '0 8px 24px rgba(0,0,0,0.25)',
        'soft-lg': '0 12px 32px rgba(0,0,0,0.3)',
        'soft-light': '0 8px 24px rgba(15,23,42,0.1)',
      },
      keyframes: {
        'fade-in': { '0%': { opacity: 0 }, '100%': { opacity: 1 } },
        'fade-in-up': {
          '0%': { opacity: 0, transform: 'translateY(6px)' },
          '100%': { opacity: 1, transform: 'translateY(0)' },
        },
        shimmer: { '0%': { backgroundPosition: '-200% 0' }, '100%': { backgroundPosition: '200% 0' } },
        'pulse-soft': { '0%, 100%': { opacity: 1 }, '50%': { opacity: 0.5 } },
      },
      animation: {
        'fade-in': 'fade-in 0.2s ease-out',
        'fade-in-up': 'fade-in-up 0.25s cubic-bezier(0.16, 1, 0.3, 1)',
        shimmer: 'shimmer 2.5s linear infinite',
        'pulse-soft': 'pulse-soft 2s ease-in-out infinite',
      },
    },
  },
  plugins: [],
}
