import type { Config } from "tailwindcss";

export default {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {

    extend: {
      colors: {
        // AI Ideation App Design System Colors
        background: '#0B0C10',
        card: '#111418',
        foreground: '#EAECEE',
        'sub-foreground': 'text-gray-300',
        accent: '#7C5CFF',
        
        // Standard colors for compatibility
        border: '#2A2D32',
        input: '#111418',
        ring: '#7C5CFF',
        primary: {
          DEFAULT: '#7C5CFF',
          foreground: '#EAECEE',
        },
        secondary: {
          DEFAULT: '#111418',
          foreground: '#AAB2BD',
        },
        destructive: {
          DEFAULT: '#EF4444',
          foreground: '#EAECEE',
        },
        muted: {
          DEFAULT: '#111418',
          foreground: '#AAB2BD',
        },
        popover: {
          DEFAULT: '#111418',
          foreground: '#EAECEE',
        },
        success: {
          DEFAULT: '#22C55E',
          foreground: '#EAECEE',
        },
        warning: {
          DEFAULT: '#F59E0B',
          foreground: '#EAECEE',
        },
      },
      borderRadius: {
        lg: '1rem',
        md: '0.75rem',
        sm: '0.5rem',
      },
      spacing: {
        '18': '4.5rem',
        '22': '5.5rem',
      },
      boxShadow: {
        'soft': '0 4px 12px rgba(0, 0, 0, 0.15)',
        'card': '0 2px 8px rgba(0, 0, 0, 0.1)',
        'glow': '0 0 20px rgba(124, 92, 255, 0.3)',
      },
      keyframes: {
        'accordion-down': {
          from: { height: '0' },
          to: { height: 'var(--radix-accordion-content-height)' },
        },
        'accordion-up': {
          from: { height: 'var(--radix-accordion-content-height)' },
          to: { height: '0' },
        },
        'scale-in': {
          '0%': { transform: 'scale(0.95)', opacity: '0' },
          '100%': { transform: 'scale(1)', opacity: '1' },
        },
        'fade-in': {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        'slide-up': {
          '0%': { transform: 'translateY(10px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
        'gradient': {
          '0%, 100%': { 'background-position': '0% 50%' },
          '50%': { 'background-position': '100% 50%' },
        },
        'pulse': {
          '0%, 100%': { 
            boxShadow: '0 0 0 0 var(--pulse-color)',
            opacity: '1'
          },
          '50%': { 
            boxShadow: '0 0 0 8px var(--pulse-color)',
            opacity: '0.5'
          },
        },
        'shine': {
          '0%': { 'background-position': '0% 0%' },
          '50%': { 'background-position': '100% 100%' },
          '100%': { 'background-position': '0% 0%' },
        },
      },
      animation: {
        'accordion-down': 'accordion-down 0.2s ease-out',
        'accordion-up': 'accordion-up 0.2s ease-out',
        'scale-in': 'scale-in 120ms ease-out',
        'fade-in': 'fade-in 200ms ease-out',
        'slide-up': 'slide-up 200ms ease-out',
        'gradient': 'gradient 8s linear infinite',
        'pulse': 'pulse var(--duration, 1.5s) ease-out infinite',
        'shine': 'shine var(--duration, 14s) linear infinite',
      },
    },
  },
  plugins: [require('tailwindcss-animate')],
} satisfies Config;
