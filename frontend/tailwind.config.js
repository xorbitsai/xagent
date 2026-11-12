/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: ["class"],
  content: [
    './src/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  prefix: "",
  theme: {
    container: {
      center: true,
      padding: "2rem",
      screens: {
        "2xl": "1400px",
      },
    },
    extend: {
      colors: {
        border: "hsl(var(--border))",
        input: "hsl(var(--input))",
        ring: "hsl(var(--ring))",
        background: "hsl(var(--background))",
        foreground: "hsl(var(--foreground))",
        primary: {
          DEFAULT: "hsl(var(--primary))",
          foreground: "hsl(var(--primary-foreground))",
        },
        secondary: {
          DEFAULT: "hsl(var(--secondary))",
          foreground: "hsl(var(--secondary-foreground))",
        },
        destructive: {
          DEFAULT: "hsl(var(--destructive))",
          foreground: "hsl(var(--destructive-foreground))",
        },
        muted: {
          DEFAULT: "hsl(var(--muted))",
          foreground: "hsl(var(--muted-foreground))",
        },
        accent: {
          DEFAULT: "hsl(var(--accent))",
          foreground: "hsl(var(--accent-foreground))",
        },
        popover: {
          DEFAULT: "hsl(var(--popover))",
          foreground: "hsl(var(--popover-foreground))",
        },
        card: {
          DEFAULT: "hsl(var(--card))",
          foreground: "hsl(var(--card-foreground))",
        },
      },
      borderRadius: {
        lg: "var(--radius)",
        md: "calc(var(--radius) - 2px)",
        sm: "calc(var(--radius) - 4px)",
      },
      keyframes: {
        "accordion-down": {
          from: { height: "0" },
          to: { height: "var(--radix-accordion-content-height)" },
        },
        "accordion-up": {
          from: { height: "var(--radix-accordion-content-height)" },
          to: { height: "0" },
        },
        "float": {
          '0%, 100%': { transform: 'translateY(0)' },
          '50%': { transform: 'translateY(-10px)' },
        },
        "wave": {
          '0%': { backgroundPosition: '0% 50%' },
          '100%': { backgroundPosition: '200% 50%' },
        },
        // Voice call orb: a large lit disc sits over the ball so the light/shade
        // edge crosses it. The disc swings back and forth between "light on the
        // left" and "light on top" (never right round), easing at each end,
        // while a second layer drifts the edge in and out on a different period
        // so the motion never visibly repeats.
        "voice-orb-swing": {
          '0%, 100%': { transform: 'rotate(-12deg)', animationTimingFunction: 'ease-in-out' },
          '22%': { transform: 'rotate(38deg)', animationTimingFunction: 'ease-in-out' },
          '45%': { transform: 'rotate(84deg)', animationTimingFunction: 'ease-in-out' },
          '64%': { transform: 'rotate(52deg)', animationTimingFunction: 'ease-in-out' },
          '82%': { transform: 'rotate(10deg)', animationTimingFunction: 'ease-in-out' },
        },
        "voice-orb-drift": {
          '0%, 100%': { transform: 'translate(0%, 0%) scale(1)', animationTimingFunction: 'ease-in-out' },
          '25%': { transform: 'translate(4%, -3%) scale(1.05)', animationTimingFunction: 'ease-in-out' },
          '50%': { transform: 'translate(-3%, 5%) scale(0.96)', animationTimingFunction: 'ease-in-out' },
          '75%': { transform: 'translate(-5%, -2%) scale(1.03)', animationTimingFunction: 'ease-in-out' },
        },
        // The composer's main action swaps between "talk" and "send".
        "voice-pop": {
          from: { opacity: '0', transform: 'scale(0.6)' },
          to: { opacity: '1', transform: 'scale(1)' },
        },
      },
      animation: {
        "accordion-down": "accordion-down 0.2s ease-out",
        "accordion-up": "accordion-up 0.2s ease-out",
        "float": 'float 3s ease-in-out infinite',
        "voice-pop": 'voice-pop 0.2s ease-out',
        "animation": {
          wave: 'wave 1.8s linear infinite',
        },
      },
    },
  },
  plugins: [require("tailwindcss-animate")],
}
