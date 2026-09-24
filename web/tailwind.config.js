/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        p1: {
          bg: "#fef2f2",
          border: "#ef4444",
          text: "#991b1b",
          chip: "#dc2626",
        },
        p2: {
          bg: "#fffbeb",
          border: "#f59e0b",
          text: "#92400e",
          chip: "#d97706",
        },
        p3: {
          bg: "#f0fdf4",
          border: "#22c55e",
          text: "#166534",
          chip: "#16a34a",
        },
        highlight: {
          bg: "#fef3c7",
          border: "#fbbf24",
        },
      },
    },
  },
  plugins: [],
};
