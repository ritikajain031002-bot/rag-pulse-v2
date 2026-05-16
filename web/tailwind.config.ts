import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  safelist: [
    "bg-nvidia",
    "bg-cyan",
    "bg-electric",
    "text-nvidia",
    "text-cyan",
    "text-electric",
    "border-nvidia/20",
    "border-cyan/20",
    "border-electric/20",
  ],
  theme: {
    extend: {
      colors: {
        nvidia: "#76b900",
        "nvidia-dim": "#5a8c00",
        electric: "#a855f7",
        cyan: "#06b6d4",
      },
      fontFamily: {
        sans: ["var(--font-sans)", "system-ui", "sans-serif"],
        mono: ["var(--font-mono)", "ui-monospace", "monospace"],
      },
      keyframes: {
        shimmer: {
          "0%": { backgroundPosition: "200% 0" },
          "100%": { backgroundPosition: "-200% 0" },
        },
        "fade-in-up": {
          "0%": { opacity: "0", transform: "translateY(10px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        "pulse-glow": {
          "0%, 100%": {
            boxShadow: "0 0 0 0 rgba(118, 185, 0, 0.5)",
            opacity: "1",
          },
          "50%": {
            boxShadow: "0 0 0 8px rgba(118, 185, 0, 0)",
            opacity: "0.7",
          },
        },
      },
      animation: {
        shimmer: "shimmer 2s linear infinite",
        "fade-in-up": "fade-in-up 0.3s ease-out",
        "pulse-glow": "pulse-glow 1.8s ease-in-out infinite",
      },
    },
  },
};

export default config;
