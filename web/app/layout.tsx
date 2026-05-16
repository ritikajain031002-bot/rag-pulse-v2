import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import "./globals.css";

const sans = Inter({ subsets: ["latin"], variable: "--font-sans", display: "swap" });
const mono = JetBrains_Mono({ subsets: ["latin"], variable: "--font-mono", display: "swap" });

export const metadata: Metadata = {
  title: "RAG Pulse · Universal Knowledge Chat",
  description:
    "Multi-modal Retrieval-Augmented Generation chatbot powered by NVIDIA NIM. Ask anything about your PDFs, images, audio, video, and the web.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={`${sans.variable} ${mono.variable}`}>
      <body className="bg-[#08080c] text-white font-sans antialiased">{children}</body>
    </html>
  );
}
