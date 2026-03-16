import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

const jetBrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-mono",
  display: "swap",
});

export const metadata: Metadata = {
  title: "DermaAI — AI-Powered Skin Lesion Classification",
  description:
    "ResNet-152 deep learning model for clinical-grade skin lesion detection and classification across 7 dermatological categories. Powered by PyTorch and HAM10000 dataset.",
  keywords: [
    "skin lesion classification",
    "AI dermatology",
    "melanoma detection",
    "ResNet-152",
    "HAM10000",
    "deep learning medical",
  ],
  authors: [{ name: "DermaAI" }],
  openGraph: {
    title: "DermaAI — AI-Powered Skin Lesion Classification",
    description: "Clinical-grade skin lesion detection powered by ResNet-152",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={`${inter.variable} ${jetBrainsMono.variable}`}>
      <body className="bg-[#0a0f1e] text-slate-200 antialiased font-sans">
        {children}
      </body>
    </html>
  );
}
