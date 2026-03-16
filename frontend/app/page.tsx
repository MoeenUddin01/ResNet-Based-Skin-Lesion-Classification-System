"use client";

import { useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import Navbar from "@/components/Navbar";
import HeroSection from "@/components/HeroSection";
import ScrollCanvasSection from "@/components/ScrollCanvasSection";
import FeaturesGrid from "@/components/FeaturesGrid";
import UploadPanel from "@/components/UploadPanel";
import ResultsDisplay from "@/components/ResultsDisplay";
import Footer from "@/components/Footer";
import type { PredictionResponse } from "@/types/prediction";

export default function HomePage() {
  const [result, setResult] = useState<PredictionResponse | null>(null);

  return (
    <main className="relative min-h-screen bg-[#0a0f1e]">
      <Navbar />
      <HeroSection />
      <ScrollCanvasSection />
      <FeaturesGrid />

      {/* ── Upload & Results Section ─────────────────────────── */}
      <section id="upload" className="py-32 px-6">
        <div className="max-w-7xl mx-auto">
          {/* Section header */}
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-100px" }}
            transition={{ duration: 0.7 }}
            className="text-center mb-16"
          >
            <div className="inline-flex items-center gap-2 border border-cyan-500/20 rounded-full px-4 py-1.5 glass-card mb-4">
              <span className="text-cyan-400/70 text-xs font-mono tracking-widest uppercase">
                Clinical Analysis
              </span>
            </div>
            <h2 className="text-4xl md:text-5xl font-black text-white mb-4">
              Start Your{" "}
              <span className="gradient-text">AI Diagnosis</span>
            </h2>
            <p className="text-slate-400 text-lg max-w-xl mx-auto">
              Upload a dermoscopic image and receive instant classification
              results with confidence scores across all clinical categories.
            </p>
          </motion.div>

          {/* Upload + Results grid */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-10 items-start max-w-5xl mx-auto">
            {/* Upload panel */}
            <motion.div
              initial={{ opacity: 0, x: -30 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6 }}
            >
              <div className="mb-4">
                <h3 className="text-white font-bold text-xl mb-1">
                  Upload Image
                </h3>
                <p className="text-slate-500 text-sm">
                  Drag & drop or click to upload a lesion image
                </p>
              </div>
              <UploadPanel
                onResult={(r) => setResult(r)}
                onClear={() => setResult(null)}
              />
            </motion.div>

            {/* Results panel */}
            <motion.div
              initial={{ opacity: 0, x: 30 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6, delay: 0.1 }}
            >
              <div className="mb-4">
                <h3 className="text-white font-bold text-xl mb-1">
                  Analysis Results
                </h3>
                <p className="text-slate-500 text-sm">
                  AI classification output will appear here
                </p>
              </div>

              <AnimatePresence mode="wait">
                {result ? (
                  <ResultsDisplay key="results" result={result} />
                ) : (
                  <motion.div
                    key="placeholder"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="glass-card rounded-3xl p-10 flex flex-col items-center justify-center text-center"
                    style={{ minHeight: "380px" }}
                  >
                    <div className="w-16 h-16 rounded-full bg-cyan-500/5 border border-cyan-500/15 flex items-center justify-center mb-4">
                      <div className="w-8 h-8 rounded-full border-2 border-dashed border-cyan-500/30 animate-spin" />
                    </div>
                    <p className="text-slate-600 text-sm font-mono">
                      Awaiting image upload…
                    </p>
                    <p className="text-slate-700 text-xs mt-2">
                      Results will display after AI analysis
                    </p>
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>
          </div>
        </div>
      </section>

      <Footer />
    </main>
  );
}
