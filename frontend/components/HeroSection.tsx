"use client";

import { useRef } from "react";
import { motion, Variants } from "framer-motion";
import { ArrowDown, Microscope, Shield, Zap } from "lucide-react";

const stats = [
    { label: "Lesion Classes", value: "7", icon: Microscope },
    { label: "Model Accuracy", value: "94%+", icon: Shield },
    { label: "Inference Time", value: "<1s", icon: Zap },
];

const wordVariants: Variants = {
    hidden: { opacity: 0, y: 40 },
    visible: (i: number) => ({
        opacity: 1,
        y: 0,
        transition: { delay: i * 0.12, duration: 0.7, ease: "easeOut" },
    }),
};

const heroWords = ["AI-Powered", "Skin Lesion", "Detection"];

export default function HeroSection() {
    const ref = useRef<HTMLElement>(null);

    return (
        <section
            id="hero"
            ref={ref}
            className="relative min-h-screen flex flex-col items-center justify-center overflow-hidden px-6 pt-24"
        >
            {/* Animated background blobs */}
            <div className="absolute inset-0 overflow-hidden pointer-events-none">
                <div className="blob blob-delay-2 absolute -top-40 -left-40 w-[600px] h-[600px] rounded-full bg-cyan-500/5 blur-3xl" />
                <div className="blob blob-delay-4 absolute top-1/2 -right-40 w-[500px] h-[500px] rounded-full bg-teal-500/5 blur-3xl" />
                <div className="blob absolute bottom-0 left-1/3 w-[400px] h-[400px] rounded-full bg-cyan-600/5 blur-3xl" />
                {/* Grid overlay */}
                <div
                    className="absolute inset-0 opacity-10"
                    style={{
                        backgroundImage: `linear-gradient(rgba(34,211,238,0.1) 1px, transparent 1px),
              linear-gradient(90deg, rgba(34,211,238,0.1) 1px, transparent 1px)`,
                        backgroundSize: "60px 60px",
                    }}
                />
                {/* Scan lines */}
                <div className="scan-overlay absolute inset-0 opacity-30" />
            </div>

            {/* Badge */}
            <motion.div
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.5 }}
                className="relative z-10 mb-8"
            >
                <div className="flex items-center gap-2 border border-cyan-500/30 rounded-full px-4 py-1.5 glass-card">
                    <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
                    <span className="text-cyan-400 text-xs font-mono tracking-widest uppercase">
                        ResNet-152 · HAM10000 Dataset · PyTorch
                    </span>
                </div>
            </motion.div>

            {/* Main headline */}
            <div className="relative z-10 text-center max-w-5xl">
                <div className="overflow-hidden">
                    {heroWords.map((word, i) => (
                        <motion.span
                            key={word}
                            custom={i}
                            variants={wordVariants}
                            initial="hidden"
                            animate="visible"
                            className={`inline-block text-6xl md:text-8xl font-black tracking-tight leading-tight mr-4 ${i === 0 ? "gradient-text" : i === 1 ? "text-white" : "gradient-text"
                                }`}
                        >
                            {word}
                        </motion.span>
                    ))}
                </div>

                <motion.p
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.6, duration: 0.7 }}
                    className="mt-6 text-slate-400 text-lg md:text-xl max-w-2xl mx-auto leading-relaxed"
                >
                    Deep learning–based dermatological image analysis. Upload a skin lesion image
                    and receive instant classification across 7 clinical categories with confidence scores.
                </motion.p>

                {/* CTA buttons */}
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.8, duration: 0.6 }}
                    className="mt-10 flex flex-col sm:flex-row gap-4 justify-center"
                >
                    <a
                        href="#upload"
                        className="group relative bg-gradient-to-r from-cyan-500 to-teal-500 text-navy font-bold px-8 py-4 rounded-2xl text-base hover:shadow-2xl hover:shadow-cyan-500/40 transition-all duration-300 hover:scale-105"
                    >
                        <span className="relative z-10">Analyze a Lesion</span>
                    </a>
                    <a
                        href="#canvas"
                        className="glass-card border border-cyan-500/20 text-slate-300 font-semibold px-8 py-4 rounded-2xl text-base hover:border-cyan-500/50 hover:text-white transition-all duration-300 hover:scale-105"
                    >
                        Explore Research ↓
                    </a>
                </motion.div>
            </div>

            {/* Stats row */}
            <motion.div
                initial={{ opacity: 0, y: 30 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 1.0, duration: 0.7 }}
                className="relative z-10 mt-20 flex flex-wrap gap-6 justify-center"
            >
                {stats.map(({ label, value, icon: Icon }, i) => (
                    <motion.div
                        key={label}
                        initial={{ opacity: 0, scale: 0.8 }}
                        animate={{ opacity: 1, scale: 1 }}
                        transition={{ delay: 1.1 + i * 0.15 }}
                        className="glass-card rounded-2xl px-6 py-4 flex items-center gap-3 glow-cyan"
                    >
                        <div className="w-10 h-10 rounded-xl bg-cyan-500/10 flex items-center justify-center">
                            <Icon className="w-5 h-5 text-cyan-400" />
                        </div>
                        <div>
                            <div className="text-2xl font-black gradient-text">{value}</div>
                            <div className="text-slate-500 text-xs font-medium">{label}</div>
                        </div>
                    </motion.div>
                ))}
            </motion.div>

            {/* Scroll indicator */}
            <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 1.5 }}
                className="absolute bottom-10 left-1/2 -translate-x-1/2 flex flex-col items-center gap-2"
            >
                <motion.div
                    animate={{ y: [0, 8, 0] }}
                    transition={{ repeat: Infinity, duration: 1.6, ease: "easeInOut" }}
                >
                    <ArrowDown className="w-5 h-5 text-cyan-400/50" />
                </motion.div>
                <span className="text-slate-600 text-xs font-mono tracking-widest">SCROLL</span>
            </motion.div>
        </section>
    );
}
