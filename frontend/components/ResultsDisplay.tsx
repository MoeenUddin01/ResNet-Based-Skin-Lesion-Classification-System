"use client";

import { motion } from "framer-motion";
import { CheckCircle, AlertTriangle, TrendingUp } from "lucide-react";
import type { PredictionResponse } from "@/types/prediction";
import { getLesionInfo } from "@/types/prediction";

interface ResultsDisplayProps {
    result: PredictionResponse;
}

// Ring SVG chart
function RingChart({ value, color }: { value: number; color: string }) {
    const radius = 52;
    const circumference = 2 * Math.PI * radius;
    const dash = circumference * value;

    return (
        <div className="relative w-36 h-36 mx-auto">
            <svg className="w-full h-full -rotate-90" viewBox="0 0 120 120">
                {/* Track */}
                <circle
                    cx="60" cy="60" r={radius}
                    fill="none"
                    stroke="rgba(34,211,238,0.08)"
                    strokeWidth="8"
                />
                {/* Progress */}
                <motion.circle
                    cx="60" cy="60" r={radius}
                    fill="none"
                    stroke={color}
                    strokeWidth="8"
                    strokeLinecap="round"
                    strokeDasharray={circumference}
                    initial={{ strokeDashoffset: circumference }}
                    animate={{ strokeDashoffset: circumference - dash }}
                    transition={{ duration: 1.2, ease: "easeOut", delay: 0.3 }}
                />
            </svg>
            {/* Center value */}
            <div className="absolute inset-0 flex flex-col items-center justify-center">
                <motion.span
                    initial={{ opacity: 0, scale: 0.5 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: 0.5, duration: 0.4 }}
                    className="text-2xl font-black text-white"
                >
                    {(value * 100).toFixed(1)}%
                </motion.span>
                <span className="text-slate-500 text-xs">confidence</span>
            </div>
        </div>
    );
}

export default function ResultsDisplay({ result }: ResultsDisplayProps) {
    const lesion = getLesionInfo(result.class_name);
    const isHighRisk = ["mel", "bcc", "akiec"].includes(result.class_name.toLowerCase());

    // Sort probabilities descending
    const sorted = Object.entries(result.probabilities).sort(([, a], [, b]) => b - a);
    const maxProb = sorted[0]?.[1] ?? 1;

    return (
        <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, ease: "easeOut" }}
            className="space-y-5"
        >
            {/* Header card */}
            <div className="glass-card rounded-3xl p-6">
                {/* Risk badge */}
                <div className="flex items-center gap-2 mb-4">
                    {isHighRisk ? (
                        <AlertTriangle className="w-4 h-4 text-amber-400" />
                    ) : (
                        <CheckCircle className="w-4 h-4 text-emerald-400" />
                    )}
                    <span
                        className={`text-xs font-mono tracking-widest uppercase ${isHighRisk ? "text-amber-400" : "text-emerald-400"
                            }`}
                    >
                        {isHighRisk ? "Requires Medical Attention" : "Generally Benign"}
                    </span>
                </div>

                {/* Ring + class info */}
                <div className="flex flex-col sm:flex-row items-center gap-6">
                    <RingChart value={result.confidence} color={lesion.color} />
                    <div className="flex-1 text-center sm:text-left">
                        <div className="text-xs font-mono text-slate-500 tracking-widest uppercase mb-1">
                            Predicted Class
                        </div>
                        <h3
                            className="text-2xl font-black mb-1"
                            style={{ color: lesion.color }}
                        >
                            {lesion.name}
                        </h3>
                        <p className="text-slate-400 text-sm leading-relaxed">
                            {lesion.description}
                        </p>
                        <div className="mt-3 inline-flex items-center gap-1.5 border rounded-full px-3 py-1 text-xs font-mono"
                            style={{ borderColor: `${lesion.color}30`, color: lesion.color, background: `${lesion.color}10` }}>
                            <TrendingUp className="w-3 h-3" />
                            CODE: {lesion.code.toUpperCase()}
                        </div>
                    </div>
                </div>
            </div>

            {/* Probability bars */}
            <div className="glass-card rounded-3xl p-6">
                <h4 className="text-slate-400 text-xs font-mono tracking-widest uppercase mb-4 flex items-center gap-2">
                    <span className="w-1.5 h-1.5 rounded-full bg-cyan-400" />
                    Class Probability Distribution
                </h4>
                <div className="space-y-3">
                    {sorted.map(([className, prob], i) => {
                        const info = getLesionInfo(className);
                        const barWidth = (prob / maxProb) * 100;
                        const isTop = i === 0;
                        return (
                            <div key={className}>
                                <div className="flex items-center justify-between mb-1">
                                    <span
                                        className={`text-xs font-medium ${isTop ? "text-white" : "text-slate-400"}`}
                                    >
                                        {info.name}
                                    </span>
                                    <span
                                        className="text-xs font-mono font-bold"
                                        style={{ color: isTop ? info.color : "rgba(148,163,184,0.7)" }}
                                    >
                                        {(prob * 100).toFixed(2)}%
                                    </span>
                                </div>
                                <div className="h-1.5 bg-white/5 rounded-full overflow-hidden">
                                    <motion.div
                                        className="h-full rounded-full"
                                        style={{ background: isTop ? info.color : "rgba(148,163,184,0.3)" }}
                                        initial={{ width: 0 }}
                                        animate={{ width: `${barWidth}%` }}
                                        transition={{ duration: 0.8, delay: 0.2 + i * 0.06, ease: "easeOut" }}
                                    />
                                </div>
                            </div>
                        );
                    })}
                </div>
            </div>

            {/* Disclaimer */}
            <p className="text-slate-600 text-xs text-center px-4 leading-relaxed">
                ⚠ This tool is for research and educational purposes only.
                Always consult a certified dermatologist for medical diagnosis.
            </p>
        </motion.div>
    );
}
