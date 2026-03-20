"use client";

import { motion, Variants } from "framer-motion";
import {
    Upload,
    BrainCircuit,
    Tag,
    BarChart3,
    LayoutDashboard,
    Zap,
} from "lucide-react";

const features = [
    {
        icon: Upload,
        title: "Image Upload",
        description:
            "Drag-and-drop JPEG/PNG skin lesion images. Instant preview with pixel-level rendering.",
        color: "#22d3ee",
        glow: "rgba(34,211,238,0.15)",
    },
    {
        icon: BrainCircuit,
        title: "AI Prediction",
        description:
            "ResNet-152 deep neural network trained on 10,000+ HAM10000 dermatoscopic images.",
        color: "#0d9488",
        glow: "rgba(13,148,136,0.15)",
    },
    {
        icon: Tag,
        title: "Lesion Class Output",
        description:
            "Classifies into 7 categories: Melanoma, Nevus, BCC, Keratosis, and more.",
        color: "#a78bfa",
        glow: "rgba(167,139,250,0.15)",
    },
    {
        icon: BarChart3,
        title: "Confidence Score",
        description:
            "Full probability distribution across all classes with animated confidence bars.",
        color: "#f59e0b",
        glow: "rgba(245,158,11,0.15)",
    },
    {
        icon: LayoutDashboard,
        title: "Medical Dashboard",
        description:
            "Clinical-grade result display with lesion metadata, risk indicators, and visual charts.",
        color: "#ec4899",
        glow: "rgba(236,72,153,0.15)",
    },
    {
        icon: Zap,
        title: "Real-time Analysis",
        description:
            "FastAPI backend delivers sub-second inference results via optimized model serving.",
        color: "#34d399",
        glow: "rgba(52,211,153,0.15)",
    },
];

const containerVariants = {
    hidden: {},
    visible: { transition: { staggerChildren: 0.12 } },
};

const cardVariants: Variants = {
    hidden: { opacity: 0, y: 40 },
    visible: {
        opacity: 1,
        y: 0,
        transition: { duration: 0.6, ease: "easeOut" },
    },
};

export default function FeaturesGrid() {
    return (
        <section id="features" className="py-32 px-6">
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
                            System Capabilities
                        </span>
                    </div>
                    <h2 className="text-4xl md:text-5xl font-black text-white mb-4">
                        Built for{" "}
                        <span className="gradient-text">Clinical Precision</span>
                    </h2>
                    <p className="text-slate-400 text-lg max-w-xl mx-auto">
                        Every module engineered for accuracy, transparency, and speed in
                        dermatological AI workflows.
                    </p>
                </motion.div>

                {/* Grid */}
                <motion.div
                    variants={containerVariants}
                    initial="hidden"
                    whileInView="visible"
                    viewport={{ once: true, margin: "-80px" }}
                    className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6"
                >
                    {features.map((f) => {
                        const Icon = f.icon;
                        return (
                            <motion.div
                                key={f.title}
                                variants={cardVariants}
                                whileHover={{ y: -6, scale: 1.02 }}
                                className="group glass-card rounded-3xl p-7 cursor-default relative overflow-hidden transition-all duration-300"
                                style={{
                                    boxShadow: `0 0 0 1px rgba(34,211,238,0.08)`,
                                }}
                                onMouseEnter={(e) => {
                                    (e.currentTarget as HTMLDivElement).style.boxShadow = `0 0 30px ${f.glow}, 0 0 0 1px rgba(34,211,238,0.2)`;
                                }}
                                onMouseLeave={(e) => {
                                    (e.currentTarget as HTMLDivElement).style.boxShadow = `0 0 0 1px rgba(34,211,238,0.08)`;
                                }}
                            >
                                {/* Background glow on hover */}
                                <div
                                    className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-500 rounded-3xl"
                                    style={{
                                        background: `radial-gradient(circle at 30% 30%, ${f.glow} 0%, transparent 60%)`,
                                    }}
                                />

                                {/* Icon */}
                                <div
                                    className="relative w-12 h-12 rounded-2xl flex items-center justify-center mb-5 transition-transform duration-300 group-hover:scale-110"
                                    style={{ background: `${f.color}15` }}
                                >
                                    <Icon
                                        className="w-6 h-6"
                                        style={{ color: f.color }}
                                        strokeWidth={1.8}
                                    />
                                </div>

                                {/* Content */}
                                <h3 className="text-white font-bold text-lg mb-2 relative z-10">
                                    {f.title}
                                </h3>
                                <p className="text-slate-400 text-sm leading-relaxed relative z-10">
                                    {f.description}
                                </p>

                                {/* Corner accent */}
                                <div
                                    className="absolute top-0 right-0 w-24 h-24 opacity-0 group-hover:opacity-10 transition-opacity duration-500"
                                    style={{
                                        background: `radial-gradient(circle, ${f.color} 0%, transparent 70%)`,
                                    }}
                                />
                            </motion.div>
                        );
                    })}
                </motion.div>
            </div>
        </section>
    );
}
