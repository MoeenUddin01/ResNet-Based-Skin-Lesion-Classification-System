"use client";

import { motion } from "framer-motion";
import { Activity, Github, ExternalLink } from "lucide-react";

const tech = ["ResNet-152", "PyTorch", "FastAPI", "Next.js", "Framer Motion", "HAM10000"];

export default function Footer() {
    return (
        <footer className="relative border-t border-cyan-500/10 py-16 px-6">
            {/* Subtle top glow */}
            <div className="absolute top-0 left-1/2 -translate-x-1/2 w-96 h-px bg-gradient-to-r from-transparent via-cyan-500/30 to-transparent" />

            <div className="max-w-7xl mx-auto">
                <div className="flex flex-col md:flex-row items-center justify-between gap-8">
                    {/* Logo */}
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        whileInView={{ opacity: 1, y: 0 }}
                        viewport={{ once: true }}
                        className="flex items-center gap-3"
                    >
                        <div className="w-8 h-8 bg-gradient-to-br from-cyan-400 to-teal-500 rounded-lg flex items-center justify-center">
                            <Activity className="w-4 h-4 text-navy" strokeWidth={2.5} />
                        </div>
                        <div>
                            <div className="text-white font-bold text-sm">DermaAI</div>
                            <div className="text-slate-600 text-xs font-mono">Skin Lesion Classification System</div>
                        </div>
                    </motion.div>

                    {/* Tech badges */}
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        whileInView={{ opacity: 1, y: 0 }}
                        viewport={{ once: true }}
                        transition={{ delay: 0.1 }}
                        className="flex flex-wrap gap-2 justify-center"
                    >
                        {tech.map((t) => (
                            <span
                                key={t}
                                className="text-xs font-mono text-slate-500 border border-slate-800 rounded-full px-3 py-1 hover:border-cyan-500/30 hover:text-cyan-400 transition-colors"
                            >
                                {t}
                            </span>
                        ))}
                    </motion.div>

                    {/* Links */}
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        whileInView={{ opacity: 1, y: 0 }}
                        viewport={{ once: true }}
                        transition={{ delay: 0.2 }}
                        className="flex items-center gap-4"
                    >
                        <a
                            href="https://github.com"
                            target="_blank"
                            rel="noopener noreferrer"
                            className="flex items-center gap-1.5 text-slate-500 hover:text-cyan-400 text-sm transition-colors"
                        >
                            <Github className="w-4 h-4" />
                            GitHub
                        </a>
                        <a
                            href="http://localhost:8000/docs"
                            target="_blank"
                            rel="noopener noreferrer"
                            className="flex items-center gap-1.5 text-slate-500 hover:text-cyan-400 text-sm transition-colors"
                        >
                            <ExternalLink className="w-4 h-4" />
                            API Docs
                        </a>
                    </motion.div>
                </div>

                {/* Bottom line */}
                <div className="mt-10 pt-6 border-t border-slate-900 text-center">
                    <p className="text-slate-700 text-xs font-mono">
                        © 2026 DermaAI · ResNet-Based Skin Lesion Classification ·{" "}
                        <span className="text-slate-600">For research purposes only</span>
                    </p>
                </div>
            </div>
        </footer>
    );
}
