"use client";

import { useEffect, useState } from "react";
import { motion, useScroll, useTransform } from "framer-motion";
import { Activity, Menu, X } from "lucide-react";

const navLinks = [
    { label: "Overview", href: "#hero" },
    { label: "Features", href: "#features" },
    { label: "Analyze", href: "#upload" },
    { label: "Research", href: "#canvas" },
];

export default function Navbar() {
    const [scrolled, setScrolled] = useState(false);
    const [mobileOpen, setMobileOpen] = useState(false);

    useEffect(() => {
        const onScroll = () => setScrolled(window.scrollY > 40);
        window.addEventListener("scroll", onScroll, { passive: true });
        return () => window.removeEventListener("scroll", onScroll);
    }, []);

    return (
        <motion.nav
            initial={{ y: -80, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ duration: 0.7, ease: "easeOut" }}
            className={`fixed top-0 left-0 right-0 z-50 transition-all duration-500 ${scrolled
                    ? "glass-card border-b border-cyan-500/10 py-3"
                    : "bg-transparent py-5"
                }`}
        >
            <div className="max-w-7xl mx-auto px-6 flex items-center justify-between">
                {/* Logo */}
                <motion.a
                    href="#hero"
                    className="flex items-center gap-2 group"
                    whileHover={{ scale: 1.02 }}
                >
                    <div className="relative w-8 h-8">
                        <div className="absolute inset-0 bg-cyan-500/20 rounded-lg blur-md group-hover:bg-cyan-500/40 transition-all" />
                        <div className="relative w-8 h-8 bg-gradient-to-br from-cyan-400 to-teal-500 rounded-lg flex items-center justify-center">
                            <Activity className="w-4 h-4 text-navy" strokeWidth={2.5} />
                        </div>
                    </div>
                    <div>
                        <span className="text-white font-bold text-sm tracking-wide">
                            DermaAI
                        </span>
                        <div className="text-cyan-400/60 text-[10px] tracking-widest uppercase font-mono">
                            ResNet-152
                        </div>
                    </div>
                </motion.a>

                {/* Desktop Links */}
                <div className="hidden md:flex items-center gap-8">
                    {navLinks.map((link, i) => (
                        <motion.a
                            key={link.href}
                            href={link.href}
                            initial={{ opacity: 0, y: -10 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: 0.1 * i + 0.3 }}
                            className="text-slate-400 hover:text-cyan-400 text-sm font-medium transition-colors duration-200 relative group"
                        >
                            {link.label}
                            <span className="absolute -bottom-0.5 left-0 w-0 h-px bg-cyan-400 group-hover:w-full transition-all duration-300" />
                        </motion.a>
                    ))}
                    <motion.a
                        href="#upload"
                        initial={{ opacity: 0, scale: 0.9 }}
                        animate={{ opacity: 1, scale: 1 }}
                        transition={{ delay: 0.6 }}
                        className="bg-gradient-to-r from-cyan-500 to-teal-500 text-navy font-semibold text-sm px-5 py-2 rounded-full hover:shadow-lg hover:shadow-cyan-500/30 transition-all duration-300 hover:scale-105"
                    >
                        Start Analysis
                    </motion.a>
                </div>

                {/* Mobile hamburger */}
                <button
                    className="md:hidden text-slate-400 hover:text-cyan-400 transition-colors"
                    onClick={() => setMobileOpen(!mobileOpen)}
                    aria-label="Toggle menu"
                >
                    {mobileOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
                </button>
            </div>

            {/* Mobile menu */}
            {mobileOpen && (
                <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: "auto" }}
                    exit={{ opacity: 0, height: 0 }}
                    className="md:hidden glass-card border-t border-cyan-500/10 px-6 py-4 flex flex-col gap-4"
                >
                    {navLinks.map((link) => (
                        <a
                            key={link.href}
                            href={link.href}
                            onClick={() => setMobileOpen(false)}
                            className="text-slate-300 hover:text-cyan-400 text-sm font-medium transition-colors"
                        >
                            {link.label}
                        </a>
                    ))}
                    <a
                        href="#upload"
                        onClick={() => setMobileOpen(false)}
                        className="bg-gradient-to-r from-cyan-500 to-teal-500 text-navy font-semibold text-sm px-5 py-2 rounded-full text-center"
                    >
                        Start Analysis
                    </a>
                </motion.div>
            )}
        </motion.nav>
    );
}
