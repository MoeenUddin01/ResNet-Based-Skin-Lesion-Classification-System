"use client";

import { useRef, useEffect } from "react";
import { motion, useScroll, useTransform, useSpring } from "framer-motion";

// Medical/dermatoscopy themed frame data
const FRAME_COUNT = 40;

function generateFrameColors(frameIndex: number): {
    bg: string;
    inner: string;
    rings: string[];
    particles: { x: number; y: number; r: number; hue: number }[];
} {
    const t = frameIndex / (FRAME_COUNT - 1);
    const hue = 180 + t * 40; // cyan to teal range

    return {
        bg: `hsl(${210 + t * 10}, 30%, ${6 + t * 4}%)`,
        inner: `hsl(${hue}, 70%, ${30 + t * 20}%)`,
        rings: [
            `hsla(${hue}, 80%, 50%, ${0.05 + t * 0.15})`,
            `hsla(${hue - 20}, 70%, 60%, ${0.03 + t * 0.1})`,
            `hsla(${hue + 20}, 90%, 70%, ${0.02 + t * 0.08})`,
        ],
        particles: Array.from({ length: 6 + Math.floor(t * 10) }, (_, i) => ({
            x: 0.5 + Math.cos((i / 8) * Math.PI * 2 + t * 2) * (0.15 + t * 0.2),
            y: 0.5 + Math.sin((i / 8) * Math.PI * 2 + t * 2) * (0.15 + t * 0.2),
            r: Math.max(0.5, 2 + Math.sin(i + t * Math.PI) * 1.5),
            hue: hue + i * 15,
        })),
    };
}

function drawFrame(ctx: CanvasRenderingContext2D, frameIndex: number, w: number, h: number) {
    const fc = generateFrameColors(frameIndex);
    const t = frameIndex / (FRAME_COUNT - 1);
    // Guard against zero/invalid canvas size
    if (!w || !h || w < 10 || h < 10) return;

    const cx = w / 2;
    const cy = h / 2;

    // Background
    ctx.fillStyle = fc.bg;
    ctx.fillRect(0, 0, w, h);

    // Grid overlay
    ctx.strokeStyle = "rgba(34,211,238,0.04)";
    ctx.lineWidth = 1;
    const gridSize = 40;
    for (let x = 0; x < w; x += gridSize) {
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
    }
    for (let y = 0; y < h; y += gridSize) {
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
    }

    // Outer scanning rings
    for (let i = 0; i < 5; i++) {
        const radius = Math.max(1, (0.15 + i * 0.07 + t * 0.04) * Math.min(w, h));
        const alpha = 0.03 + (t * 0.12) / (i + 1);
        ctx.beginPath();
        ctx.arc(cx, cy, radius, 0, Math.PI * 2);
        ctx.strokeStyle = `hsla(${190 + i * 10}, 80%, 60%, ${alpha})`;
        ctx.lineWidth = 1;
        ctx.stroke();
    }

    // Main lesion simulation: layered radial gradient
    const lesionR = Math.max(10, (0.18 + t * 0.06) * Math.min(w, h));
    const grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, Math.max(1, lesionR));
    grad.addColorStop(0, fc.inner);
    grad.addColorStop(0.5, `hsla(${190 + t * 30}, 60%, 25%, 0.9)`);
    grad.addColorStop(1, "rgba(10,15,30,0)");
    ctx.beginPath();
    ctx.arc(cx, cy, lesionR, 0, Math.PI * 2);
    ctx.fillStyle = grad;
    ctx.fill();

    // Texture dots on lesion
    for (let i = 0; i < 30; i++) {
        const angle = (i / 30) * Math.PI * 2 + t * 3;
        const r = lesionR * (0.3 + (i % 5) * 0.12);
        const px = cx + Math.cos(angle) * r;
        const py = cy + Math.sin(angle) * r * 0.85;
        ctx.beginPath();
        ctx.arc(px, py, 1.5 + (i % 3), 0, Math.PI * 2);
        ctx.fillStyle = `hsla(${200 + i * 2}, 60%, ${50 + i % 20}%, ${0.3 + t * 0.4})`;
        ctx.fill();
    }

    // Particles / diagnostic markers
    for (const p of fc.particles) {
        const px = p.x * w;
        const py = p.y * h;
        ctx.beginPath();
        ctx.arc(px, py, p.r, 0, Math.PI * 2);
        ctx.fillStyle = `hsla(${p.hue}, 80%, 70%, 0.7)`;
        ctx.fill();
        // Connecting line to center
        ctx.strokeStyle = `hsla(${p.hue}, 60%, 60%, 0.15)`;
        ctx.lineWidth = 0.5;
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(px, py);
        ctx.stroke();
    }

    // Crosshair / targeting reticle
    const rSize = lesionR * 1.3;
    ctx.strokeStyle = `rgba(34,211,238,${0.2 + t * 0.4})`;
    ctx.lineWidth = 1;
    const dashLen = 12;
    // Top
    ctx.beginPath(); ctx.moveTo(cx, cy - rSize); ctx.lineTo(cx, cy - rSize + dashLen); ctx.stroke();
    // Bottom
    ctx.beginPath(); ctx.moveTo(cx, cy + rSize - dashLen); ctx.lineTo(cx, cy + rSize); ctx.stroke();
    // Left
    ctx.beginPath(); ctx.moveTo(cx - rSize, cy); ctx.lineTo(cx - rSize + dashLen, cy); ctx.stroke();
    // Right
    ctx.beginPath(); ctx.moveTo(cx + rSize - dashLen, cy); ctx.lineTo(cx + rSize, cy); ctx.stroke();

    // Corner brackets
    const bSize = 20;
    const margin = 20;
    const corners = [
        [margin, margin],
        [w - margin, margin],
        [margin, h - margin],
        [w - margin, h - margin],
    ];
    const dirs = [[1, 1], [-1, 1], [1, -1], [-1, -1]];
    ctx.strokeStyle = `rgba(34,211,238,${0.3 + t * 0.3})`;
    ctx.lineWidth = 2;
    for (let i = 0; i < corners.length; i++) {
        const [bx, by] = corners[i];
        const [dx, dy] = dirs[i];
        ctx.beginPath(); ctx.moveTo(bx, by); ctx.lineTo(bx + dx * bSize, by); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(bx, by); ctx.lineTo(bx, by + dy * bSize); ctx.stroke();
    }

    // HUD readouts
    const progress = Math.round(t * 100);
    ctx.fillStyle = "rgba(34,211,238,0.8)";
    ctx.font = `bold 11px 'JetBrains Mono', monospace`;
    ctx.fillText(`SCAN ${String(frameIndex + 1).padStart(2, "0")}/${FRAME_COUNT}`, margin, h - margin - 12);
    ctx.fillStyle = "rgba(34,211,238,0.5)";
    ctx.font = `10px 'JetBrains Mono', monospace`;
    ctx.fillText(`ANALYSIS: ${progress}%`, margin, h - margin + 2);

    // Progress bar
    ctx.fillStyle = "rgba(34,211,238,0.1)";
    ctx.fillRect(margin, h - margin + 10, w - margin * 2, 3);
    ctx.fillStyle = `rgba(34,211,238,${0.5 + t * 0.5})`;
    ctx.fillRect(margin, h - margin + 10, (w - margin * 2) * t, 3);
}

export default function ScrollCanvasSection() {
    const sectionRef = useRef<HTMLDivElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);

    const { scrollYProgress } = useScroll({
        target: sectionRef,
        offset: ["start start", "end end"],
    });

    // Smooth out scroll progress
    const smooth = useSpring(scrollYProgress, { stiffness: 80, damping: 20 });

    // Parallax text transforms
    const textY1 = useTransform(smooth, [0, 1], ["0%", "-60%"]);
    const textY2 = useTransform(smooth, [0, 1], ["0%", "-25%"]);
    const textOpacity = useTransform(smooth, [0, 0.15, 0.8, 1], [0, 1, 1, 0]);
    const subTextOpacity = useTransform(smooth, [0.1, 0.25, 0.75, 0.9], [0, 1, 1, 0]);

    // Render frames on scroll
    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;

        const ctx = canvas.getContext("2d");
        if (!ctx) return;

        const unsubscribe = smooth.on("change", (v) => {
            const frameIndex = Math.min(
                FRAME_COUNT - 1,
                Math.max(0, Math.round(v * (FRAME_COUNT - 1)))
            );
            const w = canvas.width;
            const h = canvas.height;
            drawFrame(ctx, frameIndex, w, h);
        });

        // Draw initial frame
        drawFrame(ctx, 0, canvas.width, canvas.height);

        return () => unsubscribe();
    }, [smooth]);

    // Resize canvas to match display
    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const resize = () => {
            const rect = canvas.getBoundingClientRect();
            canvas.width = rect.width;
            canvas.height = rect.height;
            const ctx = canvas.getContext("2d");
            if (ctx) drawFrame(ctx, 0, canvas.width, canvas.height);
        };
        resize();
        window.addEventListener("resize", resize);
        return () => window.removeEventListener("resize", resize);
    }, []);

    return (
        <section id="canvas" ref={sectionRef} className="relative" style={{ height: "300vh" }}>
            {/* Sticky canvas wrapper */}
            <div className="sticky top-0 h-screen flex items-center justify-center overflow-hidden">
                {/* Canvas */}
                <canvas
                    ref={canvasRef}
                    width={1280}
                    height={720}
                    className="absolute inset-0 w-full h-full"
                />

                {/* Parallax text layer 1 — deeper depth */}
                <motion.div
                    style={{ y: textY1, opacity: textOpacity }}
                    className="relative z-10 text-center pointer-events-none select-none px-6"
                >
                    <p className="text-cyan-400/40 text-xs font-mono tracking-[0.3em] uppercase mb-4">
                        Deep Learning Diagnostics
                    </p>
                    <h2 className="text-5xl md:text-7xl font-black text-white/90 leading-tight drop-shadow-2xl">
                        See What the{" "}
                        <span className="gradient-text">Model Sees</span>
                    </h2>
                </motion.div>

                {/* Parallax text layer 2 — shallower depth */}
                <motion.div
                    style={{ y: textY2, opacity: subTextOpacity }}
                    className="absolute bottom-32 left-1/2 -translate-x-1/2 z-10 text-center pointer-events-none select-none px-6"
                >
                    <p className="text-slate-400 text-lg max-w-lg mx-auto leading-relaxed drop-shadow-2xl">
                        ResNet-152 processes 224×224 dermatoscopic patches,
                        extracting hierarchical features across 152 convolutional layers.
                    </p>
                    <div className="mt-6 flex gap-6 justify-center">
                        {[
                            ["152", "Layers"],
                            ["10K+", "Training Images"],
                            ["7", "Classes"],
                        ].map(([v, l]) => (
                            <div key={l} className="text-center">
                                <div className="text-2xl font-black gradient-text">{v}</div>
                                <div className="text-slate-500 text-xs">{l}</div>
                            </div>
                        ))}
                    </div>
                </motion.div>
            </div>
        </section>
    );
}
