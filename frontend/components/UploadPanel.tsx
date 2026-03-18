"use client";

import { useState, useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { motion, AnimatePresence } from "framer-motion";
import { Upload, ImageIcon, X, Loader2, Microscope } from "lucide-react";
import type { PredictionResponse } from "@/types/prediction";

interface UploadPanelProps {
    onResult: (result: PredictionResponse) => void;
    onClear: () => void;
}

export default function UploadPanel({ onResult, onClear }: UploadPanelProps) {
    const [preview, setPreview] = useState<string | null>(null);
    const [file, setFile] = useState<File | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const onDrop = useCallback((accepted: File[]) => {
        const f = accepted[0];
        if (!f) return;
        setFile(f);
        setError(null);
        onClear();
        const reader = new FileReader();
        reader.onload = (e) => setPreview(e.target?.result as string);
        reader.readAsDataURL(f);
    }, [onClear]);

    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop,
        accept: { "image/*": [] },
        maxFiles: 1,
        maxSize: 10 * 1024 * 1024,
    });

    const handleClear = () => {
        setPreview(null);
        setFile(null);
        setError(null);
        onClear();
    };

    const handlePredict = async () => {
        if (!file) return;
        setLoading(true);
        setError(null);

        try {
            const formData = new FormData();
            formData.append("file", file);

            const response = await fetch("/api/v1/predict", {
                method: "POST",
                body: formData,
            });

            if (!response.ok) {
                let errorMessage = "Prediction failed. Please try again.";
                const rawText = await response.text();
                try {
                    const data = JSON.parse(rawText);
                    errorMessage = data.detail ?? data.message ?? errorMessage;
                } catch {
                    if (rawText) errorMessage = `Server error (${response.status}): ${rawText.slice(0, 120)}`;
                }
                throw new Error(errorMessage);
            }

            const result: PredictionResponse = await response.json();
            onResult(result);
        } catch (err) {
            setError(err instanceof Error ? err.message : "An unexpected error occurred.");
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="w-full max-w-lg mx-auto">
            <AnimatePresence mode="wait">
                {!preview ? (
                    /* Dropzone */
                    <motion.div
                        key="dropzone"
                        initial={{ opacity: 0, scale: 0.96 }}
                        animate={{ opacity: 1, scale: 1 }}
                        exit={{ opacity: 0, scale: 0.96 }}
                        transition={{ duration: 0.3 }}
                    >
                        <div
                            {...getRootProps()}
                            className={`relative cursor-pointer rounded-3xl p-10 text-center transition-all duration-300 border-2 border-dashed ${isDragActive
                                ? "border-cyan-400 bg-cyan-500/10 glow-cyan"
                                : "border-cyan-500/20 glass-card hover:border-cyan-500/50 hover:bg-cyan-500/5"
                                } border-glow-animate`}
                        >
                            <input {...getInputProps()} />
                            <motion.div
                                animate={isDragActive ? { scale: 1.1 } : { scale: 1 }}
                                transition={{ type: "spring", stiffness: 300 }}
                                className="flex flex-col items-center gap-4"
                            >
                                <div className="relative">
                                    <div className="absolute inset-0 bg-cyan-500/10 rounded-full blur-xl animate-pulse" />
                                    <div className="relative w-20 h-20 rounded-full bg-cyan-500/10 border border-cyan-500/20 flex items-center justify-center">
                                        <Upload className="w-8 h-8 text-cyan-400" strokeWidth={1.5} />
                                    </div>
                                </div>
                                <div>
                                    <p className="text-white font-semibold text-lg">
                                        {isDragActive ? "Drop the lesion image" : "Upload a Skin Lesion Image"}
                                    </p>
                                    <p className="text-slate-500 text-sm mt-1">
                                        Drag &amp; drop or click · Any image format · max 10 MB
                                    </p>
                                </div>
                                <div className="flex gap-2 mt-2">
                                    {["JPEG", "PNG", "WEBP", "BMP"].map((f) => (
                                        <span
                                            key={f}
                                            className="text-xs font-mono text-cyan-400/60 border border-cyan-500/20 rounded-full px-3 py-0.5"
                                        >
                                            {f}
                                        </span>
                                    ))}
                                </div>
                            </motion.div>
                        </div>
                    </motion.div>
                ) : (
                    /* Preview */
                    <motion.div
                        key="preview"
                        initial={{ opacity: 0, scale: 0.96 }}
                        animate={{ opacity: 1, scale: 1 }}
                        exit={{ opacity: 0, scale: 0.96 }}
                        transition={{ duration: 0.3 }}
                        className="glass-card rounded-3xl overflow-hidden"
                    >
                        {/* Image preview */}
                        <div className="relative aspect-square bg-navy">
                            {/* eslint-disable-next-line @next/next/no-img-element */}
                            <img
                                src={preview}
                                alt="Uploaded lesion"
                                className="w-full h-full object-contain"
                            />
                            {/* Scanning overlay when loading */}
                            {loading && (
                                <div className="absolute inset-0 bg-navy/60 flex flex-col items-center justify-center gap-4">
                                    <div className="relative">
                                        <div className="w-16 h-16 rounded-full border-2 border-cyan-500/20 animate-spin border-t-cyan-400" />
                                        <div className="absolute inset-2 w-12 h-12 rounded-full border border-teal-500/30 animate-spin border-b-teal-400" style={{ animationDirection: "reverse" }} />
                                        <Loader2 className="absolute inset-0 m-auto w-5 h-5 text-cyan-400 animate-spin" />
                                    </div>
                                    <p className="text-cyan-400 text-sm font-mono tracking-widest">ANALYZING…</p>
                                </div>
                            )}
                            {/* Clear button */}
                            {!loading && (
                                <button
                                    onClick={handleClear}
                                    className="absolute top-3 right-3 w-8 h-8 bg-navy/80 border border-slate-700 rounded-full flex items-center justify-center hover:border-red-500/50 hover:text-red-400 text-slate-400 transition-all"
                                >
                                    <X className="w-4 h-4" />
                                </button>
                            )}
                            {/* Scan corner overlays */}
                            <div className="absolute top-3 left-3 w-5 h-5 border-t-2 border-l-2 border-cyan-400/50 rounded-tl" />
                            <div className="absolute top-3 right-14 w-5 h-5 border-t-2 border-r-2 border-cyan-400/50 rounded-tr" />
                            <div className="absolute bottom-3 left-3 w-5 h-5 border-b-2 border-l-2 border-cyan-400/50 rounded-bl" />
                            <div className="absolute bottom-3 right-3 w-5 h-5 border-b-2 border-r-2 border-cyan-400/50 rounded-br" />
                        </div>

                        {/* File info & action */}
                        <div className="p-5">
                            <div className="flex items-center gap-3 mb-4">
                                <ImageIcon className="w-4 h-4 text-cyan-400 flex-shrink-0" />
                                <span className="text-slate-300 text-sm truncate font-medium">{file?.name}</span>
                                <span className="text-slate-600 text-xs ml-auto flex-shrink-0">
                                    {file ? (file.size / 1024).toFixed(1) : 0} KB
                                </span>
                            </div>

                            {error && (
                                <motion.div
                                    initial={{ opacity: 0, y: -8 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    className="mb-4 p-3 rounded-xl bg-red-500/10 border border-red-500/20 text-red-400 text-sm"
                                >
                                    {error}
                                </motion.div>
                            )}

                            <motion.button
                                onClick={handlePredict}
                                disabled={loading}
                                whileHover={{ scale: loading ? 1 : 1.02 }}
                                whileTap={{ scale: loading ? 1 : 0.98 }}
                                className="w-full bg-gradient-to-r from-cyan-500 to-teal-500 text-navy font-bold py-3.5 rounded-2xl flex items-center justify-center gap-2 hover:shadow-lg hover:shadow-cyan-500/30 transition-all disabled:opacity-60 disabled:cursor-not-allowed"
                            >
                                <Microscope className="w-5 h-5" />
                                {loading ? "Analyzing…" : "Run AI Analysis"}
                            </motion.button>
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
}
