"use client";

import { useState, useRef, useEffect } from "react";
import { motion } from "framer-motion";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";

const API_BASE = process.env.NEXT_PUBLIC_PREDICT_API || "http://localhost:5001";

type PredictionResult = {
  detected_class: string;
  confidence: number;
  is_cancer: boolean;
  prob_value: number;
  prob_cancer_pct: number;
  prob_normal_pct: number;
};

const fadeUp = {
  initial: { opacity: 0, y: 20 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: 0.4 },
};

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isSampleModalOpen, setIsSampleModalOpen] = useState(false);
  const [sampleImages, setSampleImages] = useState<string[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    setFile(f || null);
    setResult(null);
    setError(null);
    if (f) {
      const url = URL.createObjectURL(f);
      setPreview(url);
      return () => URL.revokeObjectURL(url);
    } else {
      setPreview(null);
    }
  };

  const handlePredict = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const formData = new FormData();
      formData.append("file", file);
      const res = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Prediction failed");
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Prediction failed");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    const loadSamples = async () => {
      try {
        const res = await fetch("/api/sample-images");
        if (!res.ok) return;
        const data: { images?: string[] } = await res.json();
        if (Array.isArray(data.images)) {
          setSampleImages(data.images);
        }
      } catch {
        // ignore, modal will just show nothing
      }
    };
    loadSamples();
  }, []);

  const handleSampleSelect = async (src: string) => {
    try {
      setIsSampleModalOpen(false);
      setError(null);
      setResult(null);

      const res = await fetch(src);
      const blob = await res.blob();
      const filename = src.split("/").pop() || "sample-image.jpg";
      const sampleFile = new File([blob], filename, { type: blob.type || "image/jpeg" });
      const objectUrl = URL.createObjectURL(sampleFile);

      setFile(sampleFile);
      setPreview(objectUrl);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Prediction failed");
    }
  };

  const riskLevel =
    result &&
    (result.confidence < 40
      ? "Low Risk"
      : result.confidence <= 70
        ? "Moderate Risk"
        : "High Risk");

  const stageLabel =
    result?.is_cancer &&
    (result.confidence >= 40 && result.confidence < 60
      ? "Early Suspicion (40–60%)"
      : result.confidence >= 60 && result.confidence <= 80
        ? "Moderate Severity (60–80%)"
        : result.confidence > 80
          ? "High Severity (>80%)"
          : "Below 40% confidence (low confidence indication)");

  const confidenceLabel =
    result &&
    (result.confidence < 40
      ? "Low Confidence (below 40%)"
      : result.confidence <= 70
        ? "Moderate Confidence (40–70%)"
        : "High Confidence (above 70%)");

  const distData = result
    ? [
        { name: "Cancer", value: result.prob_cancer_pct, fill: result.is_cancer ? "#ef4444" : "#475569" },
        { name: "Normal", value: result.prob_normal_pct, fill: !result.is_cancer ? "#22c55e" : "#475569" },
      ]
    : [];

  const perfData = [
    { name: "Accuracy", value: 95 },
    { name: "Precision", value: 94 },
    { name: "Recall", value: 96 },
    { name: "F1-score", value: 95 },
  ];

  return (
    <div className="gradient-bg grid-pattern min-h-screen relative overflow-hidden w-full">
      <div className="page-blur-layer" aria-hidden />
      <div className="relative w-full max-w-[1400px] mx-auto px-4 py-10 pb-20 text-left sm:px-6 md:px-8 lg:px-12">
        {/* Hero */}
        <motion.header
          className="relative mb-16 pt-4"
          initial={{ opacity: 0, y: 24 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <div className="hero-orb -top-32 -right-32" aria-hidden />
          <h1 className="text-2xl font-bold tracking-tight sm:text-3xl md:text-4xl leading-tight">
            <span className="gradient-text">
              Computer-Aided Diagnosis System for Invasive Oral Cancer Detection
              using Deep Learning Techniques
            </span>
          </h1>
          <p className="mt-4 text-slate-400 leading-relaxed text-sm sm:text-base">
            This project presents a computer-aided diagnosis system for automated
            detection of invasive oral cancer from oral cavity images using deep
            learning. A transfer learning-based Convolutional Neural Network using
            MobileNetV2 is employed to classify images into Cancer and Normal
            categories. The system is designed for academic and research
            purposes to assist in early screening and decision support.
          </p>
        </motion.header>

        <div className="divider-gradient" aria-hidden />

        {/* Model & Methodology - card with animated gradient border */}
        <motion.section
          className="gradient-border-wrap mb-2"
          initial={fadeUp.initial}
          animate={fadeUp.animate}
          transition={{ ...fadeUp.transition, delay: 0.1 }}
          whileHover={{ scale: 1.002 }}
        >
          <div className="inner p-4 sm:p-6 md:p-8">
            <h2 className="text-xl font-bold text-white mb-5">
              Model Architecture and Methodology
            </h2>
            <ul className="list-disc list-inside space-y-2.5 text-slate-300 text-sm leading-relaxed">
              <li><strong className="text-slate-200">Base Model:</strong> MobileNetV2 (Transfer Learning)</li>
              <li><strong className="text-slate-200">Pre-trained on:</strong> ImageNet</li>
              <li><strong className="text-slate-200">Input image size:</strong> 224 x 224 x 3 (RGB)</li>
              <li><strong className="text-slate-200">Architecture:</strong> Global Average Pooling followed by Dense layers</li>
              <li><strong className="text-slate-200">Task:</strong> Binary classification (Cancer vs Normal)</li>
              <li><strong className="text-slate-200">Training:</strong> Optimized using Adam optimizer and Binary Cross-Entropy loss</li>
              <li><strong className="text-slate-200">Post-processing:</strong> Threshold tuning applied to reduce false negatives</li>
              <li><strong className="text-slate-200">Deployment:</strong> Final model deployed without retraining</li>
            </ul>
          </div>
        </motion.section>

        <div className="divider-gradient" aria-hidden />

        {/* Dataset & Performance */}
        <motion.section
          className="glass-card p-4 sm:p-6 md:p-8 mb-2"
          initial={fadeUp.initial}
          animate={fadeUp.animate}
          transition={{ ...fadeUp.transition, delay: 0.15 }}
          whileHover={{ scale: 1.002 }}
        >
          <h2 className="text-xl font-bold text-white mb-5">
            Dataset and Model Performance
          </h2>
          <p className="text-slate-300 text-sm font-medium mb-2">Dataset Sources:</p>
          <ul className="list-disc list-inside space-y-1 text-slate-400 text-sm mb-4">
            <li>
              <a href="https://www.kaggle.com/datasets/zaidpy/oral-cancer-dataset" target="_blank" rel="noopener noreferrer" className="text-cyan-400 hover:text-cyan-300 transition-colors">Oral Cancer Dataset (Kaggle)</a>
            </li>
            <li>
              <a href="https://www.kaggle.com/datasets/muhammadatef/oral-cancer-images-for-classification" target="_blank" rel="noopener noreferrer" className="text-cyan-400 hover:text-cyan-300 transition-colors">Oral Cancer Images for Classification (Kaggle)</a>
            </li>
          </ul>
          <p className="text-slate-300 text-sm font-medium mb-2">Dataset Summary:</p>
          <ul className="list-disc list-inside space-y-1 text-slate-400 text-sm mb-4">
            <li>Multiple public datasets were merged to improve diversity</li>
            <li>Total images used after merging datasets: <strong className="text-slate-300">1988</strong></li>
            <li>Cancer images: <strong className="text-slate-300">1185</strong> | Normal images: <strong className="text-slate-300">803</strong></li>
            <li>Dataset split into training, validation, and test sets</li>
          </ul>
          <p className="text-slate-300 text-sm font-medium mb-2">Performance Metrics (Test Set):</p>
          <ul className="list-disc list-inside space-y-1 text-slate-400 text-sm mb-2">
            <li><strong className="text-slate-300">Accuracy:</strong> 95%</li>
            <li><strong className="text-slate-300">Precision (Cancer):</strong> 0.94</li>
            <li><strong className="text-slate-300">Recall (Cancer):</strong> 0.96</li>
            <li><strong className="text-slate-300">F1-score (Cancer):</strong> 0.95</li>
          </ul>
          <p className="text-slate-300 text-sm font-medium mb-2">Confusion Matrix Summary:</p>
          <p className="text-slate-400 text-sm mb-2">True Positives: 171 | False Negatives: 7 — False Positives: 10 | True Negatives: 141</p>
          <p className="text-slate-500 text-sm italic">Recall was prioritized to reduce false negatives due to the critical medical importance of missed cancer cases.</p>
        </motion.section>

        <div className="divider-gradient" aria-hidden />

        {/* Upload Guidelines */}
        <motion.section
          className="glass-card p-4 sm:p-6 md:p-8 mb-2"
          initial={fadeUp.initial}
          animate={fadeUp.animate}
          transition={{ ...fadeUp.transition, delay: 0.2 }}
          whileHover={{ scale: 1.002 }}
        >
          <h2 className="text-xl font-bold text-white mb-5">
            Image Upload Guidelines
          </h2>
          <ul className="list-disc list-inside space-y-2.5 text-slate-400 text-sm leading-relaxed">
            <li>Upload clear oral cavity images only</li>
            <li>Supported formats: JPG, JPEG, PNG</li>
            <li>Image must be RGB (3-channel)</li>
            <li>Recommended resolution: minimum 224 x 224 pixels</li>
            <li>Avoid blurred or low-light images</li>
            <li>One image at a time</li>
          </ul>
        </motion.section>

        <div className="divider-gradient" aria-hidden />

        {/* Upload + Predict + Sample Images */}
        <motion.section
          className="glass-card p-4 sm:p-6 md:p-8 mb-2"
          initial={fadeUp.initial}
          animate={fadeUp.animate}
          transition={{ ...fadeUp.transition, delay: 0.25 }}
          whileHover={{ scale: 1.002 }}
        >
          <div className="flex flex-col gap-8 lg:flex-row lg:items-start">
            <div className="flex-1 min-w-0">
              <h2 className="text-xl font-bold text-white mb-2">Upload Image</h2>
              <p className="mt-2 mb-3 text-sm text-slate-400 leading-relaxed">
                Choose image from your device
              </p>
              <input
                ref={fileInputRef}
                type="file"
                accept="image/jpeg,image/jpg,image/png"
                onChange={handleFileChange}
                className="block w-full text-sm text-slate-200 file:mr-4 file:rounded-lg file:border-0 file:bg-cyan-600 file:px-4 file:py-2.5 file:text-white file:font-medium file:transition file:duration-200 hover:file:bg-cyan-500 hover:file:shadow-lg hover:file:shadow-cyan-500/20"
              />
              {!file && (
                <p className="mt-2 mb-3 text-sm text-slate-400 leading-relaxed">
                  Upload an image above to see the preview and the <strong>Predict</strong> button below.
                </p>
              )}
            </div>
            <div className="w-full lg:w-80 xl:w-96">
              <h3 className="text-xl font-bold text-white mb-2">Sample Images</h3>
              <p className="text-sm text-slate-400 mb-3 leading-relaxed">
                Try sample image from here
              </p>
              <motion.button
                type="button"
                onClick={() => setIsSampleModalOpen(true)}
                className="inline-flex items-center justify-center rounded-lg border border-transparent bg-cyan-600 px-4 py-2.5 text-sm font-medium text-white shadow-md shadow-cyan-500/20 transition-colors duration-200 hover:bg-cyan-500"
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                Select
              </motion.button>
            </div>
          </div>

          {preview && file && (
            <>
              <h3 className="text-base font-bold text-white mt-8 mb-3">Uploaded Image</h3>
              <div className="flex justify-start">
                <img
                  src={preview}
                  alt="Uploaded"
                  className="max-w-[360px] w-full rounded-lg border border-slate-600/60 object-contain shadow-lg"
                />
              </div>

              <h3 className="text-base font-bold text-white mt-8 mb-2">Prediction</h3>
              <p className="text-sm text-slate-500 mb-4 leading-relaxed">Click the button below to run the model on the uploaded image.</p>
              <motion.button
                type="button"
                onClick={handlePredict}
                disabled={loading}
                className="rounded-lg bg-cyan-600 px-6 py-2.5 font-medium text-white shadow-lg shadow-cyan-500/20 transition-colors duration-200 hover:bg-cyan-500 disabled:opacity-50 disabled:cursor-not-allowed"
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                {loading ? "Analyzing image..." : "Predict"}
              </motion.button>

              {error && (
                <div className="mt-4 rounded-lg bg-red-900/30 border border-red-700/50 p-4 text-sm text-red-200">
                  {error}
                </div>
              )}

              {result && (
                <div className="mt-8 space-y-6">
                  <div>
                    <p className="font-semibold text-white mb-2">Prediction Result</p>
                    <ul className="text-slate-300 text-sm space-y-1 leading-relaxed">
                      <li><strong>Detected Class:</strong> {result.detected_class}</li>
                      <li><strong>Confidence Score:</strong> {result.confidence}%</li>
                    </ul>
                  </div>
                  <div>
                    <p className="font-semibold text-white mb-2">Risk Interpretation</p>
                    <ul className="text-slate-400 text-sm space-y-1 leading-relaxed">
                      <li>Confidence &lt; 40%: Low Risk</li>
                      <li>Confidence 40–70%: Moderate Risk</li>
                      <li>Confidence &gt; 70%: High Risk</li>
                      {result.is_cancer && (
                        <li className="text-slate-200"><strong>Risk Level:</strong> {riskLevel} (based on confidence)</li>
                      )}
                    </ul>
                  </div>

                  <div className="divider-gradient" />

                  <div>
                    <h3 className="text-base font-bold text-white mb-2">Stage Estimation (Confidence-Based, Non-Clinical)</h3>
                    <p className="text-slate-400 text-sm mb-3 leading-relaxed">
                      The model does not predict clinical cancer stages. However, based on prediction confidence,
                      a tentative severity indication is shown for academic interpretation.
                    </p>
                    <div className="rounded-lg bg-slate-800/60 border border-slate-600/50 p-4 text-sm text-slate-300">
                      {result.is_cancer ? (
                        <>Tentative indication: <strong>{stageLabel}</strong></>
                      ) : (
                        "No stage estimation shown for Normal classification."
                      )}
                    </div>
                    <p className="text-slate-500 text-xs mt-2">This is not a clinical stage and must not be used for diagnosis.</p>
                  </div>

                  <div className="divider-gradient" />

                  <h3 className="text-base font-bold text-white">Prediction Confidence Distribution</h3>
                  <div className="h-[220px] w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart layout="vertical" data={distData} margin={{ top: 10, right: 30, left: 80, bottom: 10 }}>
                        <XAxis type="number" domain={[0, 105]} tick={{ fill: "#94a3b8", fontSize: 12 }} />
                        <YAxis type="category" dataKey="name" width={60} tick={{ fill: "#94a3b8", fontSize: 12 }} />
                        <Tooltip contentStyle={{ backgroundColor: "#0f172a", border: "1px solid #334155" }} />
                        <Bar dataKey="value" radius={4}>
                          {distData.map((entry, i) => (
                            <Cell key={i} fill={entry.fill} />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>

                  <h3 className="text-base font-bold text-white">Model Confidence Level</h3>
                  <div className="space-y-2">
                    <div className="h-3 w-full rounded-full bg-slate-700 overflow-hidden">
                      <div
                        className="h-full rounded-full bg-cyan-500 transition-all duration-500"
                        style={{ width: `${Math.min(result.confidence, 100)}%` }}
                      />
                    </div>
                    <p className="text-slate-500 text-sm">Current: <strong className="text-slate-400">{confidenceLabel}</strong> — {result.confidence}%</p>
                  </div>

                  <div className="divider-gradient" />

                  <h3 className="text-base font-bold text-white">Model Performance Summary (Test Dataset)</h3>
                  <p className="text-slate-500 text-sm mb-2">Test Set Performance</p>
                  <div className="h-[280px] w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={perfData} margin={{ top: 10, right: 20, left: 20, bottom: 10 }}>
                        <XAxis dataKey="name" tick={{ fill: "#94a3b8", fontSize: 12 }} />
                        <YAxis domain={[0, 105]} tick={{ fill: "#94a3b8", fontSize: 12 }} />
                        <Tooltip contentStyle={{ backgroundColor: "#0f172a", border: "1px solid #334155" }} />
                        <Bar dataKey="value" fill="#22d3ee" radius={4} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>

                  <h3 className="text-base font-bold text-white">Confusion Matrix (Test Dataset)</h3>
                  <div className="overflow-x-auto">
                    <table className="w-full max-w-xs border-collapse border border-slate-600 rounded-lg overflow-hidden">
                      <thead>
                        <tr>
                          <th className="border border-slate-600 bg-slate-800/80 p-2 text-slate-300 text-sm font-medium"></th>
                          <th className="border border-slate-600 bg-slate-800/80 p-2 text-slate-300 text-sm font-medium">Cancer</th>
                          <th className="border border-slate-600 bg-slate-800/80 p-2 text-slate-300 text-sm font-medium">Normal</th>
                        </tr>
                      </thead>
                      <tbody>
                        <tr>
                          <td className="border border-slate-600 bg-slate-800/60 p-2 text-slate-400 text-sm font-medium">Cancer</td>
                          <td className="border border-slate-600 bg-cyan-900/40 p-3 text-center text-white font-semibold">171</td>
                          <td className="border border-slate-600 bg-slate-800/40 p-3 text-center text-slate-300">7</td>
                        </tr>
                        <tr>
                          <td className="border border-slate-600 bg-slate-800/60 p-2 text-slate-400 text-sm font-medium">Normal</td>
                          <td className="border border-slate-600 bg-slate-800/40 p-3 text-center text-slate-300">10</td>
                          <td className="border border-slate-600 bg-cyan-900/40 p-3 text-center text-white font-semibold">141</td>
                        </tr>
                      </tbody>
                    </table>
                  </div>
                  <p className="text-slate-500 text-xs mt-2 leading-relaxed">
                    Top-left: Cancer predicted as Cancer (TP=171). Top-right: Cancer predicted as Normal (FN=7). Bottom-left: Normal predicted as Cancer (FP=10). Bottom-right: Normal predicted as Normal (TN=141).
                  </p>

                  <h3 className="text-base font-bold text-white">How to Interpret These Results</h3>
                  <ul className="list-disc list-inside space-y-1 text-slate-400 text-sm leading-relaxed">
                    <li>Higher confidence indicates stronger model certainty in the prediction.</li>
                    <li>The graphs above provide transparency into how the model weighs Cancer vs Normal.</li>
                    <li>False negatives are minimized in training to prioritize detection of cancer cases.</li>
                    <li>Visual outputs improve trust and explainability of the system.</li>
                  </ul>
                </div>
              )}
            </>
          )}
        </motion.section>

        <div className="divider-gradient" aria-hidden />

        {/* Disclaimer */}
        <motion.section
          className="rounded-xl border border-amber-700/40 bg-amber-950/20 backdrop-blur p-4 sm:p-6 text-amber-100/95 text-sm leading-relaxed"
          initial={fadeUp.initial}
          animate={fadeUp.animate}
          transition={{ ...fadeUp.transition, delay: 0.3 }}
        >
          <strong>Disclaimer:</strong> This system is developed strictly for academic and research purposes.
          The predictions generated by this application do not constitute medical advice and must not be used
          as a substitute for professional clinical diagnosis.
        </motion.section>

        <div className="divider-gradient" aria-hidden />

        {/* Footer */}
        <motion.footer
          className="border-t border-slate-700/50 pt-8 text-slate-500 text-sm space-y-1 text-left"
          initial={fadeUp.initial}
          animate={fadeUp.animate}
          transition={{ ...fadeUp.transition, delay: 0.35 }}
        >
          <p className="font-semibold text-slate-400">Project Author</p>
          <p><strong className="text-slate-300">Ruturaj Mane</strong></p>
          <p>
            <a href="mailto:Ruturajmane522@gmail.com" className="text-cyan-400 hover:text-cyan-300 transition-colors">Ruturajmane522@gmail.com</a>
          </p>
          <p>
            <a href="https://www.linkedin.com/in/ruturaj-mane-13a8a3264/" target="_blank" rel="noopener noreferrer" className="text-cyan-400 hover:text-cyan-300 transition-colors">LinkedIn</a>
            {" · "}
            <a href="https://github.com/Ruturajmane1003" target="_blank" rel="noopener noreferrer" className="text-cyan-400 hover:text-cyan-300 transition-colors">GitHub</a>
          </p>
        </motion.footer>

        {isSampleModalOpen && (
          <motion.div
            className="fixed inset-0 z-40 flex items-start justify-center bg-slate-950/60 backdrop-blur-xl px-4 py-8 sm:py-12"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
          >
            <motion.div
              className="w-full max-w-3xl rounded-2xl bg-slate-900/85 border border-slate-700/80 shadow-2xl shadow-slate-950/80 p-5 sm:p-7 max-h-[80vh] overflow-y-auto"
              initial={{ scale: 0.96, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ duration: 0.2 }}
            >
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-white">Sample Images</h3>
                <button
                  type="button"
                  onClick={() => setIsSampleModalOpen(false)}
                  className="rounded-full border border-slate-600/70 bg-slate-800/70 px-2 py-1 text-xs text-slate-200 hover:bg-slate-700"
                >
                  Close
                </button>
              </div>
              <p className="text-sm text-slate-400 mb-4 leading-relaxed">
                Select a sample image from the public folder to run through the same prediction workflow.
              </p>
              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-4">
                {sampleImages.map((src) => (
                  <motion.button
                    key={src}
                    type="button"
                    onClick={() => handleSampleSelect(src)}
                    className="relative overflow-hidden rounded-lg border border-slate-700/80 bg-slate-900/70 hover:border-cyan-400/80 focus:outline-none focus:ring-2 focus:ring-cyan-500 focus:ring-offset-2 focus:ring-offset-slate-900"
                    whileHover={{ scale: 1.03 }}
                    whileTap={{ scale: 0.97 }}
                  >
                    <img
                      src={src}
                      alt="Sample"
                      className="h-24 w-full object-cover"
                    />
                  </motion.button>
                ))}
              </div>
            </motion.div>
          </motion.div>
        )}
      </div>
    </div>
  );
}
