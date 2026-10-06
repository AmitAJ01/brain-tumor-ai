"use client";

import React, { useState, useEffect, useRef } from "react";
import {
  Upload,
  Brain,
  Layers,
  Activity,
  CheckCircle2,
  AlertTriangle,
  RefreshCw,
  Info,
  ChevronRight,
  Sliders,
  FileText,
  Clock,
  Sparkles,
  ShieldAlert,
} from "lucide-react";

interface ProbabilityMap {
  [key: string]: number;
}

interface SegmentationData {
  available: boolean;
  tumor_detected: boolean;
  tumor_area_pixels: number;
  total_brain_pixels: number;
  tumor_area_percentage: number;
  mask_url: string | null;
  overlay_url: string | null;
  disclaimer: string;
}

interface AnalysisResponse {
  prediction: string;
  confidence: number;
  confidence_percentage: number;
  probabilities: ProbabilityMap;
  processing_time_ms: number;
  strategy_used: string;
  original_image_url: string;
  gradcam_url: string | null;
  segmentation: SegmentationData;
  model_info: any;
  disclaimer: string;
}

const PRESET_SAMPLES = [
  { id: "1846", name: "Glioma Sample (ID 1846)", class: "glioma", file: "/samples/glioma_sample_1846.png" },
  { id: "106", name: "Meningioma Sample (ID 106)", class: "meningioma", file: "/samples/meningioma_sample_106.png" },
  { id: "1004", name: "Pituitary Sample (ID 1004)", class: "pituitary", file: "/samples/pituitary_sample_1004.png" },
];

export default function Home() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [strategy, setStrategy] = useState<string>("standard");
  const [loading, setLoading] = useState<boolean>(false);
  const [result, setResult] = useState<AnalysisResponse | null>(null);
  const [backendStatus, setBackendStatus] = useState<"checking" | "online" | "offline">("checking");
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<"dashboard" | "papers">("dashboard");
  const fileInputRef = useRef<HTMLInputElement>(null);

  const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

  // Check backend health on mount
  useEffect(() => {
    checkHealth();
  }, []);

  const checkHealth = async () => {
    try {
      setBackendStatus("checking");
      const res = await fetch(`${API_URL}/health`);
      if (res.ok) {
        setBackendStatus("online");
      } else {
        setBackendStatus("offline");
      }
    } catch {
      setBackendStatus("offline");
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setResult(null);
      setError(null);
    }
  };

  const handlePresetSelect = async (preset: typeof PRESET_SAMPLES[0]) => {
    try {
      setLoading(true);
      setError(null);
      setResult(null);
      const res = await fetch(preset.file);
      const blob = await res.blob();
      const file = new File([blob], `${preset.class}_${preset.id}.png`, { type: "image/png" });
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setLoading(false);
    } catch (err: any) {
      setError(`Failed to load preset sample: ${err.message}`);
      setLoading(false);
    }
  };

  const handleAnalyze = async () => {
    if (!selectedFile) return;

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append("file", selectedFile);
    formData.append("strategy", strategy);

    try {
      const res = await fetch(`${API_URL}/analyze`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}));
        throw new Error(errorData.detail || `Server responded with status ${res.status}`);
      }

      const data: AnalysisResponse = await res.json();
      setResult(data);
    } catch (err: any) {
      setError(err.message || "Failed to analyze image. Please ensure the backend is running.");
    } finally {
      setLoading(false);
    }
  };

  const getClassBadgeColor = (clsName: string) => {
    switch (clsName?.toLowerCase()) {
      case "glioma":
        return "bg-rose-100 text-rose-800 border-rose-300";
      case "meningioma":
        return "bg-amber-100 text-amber-800 border-amber-300";
      case "pituitary":
        return "bg-indigo-100 text-indigo-800 border-indigo-300";
      default:
        return "bg-slate-100 text-slate-800 border-slate-300";
    }
  };

  return (
    <div className="min-h-screen flex flex-col">
      {/* Main Header */}
      <header className="bg-white border-b border-slate-200 sticky top-0 z-40 shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex flex-wrap justify-between items-center gap-4">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-blue-600 to-indigo-600 flex items-center justify-center text-white shadow-md">
              <Brain className="w-6 h-6" />
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <h1 className="text-xl font-bold text-slate-900 tracking-tight">BrainTumorAI</h1>
                <span className="text-xs font-semibold px-2 py-0.5 rounded-full bg-blue-50 text-blue-700 border border-blue-200">
                  v1.0 Academic Prototype
                </span>
              </div>
              <p className="text-xs text-slate-500">
                Deep Learning Based Brain Tumor Segmentation & Classification from MRI Images
              </p>
            </div>
          </div>

          {/* Right Header: API Status & Navigation Tabs */}
          <div className="flex items-center space-x-3">
            <div className="flex items-center space-x-2 bg-slate-100 py-1.5 px-3 rounded-lg border border-slate-200 text-xs text-slate-600">
              <span
                className={`w-2 h-2 rounded-full ${
                  backendStatus === "online"
                    ? "bg-emerald-500 animate-pulse"
                    : backendStatus === "checking"
                    ? "bg-amber-400"
                    : "bg-rose-500"
                }`}
              />
              <span className="font-medium capitalize">API: {backendStatus}</span>
              <button
                onClick={checkHealth}
                className="text-slate-400 hover:text-slate-700 transition-colors ml-1"
                title="Refresh API Status"
              >
                <RefreshCw className="w-3 h-3" />
              </button>
            </div>

            {/* Navigation Tabs */}
            <div className="flex items-center space-x-1 bg-slate-100 p-1 rounded-lg border border-slate-200 text-sm">
              <button
                onClick={() => setActiveTab("dashboard")}
                className={`px-3 py-1.5 rounded-md font-medium transition-all ${
                  activeTab === "dashboard" ? "bg-white text-slate-900 shadow-sm" : "text-slate-600 hover:text-slate-900"
                }`}
              >
                Analysis Dashboard
              </button>
              <button
                onClick={() => setActiveTab("papers")}
                className={`px-3 py-1.5 rounded-md font-medium transition-all flex items-center space-x-1 ${
                  activeTab === "papers" ? "bg-white text-slate-900 shadow-sm" : "text-slate-600 hover:text-slate-900"
                }`}
              >
                <FileText className="w-4 h-4 mr-1" />
                IEEE References
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {activeTab === "dashboard" && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
            {/* Left Column: Upload & Configuration Controls */}
            <div className="lg:col-span-4 space-y-6">
              {/* Image Input Card */}
              <div className="bg-white rounded-2xl border border-slate-200 p-6 shadow-sm">
                <h2 className="text-base font-semibold text-slate-900 mb-1 flex items-center">
                  <Upload className="w-4 h-4 mr-2 text-blue-600" />
                  Upload MRI Scan
                </h2>
                <p className="text-xs text-slate-500 mb-4">
                  Select a T1-weighted contrast-enhanced brain MRI scan (.png, .jpg, .mat).
                </p>

                {/* Dropzone */}
                <div
                  onClick={() => fileInputRef.current?.click()}
                  className={`border-2 border-dashed rounded-xl p-6 text-center cursor-pointer transition-all ${
                    previewUrl
                      ? "border-blue-300 bg-blue-50/30"
                      : "border-slate-300 hover:border-blue-400 hover:bg-slate-50"
                  }`}
                >
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/*"
                    onChange={handleFileSelect}
                    className="hidden"
                  />
                  {previewUrl ? (
                    <div className="space-y-3">
                      <div className="w-36 h-36 mx-auto relative rounded-lg overflow-hidden border border-slate-300 shadow-inner bg-black">
                        <img
                          src={previewUrl}
                          alt="MRI Preview"
                          className="w-full h-full object-contain"
                        />
                      </div>
                      <p className="text-xs font-medium text-slate-700 truncate">
                        {selectedFile?.name}
                      </p>
                      <span className="text-[11px] text-blue-600 hover:underline">
                        Click to choose another image
                      </span>
                    </div>
                  ) : (
                    <div className="space-y-2 py-4">
                      <div className="w-12 h-12 mx-auto rounded-full bg-blue-50 flex items-center justify-center text-blue-600">
                        <Upload className="w-6 h-6" />
                      </div>
                      <div>
                        <span className="text-sm font-medium text-blue-600">Click to upload</span>
                        <span className="text-sm text-slate-500"> or drag and drop</span>
                      </div>
                      <p className="text-xs text-slate-400">PNG, JPG, or MAT (Max 10MB)</p>
                    </div>
                  )}
                </div>

                {/* Strategy Pattern Selection */}
                <div className="mt-5 pt-4 border-t border-slate-100">
                  <div className="flex items-center justify-between mb-2">
                    <label className="text-xs font-semibold text-slate-700 flex items-center">
                      <Sliders className="w-3.5 h-3.5 mr-1.5 text-indigo-600" />
                      Preprocessing Strategy
                    </label>
                    <span className="text-[10px] bg-indigo-50 text-indigo-700 font-mono px-1.5 py-0.5 rounded border border-indigo-200">
                      Strategy Pattern
                    </span>
                  </div>
                  <div className="grid grid-cols-2 gap-2 text-xs">
                    <button
                      type="button"
                      onClick={() => setStrategy("standard")}
                      className={`p-2.5 rounded-lg border text-left transition-all ${
                        strategy === "standard"
                          ? "border-blue-600 bg-blue-50/50 font-medium text-blue-900 shadow-sm"
                          : "border-slate-200 hover:bg-slate-50 text-slate-600"
                      }`}
                    >
                      <div className="font-semibold text-slate-900">Standard</div>
                      <div className="text-[10px] text-slate-500 mt-0.5">Min-max + 3-channel</div>
                    </button>
                    <button
                      type="button"
                      onClick={() => setStrategy("fuzzy")}
                      className={`p-2.5 rounded-lg border text-left transition-all ${
                        strategy === "fuzzy"
                          ? "border-indigo-600 bg-indigo-50/50 font-medium text-indigo-900 shadow-sm"
                          : "border-slate-200 hover:bg-slate-50 text-slate-600"
                      }`}
                    >
                      <div className="font-semibold text-slate-900">Fuzzy Logic</div>
                      <div className="text-[10px] text-slate-500 mt-0.5">16-MF CoG (2025 Paper)</div>
                    </button>
                  </div>
                </div>

                {/* Action Button */}
                <div className="mt-6">
                  <button
                    onClick={handleAnalyze}
                    disabled={!selectedFile || loading}
                    className={`w-full py-3 px-4 rounded-xl font-semibold text-sm shadow-md transition-all flex items-center justify-center space-x-2 ${
                      !selectedFile || loading
                        ? "bg-slate-200 text-slate-400 cursor-not-allowed"
                        : "bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white shadow-blue-500/20 active:scale-[0.99]"
                    }`}
                  >
                    {loading ? (
                      <>
                        <RefreshCw className="w-4 h-4 animate-spin mr-2" />
                        <span>Analyzing MRI...</span>
                      </>
                    ) : (
                      <>
                        <Activity className="w-4 h-4 mr-2" />
                        <span>Analyze MRI</span>
                      </>
                    )}
                  </button>
                </div>
              </div>

              {/* Benchmark Dataset Quick Presets */}
              <div className="bg-white rounded-2xl border border-slate-200 p-5 shadow-sm">
                <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-3">
                  Quick Benchmark Samples
                </h3>
                <p className="text-xs text-slate-500 mb-3">
                  Test the pipeline immediately with real patient slices from the test partition:
                </p>
                <div className="space-y-2">
                  {PRESET_SAMPLES.map((sample) => (
                    <button
                      key={sample.id}
                      onClick={() => handlePresetSelect(sample)}
                      className="w-full text-left p-2.5 rounded-lg border border-slate-200 hover:border-blue-300 hover:bg-blue-50/40 transition-all flex items-center justify-between text-xs"
                    >
                      <div>
                        <span className="font-semibold text-slate-800 capitalize">
                          {sample.class}
                        </span>
                        <span className="text-slate-400 ml-1.5 font-mono text-[11px]">
                          Slice #{sample.id}
                        </span>
                      </div>
                      <ChevronRight className="w-3.5 h-3.5 text-slate-400" />
                    </button>
                  ))}
                </div>
              </div>

              {/* Active Pipeline Architecture Summary */}
              <div className="bg-slate-50 rounded-2xl border border-slate-200 p-5 text-xs space-y-3">
                <h3 className="font-semibold text-slate-900 flex items-center">
                  <Info className="w-4 h-4 mr-1.5 text-blue-600" />
                  Configured ML Pipeline
                </h3>
                <div className="space-y-1.5 text-slate-600 font-mono text-[11px]">
                  <div className="flex justify-between">
                    <span>Classifier:</span>
                    <span className="font-semibold text-slate-800">EfficientNet-B0</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Segmentation:</span>
                    <span className="font-semibold text-slate-800">EfficientNet-UNet</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Explainability:</span>
                    <span className="font-semibold text-slate-800">Grad-CAM (Conv2D)</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Active Strategy:</span>
                    <span className="font-semibold text-indigo-600 uppercase">{strategy}</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Right Column: Dynamic Analysis Results & Visualizations */}
            <div className="lg:col-span-8 space-y-6">
              {error && (
                <div className="p-4 rounded-xl bg-rose-50 border border-rose-200 text-rose-800 flex items-start space-x-3 text-sm">
                  <AlertTriangle className="w-5 h-5 text-rose-600 flex-shrink-0 mt-0.5" />
                  <div>
                    <h4 className="font-semibold">Pipeline Error</h4>
                    <p className="text-xs text-rose-700 mt-1">{error}</p>
                  </div>
                </div>
              )}

              {loading && (
                <div className="bg-white rounded-2xl border border-slate-200 p-12 text-center shadow-sm">
                  <div className="relative w-16 h-16 mx-auto mb-4">
                    <div className="w-16 h-16 rounded-full border-4 border-blue-100 border-t-blue-600 animate-spin" />
                    <Brain className="w-7 h-7 text-blue-600 absolute inset-0 m-auto" />
                  </div>
                  <h3 className="text-lg font-semibold text-slate-900">Analyzing Brain MRI...</h3>
                  <p className="text-xs text-slate-500 mt-2 max-w-sm mx-auto">
                    Running selected preprocessing strategy, EfficientNet forward pass, Grad-CAM gradient calculation, and U-Net segmentation.
                  </p>
                </div>
              )}

              {!loading && !result && (
                <div className="bg-white rounded-2xl border border-slate-200 p-12 text-center shadow-sm">
                  <div className="w-14 h-14 mx-auto rounded-2xl bg-slate-100 flex items-center justify-center text-slate-400 mb-4">
                    <Brain className="w-8 h-8" />
                  </div>
                  <h3 className="text-base font-semibold text-slate-700">No MRI Scan Analyzed Yet</h3>
                  <p className="text-xs text-slate-500 mt-1 max-w-md mx-auto">
                    Upload an MRI slice on the left panel or click one of the benchmark samples to execute the full classification, segmentation, and explainability pipeline.
                  </p>
                </div>
              )}

              {!loading && result && (
                <div className="space-y-6">
                  {/* Top Result Banner */}
                  <div className="bg-white rounded-2xl border border-slate-200 p-6 shadow-sm">
                    <div className="flex flex-wrap items-center justify-between gap-4 pb-4 border-b border-slate-100">
                      <div>
                        <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
                          Primary Classification Result
                        </span>
                        <div className="flex items-center space-x-3 mt-1">
                          <span
                            className={`text-2xl font-bold px-3 py-1 rounded-xl border capitalize ${getClassBadgeColor(
                              result.prediction
                            )}`}
                          >
                            {result.prediction}
                          </span>
                          <span className="text-xs text-slate-500">
                            Confidence: <strong className="text-slate-900 text-base">{result.confidence_percentage}%</strong>
                          </span>
                        </div>
                      </div>

                      <div className="flex items-center space-x-4 text-xs text-slate-500">
                        <div className="flex items-center space-x-1.5">
                          <Clock className="w-3.5 h-3.5 text-slate-400" />
                          <span>{result.processing_time_ms} ms</span>
                        </div>
                        <div className="flex items-center space-x-1.5">
                          <Sliders className="w-3.5 h-3.5 text-slate-400" />
                          <span className="capitalize">{result.strategy_used} Strategy</span>
                        </div>
                      </div>
                    </div>

                    {/* Class Probability Distribution */}
                    <div className="mt-4">
                      <h4 className="text-xs font-semibold text-slate-600 mb-2">Class Probability Distribution</h4>
                      <div className="space-y-2">
                        {Object.entries(result.probabilities).map(([cls, prob]) => {
                          const pct = (prob * 100).toFixed(1);
                          const isTop = cls === result.prediction;
                          return (
                            <div key={cls} className="space-y-1">
                              <div className="flex justify-between text-xs">
                                <span className={`capitalize ${isTop ? "font-bold text-slate-900" : "text-slate-600"}`}>
                                  {cls}
                                </span>
                                <span className={`font-mono ${isTop ? "font-bold text-slate-900" : "text-slate-500"}`}>
                                  {pct}%
                                </span>
                              </div>
                              <div className="w-full bg-slate-100 rounded-full h-2 overflow-hidden">
                                <div
                                  className={`h-full rounded-full transition-all duration-500 ${
                                    isTop ? "bg-blue-600" : "bg-slate-300"
                                  }`}
                                  style={{ width: `${pct}%` }}
                                />
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  </div>

                  {/* Visualizations Gallery (Prompt Requirement: Original MRI, Grad-CAM, Predicted Mask, Overlay) */}
                  <div className="bg-white rounded-2xl border border-slate-200 p-6 shadow-sm space-y-4">
                    <div className="flex items-center justify-between">
                      <h3 className="text-base font-semibold text-slate-900 flex items-center">
                        <Sparkles className="w-4 h-4 mr-2 text-indigo-600" />
                        Multi-Modal Diagnostic Visualizations
                      </h3>
                      <span className="text-xs text-slate-400">224 × 224 Standardized Slice</span>
                    </div>

                    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                      {/* 1. Original MRI */}
                      <div className="border border-slate-200 rounded-xl overflow-hidden bg-black text-center">
                        <div className="bg-slate-900 text-white text-[11px] font-medium py-1.5 px-2">
                          1. Original MRI
                        </div>
                        <div className="p-2 flex items-center justify-center min-h-[180px]">
                          {result.original_image_url && (
                            <img
                              src={result.original_image_url}
                              alt="Original MRI"
                              className="max-h-40 w-auto object-contain mx-auto"
                            />
                          )}
                        </div>
                        <div className="bg-slate-950 text-slate-400 text-[10px] py-1 border-t border-slate-800">
                          Preprocessed Slice
                        </div>
                      </div>

                      {/* 2. Grad-CAM Explainability */}
                      <div className="border border-slate-200 rounded-xl overflow-hidden bg-black text-center">
                        <div className="bg-indigo-900 text-white text-[11px] font-medium py-1.5 px-2">
                          2. Grad-CAM Overlay
                        </div>
                        <div className="p-2 flex items-center justify-center min-h-[180px]">
                          {result.gradcam_url ? (
                            <img
                              src={result.gradcam_url}
                              alt="Grad-CAM Overlay"
                              className="max-h-40 w-auto object-contain mx-auto"
                            />
                          ) : (
                            <span className="text-xs text-slate-500">Grad-CAM unavailable</span>
                          )}
                        </div>
                        <div className="bg-indigo-950 text-indigo-300 text-[10px] py-1 border-t border-indigo-900">
                          Attention Heatmap
                        </div>
                      </div>

                      {/* 3. Predicted Mask */}
                      <div className="border border-slate-200 rounded-xl overflow-hidden bg-black text-center">
                        <div className="bg-slate-900 text-white text-[11px] font-medium py-1.5 px-2">
                          3. Predicted Mask
                        </div>
                        <div className="p-2 flex items-center justify-center min-h-[180px]">
                          {result.segmentation.available && result.segmentation.mask_url ? (
                            <img
                              src={result.segmentation.mask_url}
                              alt="Tumor Mask"
                              className="max-h-40 w-auto object-contain mx-auto"
                            />
                          ) : (
                            <div className="text-center px-3">
                              <span className="text-xs text-slate-400 block font-medium">Modular Placeholder</span>
                              <span className="text-[10px] text-slate-500 block mt-1">
                                UNet checkpoint ready to load
                              </span>
                            </div>
                          )}
                        </div>
                        <div className="bg-slate-950 text-slate-400 text-[10px] py-1 border-t border-slate-800">
                          Binary Tumor Map
                        </div>
                      </div>

                      {/* 4. Tumor Overlay */}
                      <div className="border border-slate-200 rounded-xl overflow-hidden bg-black text-center">
                        <div className="bg-red-950 text-red-200 text-[11px] font-medium py-1.5 px-2">
                          4. Tumor Delineation
                        </div>
                        <div className="p-2 flex items-center justify-center min-h-[180px]">
                          {result.segmentation.available && result.segmentation.overlay_url ? (
                            <img
                              src={result.segmentation.overlay_url}
                              alt="Tumor Overlay"
                              className="max-h-40 w-auto object-contain mx-auto"
                            />
                          ) : (
                            <div className="text-center px-3">
                              <span className="text-xs text-slate-400 block font-medium">Modular Placeholder</span>
                              <span className="text-[10px] text-slate-500 block mt-1">
                                Contour overlay on MRI
                              </span>
                            </div>
                          )}
                        </div>
                        <div className="bg-red-950 text-red-300 text-[10px] py-1 border-t border-red-900">
                          Red Boundary Contour
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Quantitative Tumor Area Estimation Card */}
                  {result.segmentation.available && (
                    <div className="bg-gradient-to-r from-slate-900 to-indigo-950 rounded-2xl p-6 text-white shadow-md">
                      <div className="flex items-center justify-between mb-4">
                        <h4 className="text-sm font-semibold tracking-wide uppercase text-indigo-300 flex items-center">
                          <Activity className="w-4 h-4 mr-2 text-indigo-400" />
                          Quantitative Segmentation Metrics
                        </h4>
                        <span className="text-xs font-mono bg-indigo-800/50 px-2 py-0.5 rounded text-indigo-200">
                          EfficientNet-UNet
                        </span>
                      </div>
                      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                        <div className="bg-white/10 rounded-xl p-4 backdrop-blur-sm">
                          <span className="text-xs text-slate-300 block">Tumor Pixel Area</span>
                          <span className="text-2xl font-bold font-mono text-white mt-1 block">
                            {result.segmentation.tumor_area_pixels.toLocaleString()}
                          </span>
                          <span className="text-[11px] text-slate-400">Total detected pixels</span>
                        </div>
                        <div className="bg-white/10 rounded-xl p-4 backdrop-blur-sm">
                          <span className="text-xs text-slate-300 block">Brain Tissue Parenchyma</span>
                          <span className="text-2xl font-bold font-mono text-white mt-1 block">
                            {result.segmentation.total_brain_pixels.toLocaleString()}
                          </span>
                          <span className="text-[11px] text-slate-400">Non-background area</span>
                        </div>
                        <div className="bg-white/10 rounded-xl p-4 backdrop-blur-sm">
                          <span className="text-xs text-slate-300 block">Tumor Area Ratio</span>
                          <span className="text-2xl font-bold font-mono text-emerald-400 mt-1 block">
                            {result.segmentation.tumor_area_percentage}%
                          </span>
                          <span className="text-[11px] text-slate-400">Relative 2D slice burden</span>
                        </div>
                      </div>
                      <p className="text-[11px] text-indigo-200/80 mt-4 italic">
                        {result.segmentation.disclaimer}
                      </p>
                    </div>
                  )}

                  {/* Academic Disclaimer Box */}
                  <div className="p-4 rounded-xl bg-amber-50 border border-amber-200 text-amber-900 text-xs flex items-start space-x-3">
                    <ShieldAlert className="w-5 h-5 text-amber-700 flex-shrink-0 mt-0.5" />
                    <div>
                      <strong className="font-semibold">Academic Prototype Notice:</strong>
                      <p className="mt-0.5 text-amber-800">
                        {result.disclaimer}
                      </p>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}



        {/* IEEE Research References Tab */}
        {activeTab === "papers" && (
          <div className="space-y-6 bg-white rounded-2xl border border-slate-200 p-8 shadow-sm">
            <div>
              <h2 className="text-2xl font-bold text-slate-900">Research References Analyzed</h2>
              <p className="text-sm text-slate-500 mt-1">
                The four IEEE research papers guiding the methodology, architecture, and evaluation of BrainTumorAI:
              </p>
            </div>

            <div className="space-y-4">
              {/* Paper 1 */}
              <div className="border border-slate-200 rounded-xl p-5 hover:border-blue-300 transition-all space-y-2">
                <span className="text-xs font-bold uppercase tracking-wider text-blue-600 bg-blue-50 px-2 py-0.5 rounded">
                  Primary Reference 1 (Segmentation & EfficientNet-UNet)
                </span>
                <h3 className="text-base font-bold text-slate-900">
                  Deep Learning-Based MRI Brain Tumor Segmentation With EfficientNet-Enhanced UNet
                </h3>
                <p className="text-xs text-slate-600">
                  Pradeep Kumar Tiwary, Prashant Johri, Alok Katiyar, Mayur Kumar Chhipa | <em>IEEE Access, Vol. 13, 2025</em>
                </p>
                <p className="text-xs text-slate-500 leading-relaxed">
                  Key Insights: Figshare 3064 T1-weighted CE-MRI dataset (meningioma 708, glioma 1426, pituitary 930) from 233 patients; EfficientNet encoder integrated into U-Net decoder with skip connections; Soft Dice Loss formulation.
                </p>
              </div>

              {/* Paper 2 */}
              <div className="border border-slate-200 rounded-xl p-5 hover:border-blue-300 transition-all space-y-2">
                <span className="text-xs font-bold uppercase tracking-wider text-indigo-600 bg-indigo-50 px-2 py-0.5 rounded">
                  Primary Reference 2 (YOLO & Segmentation Alignment)
                </span>
                <h3 className="text-base font-bold text-slate-900">
                  Automated Brain Tumor Segmentation and Classification in MRI Using YOLO-Based Deep Learning
                </h3>
                <p className="text-xs text-slate-600">
                  Maram Fahaad Almufareh, Muhammad Imran, Abdullah Khan, Mamoona Humayun, Muhammad Asim | <em>IEEE Access, Vol. 12, 2024</em>
                </p>
                <p className="text-xs text-slate-500 leading-relaxed">
                  Key Insights: Evaluates YOLOv5 and YOLOv7 on Figshare dataset; details .mat to PNG mask conversion and polygon approximations. YOLO documented as related work/future work per user project requirements.
                </p>
              </div>

              {/* Paper 3 */}
              <div className="border border-slate-200 rounded-xl p-5 hover:border-blue-300 transition-all space-y-2">
                <span className="text-xs font-bold uppercase tracking-wider text-purple-600 bg-purple-50 px-2 py-0.5 rounded">
                  Primary Reference 3 (Fuzzy Logic Strategy)
                </span>
                <h3 className="text-base font-bold text-slate-900">
                  Efficient Approach for Brain Tumor Detection and Classification Using Fuzzy Thresholding and Deep Learning Algorithms
                </h3>
                <p className="text-xs text-slate-600">
                  Nashaat M. Hussain Hassan, Wadii Boulila | <em>IEEE Access, Vol. 13, 2025</em>
                </p>
                <p className="text-xs text-slate-500 leading-relaxed">
                  Key Insights: 16-membership function Center-of-Gravity (CoG) fuzzy threshold calculation; 5 intensity partitions (Very Low, Low, Medium, High, Very High) to delineate tumor heterogeneity. Directly inspires our <code className="bg-slate-100 px-1 py-0.5 rounded">FuzzyPreprocessingStrategy</code>.
                </p>
              </div>

              {/* Paper 4 */}
              <div className="border border-slate-200 rounded-xl p-5 hover:border-blue-300 transition-all space-y-2">
                <span className="text-xs font-bold uppercase tracking-wider text-emerald-600 bg-emerald-50 px-2 py-0.5 rounded">
                  Primary Reference 4 (Multi-Classification Baseline)
                </span>
                <h3 className="text-base font-bold text-slate-900">
                  Multi-Classification of Brain Tumor Images Using Deep Neural Network
                </h3>
                <p className="text-xs text-slate-600">
                  Hossam H. Sultan, Nancy M. Salem, Walid Al-Atabany | <em>IEEE Access, Vol. 7, 2019</em>
                </p>
                <p className="text-xs text-slate-500 leading-relaxed">
                  Key Insights: Benchmark multi-classification on the Figshare dataset; data augmentation techniques (flipping, rotation, noise); evaluation protocol including confusion matrices, precision, sensitivity, and specificity.
                </p>
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="py-4" />
    </div>
  );
}
