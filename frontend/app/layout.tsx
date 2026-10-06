import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "BrainTumorAI - Deep Learning MRI Segmentation & Classification",
  description: "Academic prototype for deep learning brain tumor classification, segmentation, and Grad-CAM explainability.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap"
          rel="stylesheet"
        />
      </head>
      <body className="font-sans antialiased text-slate-800 bg-slate-50 min-h-screen">
        {children}
      </body>
    </html>
  );
}
