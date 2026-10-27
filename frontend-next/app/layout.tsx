import type { Metadata } from "next";
import { manrope } from "@/src/landing/fonts";
import { INTRO_SKIP_SCRIPT } from "@/src/landing/introScript";
import "./globals.css";

export const metadata: Metadata = {
  title: "compresskit — measured CNN compression",
  description:
    "Structured pruning, distillation and INT8 quantization for image-classification CNNs, with every number measured on the exported ONNX file.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`${manrope.variable} h-full antialiased`} suppressHydrationWarning>
      <head>
        {/* Static constant, no user input. May set data-intro on <html> before hydration. */}
        <script dangerouslySetInnerHTML={{ __html: INTRO_SKIP_SCRIPT }} />
      </head>
      <body className="min-h-full flex flex-col bg-ink-50 text-ink-900 font-sans">{children}</body>
    </html>
  );
}
