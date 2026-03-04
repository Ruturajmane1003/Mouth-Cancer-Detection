import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  display: "swap",
  variable: "--font-inter",
});

export const metadata: Metadata = {
  title: "Oral Cancer Detection | Computer-Aided Diagnosis",
  description:
    "Computer-aided diagnosis system for invasive oral cancer detection using deep learning.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`${inter.variable} antialiased`}>
      <body className="min-h-screen bg-[#050a12] text-slate-200 font-sans">
        {children}
      </body>
    </html>
  );
}
