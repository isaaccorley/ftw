import type { Metadata } from "next";
import { DM_Sans, JetBrains_Mono, Syne } from "next/font/google";
import "./globals.css";

const dmSans = DM_Sans({
  variable: "--font-dm-sans",
  subsets: ["latin"],
  display: "swap",
});

const jetbrainsMono = JetBrains_Mono({
  variable: "--font-jetbrains-mono",
  subsets: ["latin"],
  display: "swap",
});

const syne = Syne({
  variable: "--font-syne",
  subsets: ["latin"],
  weight: ["600", "700", "800"],
  display: "swap",
});

export const metadata: Metadata = {
  metadataBase: new URL("https://isaac.earth"),
  title: "Fields of The World",
  description:
    "Interactive demo for Fields of The World (FTW) — run field boundary delineation models directly in the browser on satellite imagery.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <meta
          httpEquiv="Content-Security-Policy"
          content={[
            "default-src 'self'",
            "script-src 'self' 'unsafe-inline' 'unsafe-eval' 'wasm-unsafe-eval'",
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com",
            "font-src 'self' https://fonts.gstatic.com",
            "img-src 'self' data: blob: https://cdn.jsdelivr.net https://raw.githubusercontent.com https://hf.co",
            "connect-src 'self' https://cdn.jsdelivr.net https://raw.githubusercontent.com https://huggingface.co https://hf.co https://*.hf.co https://data.source.coop",
            "media-src 'self' blob:",
            "worker-src 'self' blob:",
          ].join("; ")}
        />
        <link rel="preconnect" href="https://cdn.jsdelivr.net" />
        <link rel="dns-prefetch" href="https://cdn.jsdelivr.net" />
        <link rel="preconnect" href="https://raw.githubusercontent.com" />
        <link rel="dns-prefetch" href="https://raw.githubusercontent.com" />
      </head>
      <body className={`${dmSans.variable} ${jetbrainsMono.variable} ${syne.variable} antialiased`}>
        <main>{children}</main>
      </body>
    </html>
  );
}
