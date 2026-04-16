import type { Metadata } from "next";
import { Geist, Geist_Mono, PT_Serif } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
  display: "swap",
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
  display: "swap",
});

const ptSerif = PT_Serif({
  variable: "--font-pt-serif",
  subsets: ["latin"],
  weight: ["400", "700"],
  display: "swap",
});

export const metadata: Metadata = {
  metadataBase: new URL("https://isaac.earth"),
  title: "Fields of The World",
  description:
    "Interactive demo for Fields of The World (FTW) — run field boundary delineation models directly in the browser on satellite imagery.",
};

const themeScript = `
  (function() {
    var theme = localStorage.getItem('theme');
    if (theme === 'dark' || (!theme && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
      document.documentElement.classList.add('dark');
    }
  })();
`;

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
        <script dangerouslySetInnerHTML={{ __html: themeScript }} />
        <link rel="preconnect" href="https://cdn.jsdelivr.net" />
        <link rel="dns-prefetch" href="https://cdn.jsdelivr.net" />
        <link rel="preconnect" href="https://raw.githubusercontent.com" />
        <link rel="dns-prefetch" href="https://raw.githubusercontent.com" />
      </head>
      <body
        className={`${geistSans.variable} ${geistMono.variable} ${ptSerif.variable} antialiased`}
      >
        <main>{children}</main>
      </body>
    </html>
  );
}
