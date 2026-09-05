import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "Flux operator",
  description: "Playground, live engine, and bench for the Flux CPU inference server",
};

const links = [
  { href: "/", label: "Playground" },
  { href: "/live", label: "Live" },
  { href: "/bench", label: "Bench" },
];

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen font-sans antialiased">
        <header className="border-b border-zinc-800 bg-zinc-950/90">
          <div className="mx-auto flex max-w-6xl items-center justify-between px-4 py-3">
            <div>
              <div className="text-sm font-semibold tracking-wide text-zinc-100">FLUX</div>
              <div className="text-[11px] uppercase tracking-[0.2em] text-zinc-500">operator console</div>
            </div>
            <nav className="flex gap-4 text-sm">
              {links.map((link) => (
                <Link key={link.href} href={link.href} className="text-zinc-300 hover:text-white">
                  {link.label}
                </Link>
              ))}
            </nav>
          </div>
        </header>
        <main className="mx-auto max-w-6xl px-4 py-6">{children}</main>
      </body>
    </html>
  );
}
