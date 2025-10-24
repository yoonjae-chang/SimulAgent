import type { Metadata } from "next";
import { Geist } from "next/font/google";
import "./globals.css";
import Navbar from "@/components/Navbar";
import ConditionalFooter from "@/components/ConditionalFooter";


export const metadata: Metadata = {
  title: "SimulAgent",
  icons: {
    icon: "/favicon2.svg",
  },
  description: "The fastest way to prototype and generate ideas.",
};

const geistSans = Geist({
  variable: "--font-geist-sans",

  subsets: ["latin"],
});

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={`${geistSans.className} antialiased bg-background text-foreground min-h-screen flex flex-col`}>
          <Navbar />
          <main className="flex-1 pt-18">
            {children}
          </main>
          <ConditionalFooter />
      </body>
    </html>
  );
}
