import "~/styles/globals.css";

import { type Metadata } from "next";
import { Geist } from "next/font/google";

import { TooltipProvider } from "~/components/ui/tooltip";

export const metadata: Metadata = {
  title: "🦌 Deer",
  description:
    "Deep Exploration and Efficient Research, an AI tool that combines language models with specialized tools for research tasks.",
  icons: [{ rel: "icon", url: "/favicon.ico" }],
};

const geist = Geist({
  subsets: ["latin"],
  variable: "--font-geist-sans",
});

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={`${geist.variable}`}>
      <body className="h-screen w-screen overflow-hidden overscroll-none bg-[#f7f5f3]">
        <TooltipProvider>{children}</TooltipProvider>
      </body>
    </html>
  );
}
