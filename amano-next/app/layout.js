import { Ubuntu_Mono, JetBrains_Mono, Space_Mono, Fira_Code, Source_Code_Pro, Inter, Outfit, Space_Grotesk, Syncopate } from "next/font/google";
import "./globals.css";

// Initialize all fonts
const ubuntuMono = Ubuntu_Mono({
  weight: ['400', '700'],
  subsets: ['latin'],
  variable: '--font-ubuntu-mono',
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ['latin'],
  variable: '--font-jetbrains-mono',
});

const spaceMono = Space_Mono({
  weight: ['400', '700'],
  subsets: ['latin'],
  variable: '--font-space-mono',
});

const firaCode = Fira_Code({
  subsets: ['latin'],
  variable: '--font-fira-code',
});

const sourceCodePro = Source_Code_Pro({
  subsets: ['latin'],
  variable: '--font-source-code-pro',
});

const inter = Inter({
  subsets: ['latin'],
  variable: '--font-inter',
});

const outfit = Outfit({
  subsets: ['latin'],
  variable: '--font-outfit',
});

const spaceGrotesk = Space_Grotesk({
  subsets: ['latin'],
  variable: '--font-space-grotesk',
});

const syncopate = Syncopate({
  weight: ['400', '700'],
  variable: '--font-syncopate',
});

export const metadata = {
  title: "AMANO - Your Music Companion",
  description: "Personal music recommendation system",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body className={`
        ${ubuntuMono.variable} 
        ${jetbrainsMono.variable} 
        ${spaceMono.variable}
        ${firaCode.variable}
        ${sourceCodePro.variable}
        ${inter.variable}
        ${outfit.variable}
        ${spaceGrotesk.variable}
        ${syncopate.variable}
        font-mono antialiased
      `}>
        {children}
      </body>
    </html>
  );
}
