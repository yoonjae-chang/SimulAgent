'use client';

import { usePathname } from 'next/navigation';
import Footer from './Footer';

export default function ConditionalFooter() {
  const pathname = usePathname();
  const isIdeatingPage = pathname === '/ideating';

  // Don't render footer at all on ideating page
  if (isIdeatingPage) {
    return null;
  }

  return <Footer />;
}
