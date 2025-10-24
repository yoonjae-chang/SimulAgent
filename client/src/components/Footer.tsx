import Link from 'next/link';
import Image from 'next/image';

const Footer = () => {
  return (
    <footer className="bg-card border-t border-border mt-auto">
      <div className="container mx-auto px-6 py-12 max-w-6xl">
        <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-8">
          {/* Brand Section */}
          <div className="flex-1 max-w-md">
            <div className="flex items-center space-x-3 mb-4">

              <span className="text-foreground font-bold text-2xl">SimulAgent</span>
              <Image src="/logo/logo.png" alt="SimulAgent" width={35} height={35} />
            </div>
            <p className="text-sub-foreground text-base leading-relaxed">
              An End to End Solution For Data Science Research.
            </p>
          </div>

          {/* Navigation Links */}
          <div className="flex flex-col items-start md:items-end">
            <h3 className="text-foreground font-semibold mb-4 text-lg">Quick Links</h3>
            <nav className="flex flex-col space-y-3">
              <Link 
                href="/" 
                className="text-sub-foreground hover:text-accent transition-all duration-200 text-base font-medium hover:translate-x-1"
              >
                Home
              </Link>
              <Link 
                href="/ideating" 
                className="text-sub-foreground hover:text-accent transition-all duration-200 text-base font-medium hover:translate-x-1"
              >
                Start Now
              </Link>
            </nav>
          </div>
        </div>

        {/* Bottom Section */}
        <div className="border-t border-border/50 mt-12 pt-8 flex flex-col sm:flex-row justify-between items-center gap-4">
          <div className="text-sub-foreground text-sm">
            © {new Date().getFullYear()} SimulAgent. All rights reserved.
          </div>
          <div className="text-sub-foreground text-sm">
            Built by Data Science Researchers
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
