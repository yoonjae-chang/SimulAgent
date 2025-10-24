import React from 'react';
import { cn } from '@/lib/utils';

interface HyperTextProps {
  children: React.ReactNode;
  className?: string;
}

export function HyperText({ children, className }: HyperTextProps) {
  return (
    <span className={cn(
      'bg-gradient-to-r from-purple-600 via-pink-600 to-blue-600 bg-clip-text text-transparent',
      className
    )}>
      {children}
    </span>
  );
}
