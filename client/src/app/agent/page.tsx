"use client";

import { AgentDisplay } from "@/components/agent/AgentDisplay";

export default function Agent() {
  return (
    <div className="h-screen flex flex-col">
      {/* Page container */}
      <div className="flex-1 flex items-center justify-center p-4">
        {/* Agent Display Card */}
        <div className="w-full max-w-5xl h-[calc(100vh-8rem)] rounded-lg border border-border bg-background/95 backdrop-blur shadow-2xl overflow-hidden">
          <AgentDisplay />
        </div>
      </div>
    </div>
  );
}
