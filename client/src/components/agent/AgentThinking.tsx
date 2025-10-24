"use client";

import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import { Brain, Sparkles } from "lucide-react";

export function AgentThinking() {
  return (
    <div className="flex gap-3">
      {/* Avatar with animated glow */}
      <div className="flex-shrink-0 w-10 h-10 rounded-full flex items-center justify-center border border-accent/30 bg-accent/10 text-accent relative">
        <div className="absolute inset-0 bg-accent/20 rounded-full animate-ping" />
        <Brain className="w-5 h-5 relative z-10" />
      </div>

      {/* Thinking Content */}
      <div className="flex-1 space-y-1">
        <div className="flex items-center gap-2">
          <Badge
            variant="outline"
            className="text-xs border bg-purple-500/20 text-purple-400 border-purple-500/30"
          >
            <Sparkles className="w-3 h-3 mr-1 animate-pulse" />
            Agent Thinking
          </Badge>
        </div>

        <Card className="p-4 border border-purple-500/20 bg-purple-500/5">
          <div className="flex items-center gap-3">
            {/* Animated thinking dots */}
            <div className="flex gap-1">
              <div
                className={cn(
                  "w-2 h-2 rounded-full bg-accent",
                  "animate-bounce [animation-delay:0ms]"
                )}
                style={{ animationDuration: "1.4s" }}
              />
              <div
                className={cn(
                  "w-2 h-2 rounded-full bg-accent",
                  "animate-bounce [animation-delay:200ms]"
                )}
                style={{ animationDuration: "1.4s" }}
              />
              <div
                className={cn(
                  "w-2 h-2 rounded-full bg-accent",
                  "animate-bounce [animation-delay:400ms]"
                )}
                style={{ animationDuration: "1.4s" }}
              />
            </div>

            <div className="text-sm text-muted-foreground animate-pulse">
              Processing your request...
            </div>
          </div>

          {/* Animated progress bar */}
          <div className="mt-3 h-1 bg-background rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-accent/50 via-accent to-accent/50 animate-shimmer"
              style={{
                backgroundSize: "200% 100%",
              }}
            />
          </div>
        </Card>
      </div>
    </div>
  );
}
