"use client";

import { useState } from "react";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import {
  Wrench,
  ChevronDown,
  ChevronRight,
  CheckCircle,
  AlertCircle,
  Clock,
  Code2,
} from "lucide-react";

interface AgentToolCallProps {
  toolName: string;
  toolInput?: Record<string, any>;
  toolOutput?: any;
  isResult?: boolean;
  timestamp: Date;
}

export function AgentToolCall({
  toolName,
  toolInput,
  toolOutput,
  isResult = false,
  timestamp,
}: AgentToolCallProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  const getStatusIcon = () => {
    if (isResult) {
      return toolOutput?.error ? (
        <AlertCircle className="w-4 h-4 text-red-400" />
      ) : (
        <CheckCircle className="w-4 h-4 text-green-400" />
      );
    }
    return <Clock className="w-4 h-4 text-amber-400 animate-pulse" />;
  };

  const getStatusColor = () => {
    if (!isResult) return "border-amber-500/20 bg-amber-500/5";
    if (toolOutput?.error) return "border-red-500/20 bg-red-500/5";
    return "border-green-500/20 bg-green-500/5";
  };

  const getBadgeColor = () => {
    if (!isResult) return "bg-amber-500/20 text-amber-400 border-amber-500/30";
    if (toolOutput?.error)
      return "bg-red-500/20 text-red-400 border-red-500/30";
    return "bg-green-500/20 text-green-400 border-green-500/30";
  };

  return (
    <div className="flex gap-3">
      {/* Icon */}
      <div className="flex-shrink-0 w-10 h-10 rounded-full flex items-center justify-center border border-accent/30 bg-accent/10 text-accent">
        <Wrench className="w-5 h-5" />
      </div>

      {/* Tool Call Content */}
      <div className="flex-1 space-y-1">
        <div className="flex items-center gap-2">
          <Badge variant="outline" className={cn("text-xs border", getBadgeColor())}>
            {isResult ? "Tool Result" : "Tool Call"}
          </Badge>
          <span className="text-xs text-muted-foreground">
            {timestamp.toLocaleTimeString()}
          </span>
        </div>

        <Card className={cn("border transition-all", getStatusColor())}>
          {/* Header */}
          <Button
            variant="ghost"
            className="w-full p-4 hover:bg-accent/5 flex items-center justify-between"
            onClick={() => setIsExpanded(!isExpanded)}
          >
            <div className="flex items-center gap-3">
              {getStatusIcon()}
              <div className="flex items-center gap-2">
                <Code2 className="w-4 h-4 text-muted-foreground" />
                <span className="font-mono text-sm text-foreground">
                  {toolName}
                </span>
              </div>
            </div>
            {isExpanded ? (
              <ChevronDown className="w-4 h-4 text-muted-foreground" />
            ) : (
              <ChevronRight className="w-4 h-4 text-muted-foreground" />
            )}
          </Button>

          {/* Expanded Content */}
          {isExpanded && (
            <div className="px-4 pb-4 space-y-3 border-t border-border/50">
              {/* Tool Input */}
              {toolInput && (
                <div className="space-y-2">
                  <div className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
                    Input
                  </div>
                  <div className="bg-background/50 rounded-md p-3 border border-border/50">
                    <pre className="text-xs text-foreground overflow-x-auto font-mono">
                      {JSON.stringify(toolInput, null, 2)}
                    </pre>
                  </div>
                </div>
              )}

              {/* Tool Output */}
              {isResult && toolOutput && (
                <div className="space-y-2">
                  <div className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
                    {toolOutput.error ? "Error" : "Output"}
                  </div>
                  <div
                    className={cn(
                      "rounded-md p-3 border",
                      toolOutput.error
                        ? "bg-red-500/5 border-red-500/20"
                        : "bg-background/50 border-border/50"
                    )}
                  >
                    <pre className="text-xs text-foreground overflow-x-auto font-mono">
                      {typeof toolOutput === "string"
                        ? toolOutput
                        : JSON.stringify(toolOutput, null, 2)}
                    </pre>
                  </div>
                </div>
              )}

              {!isResult && (
                <div className="flex items-center gap-2 text-xs text-muted-foreground pt-2">
                  <Clock className="w-3 h-3 animate-pulse" />
                  <span>Executing...</span>
                </div>
              )}
            </div>
          )}
        </Card>
      </div>
    </div>
  );
}
