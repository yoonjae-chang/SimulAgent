"use client";

import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import {
  Bot,
  User,
  Brain,
  Wrench,
  CheckCircle,
  AlertCircle,
  Terminal,
  Code2,
} from "lucide-react";
import { AgentToolCall } from "./AgentToolCall";
import type { Message } from "./AgentDisplay";

interface AgentMessageProps {
  message: Message;
}

export function AgentMessage({ message }: AgentMessageProps) {
  const isUser = message.role === "user";
  const isSystem = message.role === "system";

  const getIcon = () => {
    if (isUser) return <User className="w-5 h-5" />;
    if (message.type === "thinking") return <Brain className="w-5 h-5" />;
    if (message.type === "tool_call" || message.type === "tool_result")
      return <Wrench className="w-5 h-5" />;
    return <Bot className="w-5 h-5" />;
  };

  const getLabel = () => {
    if (isUser) return "You";
    if (isSystem) return "System";
    if (message.type === "thinking") return "Agent Thinking";
    if (message.type === "tool_call") return "Tool Call";
    if (message.type === "tool_result") return "Tool Result";
    return "Agent";
  };

  const getBadgeColor = () => {
    if (isUser) return "bg-blue-500/20 text-blue-400 border-blue-500/30";
    if (message.type === "thinking")
      return "bg-purple-500/20 text-purple-400 border-purple-500/30";
    if (message.type === "tool_call")
      return "bg-amber-500/20 text-amber-400 border-amber-500/30";
    if (message.type === "tool_result")
      return "bg-green-500/20 text-green-400 border-green-500/30";
    return "bg-accent/20 text-accent border-accent/30";
  };

  if (message.type === "tool_call" || message.type === "tool_result") {
    return (
      <AgentToolCall
        toolName={message.toolName || "unknown"}
        toolInput={message.toolInput}
        toolOutput={message.toolOutput}
        isResult={message.type === "tool_result"}
        timestamp={message.timestamp}
      />
    );
  }

  return (
    <div
      className={cn("flex gap-3", isUser ? "flex-row-reverse" : "flex-row")}
    >
      {/* Avatar */}
      <div
        className={cn(
          "flex-shrink-0 w-10 h-10 rounded-full flex items-center justify-center border",
          isUser
            ? "bg-blue-500/10 border-blue-500/30 text-blue-400"
            : "bg-accent/10 border-accent/30 text-accent"
        )}
      >
        {getIcon()}
      </div>

      {/* Message Content */}
      <div className={cn("flex-1 space-y-1", isUser ? "items-end" : "items-start")}>
        <div className="flex items-center gap-2">
          <Badge
            variant="outline"
            className={cn("text-xs border", getBadgeColor())}
          >
            {getLabel()}
          </Badge>
          <span className="text-xs text-muted-foreground">
            {message.timestamp.toLocaleTimeString()}
          </span>
        </div>

        <Card
          className={cn(
            "p-4 border transition-all",
            isUser
              ? "bg-blue-500/5 border-blue-500/20"
              : message.type === "thinking"
              ? "bg-purple-500/5 border-purple-500/20"
              : "bg-card/50 border-border",
            "hover:border-accent/40"
          )}
        >
          <div className="text-sm text-foreground whitespace-pre-wrap break-words">
            {message.content}
          </div>
        </Card>
      </div>
    </div>
  );
}
