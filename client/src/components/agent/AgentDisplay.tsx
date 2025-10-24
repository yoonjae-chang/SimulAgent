"use client";

import { useState, useRef, useEffect } from "react";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { Send, Bot, User, Terminal } from "lucide-react";
import { AgentMessage } from "./AgentMessage";
import { AgentThinking } from "./AgentThinking";

export interface Message {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  timestamp: Date;
  type?: "message" | "thinking" | "tool_call" | "tool_result";
  toolName?: string;
  toolInput?: Record<string, any>;
  toolOutput?: any;
}

interface AgentDisplayProps {
  initialMessages?: Message[];
  onSendMessage?: (message: string) => Promise<void>;
  isProcessing?: boolean;
}

export function AgentDisplay({
  initialMessages = [],
  onSendMessage,
  isProcessing = false,
}: AgentDisplayProps) {
  const [messages, setMessages] = useState<Message[]>(initialMessages);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: input,
      timestamp: new Date(),
      type: "message",
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);

    try {
      if (onSendMessage) {
        await onSendMessage(input);
      } else {
        // Demo mode - simulate agent response
        setTimeout(() => {
          const agentMessage: Message = {
            id: (Date.now() + 1).toString(),
            role: "assistant",
            content: "I'm processing your request...",
            timestamp: new Date(),
            type: "thinking",
          };
          setMessages((prev) => [...prev, agentMessage]);

          setTimeout(() => {
            const responseMessage: Message = {
              id: (Date.now() + 2).toString(),
              role: "assistant",
              content: `I understand you want to: "${input}". Let me help you with that.`,
              timestamp: new Date(),
              type: "message",
            };
            setMessages((prev) => [...prev, responseMessage]);
            setIsLoading(false);
          }, 2000);
        }, 500);
      }
    } catch (error) {
      console.error("Error sending message:", error);
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="flex flex-col h-full w-full">
      {/* Header */}
      <div className="border-b border-border bg-card/50 backdrop-blur-sm p-4">
        <div className="flex items-center gap-3">
          <div className="relative">
            <div className="absolute inset-0 bg-accent/20 blur-xl rounded-full animate-pulse" />
            <Terminal className="w-6 h-6 text-accent relative z-10" />
          </div>
          <div>
            <h2 className="text-lg font-semibold text-foreground">
              AI Agent
            </h2>
            <p className="text-sm text-muted-foreground">
              {isLoading || isProcessing ? "Thinking..." : "Ready to help"}
            </p>
          </div>
        </div>
      </div>

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-center space-y-4 opacity-60">
            <div className="relative">
              <div className="absolute inset-0 bg-accent/10 blur-3xl rounded-full" />
              <Bot className="w-16 h-16 text-accent relative z-10" />
            </div>
            <div className="space-y-2">
              <h3 className="text-xl font-medium text-foreground">
                Start a conversation
              </h3>
              <p className="text-sm text-muted-foreground max-w-md">
                Ask me anything and I'll use my tools and reasoning to help you accomplish your goals.
              </p>
            </div>
          </div>
        ) : (
          <>
            {messages.map((message, index) => (
              <div
                key={message.id}
                className={cn(
                  "animate-in fade-in slide-in-from-bottom-2 duration-300",
                  `animation-delay-${index % 3}`
                )}
              >
                <AgentMessage message={message} />
              </div>
            ))}
            {(isLoading || isProcessing) && (
              <div className="animate-in fade-in slide-in-from-bottom-2">
                <AgentThinking />
              </div>
            )}
          </>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="border-t border-border bg-card/50 backdrop-blur-sm p-4">
        <div className="flex gap-2">
          <Input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Ask the agent to help you..."
            className="flex-1 bg-background/50 border-border focus:border-accent transition-colors"
            disabled={isLoading || isProcessing}
          />
          <Button
            onClick={handleSend}
            disabled={!input.trim() || isLoading || isProcessing}
            className={cn(
              "bg-accent hover:bg-accent/90 text-white transition-all",
              !input.trim() || isLoading || isProcessing
                ? "opacity-50 cursor-not-allowed"
                : "hover:scale-105"
            )}
          >
            <Send className="w-4 h-4" />
          </Button>
        </div>
        <p className="text-xs text-muted-foreground mt-2">
          Press Enter to send, Shift + Enter for new line
        </p>
      </div>
    </div>
  );
}
