/**
 * React component example for integrating with FastAPI Agent Streaming API
 *
 * This component demonstrates how to:
 * 1. Send requests to the streaming endpoint
 * 2. Handle different event types
 * 3. Display agent thinking and results in real-time
 */

import { useState, useRef } from 'react';

// Types for the API
interface StreamEvent {
  event: 'thinking' | 'tool_use' | 'tool_result' | 'partial_output' | 'final_output' | 'error' | 'complete';
  data: {
    content?: string;
    tool?: string;
    input?: any;
    result?: string;
    error?: string;
    message?: string;
    success?: boolean;
  };
  timestamp: string;
}

interface AgentRequest {
  input: string;
  model: string;
  mcp_servers: string[];
}

export default function AgentChat() {
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState<string[]>([]);
  const [thinking, setThinking] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [toolActivity, setToolActivity] = useState<string[]>([]);
  const abortControllerRef = useRef<AbortController | null>(null);

  const streamAgent = async (prompt: string) => {
    if (!prompt.trim()) return;

    setIsStreaming(true);
    setThinking('');
    setToolActivity([]);

    // Create abort controller for cancellation
    abortControllerRef.current = new AbortController();

    const request: AgentRequest = {
      input: prompt,
      model: 'openai/gpt-4.1',
      mcp_servers: ['joerup/exa-mcp', 'windsor/brave-search-mcp']
    };

    try {
      const response = await fetch('http://localhost:8000/api/agent/stream', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(request),
        signal: abortControllerRef.current.signal
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const reader = response.body?.getReader();
      if (!reader) throw new Error('No reader available');

      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const event: StreamEvent = JSON.parse(line.substring(6));
              handleStreamEvent(event);
            } catch (e) {
              console.error('Error parsing event:', e);
            }
          }
        }
      }
    } catch (error: any) {
      if (error.name === 'AbortError') {
        console.log('Stream aborted by user');
      } else {
        console.error('Streaming error:', error);
        setMessages(prev => [...prev, `Error: ${error.message}`]);
      }
    } finally {
      setIsStreaming(false);
      setThinking('');
      abortControllerRef.current = null;
    }
  };

  const handleStreamEvent = (event: StreamEvent) => {
    switch (event.event) {
      case 'thinking':
        setThinking(event.data.content || '');
        break;

      case 'tool_use':
        setToolActivity(prev => [...prev, `Using ${event.data.tool}...`]);
        break;

      case 'tool_result':
        setToolActivity(prev => [...prev, `Received results from ${event.data.tool}`]);
        break;

      case 'partial_output':
        // You could append partial outputs to build the message incrementally
        console.log('Partial:', event.data.content);
        break;

      case 'final_output':
        setMessages(prev => [...prev, event.data.content || '']);
        setThinking('');
        setToolActivity([]);
        break;

      case 'error':
        setMessages(prev => [...prev, `Error: ${event.data.error}`]);
        setThinking('');
        break;

      case 'complete':
        console.log('Stream complete:', event.data.message);
        break;
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isStreaming) return;

    const prompt = input;
    setInput('');
    setMessages(prev => [...prev, `You: ${prompt}`]);

    await streamAgent(prompt);
  };

  const stopStreaming = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
  };

  return (
    <div className="flex flex-col h-screen max-w-4xl mx-auto p-4">
      <h1 className="text-2xl font-bold mb-4">AI Agent Chat</h1>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto border rounded-lg p-4 mb-4 space-y-4">
        {messages.map((msg, i) => (
          <div
            key={i}
            className={`p-3 rounded-lg ${
              msg.startsWith('You:')
                ? 'bg-blue-100 ml-auto max-w-[80%]'
                : msg.startsWith('Error:')
                ? 'bg-red-100'
                : 'bg-gray-100 max-w-[80%]'
            }`}
          >
            {msg}
          </div>
        ))}

        {/* Thinking indicator */}
        {thinking && (
          <div className="bg-yellow-50 p-3 rounded-lg border-l-4 border-yellow-400">
            <div className="flex items-center space-x-2">
              <div className="animate-pulse">🤔</div>
              <span className="text-sm text-gray-700">{thinking}</span>
            </div>
          </div>
        )}

        {/* Tool activity */}
        {toolActivity.length > 0 && (
          <div className="bg-blue-50 p-3 rounded-lg border-l-4 border-blue-400">
            <div className="text-sm space-y-1">
              {toolActivity.map((activity, i) => (
                <div key={i} className="flex items-center space-x-2">
                  <span>🔧</span>
                  <span className="text-gray-700">{activity}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Input form */}
      <form onSubmit={handleSubmit} className="flex space-x-2">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask me anything..."
          disabled={isStreaming}
          className="flex-1 px-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
        {isStreaming ? (
          <button
            type="button"
            onClick={stopStreaming}
            className="px-6 py-2 bg-red-500 text-white rounded-lg hover:bg-red-600"
          >
            Stop
          </button>
        ) : (
          <button
            type="submit"
            disabled={!input.trim()}
            className="px-6 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:bg-gray-300"
          >
            Send
          </button>
        )}
      </form>

      {/* Status indicator */}
      <div className="mt-2 text-sm text-gray-500 text-center">
        {isStreaming ? (
          <span className="flex items-center justify-center space-x-2">
            <span className="animate-spin">⏳</span>
            <span>Streaming...</span>
          </span>
        ) : (
          <span>Ready</span>
        )}
      </div>
    </div>
  );
}


// Alternative: Using a custom hook for agent streaming
export function useAgentStream() {
  const [isStreaming, setIsStreaming] = useState(false);
  const [thinking, setThinking] = useState('');
  const [error, setError] = useState<string | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  const stream = async (
    input: string,
    onEvent: (event: StreamEvent) => void,
    options?: {
      model?: string;
      mcp_servers?: string[];
    }
  ) => {
    setIsStreaming(true);
    setError(null);
    setThinking('');

    abortControllerRef.current = new AbortController();

    const request: AgentRequest = {
      input,
      model: options?.model || 'openai/gpt-4.1',
      mcp_servers: options?.mcp_servers || ['joerup/exa-mcp']
    };

    try {
      const response = await fetch('http://localhost:8000/api/agent/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(request),
        signal: abortControllerRef.current.signal
      });

      if (!response.ok) throw new Error(`HTTP ${response.status}`);

      const reader = response.body?.getReader();
      if (!reader) throw new Error('No reader');

      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const event: StreamEvent = JSON.parse(line.substring(6));

            if (event.event === 'thinking') {
              setThinking(event.data.content || '');
            }

            onEvent(event);
          }
        }
      }
    } catch (err: any) {
      if (err.name !== 'AbortError') {
        setError(err.message);
      }
    } finally {
      setIsStreaming(false);
      setThinking('');
    }
  };

  const stop = () => {
    abortControllerRef.current?.abort();
  };

  return { stream, stop, isStreaming, thinking, error };
}


// Example usage of the hook:
/*
function MyComponent() {
  const { stream, stop, isStreaming, thinking } = useAgentStream();
  const [output, setOutput] = useState('');

  const handleSubmit = async (input: string) => {
    await stream(input, (event) => {
      if (event.event === 'final_output') {
        setOutput(event.data.content || '');
      }
    });
  };

  return (
    <div>
      {thinking && <p>Thinking: {thinking}</p>}
      {output && <p>Output: {output}</p>}
      <button onClick={() => handleSubmit('Hello')}>
        {isStreaming ? 'Stop' : 'Send'}
      </button>
    </div>
  );
}
*/
