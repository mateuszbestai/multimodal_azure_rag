import React, { useState, useRef, useEffect } from 'react';
import { Send, Loader2, Copy, Download, Image as ImageIcon, Search, X, ChevronRight, ChevronLeft, Maximize2, ZoomIn, ZoomOut } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import { formatDistanceToNow } from 'date-fns';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { tomorrow } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { CSSTransition, TransitionGroup } from 'react-transition-group';

interface Message {
  id: string;
  content: string;
  role: 'user' | 'assistant' | 'system';
  timestamp: Date;
  sources?: {
    pages: number[];
    images?: string[];
  };
  loading?: boolean;
}

interface SourcePreview {
  id: string;
  page: number;
  content: string;
  imageUrl?: string;
  category?: string;
  title?: string;
  date?: Date;
}

interface ImageViewerProps {
  url: string;
  onClose: () => void;
}

const ImageViewer: React.FC<ImageViewerProps> = ({ url, onClose }) => {
  const [scale, setScale] = useState(1);

  return (
    <div className="fixed inset-0 bg-black/90 z-50 flex items-center justify-center animate-fade-in">
      <div className="absolute top-4 right-4 flex gap-2">
        <button
          onClick={() => setScale(s => s + 0.25)}
          className="p-2 bg-white/10 rounded-full hover:bg-white/20 transition-colors"
        >
          <ZoomIn size={20} className="text-white" />
        </button>
        <button
          onClick={() => setScale(s => Math.max(0.5, s - 0.25))}
          className="p-2 bg-white/10 rounded-full hover:bg-white/20 transition-colors"
        >
          <ZoomOut size={20} className="text-white" />
        </button>
        <button
          onClick={onClose}
          className="p-2 bg-white/10 rounded-full hover:bg-white/20 transition-colors"
        >
          <X size={20} className="text-white" />
        </button>
      </div>
      <img
        src={url}
        alt="Full size preview"
        className="max-w-[90vw] max-h-[90vh] object-contain transition-transform duration-200"
        style={{ transform: `scale(${scale})` }}
      />
    </div>
  );
};

function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [sourcePreviews, setSourcePreviews] = useState<SourcePreview[]>([]);
  const [showReferences, setShowReferences] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [categoryFilter, setCategoryFilter] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage: Message = {
      id: crypto.randomUUID(),
      content: input,
      role: 'user',
      timestamp: new Date(),
    };

    const assistantMessage: Message = {
      id: crypto.randomUUID(),
      content: '',
      role: 'assistant',
      timestamp: new Date(),
      loading: true,
    };

    setMessages(prev => [...prev, userMessage, assistantMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await fetch('http://localhost:5001/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: input }),
      });

      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

      const data = await response.json();

      setMessages(prev =>
        prev.map(msg =>
          msg.id === assistantMessage.id
            ? {
                ...msg,
                content: data.response || 'No response could be generated',
                loading: false,
                sources: {
                  pages: data.sources?.text?.map((s: any) => s.page) || [],
                  images: data.sources?.images?.map((img: any) => img.url) || [],
                },
              }
            : msg
        )
      );

      if (data.sourcePreviews) {
        setSourcePreviews(
          data.sourcePreviews.map((preview: any) => ({
            ...preview,
            id: crypto.randomUUID(),
          }))
        );
      }
    } catch (error) {
      console.error('API request failed:', error);
      setMessages(prev =>
        prev.map(msg =>
          msg.id === assistantMessage.id
            ? {
                ...msg,
                content: 'Sorry, an error occurred while processing your request.',
                loading: false,
              }
            : msg
        )
      );
    } finally {
      setIsLoading(false);
    }
  };

  const copyToClipboard = async (text: string) => {
    try {
      await navigator.clipboard.writeText(text);
    } catch (err) {
      console.error('Failed to copy text:', err);
    }
  };

  const exportChat = () => {
    const chatExport = messages
      .map(msg => `${msg.role.toUpperCase()} (${msg.timestamp.toISOString()}):\n${msg.content}\n`)
      .join('\n---\n\n');
    const blob = new Blob([chatExport], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `chat-export-${new Date().toISOString()}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const clearChat = () => {
    setMessages([]);
    setSourcePreviews([]);
  };

  const filteredSourcePreviews = sourcePreviews.filter(preview => {
    const matchesSearch = searchTerm
      ? preview.content.toLowerCase().includes(searchTerm.toLowerCase()) ||
        preview.title?.toLowerCase().includes(searchTerm.toLowerCase())
      : true;
    const matchesCategory = categoryFilter
      ? preview.category === categoryFilter
      : true;
    return matchesSearch && matchesCategory;
  });

  const categories = Array.from(
    new Set(sourcePreviews.map(preview => preview.category).filter(Boolean))
  );

  return (
    <div className="flex h-screen bg-gray-50">
      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <header className="bg-white border-b border-gray-200 px-6 py-4 shadow-sm">
          <div className="max-w-4xl mx-auto flex justify-between items-center">
            <h1 className="text-xl font-semibold text-gray-800 flex items-center gap-2">
              <div className="w-8 h-8 bg-primary-600 rounded-full flex items-center justify-center">
                <span className="text-white font-bold">AI</span>
              </div>
              Knowledge Assistant
            </h1>
            <div className="flex items-center gap-3">
              <button
                onClick={exportChat}
                className="flex items-center gap-2 px-3 py-1.5 text-sm text-gray-600 hover:text-gray-800 hover:bg-gray-100 rounded-md transition-colors"
              >
                <Download size={16} />
                Export
              </button>
              <button
                onClick={clearChat}
                className="flex items-center gap-2 px-3 py-1.5 text-sm text-red-600 hover:text-red-800 hover:bg-red-50 rounded-md transition-colors"
              >
                <X size={16} />
                Clear
              </button>
            </div>
          </div>
        </header>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto px-6 py-6">
          <div className="max-w-4xl mx-auto">
            <TransitionGroup className="space-y-6">
              {messages.map((message) => (
                <CSSTransition key={message.id} timeout={300} classNames="message">
                  <div
                    className={`flex ${
                      message.role === 'user' ? 'justify-end' : 'justify-start'
                    } animate-slide-in`}
                  >
                    {message.role === 'assistant' && (
                      <div className="w-8 h-8 rounded-full bg-primary-600 flex items-center justify-center mr-3 mt-1">
                        <span className="text-white text-sm font-semibold">AI</span>
                      </div>
                    )}
                    <div
                      className={`max-w-[80%] rounded-lg p-4 ${
                        message.role === 'user'
                          ? 'bg-primary-600 text-white'
                          : 'bg-white shadow-sm border border-gray-200'
                      }`}
                    >
                      {message.loading ? (
                        <div className="flex items-center gap-2">
                          <Loader2 className="animate-spin" size={16} />
                          <span>Processing your request...</span>
                        </div>
                      ) : (
                        <div className="space-y-2">
                          <div className={`prose prose-sm max-w-none ${
                            message.role === 'user' ? 'prose-invert' : ''
                          }`}>
                            <ReactMarkdown
                              components={{
                                code({ node, inline, className, children, ...props }) {
                                  const match = /language-(\w+)/.exec(className || '');
                                  return !inline && match ? (
                                    <SyntaxHighlighter
                                      style={tomorrow}
                                      language={match[1]}
                                      PreTag="div"
                                      {...props}
                                    >
                                      {String(children).replace(/\n$/, '')}
                                    </SyntaxHighlighter>
                                  ) : (
                                    <code className={className} {...props}>
                                      {children}
                                    </code>
                                  );
                                },
                              }}
                            >
                              {message.content}
                            </ReactMarkdown>
                          </div>
                          <div className="flex items-center justify-between mt-2 text-sm">
                            <span className={`${
                              message.role === 'user'
                                ? 'text-primary-100'
                                : 'text-gray-400'
                            }`}>
                              {formatDistanceToNow(message.timestamp, { addSuffix: true })}
                            </span>
                            <button
                              onClick={() => copyToClipboard(message.content)}
                              className={`p-1 ${
                                message.role === 'user'
                                  ? 'text-primary-100 hover:text-white'
                                  : 'text-gray-400 hover:text-gray-600'
                              } transition-colors`}
                              title="Copy message"
                            >
                              <Copy size={16} />
                            </button>
                          </div>
                        </div>
                      )}
                    </div>
                    {message.role === 'user' && (
                      <div className="w-8 h-8 rounded-full bg-gray-200 flex items-center justify-center ml-3 mt-1">
                        <span className="text-gray-600 text-sm font-semibold">U</span>
                      </div>
                    )}
                  </div>
                </CSSTransition>
              ))}
            </TransitionGroup>
            <div ref={messagesEndRef} />
          </div>
        </div>

        {/* Input Form */}
        <div className="bg-white border-t border-gray-200 px-6 py-4">
          <form onSubmit={handleSubmit} className="max-w-4xl mx-auto">
            <div className="flex gap-4">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Ask a question..."
                className="flex-1 rounded-lg border border-gray-300 px-4 py-2 focus:outline-none focus:ring-2 focus:ring-primary-500 disabled:opacity-50"
                disabled={isLoading}
              />
              <div className="flex gap-2">
                <button
                  type="submit"
                  disabled={isLoading || !input.trim()}
                  className="bg-primary-600 text-white px-6 py-2 rounded-lg hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2 transition-colors"
                >
                  {isLoading ? (
                    <Loader2 className="animate-spin" size={16} />
                  ) : (
                    <Send size={16} />
                  )}
                  Send
                </button>
                <button
                  type="button"
                  onClick={clearChat}
                  className="bg-red-600 text-white px-6 py-2 rounded-lg hover:bg-red-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2 transition-colors"
                >
                  <X size={16} />
                  Clear
                </button>
              </div>
            </div>
          </form>
        </div>
      </div>

      {/* References Sidebar */}
      <div
        className={`w-96 bg-white border-l border-gray-200 transform transition-transform duration-300 ease-in-out ${
          showReferences ? 'translate-x-0' : 'translate-x-full'
        }`}
      >
        <div className="h-full flex flex-col">
          <div className="p-4 border-b border-gray-200">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-semibold text-gray-800">References</h2>
              <button
                onClick={() => setShowReferences(false)}
                className="p-1 text-gray-400 hover:text-gray-600 transition-colors"
              >
                <X size={20} />
              </button>
            </div>
            <div className="relative">
              <input
                type="text"
                placeholder="Search references..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full px-4 py-2 pr-10 rounded-lg border border-gray-300 focus:outline-none focus:ring-2 focus:ring-primary-500"
              />
              <Search className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400" size={18} />
            </div>
            {categories.length > 0 && (
              <div className="mt-4 flex flex-wrap gap-2">
                {categories.map((category) => (
                  <button
                    key={category}
                    onClick={() => setCategoryFilter(categoryFilter === category ? null : category || null)}
                    className={`px-3 py-1 rounded-full text-sm ${
                      categoryFilter === category
                        ? 'bg-primary-100 text-primary-800'
                        : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                    } transition-colors`}
                  >
                    {category}
                  </button>
                ))}
              </div>
            )}
          </div>
          <div className="flex-1 overflow-y-auto p-4">
            <div className="space-y-4">
              {filteredSourcePreviews.map((preview) => (
                <div
                  key={preview.id}
                  className="bg-white rounded-lg border border-gray-200 p-4 hover:border-primary-300 transition-colors"
                >
                  {preview.title && (
                    <h3 className="font-medium text-gray-900 mb-1">{preview.title}</h3>
                  )}
                  <div className="flex items-center gap-2 text-sm text-gray-500 mb-2">
                    <span>Page {preview.page}</span>
                    {preview.date && (
                      <>
                        <span>•</span>
                        <span>{formatDistanceToNow(preview.date, { addSuffix: true })}</span>
                      </>
                    )}
                  </div>
                  <p className="text-sm text-gray-600 mb-4">{preview.content}</p>
                  {preview.imageUrl && (
                    <div className="relative aspect-video bg-gray-50 rounded-md overflow-hidden group">
                      <img
                        src={preview.imageUrl}
                        alt={preview.title || `Preview from page ${preview.page}`}
                        className="w-full h-full object-contain p-2"
                        loading="lazy"
                      />
                      <div className="absolute inset-0 bg-gradient-to-t from-black/50 to-transparent opacity-0 group-hover:opacity-100 transition-opacity flex items-end justify-between p-3">
                        <button
                          onClick={() => preview.imageUrl && setSelectedImage(preview.imageUrl)}
                          className="text-white text-sm hover:text-primary-200 flex items-center gap-1"
                        >
                          <Maximize2 size={14} />
                          <span>View Full Size</span>
                        </button>
                        <a
                          href={preview.imageUrl}
                          download
                          className="text-white text-sm hover:text-primary-200 flex items-center gap-1"
                        >
                          <Download size={14} />
                          <span>Download</span>
                        </a>
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* References Toggle */}
      <button
        onClick={() => setShowReferences(!showReferences)}
        className={`fixed right-0 top-1/2 transform -translate-y-1/2 bg-white border border-gray-200 rounded-l-lg p-2 shadow-md transition-transform ${
          showReferences ? 'translate-x-96' : ''
        }`}
      >
        {showReferences ? (
          <ChevronRight className="text-gray-600" size={20} />
        ) : (
          <ChevronLeft className="text-gray-600" size={20} />
        )}
      </button>

      {/* Image Viewer */}
      {selectedImage && (
        <ImageViewer url={selectedImage} onClose={() => setSelectedImage(null)} />
      )}
    </div>
  );
}

export default App;