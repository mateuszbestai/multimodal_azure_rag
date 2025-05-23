import React, { useState, useRef, useEffect } from 'react';
import { Send, Loader2, Filter, Table, FileSpreadsheet, FileText, Image as ImageIcon, BarChart3, TrendingUp, Database } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { tomorrow } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { formatDistanceToNow } from 'date-fns';

interface DataInsight {
  type: string;
  message: string;
  details?: any;
}

interface EnhancedMessage {
  id: string;
  content: string;
  role: 'user' | 'assistant';
  timestamp: Date;
  sources?: {
    pages: number[];
    images: string[];
    sheets: Array<{name: string; type: string}>;
    tables: Array<{name: string; sheet: string; rows: number; cols: number}>;
  };
  dataInsights?: DataInsight[];
  loading?: boolean;
}

interface EnhancedChatProps {
  onSendMessage?: (message: string, filter?: string) => void;
  messages: EnhancedMessage[];
  isLoading: boolean;
}

const EnhancedChatInterface: React.FC<EnhancedChatProps> = ({ 
  onSendMessage, 
  messages, 
  isLoading 
}) => {
  const [input, setInput] = useState('');
  const [contentFilter, setContentFilter] = useState('all');
  const [showFilters, setShowFilters] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;
    
    onSendMessage?.(input, contentFilter);
    setInput('');
  };

  const filterOptions = [
    { value: 'all', label: 'All Content', icon: Database },
    { value: 'excel', label: 'Excel Data', icon: FileSpreadsheet },
    { value: 'pdf', label: 'PDF Documents', icon: FileText },
    { value: 'images', label: 'Images', icon: ImageIcon },
  ];

  const renderDataInsights = (insights: DataInsight[]) => {
    if (!insights || insights.length === 0) return null;

    return (
      <div className="mt-4 space-y-2">
        <h4 className="text-sm font-medium text-gray-700 flex items-center gap-1">
          <TrendingUp size={16} />
          Data Insights
        </h4>
        {insights.map((insight, index) => (
          <div key={index} className="bg-blue-50 border border-blue-200 rounded-lg p-3">
            <div className="flex items-start gap-2">
              {insight.type === 'table_structure' && <Table size={16} className="text-blue-600 mt-0.5" />}
              {insight.type === 'relevant_columns' && <BarChart3 size={16} className="text-blue-600 mt-0.5" />}
              <div className="flex-1">
                <p className="text-sm text-blue-800">{insight.message}</p>
                {insight.details && (
                  <div className="mt-2 text-xs text-blue-600">
                    {insight.type === 'table_structure' && insight.details.columns && (
                      <div>
                        <span className="font-medium">Columns: </span>
                        <span>{insight.details.columns.join(', ')}</span>
                        {insight.details.total_columns > insight.details.columns.length && (
                          <span> (+{insight.details.total_columns - insight.details.columns.length} more)</span>
                        )}
                      </div>
                    )}
                    {insight.type === 'relevant_columns' && insight.details.columns && (
                      <div>
                        <span className="font-medium">Matching: </span>
                        <span>{insight.details.columns.join(', ')}</span>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          </div>
        ))}
      </div>
    );
  };

  const renderSources = (sources: EnhancedMessage['sources']) => {
    if (!sources) return null;

    const hasContent = sources.pages.length > 0 || sources.images.length > 0 || 
                      sources.sheets.length > 0 || sources.tables.length > 0;

    if (!hasContent) return null;

    return (
      <div className="mt-4 p-3 bg-gray-50 rounded-lg">
        <h4 className="text-sm font-medium text-gray-700 mb-2">Sources</h4>
        <div className="grid grid-cols-2 gap-3 text-xs">
          {sources.sheets.length > 0 && (
            <div>
              <span className="font-medium text-gray-600 flex items-center gap-1">
                <FileSpreadsheet size={12} />
                Sheets ({sources.sheets.length}):
              </span>
              <ul className="mt-1 space-y-1">
                {sources.sheets.map((sheet, index) => (
                  <li key={index} className="text-gray-500">
                    {sheet.name} ({sheet.type})
                  </li>
                ))}
              </ul>
            </div>
          )}
          
          {sources.tables.length > 0 && (
            <div>
              <span className="font-medium text-gray-600 flex items-center gap-1">
                <Table size={12} />
                Tables ({sources.tables.length}):
              </span>
              <ul className="mt-1 space-y-1">
                {sources.tables.map((table, index) => (
                  <li key={index} className="text-gray-500">
                    {table.name} ({table.rows}×{table.cols})
                  </li>
                ))}
              </ul>
            </div>
          )}
          
          {sources.pages.length > 0 && (
            <div>
              <span className="font-medium text-gray-600 flex items-center gap-1">
                <FileText size={12} />
                Pages:
              </span>
              <span className="text-gray-500">
                {sources.pages.slice(0, 5).join(', ')}
                {sources.pages.length > 5 && ` (+${sources.pages.length - 5} more)`}
              </span>
            </div>
          )}
          
          {sources.images.length > 0 && (
            <div>
              <span className="font-medium text-gray-600 flex items-center gap-1">
                <ImageIcon size={12} />
                Images:
              </span>
              <span className="text-gray-500">{sources.images.length} items</span>
            </div>
          )}
        </div>
      </div>
    );
  };

  const suggestedQuestions = [
    "What data is available in the uploaded Excel files?",
    "Show me the structure of the tables in the spreadsheet",
    "What are the key metrics in the data?",
    "Compare values across different sheets",
    "What trends can you identify in the data?",
  ];

  return (
    <div className="flex flex-col h-full">
      {/* Filter Bar */}
      <div className="border-b border-gray-200 p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <button
              onClick={() => setShowFilters(!showFilters)}
              className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm transition-colors ${
                contentFilter === 'all' 
                  ? 'bg-gray-100 text-gray-600' 
                  : 'bg-primary-100 text-primary-800'
              }`}
            >
              <Filter size={16} />
              Content Filter
            </button>
            {contentFilter !== 'all' && (
              <span className="text-xs bg-primary-100 text-primary-800 px-2 py-1 rounded-full">
                {filterOptions.find(opt => opt.value === contentFilter)?.label}
              </span>
            )}
          </div>
        </div>
        
        {showFilters && (
          <div className="mt-3 flex flex-wrap gap-2">
            {filterOptions.map((option) => {
              const IconComponent = option.icon;
              return (
                <button
                  key={option.value}
                  onClick={() => {
                    setContentFilter(option.value);
                    setShowFilters(false);
                  }}
                  className={`flex items-center gap-2 px-3 py-2 rounded-lg text-sm transition-colors ${
                    contentFilter === option.value
                      ? 'bg-primary-600 text-white'
                      : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                  }`}
                >
                  <IconComponent size={16} />
                  {option.label}
                </button>
              );
            })}
          </div>
        )}
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 ? (
          <div className="text-center py-8">
            <div className="w-16 h-16 bg-primary-100 rounded-full flex items-center justify-center mx-auto mb-4">
              <Database size={28} className="text-primary-600" />
            </div>
            <h3 className="text-lg font-medium text-gray-800 mb-2">
              Ask questions about your data
            </h3>
            <p className="text-gray-500 mb-6">
              Upload Excel files or PDFs and start exploring your data with AI
            </p>
            
            {/* Suggested Questions */}
            <div className="max-w-md mx-auto">
              <p className="text-sm font-medium text-gray-700 mb-3">Try asking:</p>
              <div className="space-y-2">
                {suggestedQuestions.map((question, index) => (
                  <button
                    key={index}
                    onClick={() => setInput(question)}
                    className="w-full text-left p-2 text-sm text-gray-600 hover:bg-gray-50 rounded border hover:border-gray-300 transition-colors"
                  >
                    {question}
                  </button>
                ))}
              </div>
            </div>
          </div>
        ) : (
          messages.map((message) => (
            <div key={message.id} className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}>
              {message.role === 'assistant' && (
                <div className="w-8 h-8 rounded-full bg-primary-600 flex items-center justify-center mr-3 mt-1">
                  <span className="text-white text-sm font-semibold">AI</span>
                </div>
              )}
              <div className={`max-w-[80%] rounded-lg p-4 ${
                message.role === 'user' 
                  ? 'bg-primary-600 text-white' 
                  : 'bg-white border border-gray-200'
              }`}>
                {message.loading ? (
                  <div className="flex items-center gap-2">
                    <Loader2 className="animate-spin" size={16} />
                    <span>Analyzing your data...</span>
                  </div>
                ) : (
                  <div>
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
                    
                    {message.role === 'assistant' && (
                      <>
                        {renderDataInsights(message.dataInsights || [])}
                        {renderSources(message.sources)}
                      </>
                    )}
                    
                    <div className="flex items-center justify-between mt-2 text-sm">
                      <span className={`${
                        message.role === 'user'
                          ? 'text-primary-100'
                          : 'text-gray-400'
                      }`}>
                        {formatDistanceToNow(message.timestamp, { addSuffix: true })}
                      </span>
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
          ))
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Form */}
      <div className="border-t border-gray-200 p-4">
        <form onSubmit={handleSubmit} className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask about your data..."
            className="flex-1 rounded-lg border border-gray-300 px-4 py-2 focus:outline-none focus:ring-2 focus:ring-primary-500 disabled:opacity-50"
            disabled={isLoading}
          />
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
        </form>
      </div>
    </div>
  );
};

export default EnhancedChatInterface;