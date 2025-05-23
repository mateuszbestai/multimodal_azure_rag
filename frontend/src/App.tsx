import React, { useState, useRef, useEffect } from 'react';
import { Send, Loader2, Copy, Download, Image as ImageIcon, Search, X, ChevronRight, ChevronLeft, Maximize2, ZoomIn, ZoomOut, Plus, Trash2, MessageSquare, Edit2, Save, MoreHorizontal, ArrowLeft, Upload, Settings } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import { formatDistanceToNow } from 'date-fns';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { tomorrow } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { CSSTransition, TransitionGroup } from 'react-transition-group';
import FileUpload from './components/FileUpload';
import EnhancedChatInterface from './components/EnhancedChatInterface';

interface Message {
  id: string;
  content: string;
  role: 'user' | 'assistant' | 'system';
  timestamp: Date;
  sources?: {
    pages: number[];
    images?: string[];
    sheets?: Array<{name: string; type: string}>;
    tables?: Array<{name: string; sheet: string; rows: number; cols: number}>;
  };
  dataInsights?: Array<{
    type: string;
    message: string;
    details?: any;
  }>;
  loading?: boolean;
}

interface SourcePreview {
  id: string;
  page?: number;
  content: string;
  imageUrl?: string;
  category?: string;
  title?: string;
  date?: Date;
  contentType?: string;
  sheetName?: string;
  tableName?: string;
  nodeType?: string;
  rowCount?: number;
  colCount?: number;
}

interface ChatSession {
  id: string;
  title: string;
  createdAt: Date;
  updatedAt: Date;
  messages: Message[];
  sourcePreviews: SourcePreview[];
}

interface UploadedFile {
  docId: string;
  fileName: string;
  contentType: string;
  sheetCount?: number;
}

interface UploadResult {
  filename: string;
  file_type: string;
  summary: string;
  node_count: number;
  sheet_count?: number;
  image_count?: number;
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
  // Navigation state
  const [activeView, setActiveView] = useState<'chat' | 'upload' | 'files'>('chat');
  
  // Chat history state
  const [chatSessions, setChatSessions] = useState<ChatSession[]>([]);
  const [activeChatId, setActiveChatId] = useState<string | null>(null);
  const [showChatSidebar, setShowChatSidebar] = useState(true);
  const [isEditingTitle, setIsEditingTitle] = useState(false);
  const [editTitle, setEditTitle] = useState('');
  
  // Current chat state
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [sourcePreviews, setSourcePreviews] = useState<SourcePreview[]>([]);
  const [showReferences, setShowReferences] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [categoryFilter, setCategoryFilter] = useState<string | null>(null);
  
  // File management state
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([]);
  
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Load chat sessions from localStorage on initial render
  useEffect(() => {
    const savedSessions = localStorage.getItem('chatSessions');
    if (savedSessions) {
      try {
        const parsedSessions = JSON.parse(savedSessions, (key, value) => {
          if (key === 'timestamp' || key === 'createdAt' || key === 'updatedAt') {
            return new Date(value);
          }
          return value;
        });
        setChatSessions(parsedSessions);
        
        if (parsedSessions.length > 0) {
          const mostRecentChat = parsedSessions.sort((a, b) => 
            new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime()
          )[0];
          setActiveChatId(mostRecentChat.id);
          setMessages(mostRecentChat.messages);
          setSourcePreviews(mostRecentChat.sourcePreviews);
        } else {
          createNewChat();
        }
      } catch (error) {
        console.error('Error parsing saved chat sessions:', error);
        createNewChat();
      }
    } else {
      createNewChat();
    }
  }, []);

  // Save chat sessions to localStorage whenever they change
  useEffect(() => {
    if (chatSessions.length > 0) {
      localStorage.setItem('chatSessions', JSON.stringify(chatSessions));
    }
  }, [chatSessions]);

  // Set current chat data when active chat changes
  useEffect(() => {
    if (activeChatId) {
      const activeChat = chatSessions.find(chat => chat.id === activeChatId);
      if (activeChat) {
        setMessages(activeChat.messages);
        setSourcePreviews(activeChat.sourcePreviews);
      }
    }
  }, [activeChatId, chatSessions]);

  // Scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const createNewChat = () => {
    const newChatId = crypto.randomUUID();
    const newChat: ChatSession = {
      id: newChatId,
      title: `New Chat ${new Date().toLocaleDateString()}`,
      createdAt: new Date(),
      updatedAt: new Date(),
      messages: [],
      sourcePreviews: []
    };
    
    setChatSessions(prev => [newChat, ...prev]);
    setActiveChatId(newChatId);
    setMessages([]);
    setSourcePreviews([]);
    setActiveView('chat');
  };

  const deleteChat = (chatId: string) => {
    setChatSessions(prev => prev.filter(chat => chat.id !== chatId));
    
    if (chatId === activeChatId) {
      const remainingChats = chatSessions.filter(chat => chat.id !== chatId);
      if (remainingChats.length > 0) {
        setActiveChatId(remainingChats[0].id);
        setMessages(remainingChats[0].messages);
        setSourcePreviews(remainingChats[0].sourcePreviews);
      } else {
        createNewChat();
      }
    }
  };

  const startTitleEdit = (chatId: string) => {
    const chat = chatSessions.find(c => c.id === chatId);
    if (chat) {
      setEditTitle(chat.title);
      setIsEditingTitle(true);
    }
  };

  const saveTitle = () => {
    if (activeChatId && editTitle.trim()) {
      setChatSessions(prev => 
        prev.map(chat => 
          chat.id === activeChatId 
            ? { ...chat, title: editTitle.trim() } 
            : chat
        )
      );
      setIsEditingTitle(false);
    }
  };

  const updateChatSession = (newMessages: Message[], newSourcePreviews: SourcePreview[]) => {
    if (activeChatId) {
      setChatSessions(prev => 
        prev.map(chat => 
          chat.id === activeChatId 
            ? { 
                ...chat, 
                messages: newMessages, 
                sourcePreviews: newSourcePreviews,
                updatedAt: new Date()
              } 
            : chat
        )
      );
    }
  };

  const handleSendMessage = async (message: string, contentFilter: string = 'all') => {
    if (!message.trim() || isLoading || !activeChatId) return;

    const userMessage: Message = {
      id: crypto.randomUUID(),
      content: message,
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

    const updatedMessages = [...messages, userMessage, assistantMessage];
    setMessages(updatedMessages);
    updateChatSession(updatedMessages, sourcePreviews);
    setIsLoading(true);

    try {
      const response = await fetch('http://localhost:5001/api/chat_enhanced', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          message: message,
          content_filter: contentFilter 
        }),
      });

      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

      const data = await response.json();

      const finalMessages = updatedMessages.map(msg =>
        msg.id === assistantMessage.id
          ? {
              ...msg,
              content: data.response || 'No response could be generated',
              loading: false,
              sources: data.sources || { pages: [], images: [], sheets: [], tables: [] },
              dataInsights: data.dataInsights || [],
            }
          : msg
      );

      setMessages(finalMessages);

      let updatedSourcePreviews = sourcePreviews;
      if (data.sourcePreviews) {
        updatedSourcePreviews = [
          ...sourcePreviews,
          ...data.sourcePreviews.map((preview: any) => ({
            ...preview,
            id: preview.id || crypto.randomUUID(),
          }))
        ];
        setSourcePreviews(updatedSourcePreviews);
      }

      // Update chat title based on first user message if it's a new chat
      const activeChat = chatSessions.find(chat => chat.id === activeChatId);
      if (activeChat && activeChat.messages.length === 0) {
        const newTitle = userMessage.content.length > 30 
          ? userMessage.content.substring(0, 30) + '...' 
          : userMessage.content;
        
        setChatSessions(prev => 
          prev.map(chat => 
            chat.id === activeChatId 
              ? { ...chat, title: newTitle } 
              : chat
          )
        );
      }

      updateChatSession(finalMessages, updatedSourcePreviews);

    } catch (error) {
      console.error('API request failed:', error);
      const errorMessages = updatedMessages.map(msg =>
        msg.id === assistantMessage.id
          ? {
              ...msg,
              content: 'Sorry, an error occurred while processing your request.',
              loading: false,
            }
          : msg
      );
      
      setMessages(errorMessages);
      updateChatSession(errorMessages, sourcePreviews);
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
    if (!activeChatId) return;
    
    const activeChat = chatSessions.find(chat => chat.id === activeChatId);
    if (!activeChat) return;
    
    const chatExport = activeChat.messages
      .map(msg => `${msg.role.toUpperCase()} (${msg.timestamp.toISOString()}):\n${msg.content}\n`)
      .join('\n---\n\n');
    
    const blob = new Blob([chatExport], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `chat-export-${activeChat.title}-${new Date().toISOString()}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const clearChat = () => {
    if (activeChatId) {
      setMessages([]);
      setSourcePreviews([]);
      updateChatSession([], []);
    }
  };

  const handleUploadComplete = (result: UploadResult) => {
    console.log('File uploaded successfully:', result);
    // Optionally switch to chat view after successful upload
    setActiveView('chat');
  };

  const handleFilesChange = (files: UploadedFile[]) => {
    setUploadedFiles(files);
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

  const activeChat = activeChatId 
    ? chatSessions.find(chat => chat.id === activeChatId) 
    : null;

  const navigationItems = [
    { id: 'chat', label: 'Chat', icon: MessageSquare },
    { id: 'upload', label: 'Upload', icon: Upload },
    { id: 'files', label: 'Files', icon: Settings },
  ];

  return (
    <div className="flex h-screen bg-gray-50">
      {/* Chat History Sidebar */}
      <div 
        className={`w-72 bg-gray-900 transform transition-transform duration-300 ease-in-out ${
          showChatSidebar ? 'translate-x-0' : '-translate-x-full'
        } absolute md:relative z-20 h-full overflow-hidden shadow-lg md:shadow-none`}
      >
        <div className="flex flex-col h-full">
          <div className="p-4 border-b border-gray-800">
            <button 
              onClick={createNewChat}
              className="w-full flex items-center justify-center gap-2 bg-primary-600 hover:bg-primary-700 text-white py-2 px-4 rounded-lg transition-colors mb-4"
            >
              <Plus size={18} />
              New Chat
            </button>
            
            {/* Navigation */}
            <div className="flex gap-1 bg-gray-800 rounded-lg p-1">
              {navigationItems.map((item) => {
                const IconComponent = item.icon;
                return (
                  <button
                    key={item.id}
                    onClick={() => setActiveView(item.id as any)}
                    className={`flex-1 flex items-center justify-center gap-1 py-2 px-3 rounded-md text-sm transition-colors ${
                      activeView === item.id
                        ? 'bg-primary-600 text-white'
                        : 'text-gray-300 hover:text-white hover:bg-gray-700'
                    }`}
                  >
                    <IconComponent size={16} />
                    {item.label}
                  </button>
                );
              })}
            </div>
          </div>
          
          <div className="flex-1 overflow-y-auto">
            <div className="p-2">
              <h2 className="text-gray-400 text-xs uppercase font-semibold mb-2 px-2">Chat History</h2>
              <div className="space-y-1">
                {chatSessions.map(chat => (
                  <div 
                    key={chat.id}
                    className={`group flex items-center justify-between p-2 rounded-lg cursor-pointer transition-colors ${
                      chat.id === activeChatId 
                        ? 'bg-gray-800 text-white' 
                        : 'text-gray-300 hover:bg-gray-800/70'
                    }`}
                    onClick={() => {
                      setActiveChatId(chat.id);
                      setActiveView('chat');
                    }}
                  >
                    <div className="flex items-center gap-2 overflow-hidden">
                      <MessageSquare size={16} />
                      <span className="text-sm truncate">{chat.title}</span>
                    </div>
                    <div className={`opacity-0 group-hover:opacity-100 transition-opacity`}>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          startTitleEdit(chat.id);
                        }}
                        className="p-1 text-gray-400 hover:text-white transition-colors"
                      >
                        <Edit2 size={14} />
                      </button>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          deleteChat(chat.id);
                        }}
                        className="p-1 text-gray-400 hover:text-red-500 transition-colors"
                      >
                        <Trash2 size={14} />
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
          
          <div className="p-4 border-t border-gray-800 text-gray-400 text-xs">
            <p>Your chats are stored locally in this browser</p>
          </div>
        </div>
      </div>

      {/* Chat Sidebar Toggle */}
      <button
        onClick={() => setShowChatSidebar(!showChatSidebar)}
        className="fixed left-0 top-6 z-30 md:hidden bg-primary-600 text-white rounded-r-lg p-2"
      >
        {showChatSidebar ? <ArrowLeft size={20} /> : <MessageSquare size={20} />}
      </button>

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <header className="bg-white border-b border-gray-200 px-6 py-4 shadow-sm">
          <div className="max-w-4xl mx-auto flex justify-between items-center">
            {isEditingTitle && activeChat ? (
              <div className="flex items-center gap-2">
                <input
                  type="text"
                  value={editTitle}
                  onChange={(e) => setEditTitle(e.target.value)}
                  className="text-lg border-b border-gray-300 focus:outline-none focus:border-primary-500 font-medium"
                  autoFocus
                />
                <button 
                  onClick={saveTitle}
                  className="p-1 text-gray-600 hover:text-primary-600"
                >
                  <Save size={18} />
                </button>
              </div>
            ) : (
              <h1 className="text-xl font-semibold text-gray-800 flex items-center gap-2">
                <div className="w-8 h-8 bg-primary-600 rounded-full flex items-center justify-center">
                  <span className="text-white font-bold">AI</span>
                </div>
                <span className="truncate max-w-sm">
                  {activeView === 'chat' && activeChat ? activeChat.title : 
                   activeView === 'upload' ? 'Upload Documents' :
                   activeView === 'files' ? 'File Management' : 'Knowledge Assistant'}
                </span>
                {activeView === 'chat' && activeChat && (
                  <button 
                    onClick={() => startTitleEdit(activeChat.id)}
                    className="p-1 text-gray-400 hover:text-gray-600"
                  >
                    <Edit2 size={16} />
                  </button>
                )}
              </h1>
            )}
            
            {activeView === 'chat' && (
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
            )}
          </div>
        </header>

        {/* Main Content */}
        <div className="flex-1 overflow-hidden">
          {activeView === 'chat' && (
            <EnhancedChatInterface
              messages={messages}
              isLoading={isLoading}
              onSendMessage={handleSendMessage}
            />
          )}
          
          {activeView === 'upload' && (
            <div className="p-6 max-w-4xl mx-auto">
              <FileUpload
                onUploadComplete={handleUploadComplete}
                onFilesChange={handleFilesChange}
              />
            </div>
          )}
          
          {activeView === 'files' && (
            <div className="p-6 max-w-4xl mx-auto">
              <div className="bg-white rounded-lg shadow-sm border p-6">
                <h2 className="text-lg font-semibold text-gray-800 mb-4">File Management</h2>
                <p className="text-gray-600 mb-6">
                  Manage your uploaded documents and their processing status.
                </p>
                
                {uploadedFiles.length > 0 ? (
                  <div className="space-y-3">
                    {uploadedFiles.map((file) => (
                      <div key={file.docId} className="flex items-center justify-between p-4 border rounded-lg hover:bg-gray-50">
                        <div className="flex items-center gap-3">
                          {file.contentType.includes('excel') ? (
                            <div className="w-10 h-10 bg-green-100 rounded-lg flex items-center justify-center">
                              <span className="text-green-600 text-xs font-bold">XLS</span>
                            </div>
                          ) : (
                            <div className="w-10 h-10 bg-red-100 rounded-lg flex items-center justify-center">
                              <span className="text-red-600 text-xs font-bold">PDF</span>
                            </div>
                          )}
                          <div>
                            <p className="font-medium text-gray-900">{file.fileName}</p>
                            <p className="text-sm text-gray-500 capitalize">
                              {file.contentType.replace('_', ' ')}
                              {file.sheetCount && ` • ${file.sheetCount} sheets`}
                            </p>
                          </div>
                        </div>
                        <div className="flex items-center gap-2">
                          <span className="px-2 py-1 bg-green-100 text-green-800 text-xs rounded-full">
                            Processed
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-center py-8">
                    <Upload className="mx-auto text-gray-300 mb-4" size={48} />
                    <p className="text-gray-500">No files uploaded yet</p>
                    <button
                      onClick={() => setActiveView('upload')}
                      className="mt-4 bg-primary-600 text-white px-4 py-2 rounded-lg hover:bg-primary-700 transition-colors"
                    >
                      Upload Your First File
                    </button>
                  </div>
                )}
              </div>
            </div>
          )}
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
                    {preview.page && <span>Page {preview.page}</span>}
                    {preview.sheetName && <span>Sheet: {preview.sheetName}</span>}
                    {preview.tableName && <span>Table: {preview.tableName}</span>}
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
                        alt={preview.title || `Preview from ${preview.contentType}`}
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