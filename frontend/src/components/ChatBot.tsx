import { useState, useRef, useEffect, useCallback } from 'react';
import {
  listConversations,
  getConversation,
  createConversation,
  deleteConversation,
  sendConversationMessage,
  ApiException,
} from '../services/api';
import type { ConversationSummary, ConversationMessage, VisualizationForChat } from '../services/api';

interface ChatBotProps {
  context?: Record<string, unknown>;
  contextType?: string;
  title?: string;
  visualizations?: VisualizationForChat[]; // Current visualizations for AI to see
}

export default function ChatBot({ context, contextType, title = 'AI Assistant', visualizations }: ChatBotProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [showSidebar, setShowSidebar] = useState(false);
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [currentConversationId, setCurrentConversationId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ConversationMessage[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isLoadingConversations, setIsLoadingConversations] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Load conversations when sidebar is opened
  const loadConversations = useCallback(async () => {
    setIsLoadingConversations(true);
    try {
      const response = await listConversations(contextType);
      setConversations(response.items);
    } catch (err) {
      console.error('Failed to load conversations:', err);
    } finally {
      setIsLoadingConversations(false);
    }
  }, [contextType]);

  useEffect(() => {
    if (showSidebar) {
      loadConversations();
    }
  }, [showSidebar, loadConversations]);

  // Load a specific conversation
  const loadConversation = async (conversationId: string) => {
    try {
      const conversation = await getConversation(conversationId);
      setMessages(conversation.messages);
      setCurrentConversationId(conversationId);
      setShowSidebar(false);
      setError(null);
    } catch (err) {
      if (err instanceof ApiException) {
        setError(err.detail);
      }
    }
  };

  // Start a new conversation
  const startNewConversation = async () => {
    try {
      const conversation = await createConversation({
        title: 'New Conversation',
        context_type: contextType,
        context_data: context,
      });
      setCurrentConversationId(conversation.id);
      setMessages([]);
      setShowSidebar(false);
      setError(null);
      // Refresh the conversation list
      loadConversations();
    } catch (err) {
      if (err instanceof ApiException) {
        setError(err.detail);
      }
    }
  };

  // Delete a conversation
  const handleDeleteConversation = async (conversationId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (!confirm('Delete this conversation?')) return;

    try {
      await deleteConversation(conversationId);
      if (currentConversationId === conversationId) {
        setCurrentConversationId(null);
        setMessages([]);
      }
      loadConversations();
    } catch (err) {
      console.error('Failed to delete conversation:', err);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage = input.trim();
    setInput('');
    setError(null);

    // If no conversation exists, create one first
    let conversationId = currentConversationId;
    if (!conversationId) {
      try {
        const conversation = await createConversation({
          title: 'New Conversation',
          context_type: contextType,
          context_data: context,
        });
        conversationId = conversation.id;
        setCurrentConversationId(conversationId);
      } catch (err) {
        if (err instanceof ApiException) {
          setError(err.detail);
        } else {
          setError('Failed to create conversation.');
        }
        return;
      }
    }

    // Optimistically add user message
    const tempUserMsg: ConversationMessage = {
      id: 'temp-user',
      conversation_id: conversationId,
      role: 'user',
      content: userMessage,
      created_at: new Date().toISOString(),
    };
    setMessages((prev) => [...prev, tempUserMsg]);
    setIsLoading(true);

    try {
      const response = await sendConversationMessage(conversationId, userMessage, visualizations);

      // Replace temp message with actual messages
      setMessages((prev) => [
        ...prev.filter((m) => m.id !== 'temp-user'),
        response.user_message,
        response.assistant_message,
      ]);

      // Refresh conversations to update titles
      loadConversations();
    } catch (err) {
      // Remove temp message on error
      setMessages((prev) => prev.filter((m) => m.id !== 'temp-user'));
      if (err instanceof ApiException) {
        setError(err.detail);
      } else {
        setError('Failed to get response. Please try again.');
      }
    } finally {
      setIsLoading(false);
    }
  };

  const clearChat = () => {
    setMessages([]);
    setCurrentConversationId(null);
    setError(null);
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffHours = diffMs / (1000 * 60 * 60);

    if (diffHours < 24) {
      return date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
    } else if (diffHours < 48) {
      return 'Yesterday';
    } else {
      return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
    }
  };

  return (
    <div className="chatbot-container">
      {/* Toggle Button */}
      <button
        className="chatbot-toggle"
        onClick={() => setIsOpen(!isOpen)}
        title={isOpen ? 'Close chat' : 'Open AI Assistant'}
      >
        {isOpen ? (
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <line x1="18" y1="6" x2="6" y2="18"></line>
            <line x1="6" y1="6" x2="18" y2="18"></line>
          </svg>
        ) : (
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
          </svg>
        )}
      </button>

      {/* Chat Panel */}
      {isOpen && (
        <div className={`chatbot-panel ${showSidebar ? 'with-sidebar' : ''}`}>
          {/* Sidebar for conversation history */}
          {showSidebar && (
            <div className="chatbot-sidebar">
              <div className="chatbot-sidebar-header">
                <h5>Conversations</h5>
                <button
                  className="chatbot-sidebar-close"
                  onClick={() => setShowSidebar(false)}
                  title="Close sidebar"
                >
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <line x1="18" y1="6" x2="6" y2="18"></line>
                    <line x1="6" y1="6" x2="18" y2="18"></line>
                  </svg>
                </button>
              </div>

              <button className="chatbot-new-chat" onClick={startNewConversation}>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <line x1="12" y1="5" x2="12" y2="19"></line>
                  <line x1="5" y1="12" x2="19" y2="12"></line>
                </svg>
                New Chat
              </button>

              <div className="chatbot-conversation-list">
                {isLoadingConversations ? (
                  <div className="chatbot-loading-conversations">Loading...</div>
                ) : conversations.length === 0 ? (
                  <div className="chatbot-no-conversations">No previous conversations</div>
                ) : (
                  conversations.map((conv) => (
                    <div
                      key={conv.id}
                      className={`chatbot-conversation-item ${
                        currentConversationId === conv.id ? 'active' : ''
                      }`}
                      onClick={() => loadConversation(conv.id)}
                    >
                      <div className="chatbot-conversation-title">{conv.title}</div>
                      <div className="chatbot-conversation-meta">
                        <span>{conv.message_count} messages</span>
                        <span>{formatDate(conv.updated_at)}</span>
                      </div>
                      <button
                        className="chatbot-conversation-delete"
                        onClick={(e) => handleDeleteConversation(conv.id, e)}
                        title="Delete conversation"
                      >
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                          <polyline points="3 6 5 6 21 6"></polyline>
                          <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                        </svg>
                      </button>
                    </div>
                  ))
                )}
              </div>
            </div>
          )}

          {/* Main chat area */}
          <div className="chatbot-main">
            <div className="chatbot-header">
              <div className="chatbot-header-left">
                <button
                  className="chatbot-menu-btn"
                  onClick={() => setShowSidebar(!showSidebar)}
                  title="Conversation history"
                >
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <line x1="3" y1="12" x2="21" y2="12"></line>
                    <line x1="3" y1="6" x2="21" y2="6"></line>
                    <line x1="3" y1="18" x2="21" y2="18"></line>
                  </svg>
                </button>
                <h4>{title}</h4>
              </div>
              <div className="chatbot-header-actions">
                <button onClick={clearChat} className="chatbot-clear" title="New chat">
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <line x1="12" y1="5" x2="12" y2="19"></line>
                    <line x1="5" y1="12" x2="19" y2="12"></line>
                  </svg>
                </button>
              </div>
            </div>

            <div className="chatbot-messages">
              {messages.length === 0 && (
                <div className="chatbot-welcome">
                  <p>Hi! I'm your AI assistant. I can help you understand:</p>
                  <ul>
                    <li>AI pipeline analysis and recommendations</li>
                    <li>Your data visualizations and charts</li>
                    <li>Experiment results and metrics</li>
                    <li>Model performance comparisons</li>
                    <li>Feature importance analysis</li>
                    <li>Best practices and recommendations</li>
                  </ul>
                  <p>Ask me anything about your data, visualizations, or AI insights!</p>
                </div>
              )}

              {messages.map((msg) => (
                <div key={msg.id} className={`chatbot-message ${msg.role}`}>
                  <div className="message-content">{msg.content}</div>
                </div>
              ))}

              {isLoading && (
                <div className="chatbot-message assistant">
                  <div className="message-content typing">
                    <span></span>
                    <span></span>
                    <span></span>
                  </div>
                </div>
              )}

              {error && <div className="chatbot-error">{error}</div>}

              <div ref={messagesEndRef} />
            </div>

            <form onSubmit={handleSubmit} className="chatbot-input-form">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Ask a question..."
                disabled={isLoading}
                className="chatbot-input"
              />
              <button
                type="submit"
                disabled={isLoading || !input.trim()}
                className="chatbot-send"
              >
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <line x1="22" y1="2" x2="11" y2="13"></line>
                  <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
                </svg>
              </button>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
