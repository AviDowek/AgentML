import { useState, useEffect, useRef, useCallback } from 'react';

interface TrainingLogEntry {
  timestamp: string;
  raw_log: string;
  interpreted: string | null;
  log_type: 'info' | 'progress' | 'warning' | 'error' | 'milestone' | 'model_start' | 'model_complete';
}

interface TrainingLogStreamProps {
  experimentId: string;
  isRunning: boolean;
}

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8001';

export function TrainingLogStream({ experimentId, isRunning }: TrainingLogStreamProps) {
  const [logs, setLogs] = useState<TrainingLogEntry[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [showRaw, setShowRaw] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);
  const nextIndexRef = useRef(0);
  const pollIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const lastExperimentIdRef = useRef<string | null>(null);
  // Use ref for auto-scroll to avoid state update race conditions
  const autoScrollRef = useRef(true);
  const [showScrollButton, setShowScrollButton] = useState(false);

  // Auto-scroll to bottom when new logs arrive (within container only)
  useEffect(() => {
    if (autoScrollRef.current && containerRef.current) {
      // Use requestAnimationFrame to ensure DOM is updated before scrolling
      requestAnimationFrame(() => {
        if (containerRef.current && autoScrollRef.current) {
          containerRef.current.scrollTop = containerRef.current.scrollHeight;
        }
      });
    }
  }, [logs]);

  // Detect manual scroll to disable auto-scroll
  const handleScroll = useCallback(() => {
    if (!containerRef.current) return;
    const { scrollTop, scrollHeight, clientHeight } = containerRef.current;
    const isAtBottom = scrollHeight - scrollTop - clientHeight < 50;
    autoScrollRef.current = isAtBottom;
    setShowScrollButton(!isAtBottom);
  }, []);

  // Start/stop polling based on isRunning
  useEffect(() => {
    // Stop polling if not running
    if (!isRunning || !experimentId) {
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
        pollIntervalRef.current = null;
      }
      return;
    }

    // Only reset if experiment ID changed (not on every re-render)
    if (lastExperimentIdRef.current !== experimentId) {
      lastExperimentIdRef.current = experimentId;
      nextIndexRef.current = 0;
      setLogs([]);
    }

    // Don't restart if already polling
    if (pollIntervalRef.current) {
      return;
    }

    // Fetch immediately
    const doFetch = async () => {
      if (!experimentId) return;
      try {
        const response = await fetch(
          `${API_URL}/training/${experimentId}/logs?start_index=${nextIndexRef.current}`
        );
        if (!response.ok) {
          throw new Error(`Failed to fetch logs: ${response.status}`);
        }
        const data = await response.json();
        if (data.logs && data.logs.length > 0) {
          // Add all logs that have interpretations
          // The backend now interprets logs before returning them
          const newLogs = data.logs.filter((log: TrainingLogEntry) => log.interpreted);
          if (newLogs.length > 0) {
            setLogs((prev) => {
              // Dedupe by timestamp+raw_log to avoid duplicates
              const existingKeys = new Set(prev.map(l => `${l.timestamp}:${l.raw_log}`));
              const uniqueNew = newLogs.filter(
                (l: TrainingLogEntry) => !existingKeys.has(`${l.timestamp}:${l.raw_log}`)
              );
              if (uniqueNew.length === 0) return prev;
              return [...prev.slice(-500), ...uniqueNew];
            });
          }
          // Always advance the index to avoid re-fetching the same logs
          nextIndexRef.current = data.next_index;
        }
        setError(null);
      } catch (e) {
        console.error('Failed to fetch training logs:', e);
        setError(e instanceof Error ? e.message : 'Failed to fetch logs');
      }
    };

    doFetch();

    // Poll every 1 second
    pollIntervalRef.current = setInterval(doFetch, 1000);

    return () => {
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
        pollIntervalRef.current = null;
      }
    };
  }, [experimentId, isRunning]);

  const getLogIcon = (logType: string) => {
    switch (logType) {
      case 'milestone':
        return '🏆';
      case 'model_start':
        return '🚀';
      case 'model_complete':
        return '✅';
      case 'progress':
        return '📈';
      case 'warning':
        return '⚠️';
      case 'error':
        return '❌';
      default:
        return '📋';
    }
  };

  const getLogColor = (logType: string) => {
    switch (logType) {
      case 'milestone':
        return '#059669'; // green
      case 'model_complete':
        return '#10b981'; // emerald
      case 'model_start':
        return '#3b82f6'; // blue
      case 'progress':
        return '#6366f1'; // indigo
      case 'warning':
        return '#f59e0b'; // amber
      case 'error':
        return '#ef4444'; // red
      default:
        return '#6b7280'; // gray
    }
  };

  if (!isRunning && logs.length === 0) {
    return null;
  }

  return (
    <div className="training-log-stream" style={{ marginTop: '1.5rem' }}>
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: '0.75rem',
      }}>
        <h4 style={{ margin: 0, display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          Training Logs
          {isRunning && (
            <span style={{
              width: '8px',
              height: '8px',
              borderRadius: '50%',
              backgroundColor: '#10b981',
              animation: 'pulse 2s infinite',
            }} title="Live" />
          )}
        </h4>
        <div style={{ display: 'flex', gap: '0.5rem' }}>
          <button
            onClick={() => setShowRaw(!showRaw)}
            style={{
              padding: '0.25rem 0.5rem',
              fontSize: '0.75rem',
              borderRadius: '4px',
              border: '1px solid #d1d5db',
              background: showRaw ? '#e5e7eb' : 'white',
              cursor: 'pointer',
            }}
          >
            {showRaw ? 'Show Interpreted' : 'Show Raw'}
          </button>
          <button
            onClick={() => setLogs([])}
            style={{
              padding: '0.25rem 0.5rem',
              fontSize: '0.75rem',
              borderRadius: '4px',
              border: '1px solid #d1d5db',
              background: 'white',
              cursor: 'pointer',
            }}
          >
            Clear
          </button>
        </div>
      </div>

      {error && (
        <div style={{
          padding: '0.5rem',
          marginBottom: '0.5rem',
          backgroundColor: '#fef2f2',
          border: '1px solid #fecaca',
          borderRadius: '4px',
          color: '#991b1b',
          fontSize: '0.875rem',
        }}>
          {error}
        </div>
      )}

      <div
        ref={containerRef}
        onScroll={handleScroll}
        style={{
          backgroundColor: '#1f2937',
          borderRadius: '8px',
          padding: '1rem',
          maxHeight: '400px',
          overflowY: 'auto',
          fontFamily: 'Monaco, Consolas, "Courier New", monospace',
          fontSize: '0.8125rem',
          lineHeight: '1.6',
        }}
      >
        {logs.length === 0 ? (
          <div style={{ color: '#9ca3af', textAlign: 'center', padding: '2rem' }}>
            {isRunning ? 'Waiting for training logs...' : 'No logs captured'}
          </div>
        ) : (
          logs.map((log, index) => (
            <div
              key={index}
              style={{
                display: 'flex',
                gap: '0.5rem',
                marginBottom: '0.25rem',
                color: getLogColor(log.log_type),
              }}
            >
              <span style={{ flexShrink: 0 }}>{getLogIcon(log.log_type)}</span>
              <span style={{ color: '#9ca3af', flexShrink: 0 }}>
                {new Date(log.timestamp).toLocaleTimeString()}
              </span>
              <span style={{ color: showRaw ? '#d1d5db' : 'inherit' }}>
                {showRaw ? log.raw_log : log.interpreted}
              </span>
            </div>
          ))
        )}
      </div>

      {showScrollButton && logs.length > 0 && (
        <button
          onClick={() => {
            autoScrollRef.current = true;
            setShowScrollButton(false);
            if (containerRef.current) {
              containerRef.current.scrollTop = containerRef.current.scrollHeight;
            }
          }}
          style={{
            marginTop: '0.5rem',
            padding: '0.25rem 0.75rem',
            fontSize: '0.75rem',
            borderRadius: '4px',
            border: 'none',
            background: '#3b82f6',
            color: 'white',
            cursor: 'pointer',
          }}
        >
          Scroll to bottom
        </button>
      )}

      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.5; }
        }
      `}</style>
    </div>
  );
}

export default TrainingLogStream;
