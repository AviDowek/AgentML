import { useState, useEffect } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { useNavigate } from 'react-router-dom';
import {
  getAdminStats,
  getAdminUsers,
  getAdminLogs,
  getAdminAgentRunLogs,
  type PlatformStats,
  type AdminUser,
  type ActivityLog,
} from '../services/api';

export default function Admin() {
  const { user } = useAuth();
  const navigate = useNavigate();
  const [stats, setStats] = useState<PlatformStats | null>(null);
  const [users, setUsers] = useState<AdminUser[]>([]);
  const [logs, setLogs] = useState<ActivityLog[]>([]);
  const [agentLogs, setAgentLogs] = useState<ActivityLog[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'overview' | 'users' | 'experiments' | 'agents'>('overview');
  const [userSearch, setUserSearch] = useState('');
  const [logDays, setLogDays] = useState(7);

  useEffect(() => {
    if (user && !user.is_admin) {
      navigate('/');
    }
  }, [user, navigate]);

  useEffect(() => {
    loadData();
  }, []);

  useEffect(() => {
    if (activeTab === 'experiments') {
      loadLogs();
    } else if (activeTab === 'agents') {
      loadAgentLogs();
    }
  }, [activeTab, logDays]);

  const loadData = async () => {
    try {
      setLoading(true);
      const [statsData, usersData, logsData] = await Promise.all([
        getAdminStats(),
        getAdminUsers(),
        getAdminLogs(7),
      ]);
      setStats(statsData);
      setUsers(usersData);
      setLogs(logsData);
    } catch (err: any) {
      setError(err.message || 'Failed to load admin data');
    } finally {
      setLoading(false);
    }
  };

  const loadLogs = async () => {
    try {
      const data = await getAdminLogs(logDays);
      setLogs(data);
    } catch {}
  };

  const loadAgentLogs = async () => {
    try {
      const data = await getAdminAgentRunLogs(logDays);
      setAgentLogs(data);
    } catch {}
  };

  const searchUsers = async () => {
    try {
      const data = await getAdminUsers(userSearch || undefined);
      setUsers(data);
    } catch {}
  };

  useEffect(() => {
    const timer = setTimeout(() => {
      if (activeTab === 'users') searchUsers();
    }, 300);
    return () => clearTimeout(timer);
  }, [userSearch]);

  const formatDate = (d: string | null) => {
    if (!d) return '—';
    return new Date(d).toLocaleDateString('en-US', {
      month: 'short', day: 'numeric', year: 'numeric',
      hour: '2-digit', minute: '2-digit',
    });
  };

  const statusColor = (status: string | null) => {
    if (!status) return '#888';
    const colors: Record<string, string> = {
      completed: '#22c55e', running: '#3b82f6', failed: '#ef4444',
      cancelled: '#f59e0b', pending: '#888',
    };
    return colors[status] || '#888';
  };

  if (loading) return <div style={{ padding: 40, textAlign: 'center' }}>Loading admin dashboard...</div>;
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>;

  return (
    <div style={{ padding: '24px 32px', maxWidth: 1400, margin: '0 auto' }}>
      <h1 style={{ fontSize: 24, fontWeight: 700, marginBottom: 4 }}>Admin Dashboard</h1>
      <p style={{ color: '#888', marginBottom: 24, fontSize: 14 }}>Platform analytics and management</p>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 0, borderBottom: '1px solid #333', marginBottom: 24 }}>
        {(['overview', 'users', 'experiments', 'agents'] as const).map(tab => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            style={{
              padding: '10px 20px',
              background: 'none',
              border: 'none',
              borderBottom: activeTab === tab ? '2px solid #3b82f6' : '2px solid transparent',
              color: activeTab === tab ? '#fff' : '#888',
              cursor: 'pointer',
              fontSize: 14,
              fontWeight: activeTab === tab ? 600 : 400,
              textTransform: 'capitalize',
            }}
          >
            {tab === 'agents' ? 'Agent Runs' : tab}
          </button>
        ))}
      </div>

      {/* Overview Tab */}
      {activeTab === 'overview' && stats && (
        <div>
          {/* Stat Cards */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 16, marginBottom: 32 }}>
            <StatCard label="Total Users" value={stats.total_users} sub={`+${stats.new_users_7d} this week`} />
            <StatCard label="Active Users (7d)" value={stats.active_users_7d} sub={`${stats.active_users_30d} in 30d`} />
            <StatCard label="Total Projects" value={stats.total_projects} />
            <StatCard label="Total Experiments" value={stats.total_experiments} sub={`+${stats.experiments_last_7d} this week`} />
            <StatCard label="Agent Runs" value={stats.total_agent_runs} />
            <StatCard label="Auto DS Sessions" value={stats.total_auto_ds_sessions} />
          </div>

          {/* Experiment Status Breakdown */}
          <div style={{ background: '#1a1a2e', borderRadius: 8, padding: 20, marginBottom: 24 }}>
            <h3 style={{ fontSize: 16, fontWeight: 600, marginBottom: 16 }}>Experiments by Status</h3>
            <div style={{ display: 'flex', gap: 24, flexWrap: 'wrap' }}>
              {Object.entries(stats.experiments_by_status).map(([status, count]) => (
                <div key={status} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <span style={{
                    width: 10, height: 10, borderRadius: '50%',
                    background: statusColor(status), display: 'inline-block',
                  }} />
                  <span style={{ color: '#ccc', fontSize: 14 }}>{status}: <strong>{count}</strong></span>
                </div>
              ))}
            </div>
          </div>

          {/* Recent Activity Preview */}
          <div style={{ background: '#1a1a2e', borderRadius: 8, padding: 20 }}>
            <h3 style={{ fontSize: 16, fontWeight: 600, marginBottom: 16 }}>Recent Experiments</h3>
            <LogTable logs={logs.slice(0, 10)} formatDate={formatDate} statusColor={statusColor} />
          </div>
        </div>
      )}

      {/* Users Tab */}
      {activeTab === 'users' && (
        <div>
          <div style={{ marginBottom: 16 }}>
            <input
              type="text"
              placeholder="Search users by email or name..."
              value={userSearch}
              onChange={e => setUserSearch(e.target.value)}
              style={{
                padding: '8px 14px', width: 350, background: '#1a1a2e',
                border: '1px solid #333', borderRadius: 6, color: '#fff', fontSize: 14,
              }}
            />
            <span style={{ marginLeft: 12, color: '#888', fontSize: 13 }}>{users.length} users</span>
          </div>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '1px solid #333', color: '#888', textAlign: 'left' }}>
                  <th style={{ padding: '10px 12px' }}>Email</th>
                  <th style={{ padding: '10px 12px' }}>Name</th>
                  <th style={{ padding: '10px 12px' }}>Projects</th>
                  <th style={{ padding: '10px 12px' }}>Experiments</th>
                  <th style={{ padding: '10px 12px' }}>Signed Up</th>
                  <th style={{ padding: '10px 12px' }}>Last Active</th>
                  <th style={{ padding: '10px 12px' }}>Status</th>
                </tr>
              </thead>
              <tbody>
                {users.map(u => (
                  <tr key={u.id} style={{ borderBottom: '1px solid #222' }}>
                    <td style={{ padding: '10px 12px', color: '#fff' }}>
                      {u.email}
                      {u.is_admin && <span style={{ marginLeft: 6, fontSize: 11, color: '#f59e0b', background: '#f59e0b22', padding: '1px 6px', borderRadius: 3 }}>ADMIN</span>}
                    </td>
                    <td style={{ padding: '10px 12px', color: '#ccc' }}>{u.full_name || '—'}</td>
                    <td style={{ padding: '10px 12px', color: '#ccc' }}>{u.project_count}</td>
                    <td style={{ padding: '10px 12px', color: '#ccc' }}>{u.experiment_count}</td>
                    <td style={{ padding: '10px 12px', color: '#888', fontSize: 12 }}>{formatDate(u.created_at)}</td>
                    <td style={{ padding: '10px 12px', color: '#888', fontSize: 12 }}>{formatDate(u.last_activity)}</td>
                    <td style={{ padding: '10px 12px' }}>
                      <span style={{
                        fontSize: 11, padding: '2px 8px', borderRadius: 3,
                        background: u.is_active ? '#22c55e22' : '#ef444422',
                        color: u.is_active ? '#22c55e' : '#ef4444',
                      }}>
                        {u.is_active ? 'Active' : 'Inactive'}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Experiments Tab */}
      {activeTab === 'experiments' && (
        <div>
          <div style={{ marginBottom: 16, display: 'flex', gap: 8, alignItems: 'center' }}>
            <span style={{ color: '#888', fontSize: 13 }}>Show last</span>
            {[7, 14, 30, 90].map(d => (
              <button
                key={d}
                onClick={() => setLogDays(d)}
                style={{
                  padding: '4px 12px', borderRadius: 4, border: 'none', cursor: 'pointer',
                  background: logDays === d ? '#3b82f6' : '#1a1a2e', color: logDays === d ? '#fff' : '#888',
                  fontSize: 13,
                }}
              >
                {d}d
              </button>
            ))}
            <span style={{ marginLeft: 12, color: '#888', fontSize: 13 }}>{logs.length} entries</span>
          </div>
          <LogTable logs={logs} formatDate={formatDate} statusColor={statusColor} />
        </div>
      )}

      {/* Agent Runs Tab */}
      {activeTab === 'agents' && (
        <div>
          <div style={{ marginBottom: 16, display: 'flex', gap: 8, alignItems: 'center' }}>
            <span style={{ color: '#888', fontSize: 13 }}>Show last</span>
            {[7, 14, 30, 90].map(d => (
              <button
                key={d}
                onClick={() => setLogDays(d)}
                style={{
                  padding: '4px 12px', borderRadius: 4, border: 'none', cursor: 'pointer',
                  background: logDays === d ? '#3b82f6' : '#1a1a2e', color: logDays === d ? '#fff' : '#888',
                  fontSize: 13,
                }}
              >
                {d}d
              </button>
            ))}
            <span style={{ marginLeft: 12, color: '#888', fontSize: 13 }}>{agentLogs.length} entries</span>
          </div>
          <LogTable logs={agentLogs} formatDate={formatDate} statusColor={statusColor} />
        </div>
      )}
    </div>
  );
}

function StatCard({ label, value, sub }: { label: string; value: number; sub?: string }) {
  return (
    <div style={{
      background: '#1a1a2e', borderRadius: 8, padding: '18px 20px',
      border: '1px solid #2a2a4a',
    }}>
      <div style={{ color: '#888', fontSize: 12, marginBottom: 6, textTransform: 'uppercase', letterSpacing: 0.5 }}>{label}</div>
      <div style={{ fontSize: 28, fontWeight: 700, color: '#fff' }}>{value.toLocaleString()}</div>
      {sub && <div style={{ color: '#666', fontSize: 12, marginTop: 4 }}>{sub}</div>}
    </div>
  );
}

function LogTable({ logs, formatDate, statusColor }: {
  logs: ActivityLog[];
  formatDate: (d: string | null) => string;
  statusColor: (s: string | null) => string;
}) {
  if (logs.length === 0) {
    return <div style={{ color: '#666', padding: 20, textAlign: 'center' }}>No activity found</div>;
  }

  return (
    <div style={{ overflowX: 'auto' }}>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
        <thead>
          <tr style={{ borderBottom: '1px solid #333', color: '#888', textAlign: 'left' }}>
            <th style={{ padding: '8px 12px' }}>Time</th>
            <th style={{ padding: '8px 12px' }}>User</th>
            <th style={{ padding: '8px 12px' }}>Action</th>
            <th style={{ padding: '8px 12px' }}>Project</th>
            <th style={{ padding: '8px 12px' }}>Status</th>
            <th style={{ padding: '8px 12px' }}>Details</th>
          </tr>
        </thead>
        <tbody>
          {logs.map(log => (
            <tr key={log.id} style={{ borderBottom: '1px solid #222' }}>
              <td style={{ padding: '8px 12px', color: '#888', fontSize: 12, whiteSpace: 'nowrap' }}>{formatDate(log.timestamp)}</td>
              <td style={{ padding: '8px 12px', color: '#ccc' }}>{log.user_email || '—'}</td>
              <td style={{ padding: '8px 12px', color: '#fff' }}>{log.action}</td>
              <td style={{ padding: '8px 12px', color: '#888' }}>{log.project_name || '—'}</td>
              <td style={{ padding: '8px 12px' }}>
                {log.status && (
                  <span style={{
                    fontSize: 11, padding: '2px 8px', borderRadius: 3,
                    background: `${statusColor(log.status)}22`,
                    color: statusColor(log.status),
                  }}>
                    {log.status}
                  </span>
                )}
              </td>
              <td style={{ padding: '8px 12px', color: '#666', fontSize: 12, maxWidth: 300, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                {log.details || ''}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
