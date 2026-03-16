/**
 * Accept Invite Page
 * Handles accepting sharing invitations from email links
 */
import { useState, useEffect } from 'react';
import { useNavigate, useSearchParams, Link } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import * as api from '../services/api';
import { ApiException } from '../services/api';

export default function AcceptInvite() {
  const [searchParams] = useSearchParams();
  const token = searchParams.get('token');
  const navigate = useNavigate();
  const { isAuthenticated, isLoading: authLoading } = useAuth();

  const [status, setStatus] = useState<'loading' | 'success' | 'error' | 'login-required'>('loading');
  const [message, setMessage] = useState('');
  const [resourceType, setResourceType] = useState<string | null>(null);
  const [resourceId, setResourceId] = useState<string | null>(null);

  useEffect(() => {
    if (!token) {
      setStatus('error');
      setMessage('Invalid invitation link. No token provided.');
      return;
    }

    if (authLoading) {
      return;
    }

    if (!isAuthenticated) {
      setStatus('login-required');
      setMessage('Please log in to accept this invitation.');
      return;
    }

    // Try to accept the invitation
    const acceptInvitation = async () => {
      try {
        const result = await api.acceptInvite(token);
        setStatus('success');
        setMessage(result.message);
        setResourceType(result.resource_type);
        setResourceId(result.resource_id);
      } catch (err) {
        setStatus('error');
        if (err instanceof ApiException) {
          setMessage(err.detail);
        } else {
          setMessage('Failed to accept invitation. Please try again.');
        }
      }
    };

    acceptInvitation();
  }, [token, isAuthenticated, authLoading]);

  const handleNavigateToResource = () => {
    if (resourceType === 'project' && resourceId) {
      navigate(`/projects/${resourceId}`);
    } else {
      navigate('/');
    }
  };

  if (authLoading || status === 'loading') {
    return (
      <div className="auth-page">
        <div className="auth-container">
          <h1>Accepting Invitation...</h1>
          <div className="loading-spinner">Loading...</div>
        </div>
      </div>
    );
  }

  if (status === 'login-required') {
    return (
      <div className="auth-page">
        <div className="auth-container">
          <h1>Login Required</h1>
          <p className="auth-subtitle">{message}</p>
          <div className="auth-actions">
            <Link
              to={`/login?redirect=${encodeURIComponent(`/accept-invite?token=${token}`)}`}
              className="btn btn-primary"
            >
              Log In
            </Link>
            <Link
              to={`/signup?redirect=${encodeURIComponent(`/accept-invite?token=${token}`)}`}
              className="btn btn-secondary"
            >
              Sign Up
            </Link>
          </div>
        </div>
      </div>
    );
  }

  if (status === 'error') {
    return (
      <div className="auth-page">
        <div className="auth-container">
          <h1>Invitation Error</h1>
          <div className="auth-error">{message}</div>
          <Link to="/" className="btn btn-primary">
            Go to Home
          </Link>
        </div>
      </div>
    );
  }

  return (
    <div className="auth-page">
      <div className="auth-container">
        <h1>Invitation Accepted!</h1>
        <div className="auth-success">{message}</div>
        <p className="auth-subtitle">
          You now have access to the shared project.
        </p>
        <button onClick={handleNavigateToResource} className="btn btn-primary">
          View Project
        </button>
      </div>
    </div>
  );
}
