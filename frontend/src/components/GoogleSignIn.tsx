/**
 * Google Sign-In Button Component
 * Uses Google Identity Services (GIS) for OAuth
 */
import { useEffect, useRef, useState } from 'react';
import { useAuth } from '../contexts/AuthContext';

// Extend Window interface for Google GIS
declare global {
  interface Window {
    google?: {
      accounts: {
        id: {
          initialize: (config: {
            client_id: string;
            callback: (response: { credential: string }) => void;
            auto_select?: boolean;
          }) => void;
          renderButton: (
            element: HTMLElement,
            options: {
              theme?: 'outline' | 'filled_blue' | 'filled_black';
              size?: 'large' | 'medium' | 'small';
              type?: 'standard' | 'icon';
              text?: 'signin_with' | 'signup_with' | 'continue_with' | 'signin';
              width?: number;
            }
          ) => void;
          prompt: () => void;
        };
      };
    };
  }
}

interface GoogleSignInProps {
  onError?: (error: string) => void;
  onSuccess?: () => void;
}

// Get the Google Client ID from environment
const GOOGLE_CLIENT_ID = import.meta.env.VITE_GOOGLE_CLIENT_ID || '';

export default function GoogleSignIn({ onError, onSuccess }: GoogleSignInProps) {
  const buttonRef = useRef<HTMLDivElement>(null);
  const { googleLogin } = useAuth();
  const [isLoading, setIsLoading] = useState(false);
  const [isConfigured, setIsConfigured] = useState(false);

  useEffect(() => {
    // Check if Google Client ID is configured
    if (!GOOGLE_CLIENT_ID) {
      console.log('Google OAuth not configured - VITE_GOOGLE_CLIENT_ID not set');
      setIsConfigured(false);
      return;
    }

    setIsConfigured(true);

    // Wait for Google script to load
    const initGoogle = () => {
      if (!window.google?.accounts?.id || !buttonRef.current) {
        return;
      }

      window.google.accounts.id.initialize({
        client_id: GOOGLE_CLIENT_ID,
        callback: async (response) => {
          if (response.credential) {
            setIsLoading(true);
            try {
              await googleLogin(response.credential);
              onSuccess?.();
            } catch (err) {
              console.error('Google login error:', err);
              onError?.(err instanceof Error ? err.message : 'Google sign-in failed');
            } finally {
              setIsLoading(false);
            }
          }
        },
      });

      window.google.accounts.id.renderButton(buttonRef.current, {
        theme: 'filled_blue',
        size: 'large',
        type: 'standard',
        text: 'continue_with',
        width: 350,
      });
    };

    // Check if script is already loaded
    if (window.google?.accounts?.id) {
      initGoogle();
    } else {
      // Wait for script to load
      const checkGoogle = setInterval(() => {
        if (window.google?.accounts?.id) {
          clearInterval(checkGoogle);
          initGoogle();
        }
      }, 100);

      // Cleanup after 10 seconds
      setTimeout(() => clearInterval(checkGoogle), 10000);
    }
  }, [googleLogin, onError, onSuccess]);

  if (!isConfigured) {
    return (
      <button
        type="button"
        className="btn btn-oauth btn-google"
        onClick={() => onError?.('Google Sign-In requires VITE_GOOGLE_CLIENT_ID to be configured in .env')}
        disabled={isLoading}
      >
        <svg viewBox="0 0 24 24" width="20" height="20">
          <path
            fill="#4285f4"
            d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"
          />
          <path
            fill="#34a853"
            d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
          />
          <path
            fill="#fbbc05"
            d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"
          />
          <path
            fill="#ea4335"
            d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"
          />
        </svg>
        Continue with Google
      </button>
    );
  }

  return (
    <div className="google-signin-container">
      {isLoading && <div className="google-signin-loading">Signing in...</div>}
      <div ref={buttonRef} style={{ display: isLoading ? 'none' : 'block' }} />
    </div>
  );
}
