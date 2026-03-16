/**
 * Authentication Context
 * Provides auth state and methods throughout the app
 */
import {
  createContext,
  useContext,
  useState,
  useEffect,
  useCallback,
  type ReactNode,
} from 'react';
import type { User, UserCreate, UserLogin, GoogleAuthRequest } from '../types/api';
import * as api from '../services/api';

interface AuthContextType {
  user: User | null;
  isLoading: boolean;
  isAuthenticated: boolean;
  login: (data: UserLogin) => Promise<void>;
  signup: (data: UserCreate) => Promise<void>;
  googleLogin: (credential: string) => Promise<void>;
  logout: () => void;
  refreshUser: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  // Check for existing session on mount
  useEffect(() => {
    const checkAuth = async () => {
      const token = api.getAuthToken();
      if (token) {
        try {
          const currentUser = await api.getCurrentUser();
          setUser(currentUser);
        } catch {
          // Token invalid or expired
          api.clearAuthToken();
        }
      }
      setIsLoading(false);
    };
    checkAuth();
  }, []);

  const login = useCallback(async (data: UserLogin) => {
    const response = await api.login(data);
    api.setAuthToken(response.access_token);
    const currentUser = await api.getCurrentUser();
    setUser(currentUser);
  }, []);

  const signup = useCallback(async (data: UserCreate) => {
    // Create the account
    await api.signup(data);
    // Auto-login after signup
    const loginResponse = await api.login({
      email: data.email,
      password: data.password,
    });
    api.setAuthToken(loginResponse.access_token);
    const currentUser = await api.getCurrentUser();
    setUser(currentUser);
  }, []);

  const googleLogin = useCallback(async (credential: string) => {
    const response = await api.googleAuth({ credential } as GoogleAuthRequest);
    api.setAuthToken(response.access_token);
    const currentUser = await api.getCurrentUser();
    setUser(currentUser);
  }, []);

  const logout = useCallback(() => {
    api.logout();
    setUser(null);
  }, []);

  const refreshUser = useCallback(async () => {
    if (api.getAuthToken()) {
      try {
        const currentUser = await api.getCurrentUser();
        setUser(currentUser);
      } catch {
        api.clearAuthToken();
        setUser(null);
      }
    }
  }, []);

  const value: AuthContextType = {
    user,
    isLoading,
    isAuthenticated: !!user,
    login,
    signup,
    googleLogin,
    logout,
    refreshUser,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}
