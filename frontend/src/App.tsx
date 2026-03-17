import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { AuthProvider } from './contexts/AuthContext';
import ProtectedRoute from './components/ProtectedRoute';
import MainLayout from './layouts/MainLayout';
import Home from './pages/Home';
import Projects from './pages/Projects';
import ProjectDetail from './pages/ProjectDetail';
import Experiments from './pages/Experiments';
import ExperimentDetail from './pages/ExperimentDetail';
import DatasetResults from './pages/DatasetResults';
import Models from './pages/Models';
import ModelDetail from './pages/ModelDetail';
import AutoDS from './pages/AutoDS';
import AutoDSDetail from './pages/AutoDSDetail';
import Settings from './pages/Settings';
import Login from './pages/Login';
import Signup from './pages/Signup';
import AcceptInvite from './pages/AcceptInvite';
import './App.css';

function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<MainLayout />}>
            {/* Public routes */}
            <Route index element={<Home />} />
            <Route path="login" element={<Login />} />
            <Route path="signup" element={<Signup />} />
            <Route path="accept-invite" element={<AcceptInvite />} />

            {/* Protected routes - require authentication */}
            <Route path="projects" element={<ProtectedRoute><Projects /></ProtectedRoute>} />
            <Route path="projects/:projectId" element={<ProtectedRoute><ProjectDetail /></ProtectedRoute>} />
            <Route path="experiments" element={<ProtectedRoute><Experiments /></ProtectedRoute>} />
            <Route path="experiments/:experimentId" element={<ProtectedRoute><ExperimentDetail /></ProtectedRoute>} />
            <Route path="dataset-results/:datasetSpecId" element={<ProtectedRoute><DatasetResults /></ProtectedRoute>} />
            <Route path="models" element={<ProtectedRoute><Models /></ProtectedRoute>} />
            <Route path="models/:modelId" element={<ProtectedRoute><ModelDetail /></ProtectedRoute>} />
            <Route path="auto-ds" element={<ProtectedRoute><AutoDS /></ProtectedRoute>} />
            <Route path="auto-ds/:sessionId" element={<ProtectedRoute><AutoDSDetail /></ProtectedRoute>} />
            <Route path="settings" element={<ProtectedRoute><Settings /></ProtectedRoute>} />
          </Route>
        </Routes>
      </BrowserRouter>
    </AuthProvider>
  );
}

export default App;
