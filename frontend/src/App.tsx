import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { AuthProvider } from './contexts/AuthContext';
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
            <Route index element={<Home />} />
            <Route path="projects" element={<Projects />} />
            <Route path="projects/:projectId" element={<ProjectDetail />} />
            <Route path="experiments" element={<Experiments />} />
            <Route path="experiments/:experimentId" element={<ExperimentDetail />} />
            <Route path="dataset-results/:datasetSpecId" element={<DatasetResults />} />
            <Route path="models" element={<Models />} />
            <Route path="models/:modelId" element={<ModelDetail />} />
            <Route path="auto-ds" element={<AutoDS />} />
            <Route path="auto-ds/:sessionId" element={<AutoDSDetail />} />
            <Route path="settings" element={<Settings />} />
            <Route path="login" element={<Login />} />
            <Route path="signup" element={<Signup />} />
            <Route path="accept-invite" element={<AcceptInvite />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </AuthProvider>
  );
}

export default App;
