import { Navigate, Route, Routes } from "react-router-dom";
import CallListPage from "./pages/CallListPage";
import LeadDetailPage from "./pages/LeadDetailPage";
import ModelPage from "./pages/ModelPage";

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<CallListPage />} />
      <Route path="/leads" element={<Navigate to="/" replace />} />
      <Route path="/leads/" element={<Navigate to="/" replace />} />
      <Route path="/leads/:leadId" element={<LeadDetailPage />} />
      <Route path="/model" element={<ModelPage />} />
      <Route
        path="*"
        element={
          <div className="min-h-screen flex items-center justify-center p-6 bg-gradient-to-br from-slate-50 via-white to-indigo-50">
            <div className="bg-white rounded-2xl shadow-xl border border-slate-200 p-10 text-center max-w-md w-full">
              <div className="text-6xl mb-4">🔍</div>
              <h1 className="text-2xl font-bold text-slate-900 mb-2">
                Page not found
              </h1>
              <p className="text-slate-500 mb-6">
                The page you requested doesn't exist or was moved.
              </p>
              <a
                href="/"
                className="inline-flex items-center justify-center w-full px-5 py-3 rounded-xl text-sm font-semibold text-white bg-gradient-to-r from-indigo-600 to-blue-600 shadow-lg shadow-indigo-500/25 hover:shadow-indigo-500/40 hover:from-indigo-700 hover:to-blue-700 transition"
              >
                ← Back to call list
              </a>
            </div>
          </div>
        }
      />
    </Routes>
  );
}
