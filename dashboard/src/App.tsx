import React, { useState } from 'react';
import { Play, Send, Code, Video, RefreshCw } from 'lucide-react';
import axios from 'axios';

const App: React.FC = () => {
  const [prompt, setPrompt] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<{ video_url: string; tsx_preview: string } | null>(null);
  const [error, setError] = useState('');

  const handleGenerate = async () => {
    if (!prompt) return;
    setLoading(true);
    setError('');
    try {
      const response = await axios.post('http://localhost:8000/generate', { prompt });
      setResult(response.data);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to generate explanation');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#000b1e] text-white font-sans p-8">
      <header className="flex items-center gap-4 mb-12">
        <div className="w-12 h-12 bg-[#00f2ff] rounded-lg flex items-center justify-center">
          <Video className="text-[#000b1e]" />
        </div>
        <h1 className="text-3xl font-bold tracking-tight">Nexus <span className="text-[#00f2ff]">Explainer</span></h1>
      </header>

      <main className="max-w-6xl mx-auto grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Left Column: Input */}
        <section className="space-y-6">
          <div className="bg-[#00152a] p-6 rounded-2xl border border-white/10">
            <label className="block text-sm font-medium text-gray-400 mb-2">Conceptual Prompt</label>
            <textarea
              className="w-full h-32 bg-black/30 border border-white/10 rounded-xl p-4 text-white focus:ring-2 focus:ring-[#00f2ff] outline-none transition-all resize-none"
              placeholder="e.g. Explain the Central Limit Theorem using a Galton Board simulation..."
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
            />
            <button
              onClick={handleGenerate}
              disabled={loading || !prompt}
              className="mt-4 w-full bg-[#00f2ff] hover:bg-[#00d1dc] disabled:opacity-50 text-[#000b1e] font-bold py-3 rounded-xl flex items-center justify-center gap-2 transition-all"
            >
              {loading ? <RefreshCw className="animate-spin" /> : <Send size={18} />}
              {loading ? 'Generating Logic...' : 'Generate Explanation'}
            </button>
          </div>

          {error && (
            <div className="bg-red-500/10 border border-red-500/50 text-red-500 p-4 rounded-xl text-sm">
              {error}
            </div>
          )}

          <div className="bg-[#00152a] p-6 rounded-2xl border border-white/10">
            <h3 className="flex items-center gap-2 text-gray-300 font-semibold mb-4">
              <Play size={18} /> Preview Control
            </h3>
            <p className="text-sm text-gray-400 mb-4">
              Once generated, you can view the video in the Remotion Preview or render the final MP4.
            </p>
            <div className="flex gap-4">
              <button className="flex-1 bg-white/10 hover:bg-white/20 py-2 rounded-lg text-sm transition-all">
                Open Preview
              </button>
              <button className="flex-1 bg-white/10 hover:bg-white/20 py-2 rounded-lg text-sm transition-all">
                Download MP4
              </button>
            </div>
          </div>
        </section>

        {/* Right Column: Code/Preview */}
        <section className="space-y-6">
          <div className="bg-[#00152a] p-6 rounded-2xl border border-white/10 h-full flex flex-col">
            <h3 className="flex items-center gap-2 text-gray-300 font-semibold mb-4">
              <Code size={18} /> Generated Remotion TSX
            </h3>
            <div className="flex-1 bg-black/50 rounded-xl p-4 font-mono text-xs overflow-auto border border-white/5">
              {result ? (
                <pre className="text-[#00f2ff]">
                  {result.tsx_preview}
                </pre>
              ) : (
                <div className="h-full flex items-center justify-center text-gray-600">
                  Awaiting generation...
                </div>
              )}
            </div>
          </div>
        </section>
      </main>
    </div>
  );
};

export default App;
