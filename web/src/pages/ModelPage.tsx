import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { getModelCard } from "../api";
import type { ModelCardResponse } from "../types";

export default function ModelPage() {
  const [card, setCard] = useState<ModelCardResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      setLoading(true);
      setError(null);
      try {
        const c = await getModelCard();
        if (!cancelled) setCard(c);
      } catch (e: any) {
        if (!cancelled) {
          setCard(null);
          setError(e?.message || "Couldn't load the model card.");
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    load();
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <div className="min-h-screen">
      <header className="header-gradient sticky top-0 z-20">
        <div className="max-w-5xl mx-auto px-3 md:px-6 py-3 md:py-4 flex flex-wrap items-center gap-3">
          <Link to="/" className="chip chip-soft hover:bg-slate-100">
            ← Call list
          </Link>
          <div className="flex items-center gap-2 ml-2">
            <div
              className="w-8 h-8 rounded-xl flex items-center justify-center text-white font-black
                bg-gradient-to-br from-indigo-600 via-blue-600 to-sky-500 shadow-md shadow-indigo-500/30"
            >
              L
            </div>
            <span className="brand-title hidden sm:block">LeadIQ</span>
          </div>
        </div>
      </header>

      <main className="max-w-5xl mx-auto px-3 md:px-6 py-5 md:py-7 space-y-5">
        {loading ? (
          <div className="card p-10 flex flex-col items-center justify-center gap-4">
            <div className="w-14 h-14 rounded-2xl border-4 border-indigo-500 border-t-transparent animate-spin" />
            <div className="text-slate-600 font-medium">Loading model card…</div>
          </div>
        ) : (
          <>
            {error && (
              <div className="card border border-rose-200 bg-rose-50/60 p-5 text-sm text-rose-700">
                <b className="text-rose-800 block mb-1">
                  Couldn't reach the model card API
                </b>
                <code className="block break-all text-xs bg-white/60 p-2 rounded mt-1 border border-rose-200">
                  {error}
                </code>
                <p className="mt-2 text-xs text-rose-600">
                  Make sure the scoring API is running, then retry.
                </p>
              </div>
            )}

            {card ? (
              <>
                {/* HERO */}
                <section className="card p-6 md:p-7 relative overflow-hidden">
                  <div className="absolute -right-16 -top-20 w-72 h-72 rounded-full bg-gradient-to-br from-indigo-400 to-sky-400 opacity-10 blur-3xl" />
                  <div className="relative flex flex-wrap items-end justify-between gap-4">
                    <div className="min-w-0">
                      <div className="text-[11px] uppercase tracking-widest font-bold text-slate-400 mb-1">
                        Champion model
                      </div>
                      <h1 className="text-3xl md:text-4xl font-extrabold text-slate-900 break-all">
                        {card.version}
                      </h1>
                      <div className="mt-2 flex flex-wrap items-center gap-2">
                        <span className="chip chip-soft-active">
                          🏆 {card.status}
                        </span>
                        {card.feature_list.length > 0 && (
                          <span className="chip chip-soft font-mono">
                            {card.feature_list.length} features
                          </span>
                        )}
                        {card.created_at && (
                          <span className="chip chip-soft">
                            🗓️{" "}
                            {new Date(card.created_at).toLocaleDateString()}
                          </span>
                        )}
                      </div>
                      {card.training_window && (
                        <div className="mt-3 text-sm text-slate-500 font-mono break-all">
                          Training window: {card.training_window}
                        </div>
                      )}
                    </div>
                    <div className="text-left md:text-right">
                      <div className="text-[11px] uppercase tracking-widest font-bold text-slate-400 mb-1">
                        Primary quality
                      </div>
                      <div className="flex flex-wrap gap-3 md:gap-5 items-end">
                        <MetricMini
                          label="PR-AUC"
                          value={fmt(card.metrics.pr_auc)}
                        />
                        <MetricMini
                          label="ECE"
                          value={fmt(card.metrics.ece_10bin)}
                          warn={
                            card.metrics.ece_10bin != null &&
                            card.metrics.ece_10bin > 0.05
                          }
                        />
                      </div>
                    </div>
                  </div>
                </section>

                {/* METRICS */}
                <section className="grid grid-cols-2 md:grid-cols-4 gap-3 md:gap-4">
                  <Metric
                    label="PR-AUC"
                    value={fmt(card.metrics.pr_auc)}
                    desc="Precision–recall AUC (key metric for imbalanced admissions)"
                    accent="from-indigo-500 to-blue-500"
                    icon="🎯"
                  />
                  <Metric
                    label="ECE (10 bins)"
                    value={fmt(card.metrics.ece_10bin)}
                    desc="Expected calibration error — lower is better"
                    accent="from-amber-400 to-orange-500"
                    icon="🎚️"
                    warn={
                      card.metrics.ece_10bin != null &&
                      card.metrics.ece_10bin > 0.05
                    }
                  />
                  <Metric
                    label="Brier score"
                    value={fmt(card.metrics.brier_score)}
                    desc="Mean squared error of probabilities"
                    accent="from-emerald-400 to-green-600"
                    icon="📏"
                  />
                  <Metric
                    label="ROC-AUC"
                    value={fmt(card.metrics.roc_auc)}
                    desc="Overall ranking quality"
                    accent="from-rose-500 to-red-500"
                    icon="📈"
                  />
                </section>

                {/* PROVENANCE */}
                <section className="card p-5 md:p-6">
                  <h2 className="text-lg font-bold text-slate-900 mb-3">
                    Provenance
                  </h2>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                    <div className="p-4 rounded-2xl bg-slate-50 border border-slate-100">
                      <div className="section-title">Data SHA-256</div>
                      <div className="font-mono text-[12px] break-all text-slate-700">
                        {card.data_sha256}
                      </div>
                    </div>
                    <div className="p-4 rounded-2xl bg-slate-50 border border-slate-100">
                      <div className="section-title">Full metrics</div>
                      <dl className="grid grid-cols-2 gap-y-1 text-xs">
                        {Object.entries(card.metrics).map(([k, v]) => (
                          <div
                            key={k}
                            className="flex justify-between gap-2 border-b border-slate-200/60 py-1 last:border-b-0"
                          >
                            <dt className="text-slate-500 mr-2 truncate">
                              {k}
                            </dt>
                            <dd className="font-mono tabular-nums text-slate-800">
                              {fmt(v)}
                            </dd>
                          </div>
                        ))}
                      </dl>
                    </div>
                  </div>
                </section>

                {/* FEATURES */}
                {card.feature_list.length > 0 && (
                  <section className="card p-5 md:p-6">
                    <h2 className="text-lg font-bold text-slate-900 mb-1">
                      Feature list
                    </h2>
                    <p className="text-sm text-slate-500 mb-4">
                      {card.feature_list.length} features consumed by the
                      champion model
                    </p>
                    <div className="flex flex-wrap gap-1.5 max-h-80 overflow-y-auto pr-1">
                      {card.feature_list.map((f) => (
                        <span
                          key={f}
                          className="chip chip-soft font-mono text-[11px]"
                          title={f}
                        >
                          <FeatureBadge name={f} />
                        </span>
                      ))}
                    </div>
                  </section>
                )}
              </>
            ) : (
              <div className="card p-10 text-center max-w-xl mx-auto">
                <div className="text-5xl mb-4">🤷</div>
                <h2 className="text-xl font-bold text-slate-900 mb-1">
                  No model registered yet
                </h2>
                <p className="text-slate-500">
                  Register a champion model in the registry (Module 5 setup
                  commands) and it will appear here automatically.
                </p>
              </div>
            )}
          </>
        )}
      </main>
    </div>
  );
}

function FeatureBadge({ name }: { name: string }) {
  if (name.startsWith("num__")) {
    return (
      <span className="flex items-center gap-1">
        <span className="text-indigo-600">#</span>
        {name.slice(5)}
      </span>
    );
  }
  if (name.startsWith("cat__")) {
    return (
      <span className="flex items-center gap-1">
        <span className="text-emerald-600">◉</span>
        {name.slice(5)}
      </span>
    );
  }
  return <>{name}</>;
}

function Metric({
  label,
  value,
  desc,
  accent,
  icon,
  warn,
}: {
  label: string;
  value: string;
  desc?: string;
  accent: string;
  icon: string;
  warn?: boolean;
}) {
  return (
    <div className={`stat-card ${warn ? "ring-2 ring-amber-300/60" : ""}`}>
      <div
        className={`absolute -right-8 -top-8 w-24 h-24 rounded-full bg-gradient-to-br ${accent} opacity-10 blur-xl`}
      />
      <div className="relative flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="section-title">{label}</div>
          <div className="mt-1 text-3xl font-extrabold tabular-nums text-slate-900">
            {value}
          </div>
          {desc && (
            <div className="mt-1 text-xs text-slate-500 line-clamp-2">
              {desc}
            </div>
          )}
        </div>
        <div
          className={`w-10 h-10 rounded-xl flex items-center justify-center text-lg bg-gradient-to-br ${accent} text-white shadow-sm`}
        >
          {icon}
        </div>
      </div>
    </div>
  );
}

function MetricMini({
  label,
  value,
  warn,
}: {
  label: string;
  value: string;
  warn?: boolean;
}) {
  return (
    <div>
      <div className="text-[10px] uppercase tracking-widest font-bold text-slate-400">
        {label}
      </div>
      <div
        className={`text-2xl md:text-3xl font-black tabular-nums ${
          warn ? "text-amber-600" : "text-slate-900"
        }`}
      >
        {value}
      </div>
    </div>
  );
}

function fmt(v: number | undefined | null): string {
  if (v === undefined || v === null || isNaN(v as number)) return "—";
  const num = v as number;
  if (Math.abs(num) >= 1) return num.toFixed(4).replace(/\.?0+$/, "");
  return num.toFixed(4).replace(/\.?0+$/, "") || "0";
}
