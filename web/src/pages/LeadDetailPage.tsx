import { useEffect, useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
import BandChip from "../components/BandChip";
import {
  getSimilarLeads,
  loadLeadMetadata,
  scoreLead,
} from "../api";
import type { Language, LeadMetadata, ScoreResult, SimilarLeadsResponse } from "../types";

export default function LeadDetailPage() {
  const { leadId } = useParams();
  const [score, setScore] = useState<ScoreResult | null>(null);
  const [similar, setSimilar] = useState<SimilarLeadsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [language, setLanguage] = useState<Language>("en");
  const [meta, setMeta] = useState<LeadMetadata | null>(null);

  useEffect(() => {
    if (!leadId) return;
    const id: string = leadId;
    let cancelled = false;
    async function load() {
      setLoading(true);
      setError(null);
      try {
        let m: LeadMetadata;
        try {
          m = await loadLeadMetadata(id);
        } catch {
          m = {
            lead_id: id,
            full_name: null,
            source: "—",
            branch: "—",
            course_interest: "—",
            age: null,
            home_locality: null,
            phone: null,
            email: null,
            created_at: new Date().toISOString(),
            assigned_counselor_id: null,
          } as LeadMetadata;
        }
        setMeta(m);

        let s: ScoreResult;
        s = await scoreLead(id);
        if (cancelled) return;
        setScore(s);

        try {
          setSimilar(await getSimilarLeads(id, 10));
        } catch {
          /* optional */
        }
      } catch (e: any) {
        if (!cancelled) setError(e?.message || "Failed to load");
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    load();
    return () => {
      cancelled = true;
    };
  }, [leadId]);

  const pct = useMemo(
    () => (score ? Math.round(score.probability * 100) : 0),
    [score]
  );

  const grad =
    score?.band === "P1"
      ? "from-rose-500 to-red-600"
      : score?.band === "P2"
      ? "from-amber-400 to-orange-500"
      : "from-emerald-400 to-green-600";

  return (
    <div className="min-h-screen">
      <header className="header-gradient sticky top-0 z-20">
        <div className="max-w-5xl mx-auto px-3 md:px-6 py-3 md:py-4 flex flex-wrap items-center gap-3">
          <Link
            to="/"
            className="chip chip-soft hover:bg-slate-100"
          >
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

          <div className="ml-auto flex items-center gap-2 flex-wrap">
            {score && <BandChip band={score.band} size="sm" />}
            <div className="lang-toggle">
              {(["en", "hi"] as const).map((l) => (
                <button
                  key={l}
                  type="button"
                  onClick={() => setLanguage(l)}
                  className={language === l ? "active" : ""}
                >
                  {l === "en" ? "English" : "हिंग्लिश"}
                </button>
              ))}
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-5xl mx-auto px-3 md:px-6 py-5 md:py-7 space-y-5">
        {loading ? (
          <div className="card p-10 flex flex-col items-center justify-center gap-4">
            <div className="w-14 h-14 rounded-2xl border-4 border-indigo-500 border-t-transparent animate-spin" />
            <div className="text-slate-600 font-medium">
              Loading enquiry details…
            </div>
          </div>
        ) : error ? (
          <div className="card p-8">
            <div className="font-semibold text-slate-800 mb-2">
              Couldn't load this enquiry
            </div>
            <div className="text-sm text-slate-500 mb-4 break-words">
              {error}
            </div>
            <Link to="/" className="btn btn-primary">
              ← Back to call list
            </Link>
          </div>
        ) : (
          score &&
          meta && (
            <>
              {/* HERO / PROBABILITY */}
              <section className="card p-5 md:p-7 relative overflow-hidden">
                <div
                  className={`absolute -right-20 -top-24 w-80 h-80 rounded-full bg-gradient-to-br ${grad} opacity-10 blur-3xl`}
                />
                <div className="relative grid grid-cols-1 md:grid-cols-[1fr_auto] gap-6 items-center">
                  <div className="flex items-start gap-4 md:gap-5">
                    <div
                      className={`w-16 h-16 md:w-20 md:h-20 rounded-2xl flex items-center justify-center text-2xl md:text-3xl font-black text-white
                        bg-gradient-to-br ${grad} shadow-xl`}
                    >
                      {initials(meta.full_name || meta.lead_id)}
                    </div>
                    <div className="min-w-0 flex-1">
                      <div className="flex flex-wrap items-center gap-2 mb-2">
                        {score.duplicate_cluster_size > 1 && (
                          <span className="chip chip-soft">
                            👥 {score.duplicate_cluster_size} entries merged
                          </span>
                        )}
                        <span className="chip chip-soft font-mono">
                          {meta.lead_id}
                        </span>
                      </div>
                      <h1 className="text-2xl md:text-3xl font-extrabold text-slate-900 truncate">
                        {meta.full_name || "Enquiry"}
                      </h1>
                      <div className="mt-2 flex flex-wrap gap-x-5 gap-y-1 text-sm text-slate-600">
                        <Pill label="Source" value={meta.source} icon="✨" />
                        <Pill label="Branch" value={meta.branch} icon="📍" />
                        <Pill label="Course" value={meta.course_interest} icon="📚" />
                        {meta.age && (
                          <Pill label="Age" value={`${meta.age} yrs`} icon="🎂" />
                        )}
                        {meta.home_locality && (
                          <Pill label="Locality" value={meta.home_locality} icon="🏘️" />
                        )}
                        {meta.assigned_counselor_id && (
                          <Pill
                            label="Counselor"
                            value={meta.assigned_counselor_id}
                            icon="👤"
                          />
                        )}
                      </div>
                    </div>
                  </div>

                  <div className="text-center md:text-right">
                    <div className="text-[11px] uppercase tracking-widest font-bold text-slate-400 mb-1">
                      Admission probability
                    </div>
                    <div
                      className={`text-5xl md:text-6xl font-black tabular-nums leading-none bg-gradient-to-br ${grad} bg-clip-text text-transparent`}
                    >
                      {pct}%
                    </div>
                    <div className="prob-bar w-40 md:ml-auto mt-3">
                      <div
                        className={`prob-bar-fill bg-gradient-to-r ${grad}`}
                        style={{ width: `${pct}%` }}
                      />
                    </div>
                    <div className="mt-2 text-xs text-slate-500">
                      Model <span className="font-mono">{score.model_version}</span>
                    </div>
                  </div>
                </div>
              </section>

              {/* TOP REASONS */}
              <section className="card p-5 md:p-6">
                <h2 className="text-lg font-bold text-slate-900 mb-1">
                  Why this score
                </h2>
                <p className="text-sm text-slate-500 mb-4">
                  The top three reasons the model prioritised this enquiry.
                </p>
                <ol className="space-y-3">
                  {score.reasons.map((r, i) => (
                    <li key={i} className="reason-item">
                      <div
                        className={`w-9 h-9 rounded-xl flex items-center justify-center font-black text-white flex-shrink-0
                          bg-gradient-to-br ${grad} shadow-sm`}
                      >
                        {i + 1}
                      </div>
                      <div className="min-w-0 flex-1">
                        <div className="text-slate-900 font-medium">
                          {language === "en" ? r.en : r.hi}
                        </div>
                        {language === "en" && r.hi && r.hi !== r.en && (
                          <div className="text-xs text-slate-500 mt-1">
                            हिंग्लिश: {r.hi}
                          </div>
                        )}
                      </div>
                    </li>
                  ))}
                </ol>
              </section>

              {/* SIMILAR PAST ENQUIRIES */}
              {similar && similar.similar.length > 0 && (
                <section className="card p-5 md:p-6">
                  <div className="flex flex-wrap items-end justify-between gap-3 mb-3">
                    <div>
                      <h2 className="text-lg font-bold text-slate-900">
                        Similar past enquiries
                      </h2>
                      <p className="text-sm text-slate-500">
                        Nearest {similar.total} neighbours by M3 chat embedding
                        ·{" "}
                        <b className="text-slate-700">
                          {similar.admissions} admissions
                        </b>{" "}
                        · model {similar.model_version}
                      </p>
                    </div>
                  </div>

                  <div className="overflow-x-auto -mx-2 px-2">
                    <table className="w-full text-sm border-separate border-spacing-y-1">
                      <thead>
                        <tr className="text-left text-[11px] uppercase tracking-wider font-bold text-slate-500">
                          <th className="py-2 pr-4">Lead id</th>
                          <th className="py-2 pr-4">Similarity</th>
                          <th className="py-2 pr-4 w-full">Bar</th>
                          <th className="py-2 pr-4 text-right">Outcome</th>
                        </tr>
                      </thead>
                      <tbody>
                        {similar.similar.map((s) => {
                          const sim = Math.max(0, Math.min(1, s.similarity));
                          return (
                            <tr key={s.lead_id} className="group">
                              <td className="py-2 pr-4 font-mono text-xs text-slate-700 group-hover:text-indigo-600">
                                <Link to={`/leads/${s.lead_id}`}>
                                  {s.lead_id}
                                </Link>
                              </td>
                              <td className="py-2 pr-4 tabular-nums font-semibold text-slate-800">
                                {Math.round(sim * 100)}%
                              </td>
                              <td className="py-2 pr-4 w-full">
                                <div className="prob-bar">
                                  <div
                                    className="prob-bar-fill bg-gradient-to-r from-indigo-500 to-sky-500"
                                    style={{ width: `${sim * 100}%` }}
                                  />
                                </div>
                              </td>
                              <td className="py-2 pr-4 text-right">
                                {s.outcome === 1 ? (
                                  <span className="chip chip-p1">✓ Admitted</span>
                                ) : (
                                  <span className="chip chip-soft">
                                    Not admitted
                                  </span>
                                )}
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </section>
              )}

              {/* META FOOTER */}
              <section className="card p-4 md:p-5 text-xs text-slate-500 flex flex-wrap items-center gap-x-5 gap-y-2">
                <span>
                  Score as-of: <b className="text-slate-700">{formatDate(score.asof)}</b>
                </span>
                <span>·</span>
                <span>
                  Enrolled at: <b className="text-slate-700">{formatDate(meta.created_at)}</b>
                </span>
              </section>
            </>
          )
        )}
      </main>
    </div>
  );
}

function Pill({
  label,
  value,
  icon,
}: {
  label: string;
  value: string;
  icon: string;
}) {
  return (
    <span className="inline-flex items-center gap-1.5">
      <span className="text-slate-400">{icon}</span>
      <span>
        <span className="sr-only">{label}: </span>
        <span className="font-medium text-slate-700">{value}</span>
      </span>
    </span>
  );
}

function initials(name: string): string {
  const parts = name
    .trim()
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 2);
  if (parts.length === 0) return "?";
  if (parts.length === 1) {
    const c = parts[0][0]?.toUpperCase();
    return c || "?";
  }
  return (
    (parts[0][0]?.toUpperCase() || "") +
    (parts[1][0]?.toUpperCase() || "")
  );
}

function formatDate(s: string) {
  try {
    const d = new Date(s);
    if (isNaN(d.getTime())) return s;
    return d.toLocaleString(undefined, {
      dateStyle: "medium",
      timeStyle: "short",
    });
  } catch {
    return s;
  }
}
