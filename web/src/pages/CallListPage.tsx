import { useEffect, useMemo, useRef, useState } from "react";
import { loadLeadMetadatas, scoreMetadatas } from "../api";
import FiltersBar from "../components/FiltersBar";
import CallListRow from "../components/CallListRow";
import type { CallListItem, Filters, Language, ScoreResult } from "../types";
import { Link, useNavigate } from "react-router-dom";
import BandChip from "../components/BandChip";

export default function CallListPage() {
  const [rows, setRows] = useState<CallListItem[]>([]);
  const [totalMetas, setTotalMetas] = useState(0);
  const [totalLeads, setTotalLeads] = useState(0);
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(100);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [language, setLanguage] = useState<Language>("en");
  const [filters, setFilters] = useState<Filters>({
    branch: new Set(),
    source: new Set(),
    band: new Set(),
    search: "",
  });
  const [refreshTick, setRefreshTick] = useState(0);
  const navigate = useNavigate();

  // In-memory cache of scored leads to make page navigation instant
  const scoreCacheRef = useRef<Map<string, ScoreResult>>(new Map());

  useEffect(() => {
    let cancelled = false;
    async function load() {
      setLoading(true);
      setError(null);
      setRows([]);
      try {
        const offset = (page - 1) * pageSize;
        const res = await loadLeadMetadatas(pageSize, offset);
        if (cancelled) return;
        setTotalLeads(res.total);
        setTotalMetas(res.leads.length);

        // Score this page's leads (streaming partial batches, using cache if previously visited)
        const items = await scoreMetadatas(
          res.leads,
          (partial) => {
            if (!cancelled) setRows(partial);
          },
          scoreCacheRef.current
        );
        if (!cancelled) setRows(items);
      } catch (err: any) {
        if (!cancelled) setError(err?.message || "Failed to load");
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    load();
    return () => {
      cancelled = true;
    };
  }, [page, pageSize, refreshTick]);

  const totalPages = Math.max(1, Math.ceil(totalLeads / pageSize));

  const handlePageChange = (newPage: number) => {
    const clamped = Math.max(1, Math.min(newPage, totalPages));
    if (clamped !== page) {
      setPage(clamped);
      window.scrollTo({ top: 0, behavior: "smooth" });
    }
  };

  const handlePageSizeChange = (newSize: number) => {
    if (newSize !== pageSize) {
      setPageSize(newSize);
      setPage(1);
      window.scrollTo({ top: 0, behavior: "smooth" });
    }
  };

  const handleRefresh = () => {
    scoreCacheRef.current.clear();
    setRefreshTick((t) => t + 1);
  };

  const options = useMemo(() => {
    const KNOWN_BRANCHES = ["Central", "Lakeview", "Riverside"];
    const KNOWN_SOURCES = [
      "Click-to-WhatsApp Ad",
      "Facebook Lead Form",
      "Instagram Lead Form",
      "Referral",
      "Walk-in",
      "Website Form",
      "YouTube Ad",
    ];
    const branches = new Set<string>(KNOWN_BRANCHES);
    const sources = new Set<string>(KNOWN_SOURCES);
    for (const r of rows) {
      if (r.metadata.branch) branches.add(r.metadata.branch);
      if (r.metadata.source) sources.add(r.metadata.source);
    }
    return {
      branches: [...branches].sort(),
      sources: [...sources].sort(),
    };
  }, [rows]);

  const filtered = useMemo(() => {
    const q = filters.search.trim().toLowerCase();
    const result = rows.filter((r) => {
      if (filters.branch.size && !filters.branch.has(r.metadata.branch))
        return false;
      if (filters.source.size && !filters.source.has(r.metadata.source))
        return false;
      if (filters.band.size && !filters.band.has(r.score.band)) return false;
      if (q) {
        const hay = [
          r.metadata.full_name ?? "",
          r.metadata.lead_id,
          r.metadata.course_interest,
          r.metadata.home_locality ?? "",
          r.metadata.source ?? "",
          r.metadata.branch ?? "",
          r.metadata.assigned_counselor_id ?? "",
        ]
          .join(" ")
          .toLowerCase();
        if (!hay.includes(q)) return false;
      }
      return true;
    });
    return [...result].sort(
      (a, b) =>
        b.score.probability - a.score.probability ||
        (b.urgent ? 1 : 0) - (a.urgent ? 1 : 0)
    );
  }, [rows, filters]);

  const counts = useMemo(() => {
    let P1 = 0;
    let P2 = 0;
    let P3 = 0;
    let urgent = 0;
    const total = filtered.length;
    let avgProb = 0;
    for (const r of filtered) {
      if (r.score.band === "P1") P1++;
      else if (r.score.band === "P2") P2++;
      else if (r.score.band === "P3") P3++;
      if (r.urgent) urgent++;
      avgProb += r.score.probability;
    }
    avgProb = total ? avgProb / total : 0;
    return { P1, P2, P3, urgent, total, avgProb };
  }, [filtered]);

  const openLead = (id: string) => {
    navigate(`/leads/${encodeURIComponent(id)}`);
  };

  const modelVersion = rows.length > 0 ? rows[0].score.model_version : null;

  return (
    <div className="min-h-screen">
      {/* HEADER */}
      <header className="header-gradient sticky top-0 z-20">
        <div className="max-w-6xl mx-auto px-3 md:px-6 py-3 md:py-4 flex flex-wrap items-center gap-3">
          <Link to="/" className="flex items-center gap-2 group">
            <div
              className="w-9 h-9 rounded-xl flex items-center justify-center text-white font-black text-lg
                bg-gradient-to-br from-indigo-600 via-blue-600 to-sky-500
                shadow-lg shadow-indigo-500/30 group-hover:shadow-indigo-500/50 transition"
            >
              L
            </div>
            <span className="brand-title">LeadIQ</span>
          </Link>

          <nav className="flex gap-2 ml-3 md:ml-5">
            <Link to="/" className="chip chip-soft-active">
              📋 Call list
            </Link>
            <Link to="/model" className="chip chip-soft hover:bg-slate-50">
              📊 Model
            </Link>
          </nav>

          <div className="ml-auto flex items-center gap-2 flex-wrap">
            {modelVersion && (
              <span
                className="chip chip-soft font-mono"
                title="Champion model version that scored this list"
              >
                {modelVersion}
              </span>
            )}
            <button
              onClick={handleRefresh}
              className="btn !py-2 !px-3 text-xs"
              title="Refresh data and re-score"
            >
              ↻ Refresh
            </button>
          </div>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-3 md:px-6 py-5 md:py-7">
        {/* STATS BAR */}
        {!loading && !error && rows.length > 0 && (
          <section className="grid grid-cols-2 md:grid-cols-4 gap-3 md:gap-4 mb-5">
            <StatCard
              label="Total Enquiries"
              big={totalLeads > 0 ? totalLeads.toLocaleString() : rows.length}
              icon="👥"
              accent="from-indigo-500 to-blue-500"
              sub={`Page ${page} of ${totalPages} (${rows.length} loaded)`}
            />
            <StatCard
              label="Priority (P1)"
              big={counts.P1}
              icon="🚨"
              accent="from-rose-500 to-red-500"
              sub={`${pct(counts.P1, counts.total)} of page`}
            />
            <StatCard
              label="Follow-up (P2)"
              big={counts.P2}
              icon="💡"
              accent="from-amber-400 to-orange-500"
              sub={`${pct(counts.P2, counts.total)} of page`}
            />
            <StatCard
              label="Waiting > 15 min"
              big={counts.urgent}
              icon="⏰"
              accent="from-yellow-400 to-amber-500"
              sub={
                counts.urgent > 0
                  ? "Highlighted amber rows"
                  : "All P1 caught up"
              }
            />
          </section>
        )}

        <FiltersBar
          filters={filters}
          options={options}
          onChange={setFilters}
          language={language}
          onLanguageChange={setLanguage}
        />

        {loading && rows.length === 0 ? (
          <LoadingState />
        ) : error && rows.length === 0 ? (
          <ErrorState
            message={error}
            onRetry={handleRefresh}
          />
        ) : filtered.length === 0 ? (
          <EmptyState
            onClear={() =>
              setFilters({
                branch: new Set(),
                source: new Set(),
                band: new Set(),
                search: "",
              })
            }
          />
        ) : (
          <>
            {/* TOP CONTROLS & STREAMING PROGRESS */}
            {loading && (
              <div className="card mb-4 p-3 flex items-center gap-3 text-sm text-slate-500">
                <div className="w-5 h-5 rounded-lg border-2 border-indigo-500 border-t-transparent animate-spin flex-shrink-0" />
                Scoring page {page} enquiries… {rows.length} of {totalMetas} so far.
              </div>
            )}

            {/* TOP COMPACT PAGINATION HEADER */}
            <div className="flex flex-wrap items-center justify-between gap-2 mb-3 px-1 text-xs text-slate-500">
              <div className="flex items-center gap-2">
                <span>
                  Page <b className="text-slate-800 font-semibold">{page}</b> of{" "}
                  <b className="text-slate-800 font-semibold">{totalPages}</b>
                </span>
                <span>·</span>
                <span>
                  Showing{" "}
                  <b className="text-slate-800 font-semibold">
                    {(page - 1) * pageSize + 1}–{Math.min(page * pageSize, totalLeads)}
                  </b>{" "}
                  of <b className="text-slate-800 font-semibold">{totalLeads.toLocaleString()}</b>
                </span>
              </div>
              <div className="flex items-center gap-1.5">
                <button
                  type="button"
                  disabled={loading || page <= 1}
                  onClick={() => handlePageChange(page - 1)}
                  className="px-2.5 py-1 rounded-lg border border-slate-200 bg-white text-slate-700 hover:bg-slate-50 disabled:opacity-40 transition font-medium"
                >
                  ‹ Prev
                </button>
                <button
                  type="button"
                  disabled={loading || page >= totalPages}
                  onClick={() => handlePageChange(page + 1)}
                  className="px-2.5 py-1 rounded-lg border border-slate-200 bg-white text-slate-700 hover:bg-slate-50 disabled:opacity-40 transition font-medium"
                >
                  Next ›
                </button>
              </div>
            </div>

            {/* CARD LIST */}
            <div className="flex flex-col gap-4 md:gap-5">
              {filtered.map((r) => (
                <article
                  key={r.metadata.lead_id}
                  onMouseDown={(e) => {
                    // Prevent React Router <Link> hard-reload edge-cases
                    if (e.button !== 0) return;
                  }}
                  onClick={() => openLead(r.metadata.lead_id)}
                >
                  <CallListRow item={r} language={language} />
                </article>
              ))}
            </div>

            {/* SUMMARY BAR */}
            <div className="card mt-5 p-4 md:p-5 flex flex-wrap items-center justify-between gap-3 text-sm text-slate-500">
              <div className="flex flex-wrap items-center gap-3">
                <span>
                  Showing <b className="text-slate-700">{filtered.length}</b> on this page{" "}
                  (Total <b className="text-slate-700">{totalLeads.toLocaleString()}</b> across system)
                </span>
                <span>·</span>
                <span>
                  Avg probability{" "}
                  <b className="text-slate-700">
                    {Math.round(counts.avgProb * 100)}%
                  </b>
                </span>
              </div>
              <span className="flex flex-wrap items-center gap-2">
                <BandChip band="P1" size="sm" /> Call today ·
                <BandChip band="P2" size="sm" /> This week ·
                <BandChip band="P3" size="sm" /> Nurture
              </span>
            </div>

            {/* MAIN PAGINATION CONTROLS */}
            <PaginationControls
              page={page}
              pageSize={pageSize}
              total={totalLeads}
              totalPages={totalPages}
              onPageChange={handlePageChange}
              onPageSizeChange={handlePageSizeChange}
              disabled={loading}
            />
          </>
        )}
      </main>
    </div>
  );
}

function PaginationControls({
  page,
  pageSize,
  total,
  totalPages,
  onPageChange,
  onPageSizeChange,
  disabled,
}: {
  page: number;
  pageSize: number;
  total: number;
  totalPages: number;
  onPageChange: (p: number) => void;
  onPageSizeChange: (s: number) => void;
  disabled?: boolean;
}) {
  const [jumpInput, setJumpInput] = useState("");

  const pageNumbers = useMemo(() => {
    if (totalPages <= 7) {
      return Array.from({ length: totalPages }, (_, i) => i + 1);
    }
    const pages: (number | string)[] = [];
    if (page <= 4) {
      pages.push(1, 2, 3, 4, 5, "...", totalPages);
    } else if (page >= totalPages - 3) {
      pages.push(1, "...", totalPages - 4, totalPages - 3, totalPages - 2, totalPages - 1, totalPages);
    } else {
      pages.push(1, "...", page - 1, page, page + 1, "...", totalPages);
    }
    return pages;
  }, [page, totalPages]);

  const startIdx = total === 0 ? 0 : (page - 1) * pageSize + 1;
  const endIdx = Math.min(page * pageSize, total);

  const handleJumpSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const p = parseInt(jumpInput, 10);
    if (!isNaN(p) && p >= 1 && p <= totalPages) {
      onPageChange(p);
      setJumpInput("");
    }
  };

  return (
    <div className="card p-4 md:p-5 flex flex-col md:flex-row items-center justify-between gap-4 mt-5">
      {/* Left: Info & Rows per page */}
      <div className="flex flex-wrap items-center gap-4 text-xs md:text-sm text-slate-500 w-full md:w-auto justify-between md:justify-start">
        <span>
          Showing <b className="text-slate-800 font-semibold">{startIdx}</b>–
          <b className="text-slate-800 font-semibold">{endIdx}</b> of{" "}
          <b className="text-slate-800 font-semibold">{total.toLocaleString()}</b>
        </span>

        <div className="flex items-center gap-1.5">
          <span className="text-slate-400">Rows:</span>
          {[25, 50, 100, 250, 500].map((sz) => (
            <button
              key={sz}
              type="button"
              disabled={disabled}
              onClick={() => onPageSizeChange(sz)}
              className={`px-2.5 py-1 text-xs rounded-lg font-medium transition ${
                pageSize === sz
                  ? "bg-indigo-600 text-white shadow-sm"
                  : "bg-slate-100 text-slate-600 hover:bg-slate-200"
              }`}
            >
              {sz}
            </button>
          ))}
        </div>
      </div>

      {/* Right: Page Buttons & Quick Jump */}
      <div className="flex flex-wrap items-center gap-1.5 justify-center w-full md:w-auto">
        <button
          type="button"
          disabled={disabled || page <= 1}
          onClick={() => onPageChange(1)}
          className="px-2.5 py-1.5 text-xs rounded-lg border border-slate-200 bg-white text-slate-700 hover:bg-slate-50 disabled:opacity-40 disabled:cursor-not-allowed transition font-medium"
          title="First page"
        >
          «
        </button>
        <button
          type="button"
          disabled={disabled || page <= 1}
          onClick={() => onPageChange(page - 1)}
          className="px-3 py-1.5 text-xs rounded-lg border border-slate-200 bg-white text-slate-700 hover:bg-slate-50 disabled:opacity-40 disabled:cursor-not-allowed transition font-medium"
          title="Previous page"
        >
          ‹ Prev
        </button>

        <div className="flex items-center gap-1">
          {pageNumbers.map((p, idx) =>
            typeof p === "number" ? (
              <button
                key={p}
                type="button"
                disabled={disabled}
                onClick={() => onPageChange(p)}
                className={`min-w-[32px] h-8 px-2 text-xs rounded-lg font-semibold transition ${
                  page === p
                    ? "bg-gradient-to-br from-indigo-600 to-blue-600 text-white shadow-md shadow-indigo-500/25"
                    : "border border-slate-200 bg-white text-slate-700 hover:bg-slate-50"
                }`}
              >
                {p}
              </button>
            ) : (
              <span key={`ellipsis-${idx}`} className="px-1 text-slate-400 text-xs">
                …
              </span>
            )
          )}
        </div>

        <button
          type="button"
          disabled={disabled || page >= totalPages}
          onClick={() => onPageChange(page + 1)}
          className="px-3 py-1.5 text-xs rounded-lg border border-slate-200 bg-white text-slate-700 hover:bg-slate-50 disabled:opacity-40 disabled:cursor-not-allowed transition font-medium"
          title="Next page"
        >
          Next ›
        </button>
        <button
          type="button"
          disabled={disabled || page >= totalPages}
          onClick={() => onPageChange(totalPages)}
          className="px-2.5 py-1.5 text-xs rounded-lg border border-slate-200 bg-white text-slate-700 hover:bg-slate-50 disabled:opacity-40 disabled:cursor-not-allowed transition font-medium"
          title="Last page"
        >
          »
        </button>

        {totalPages > 5 && (
          <form onSubmit={handleJumpSubmit} className="flex items-center gap-1 ml-2">
            <input
              type="number"
              min={1}
              max={totalPages}
              placeholder="#"
              value={jumpInput}
              onChange={(e) => setJumpInput(e.target.value)}
              className="w-12 h-8 text-xs text-center border border-slate-200 rounded-lg px-1 focus:outline-none focus:ring-1 focus:ring-indigo-500"
              title="Go to page number"
            />
            <button
              type="submit"
              disabled={disabled || !jumpInput}
              className="h-8 px-2 text-xs rounded-lg bg-slate-100 hover:bg-slate-200 text-slate-700 font-medium disabled:opacity-40 transition"
            >
              Go
            </button>
          </form>
        )}
      </div>
    </div>
  );
}

function StatCard({
  label,
  big,
  icon,
  accent,
  sub,
}: {
  label: string;
  big: number | string;
  icon: string;
  accent: string;
  sub?: string;
}) {
  return (
    <div className="stat-card relative transition-all hover:-translate-y-0.5 hover:shadow-md duration-300">
      <div
        className={`absolute -right-8 -top-8 w-24 h-24 rounded-full bg-gradient-to-br ${accent} opacity-10 blur-xl`}
      />
      <div className="relative flex items-start justify-between gap-2">
        <div className="min-w-0">
          <div className="text-[11px] uppercase tracking-wide font-bold text-slate-500">
            {label}
          </div>
          <div className="mt-1 text-2xl md:text-3xl font-extrabold tabular-nums text-slate-900">
            {big}
          </div>
          {sub && (
            <div className="mt-0.5 text-xs text-slate-500 truncate">{sub}</div>
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

function pct(part: number, whole: number) {
  if (!whole) return "0%";
  return `${Math.round((part / whole) * 100)}%`;
}

function LoadingState() {
  return (
    <div className="card p-8">
      <div className="flex items-center gap-3 mb-6">
        <div className="h-10 w-10 rounded-2xl bg-gradient-to-br from-indigo-500 to-blue-500 flex items-center justify-center text-white animate-pulse">
          ⏳
        </div>
        <div>
          <div className="font-semibold text-slate-800">
            Loading your call list…
          </div>
          <div className="text-sm text-slate-500">
            Scoring enquiries with the champion model.
          </div>
        </div>
      </div>

      <div className="space-y-4">
        {Array.from({ length: 5 }).map((_, i) => (
          <div
            key={i}
            className="rounded-2xl border border-slate-100 bg-white p-4 flex gap-4"
          >
            <div className="w-12 h-12 md:w-14 md:h-14 rounded-2xl skeleton" />
            <div className="flex-1 space-y-2">
              <div className="h-4 w-1/3 rounded skeleton" />
              <div className="h-3 w-2/3 rounded skeleton" />
              <div className="h-3 w-1/2 rounded skeleton mt-2" />
            </div>
            <div className="w-20 md:w-28 space-y-2 hidden sm:block">
              <div className="h-8 rounded skeleton" />
              <div className="h-2 rounded skeleton" />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function EmptyState({ onClear }: { onClear: () => void }) {
  return (
    <div className="card p-10 text-center max-w-xl mx-auto">
      <div className="w-20 h-20 mx-auto rounded-3xl flex items-center justify-center text-4xl mb-5 bg-gradient-to-br from-slate-100 to-slate-200">
        📭
      </div>
      <h2 className="text-xl font-bold text-slate-900 mb-1">
        No enquiries match
      </h2>
      <p className="text-slate-500 mb-5">
        Try clearing some filters or broadening your search to see more leads.
      </p>
      <button onClick={onClear} className="btn btn-primary">
        Clear all filters
      </button>
    </div>
  );
}

function ErrorState({
  message,
  onRetry,
}: {
  message: string;
  onRetry: () => void;
}) {
  return (
    <div className="card p-8 max-w-xl mx-auto">
      <div className="flex items-start gap-4">
        <div className="w-14 h-14 rounded-2xl flex items-center justify-center text-3xl flex-shrink-0 bg-gradient-to-br from-rose-100 to-red-100">
          ⚠️
        </div>
        <div className="flex-1">
          <h2 className="text-lg font-bold text-slate-900 mb-1">
            Couldn't load the call list
          </h2>
          <p className="text-sm text-slate-600 mb-4 break-words">{message}</p>
          <div className="flex flex-wrap gap-2">
            <button onClick={onRetry} className="btn btn-primary">
              ↻ Retry
            </button>
            <a href="/" className="btn">
              Go home
            </a>
          </div>
        </div>
      </div>
    </div>
  );
}
