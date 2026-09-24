import BandChip from "./BandChip";
import type { CallListItem, Language } from "../types";

interface Props {
  item: CallListItem;
  language: Language;
}

const branchIcon: Record<string, string> = {
  Central: "🏙️",
  Lakeview: "🌊",
  Riverside: "🌉",
};

const sourceIcon: Record<string, string> = {
  "Click-to-WhatsApp Ad": "💬",
  "Facebook Lead Form": "📘",
  "Instagram Lead Form": "📸",
  "Website Form": "🌐",
  "Walk-in": "🚶",
  Referral: "🤝",
  "YouTube Ad": "▶️",
  Unknown: "✨",
};

export default function CallListRow({ item, language }: Props) {
  const { metadata, score } = item;

  const name =
    metadata.full_name?.trim() ||
    metadata.course_interest ||
    metadata.lead_id;

  const pct = Math.round(score.probability * 100);
  const ageStr = formatAge(item.age_hours);

  const reasons = score.reasons
    .map((r) => (language === "en" ? r.en : r.hi))
    .filter(Boolean);

  const grad =
    score.band === "P1"
      ? "from-rose-500 to-red-600"
      : score.band === "P2"
      ? "from-amber-400 to-orange-500"
      : "from-emerald-400 to-green-600";

  const borderByBand =
    score.band === "P1"
      ? "border-l-rose-500"
      : score.band === "P2"
      ? "border-l-amber-400"
      : "border-l-emerald-400";

  return (
    <section
      className={`group relative overflow-hidden rounded-2xl transition-all duration-300 ease-out
        ${
          item.urgent
            ? "bg-gradient-to-br from-amber-50 via-yellow-50/70 to-white border-2 border-amber-300 shadow-[0_8px_30px_-10px_rgba(245,158,11,0.45)] scale-[1.005]"
            : "bg-white/75 backdrop-blur-md border border-slate-200/80 shadow-[0_2px_8px_-3px_rgba(15,23,42,0.1),0_14px_40px_-24px_rgba(79,70,229,0.25)] hover:-translate-y-1 hover:translate-x-[1px] hover:shadow-[0_10px_20px_-8px_rgba(15,23,42,0.12),0_30px_60px_-30px_rgba(79,70,229,0.45)]"
        }
        border-l-[6px] ${borderByBand}
        cursor-pointer
      `}
      title={metadata.lead_id}
    >
      {/* Decorative gradient blob */}
      <div
        className={`absolute -right-16 -top-16 w-48 h-48 rounded-full bg-gradient-to-br ${grad} opacity-[0.08] blur-2xl pointer-events-none transition-opacity duration-300 group-hover:opacity-20`}
      />

      <div className="relative p-4 md:p-5 flex flex-col lg:flex-row lg:items-center gap-4">
        {/* =========================================================
            LEFT: Identity + Metadata
        ========================================================= */}
        <div className="flex items-start gap-3 md:gap-4 lg:min-w-[300px] lg:max-w-[360px] flex-1 min-w-0">
          {/* Avatar */}
          <div
            className={`flex-shrink-0 w-12 h-12 md:w-14 md:h-14 rounded-2xl flex items-center justify-center text-lg md:text-xl font-bold text-white
              bg-gradient-to-br ${grad}
              shadow-[0_8px_16px_-8px_rgba(0,0,0,0.3)]
              transition-transform duration-300
              group-hover:scale-105 group-hover:rotate-[-2deg]`}
            title={name}
          >
            {initials(name)}
          </div>

          {/* Identity */}
          <div className="min-w-0 flex-1">
            <div className="flex flex-wrap items-center gap-2 mb-1.5">
              <BandChip band={score.band} size="sm" />

              {score.duplicate_cluster_size > 1 && (
                <span
                  title={`${score.duplicate_cluster_size} duplicate entries merged into one record`}
                  className="chip chip-soft text-[11px]"
                >
                  👥 {score.duplicate_cluster_size} merged
                </span>
              )}

              {item.urgent && (
                <span className="chip chip-urgent text-[11px]">
                  ⚠️ Waiting
                </span>
              )}

              {metadata.assigned_counselor_id && (
                <span className="chip chip-soft text-[11px]">
                  👤 {metadata.assigned_counselor_id}
                </span>
              )}
            </div>

            <div className="flex flex-wrap items-baseline gap-2 min-w-0">
              <h3 className="font-bold text-slate-900 truncate text-[15px] md:text-base">
                {name}
              </h3>

              <span className="font-mono text-[11px] md:text-xs text-slate-400 flex-shrink-0">
                {metadata.lead_id}
              </span>
            </div>

            {/* Source + Branch */}
            <div className="flex flex-wrap items-center gap-x-3 gap-y-1 mt-1.5 text-xs text-slate-500">
              <span title={metadata.source}>
                {sourceIcon[metadata.source] || "✨"}{" "}
                <span className="truncate max-w-[130px] inline-block align-bottom">
                  {metadata.source}
                </span>
              </span>

              <span title={metadata.branch}>
                {branchIcon[metadata.branch] || "📍"} {metadata.branch}
              </span>
            </div>

            {/* Course + Age + Locality */}
            <div className="flex flex-wrap items-center gap-x-3 gap-y-1 mt-0.5 text-xs text-slate-500">
              <span>📚 {metadata.course_interest}</span>

              <span>⏱️ {ageStr} old</span>

              {metadata.home_locality && (
                <span>🏘️ {metadata.home_locality}</span>
              )}
            </div>
          </div>
        </div>

        {/* =========================================================
            MIDDLE: Probability + Mobile Reasons
        ========================================================= */}
        <div
          className="
            w-full
            lg:w-auto
            flex
            items-center
            justify-between
            lg:justify-end
            gap-4
            lg:gap-6
            lg:pr-4
            lg:min-w-[220px]
          "
        >
          {/* Mobile reasons */}
          <div className="flex-1 min-w-0 lg:hidden">
            <div className="text-[11px] uppercase tracking-wide font-bold text-slate-400 mb-1">
              Why call first
            </div>

            <ul className="space-y-1">
              {reasons.slice(0, 2).map((r, i) => (
                <li
                  key={i}
                  className="flex items-start gap-1.5 text-[12px] text-slate-600 leading-snug"
                >
                  <span
                    className={`mt-1 w-1.5 h-1.5 rounded-full bg-gradient-to-br ${grad} flex-shrink-0`}
                  />

                  <span className="line-clamp-1 min-w-0">
                    {r}
                  </span>
                </li>
              ))}
            </ul>
          </div>

          {/* Probability */}
          <div className="text-right flex-shrink-0 min-w-[78px]">
            <div className="text-[10px] sm:text-[11px] uppercase tracking-widest font-bold text-slate-400 mb-1">
              Chance
            </div>

            <div
              className={`text-3xl sm:text-4xl font-black tabular-nums leading-none
                bg-gradient-to-br ${grad}
                bg-clip-text text-transparent
                transition-transform duration-300
                group-hover:scale-105
                origin-right`}
            >
              {pct}%
            </div>
          </div>
        </div>

        {/* =========================================================
            RIGHT: Progress Bar + Reasons - Desktop Only
        ========================================================= */}
        <div
          className="
            lg:flex-1
            lg:pl-4
            lg:border-l
            lg:border-slate-200/70
            min-w-0
            hidden
            lg:block
          "
        >
          {/* Progress bar */}
          <div className="prob-bar mb-3 group-hover:opacity-90 transition-opacity">
            <div
              className={`prob-bar-fill bg-gradient-to-r ${grad}`}
              style={{
                width: `${Math.max(3, pct)}%`,
              }}
            />
          </div>

          <div className="text-[11px] uppercase tracking-wide font-bold text-slate-400 mb-1.5">
            Why call first
          </div>

          <ul className="space-y-1.5">
            {reasons.slice(0, 3).map((r, i) => (
              <li
                key={i}
                className="flex items-start gap-2 text-[13px] text-slate-700 leading-snug"
              >
                <span
                  className={`mt-1 w-1.5 h-1.5 rounded-full bg-gradient-to-br ${grad} flex-shrink-0 shadow-sm`}
                />

                <span className="line-clamp-2">
                  {r}
                </span>
              </li>
            ))}
          </ul>
        </div>

        {/* =========================================================
            MOBILE CTA
        ========================================================= */}
        <div
          className="
            lg:hidden
            mt-1
            flex
            items-center
            justify-between
            gap-3
            border-t
            border-dashed
            border-slate-200
            pt-3
          "
        >
          <span className="text-[12px] font-semibold text-slate-500 truncate">
            Tap for details, reasons & similar leads →
          </span>

          <span
            className={`chip bg-gradient-to-br ${grad} text-white flex-shrink-0`}
            aria-hidden
          >
            Open →
          </span>
        </div>
      </div>
    </section>
  );
}

/* =========================================================
   Helpers
========================================================= */

function initials(name: string): string {
  const parts = name
    .trim()
    .split(/[\s·\-–—]+/)
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

function formatAge(hours: number): string {
  if (hours < 1) {
    const m = Math.max(1, Math.round(hours * 60));
    return `${m}m`;
  }

  if (hours < 48) {
    return `${Math.round(hours)}h`;
  }

  const d = Math.round(hours / 24);
  return `${d}d`;
}
