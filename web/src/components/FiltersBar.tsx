import type { Band, Filters } from "../types";

interface Props {
  filters: Filters;
  options: {
    branches: string[];
    sources: string[];
  };
  onChange: (next: Filters) => void;
  language: "en" | "hi";
  onLanguageChange: (lang: "en" | "hi") => void;
}

export default function FiltersBar({
  filters,
  options,
  onChange,
  language,
  onLanguageChange,
}: Props) {
  const toggle = <K extends keyof Filters>(
    key: K & ("branch" | "source" | "band"),
    value: string
  ) => {
    const current = filters[key] as Set<string>;
    const next = new Set(current);
    if (next.has(value)) next.delete(value);
    else next.add(value);
    onChange({ ...filters, [key]: next });
  };

  const bands: Band[] = ["P1", "P2", "P3"];
  const bandColor = (b: Band, on: boolean) => {
    if (!on) return "chip-soft";
    return b === "P1" ? "chip-p1" : b === "P2" ? "chip-p2" : "chip-p3";
  };

  return (
    <div className="card p-4 md:p-5 mb-5">
      <div className="flex flex-col gap-4">
        <div className="flex flex-col md:flex-row gap-3 md:items-center">
          <div className="flex-1 min-w-0">
            <input
              className="search-input w-full"
              placeholder="Search by name, lead id, course, locality…"
              value={filters.search}
              onChange={(e) =>
                onChange({ ...filters, search: e.target.value })
              }
            />
          </div>
          <div className="lang-toggle self-start md:self-auto">
            {(["en", "hi"] as const).map((l) => (
              <button
                key={l}
                type="button"
                onClick={() => onLanguageChange(l)}
                className={language === l ? "active" : ""}
              >
                {l === "en" ? "English" : "हिंग्लिश"}
              </button>
            ))}
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <div className="section-title">Band</div>
            <div className="flex flex-wrap gap-2">
              {bands.map((b) => {
                const on = filters.band.has(b);
                return (
                  <button
                    key={b}
                    type="button"
                    onClick={() => toggle("band", b)}
                    className={`chip ${bandColor(b, on)}`}
                  >
                    {b}
                    {on && (
                      <span className="opacity-80 ml-0.5">✓</span>
                    )}
                  </button>
                );
              })}
            </div>
          </div>

          <div>
            <div className="section-title">Branch</div>
            <div className="flex flex-wrap gap-2">
              {options.branches.map((b) => {
                const on = filters.branch.has(b);
                return (
                  <button
                    key={b}
                    type="button"
                    onClick={() => toggle("branch", b)}
                    className={`chip ${on ? "chip-soft-active" : "chip-soft"}`}
                  >
                    {b}
                    {on && <span className="opacity-90 ml-0.5">✓</span>}
                  </button>
                );
              })}
            </div>
          </div>

          <div>
            <div className="section-title">Source</div>
            <div className="flex flex-wrap gap-2">
              {filters.source.size > 0 &&
                [...filters.source].map((s) => (
                  <span
                    key={s}
                    className="chip chip-soft-active"
                    title={s}
                  >
                    <span className="max-w-[140px] truncate">{s}</span>
                    <button
                      type="button"
                      className="ml-1 opacity-80 hover:opacity-100"
                      onClick={() => toggle("source", s)}
                      aria-label={`Remove ${s}`}
                    >
                      ×
                    </button>
                  </span>
                ))}
              <select
                className="input max-w-xs cursor-pointer"
                value=""
                onChange={(e) => {
                  if (e.target.value) toggle("source", e.target.value);
                  e.target.value = "";
                }}
                title="Add source filter"
              >
                <option value="">+ Add source…</option>
                {options.sources.map((s) => (
                  <option key={s} value={s}>
                    {s}
                  </option>
                ))}
              </select>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
