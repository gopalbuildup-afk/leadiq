import type { Band } from "../types";

interface Props {
  band: Band;
  size?: "sm" | "md" | "lg";
}

const sizeMap = {
  sm: "px-2 py-0.5 text-[11px]",
  md: "px-2.5 py-1 text-xs",
  lg: "px-3 py-1.5 text-sm",
} as const;

export default function BandChip({ band, size = "md" }: Props) {
  const cls =
    band === "P1" ? "chip-p1" : band === "P2" ? "chip-p2" : "chip-p3";
  const subtitle =
    band === "P1" ? "PRIORITY" : band === "P2" ? "FOLLOW UP" : "STANDARD";
  return (
    <span className={`chip ${cls} ${sizeMap[size]} gap-1.5`}>
      <span className="opacity-80">●</span>
      {band}
      <span className={`opacity-80 ${size === "sm" ? "hidden" : "inline"}`}>
        · {subtitle}
      </span>
    </span>
  );
}
