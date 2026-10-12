//! Turn accounting is owned by the backend. This component only formats the current snapshot.
import { useEffect, useRef, useState } from "react";
import { fmtTokens } from "../../format";
import { useT } from "../../i18n";
import { agentTurnStats, type AgentTurnStats } from "../../ipc/commands";
import type { Session } from "../../types";
import { Section } from "./parts";
import "./turn-section.css";

export type TurnStats = AgentTurnStats;

const tokens = (value: number | null | undefined) => value == null ? "—" : fmtTokens(value);
const contextTokens = (value: number | null | undefined) => value != null && value >= 1000 && value < 1_000_000
  ? `${(value / 1000).toFixed(1)}k` : tokens(value);
// 主吞吐率只在有限正数时可信；0 或非有限值一律视为未知。
const positiveRate = (v: number | null | undefined) => v != null && Number.isFinite(v) && v > 0;
// 缓存占比只接受 0..100 的有限值。
const validPercent = (v: number | null | undefined) => v != null && Number.isFinite(v) && v >= 0 && v <= 100;

export function TurnStatsView({ stats, state = "ready" }: {
  stats: TurnStats | null;
  state?: "loading" | "waiting" | "error" | "ready";
}) {
  const t = useT();
  const percent = stats?.contextPercent;
  const tone = percent != null && percent >= 90 ? "crit" : percent != null && percent >= 70 ? "warn" : "ok";
  const context = `${contextTokens(stats?.contextTokens)} / ${contextTokens(stats?.contextLimit)}`;
  const rate = positiveRate(stats?.generationTokensPerSecond) ? stats!.generationTokensPerSecond! : null;
  const estimatedRate = positiveRate(stats?.estimatedOutputTokensPerSecond)
    ? stats!.estimatedOutputTokensPerSecond! : null;
  const basis = stats?.outputThroughputBasis === "request" || stats?.outputThroughputBasis === "stream"
    ? stats.outputThroughputBasis : null;
  const cached = stats?.cachedTokens ?? null;
  const reasoning = stats?.reasoningTokens ?? null;
  const hit = validPercent(stats?.cacheHitPercent) ? stats!.cacheHitPercent! : null;
  const model = stats?.model;
  // 每组只在有内容时渲染；仅有估算速率的快照也算有本轮数据。
  const hasContext = stats?.contextTokens != null || stats?.contextLimit != null || percent != null;
  const hasTurn = stats?.inputTokens != null || stats?.tokens != null || stats?.totalTokens != null
    || stats?.cacheHitPercent != null || rate != null || stats?.filesTouched != null
    || reasoning != null || estimatedRate != null;
  const hasSession = stats?.sessionTokens != null;
  const hasData = model != null || hasContext || hasTurn || hasSession;
  return <Section id="turn" title={t("panel.turnSection")} className="turn-section" ariaLabel={t("panel.turnSection")}
    busy={state === "loading"}>
    {model != null && (
      <div className="kv" title={t("panel.modelHint")}>
        <span className="k">{t("panel.model")}</span><span className="v">{model}</span>
      </div>
    )}
    {hasContext && (
      <div className={`turn-context ${tone}`} title={t("panel.contextHint")}>
        <div className="kv">
          <span className="k">{t("panel.context")}</span>
          <span className="v">
            {context}
            {percent != null && <span className="turn-pill">{Math.round(percent)}%</span>}
          </span>
        </div>
        {percent != null && (
          <div className="turn-meter" role="progressbar"
            aria-label={t("panel.contextUsage")} aria-valuemin={0} aria-valuemax={100}
            aria-valuenow={Math.min(100, Math.max(0, percent))}
            aria-valuetext={context}>
            <i style={{ width: `${Math.min(100, Math.max(0, percent))}%` }} />
          </div>
        )}
      </div>
    )}
    {hasTurn && (
      <div className="turn-rows">
        <div className="turn-divider" />
        <div className="turn-metrics">
          <div className="turn-metric" title={t("panel.turnUsageHint")}>
            <span className="m-k">turn usage</span>
            <span className="m-v" title={stats?.totalTokens?.toLocaleString()}>{tokens(stats?.totalTokens)}</span>
          </div>
          <div className="turn-metric" title={t("panel.inputTokensHint")}>
            <span className="m-k">input tokens</span>
            <span className="m-v" title={stats?.inputTokens?.toLocaleString()}>{tokens(stats?.inputTokens)}</span>
          </div>
          <div className="turn-metric" title={t("panel.outputTokensHint")}>
            <span className="m-k">output tokens</span>
            <span className="m-v" title={stats?.tokens == null ? undefined
              : `${stats.tokens.toLocaleString()}${reasoning != null ? ` · ${t("panel.reasoningOf", reasoning.toLocaleString())}` : ""}`}>
              {tokens(stats?.tokens)}
              {reasoning != null && <span className="m-sub">{t("panel.reasoningOf", reasoning.toLocaleString())}</span>}
            </span>
          </div>
          <div className="turn-metric" title={cached != null
            ? `${t("panel.cacheHitHint")} ${t("panel.cacheHitKnownHint", cached.toLocaleString())}`
            : `${t("panel.cacheHitHint")} ${t("panel.cacheHitUnknownHint")}`}>
            <span className="m-k">cache read share</span>
            <span className="m-v" title={cached?.toLocaleString()}>
              {hit == null ? "—" : `${hit.toFixed(1)}%`}
            </span>
          </div>
          <div className="turn-metric" title={t("panel.averageOutputHint")}>
            <span className="m-k">output throughput</span>
            <span className="m-v">
              {rate == null ? "—" : `${rate.toFixed(1)} tok/s`}
              {/* 来源细节只跟主值展示；无主值时的估算只作次要文本，不冒充主值。 */}
              {rate != null && basis != null && (
                <span className="m-sub">
                  {basis === "request" ? t("panel.throughputBasisRequest") : t("panel.throughputBasisStream")}
                </span>
              )}
              {rate == null && estimatedRate != null && (
                <span className="m-sub">{t("panel.estimatedThroughput", estimatedRate.toFixed(1))}</span>
              )}
            </span>
          </div>
          <div className="turn-metric" title={t("panel.filesChangedHint")}>
            <span className="m-k">recorded edited files</span>
            <span className="m-v" title={stats?.filesTouched?.toLocaleString()}>
              {stats?.filesTouched == null ? "—" : String(stats.filesTouched)}
            </span>
          </div>
        </div>
      </div>
    )}
    {hasSession && (
      <div className="turn-rows" title={t("panel.sessionTotalHint")}>
        <div className="turn-divider" />
        <div className="kv">
          <span className="k">{t("panel.sessionTotal")}</span>
          <span className="v turn-strong">{tokens(stats?.sessionTokens)}<small>{t("panel.tokensUnit")}</small></span>
        </div>
      </div>
    )}
    {!hasData && <p className="turn-status" role="status">{
      state === "loading" ? t("panel.loadingStats") : state === "waiting" ? t("panel.statsAfterFirstTurn") :
      state === "error" ? t("panel.statsUnavailable") : t("panel.noStats")
    }</p>}
  </Section>;
}

export function TurnSection({ session, agentState, currentTool }: {
  session: Session; agentState?: string | null; currentTool?: string | null;
}) {
  const [stats, setStats] = useState<TurnStats | null>(null);
  const [state, setState] = useState<"loading" | "waiting" | "error" | "ready">("loading");
  const previous = useRef({ identity: "", agentState });
  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const identity = `${session.id}:${session.agentSessionId ?? ""}`;
    const reset = previous.current.identity !== identity || (agentState === "working" && previous.current.agentState !== "working");
    previous.current = { identity, agentState };
    if (reset) { setStats(null); setState("loading"); }
    if (!session.agentSessionId) { setState("waiting"); return; }
    const sample = async (settled = false) => {
      try {
        const next = await agentTurnStats(session.id);
        if (!cancelled) { setStats(next); setState("ready"); }
      } catch {
        if (!cancelled) { setStats(null); setState("error"); }
      }
      // Serialize reads. A final delayed read absorbs the provider's turn-end disk flush.
      if (!cancelled && (agentState === "working" || !settled)) {
        timer = setTimeout(() => { void sample(true); }, agentState === "working" ? 1500 : 750);
      }
    };
    void sample();
    return () => { cancelled = true; clearTimeout(timer); };
  }, [session.id, session.agentSessionId, agentState, currentTool]);
  return <TurnStatsView stats={stats} state={state} />;
}
