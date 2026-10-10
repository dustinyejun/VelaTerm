//! State categories behind the sidebar dots, the status filters, and the status bar counts.

import { describe, expect, it } from "vitest";
import { countByAgentState, effectiveStatus, matchesAgentState, type SessionRuntime } from "./types";

const runtime = (agentState: SessionRuntime["agentState"]): SessionRuntime =>
  ({ status: "running", agent: "claude", agentState }) as SessionRuntime;

describe("Codex process lifecycle before the first hook", () => {
  const started: SessionRuntime = {
    status: "idle",
    alive: true,
    agent: "codex",
    agentStateSource: "hooks",
    agentHookReady: false,
    authoritative: true,
  };

  it("shows a backend-confirmed process as running without counting an empty session as agent activity", () => {
    expect(effectiveStatus(started)).toBe("running");
    expect(countByAgentState([{ id: "codex" }], { codex: started }, {})).toEqual({
      working: 0, asking: 0, waiting: 0, background: 0,
    });
    expect(effectiveStatus({ ...started, alive: false })).toBe("idle");
  });

  it.each(["exited", "error"] as const)("preserves %s despite stale process or hook state", (status) => {
    expect(effectiveStatus({ ...started, status, agentState: "working" })).toBe(status);
  });
});

describe("background agent state", () => {
  it("counts as background only once its reply has been read", () => {
    expect(matchesAgentState("background", "background", false)).toBe(true);
    expect(matchesAgentState("background", "background", true)).toBe(false);
    expect(matchesAgentState("asking", "background", true)).toBe(true);
    expect(matchesAgentState("working", "background", false)).toBe(false);
    expect(matchesAgentState("waiting", "background", false)).toBe(false);
  });

  it("puts each session in exactly one count", () => {
    const sessions = [{ id: "a" }, { id: "b" }, { id: "c" }, { id: "d" }];
    const runtimes = {
      a: runtime("background"),
      b: runtime("background"),
      c: runtime("working"),
      d: runtime("waiting"),
    };
    expect(countByAgentState(sessions, runtimes, { b: 1 })).toEqual({
      working: 1,
      asking: 1,
      waiting: 1,
      background: 1,
    });
  });
});
