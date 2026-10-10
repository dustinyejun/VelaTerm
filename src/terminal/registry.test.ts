import type { Terminal } from "@xterm/xterm";
import { afterEach, describe, expect, it, vi } from "vitest";

import { ptyRedraw } from "../ipc/commands";
import {
  clearTerminalTextureAtlases,
  redrawTerminal,
  registerRedraw,
  registerTerminal,
  unregisterRedraw,
  unregisterTerminal,
} from "./registry";

vi.mock("../ipc/commands", () => ({ ptyRedraw: vi.fn().mockResolvedValue(undefined) }));

const cleanup: (() => void)[] = [];
afterEach(() => {
  cleanup.splice(0).forEach(fn => fn());
  vi.clearAllMocks();
});

function terminal(id: string, clear = vi.fn()) {
  const term = { clearTextureAtlas: clear, refresh: vi.fn(), rows: 24 } as unknown as Terminal;
  registerTerminal(id, term);
  cleanup.push(() => unregisterTerminal(id, term));
  return term;
}

describe("shared WebGL atlas recovery", () => {
  it("invalidates visible and hidden peers before a manual redraw callback can repaint", () => {
    const invalidated: string[] = [];
    terminal("visible", vi.fn(() => invalidated.push("visible")));
    terminal("hidden", vi.fn(() => invalidated.push("hidden")));
    const redraw = vi.fn(() => expect(invalidated).toEqual(["visible", "hidden"]));
    registerRedraw("visible", redraw);
    cleanup.push(() => unregisterRedraw("visible", redraw));

    redrawTerminal("visible");

    expect(redraw).toHaveBeenCalledOnce();
  });

  it("recovers remaining peers if one renderer is already lost or disposed", () => {
    terminal("lost", vi.fn(() => { throw new Error("context lost"); }));
    const peer = terminal("peer");

    expect(() => clearTerminalTextureAtlases()).not.toThrow();
    expect(peer.clearTextureAtlas).toHaveBeenCalledOnce();
  });

  it("also invalidates peers when the manual redraw uses its fallback handler", () => {
    const requested = terminal("requested");
    const peer = terminal("peer");

    redrawTerminal("requested");

    expect(requested.clearTextureAtlas).toHaveBeenCalledOnce();
    expect(peer.clearTextureAtlas).toHaveBeenCalledOnce();
    expect(requested.refresh).toHaveBeenCalledWith(0, 23);
    expect(ptyRedraw).toHaveBeenCalledWith("requested");
  });
});
