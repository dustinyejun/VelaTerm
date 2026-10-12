//! Provider-reported turn accounting. Read complete histories once, then consume only appended lines.
//! Token usage is deduplicated by message/response identity; absent usage is never represented as zero.
//! Claude, Codex and OpenCode are read from their own recordings; Pi and OMP keep one JSONL tree each,
//! whose active branch supplies the same numbers.

use super::{resume, AgentContextInfo};
use crate::agent::opencode_store::{self, OpencodeMessage};
use crate::models::SessionKind;
use serde::Serialize;
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::io::{BufRead, BufReader, Seek, SeekFrom};
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};
use std::time::SystemTime;

mod timing;
mod grok;
mod antigravity;
use timing::RecordedTiming;

#[derive(Clone, Debug, Default, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct TurnStats {
    /// Model the latest context reading came from, when the provider names one.
    pub model: Option<String>,
    pub tokens: Option<u64>,
    pub input_tokens: Option<u64>,
    pub total_tokens: Option<u64>,
    pub cached_tokens: Option<u64>,
    pub cache_hit_percent: Option<f64>,
    pub session_tokens: Option<u64>,
    pub tools_used: Option<u32>,
    /// Successful, explicitly recorded file edits, not the working tree's pre-existing changes.
    pub files_touched: Option<u32>,
    pub context_tokens: Option<u64>,
    pub context_limit: Option<u64>,
    pub context_percent: Option<f64>,
    /// 与本轮用量完整匹配的模型请求或流式吞吐率。
    pub generation_tokens_per_second: Option<f64>,
    pub output_throughput_basis: Option<ThroughputBasis>,
    /// 仅有回合时间戳时的响应估算，不冒充模型请求吞吐率。
    pub estimated_output_tokens_per_second: Option<f64>,
    pub reasoning_tokens: Option<u64>,
}

#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ThroughputBasis {
    Request,
    Stream,
}
impl TurnStats {
    pub fn with_context(&mut self, context: &AgentContextInfo) {
        self.model = context.model.clone();
        self.context_tokens = context.context_tokens;
        self.context_limit = (context.context_limit > 0).then_some(context.context_limit);
        self.context_percent = self
            .context_tokens
            .zip(self.context_limit)
            .map(|(used, limit)| used as f64 * 100.0 / limit as f64);
    }
}

#[derive(Clone, Copy, Debug)]
struct Usage {
    input: u64,
    output: u64,
    cached: u64,
    cache_known: bool,
    reasoning: Option<u64>,
}
impl Default for Usage {
    fn default() -> Self {
        Self {
            input: 0,
            output: 0,
            cached: 0,
            cache_known: true,
            reasoning: Some(0),
        }
    }
}
impl Usage {
    fn parse(value: &Value, claude: bool) -> Option<Self> {
        let input = value.get("input_tokens")?.as_u64()?;
        let output = value.get("output_tokens")?.as_u64()?;
        let cache_value = value
            .get(if claude {
                "cache_read_input_tokens"
            } else {
                "cached_input_tokens"
            })
            .and_then(Value::as_u64);
        let cached = cache_value.unwrap_or(0);
        let created = value
            .get("cache_creation_input_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        Some(Self {
            input: if claude {
                input.saturating_add(cached).saturating_add(created)
            } else {
                input
            },
            output,
            cached,
            cache_known: cache_value.is_some(),
            reasoning: value
                .pointer(if claude {
                    "/output_tokens_details/thinking_tokens"
                } else {
                    "/reasoning_output_tokens"
                })
                .and_then(Value::as_u64)
                .filter(|count| *count <= output),
        })
    }
    fn total(self) -> u64 {
        self.input.saturating_add(self.output)
    }
    fn add(&mut self, other: Self) {
        self.input = self.input.saturating_add(other.input);
        self.output = self.output.saturating_add(other.output);
        self.cached = self.cached.saturating_add(other.cached);
        self.cache_known &= other.cache_known;
        self.reasoning = self.reasoning.zip(other.reasoning)
            .map(|(a, b)| a.saturating_add(b));
    }
    fn difference(self, base: Self) -> Self {
        Self {
            input: self.input.saturating_sub(base.input),
            output: self.output.saturating_sub(base.output),
            cached: self.cached.saturating_sub(base.cached),
            cache_known: self.cache_known && base.cache_known,
            reasoning: self.reasoning.zip(base.reasoning)
                .map(|(a, b)| a.saturating_sub(b)),
        }
    }
}

#[derive(Default)]
struct Accounting {
    messages: HashMap<String, Usage>,
    turn_messages: HashSet<String>,
    tools: HashSet<String>,
    pending_files: HashMap<String, Vec<String>>,
    files: HashSet<String>,
    session: Option<Usage>,
    baseline: Option<Usage>,
    turn: Option<Usage>,
    native_turn: Option<String>,
    native_turn_usage: bool,
    has_turn: bool,
    line_number: u64,
    timing: RecordedTiming,
}
impl Accounting {
    fn start(&mut self) {
        self.baseline = self.session;
        self.turn = None;
        self.native_turn_usage = false;
        self.turn_messages.clear();
        self.tools.clear();
        self.pending_files.clear();
        self.files.clear();
        self.has_turn = true;
        self.timing = RecordedTiming::default();
    }
    fn consume(&mut self, line: &str, kind: SessionKind) {
        self.line_number += 1;
        let Ok(v) = serde_json::from_str::<Value>(line) else {
            return;
        };
        if kind == SessionKind::Claude {
            self.claude(&v, line);
        } else {
            self.codex(&v, line);
        }
    }
    fn claude(&mut self, v: &Value, line: &str) {
        if v.get("isSidechain").and_then(Value::as_bool) == Some(true) {
            return;
        }
        if super::parse_claude_line(line).is_some_and(|p| p.role == "user") {
            self.start();
            self.timing.start(v);
        }
        if v.get("subtype").and_then(Value::as_str) == Some("compact_boundary") {
            self.timing.invalidate();
        }
        let Some(message) = v.get("message") else {
            return;
        };
        let blocks = message.get("content").and_then(Value::as_array);
        if v.get("type").and_then(Value::as_str) == Some("user") {
            for block in blocks.into_iter().flatten() {
                if block.get("type").and_then(Value::as_str) != Some("tool_result") {
                    continue;
                }
                let id = block
                    .get("tool_use_id")
                    .and_then(Value::as_str)
                    .unwrap_or("");
                self.timing.finish_tool(id, v);
                if let Some(paths) = self.pending_files.remove(id) {
                    if block.get("is_error").and_then(Value::as_bool) != Some(true) {
                        self.files.extend(paths);
                    }
                }
            }
            return;
        }
        if v.get("type").and_then(Value::as_str) != Some("assistant")
            || message.get("model").and_then(Value::as_str) == Some("<synthetic>")
        {
            return;
        }
        let id = message
            .get("id")
            .and_then(Value::as_str)
            .map(str::to_string)
            .unwrap_or_else(|| format!("line-{}", self.line_number));
        if let Some(usage) = message.get("usage").and_then(|u| Usage::parse(u, true)) {
            let previous = self.messages.insert(id.clone(), usage).unwrap_or_default();
            let mut session = self.session.unwrap_or_default().difference(previous);
            session.add(usage);
            self.session = Some(session);
            self.turn_messages.insert(id.clone());
            let mut turn = Usage::default();
            for message_id in &self.turn_messages {
                if let Some(usage) = self.messages.get(message_id) {
                    turn.add(*usage);
                }
            }
            self.turn = Some(turn);
            self.timing.output(turn.output, v, true);
        }
        for (index, block) in blocks.into_iter().flatten().enumerate() {
            if block.get("type").and_then(Value::as_str) != Some("tool_use") {
                continue;
            }
            let call_id = block
                .get("id")
                .and_then(Value::as_str)
                .map(str::to_string)
                .unwrap_or_else(|| format!("{id}-{index}"));
            self.timing.start_tool(block.get("id").and_then(Value::as_str).unwrap_or(""), v);
            if !self.tools.insert(call_id.clone()) {
                continue;
            }
            let name = block.get("name").and_then(Value::as_str).unwrap_or("");
            if super::is_edit_tool(name) {
                if let Some(path) = block.get("input").and_then(super::tool_file_path) {
                    self.pending_files.insert(call_id, vec![path]);
                }
            }
        }
    }
    fn codex(&mut self, v: &Value, line: &str) {
        if v.get("type").and_then(Value::as_str) == Some("compacted") {
            self.timing.invalidate();
        }
        let Some(p) = v.get("payload") else { return };
        let row_type = v.get("type").and_then(Value::as_str).unwrap_or("");
        let kind = p.get("type").and_then(Value::as_str).unwrap_or("");
        if row_type == "event_msg" && kind == "task_started" {
            let id = p.get("turn_id").and_then(Value::as_str);
            if id.is_none() || self.native_turn.as_deref() != id {
                self.start();
                self.timing.start(v);
                self.native_turn = Some(
                    id.map(str::to_string)
                        .unwrap_or_else(|| format!("legacy-{}", self.line_number)),
                );
            }
        } else if self.native_turn.is_none()
            && super::parse_codex_line(line).is_some_and(|p| p.role == "user")
        {
            self.start();
            self.timing.start(v);
        }
        if row_type == "token_usage_record" {
            // Ignore delayed usage from another turn rather than pairing it with this turn's clock.
            if p.get("turn_id").and_then(Value::as_str)
                .zip(self.native_turn.as_deref()).is_some_and(|(id, current)| id != current) {
                return;
            }
            if let Some(usage) = p
                .get("thread_token_usage")
                .and_then(|u| Usage::parse(u, false))
            {
                self.session = Some(usage);
            }
            if let Some(usage) = p
                .get("turn_token_usage")
                .and_then(|u| Usage::parse(u, false))
            {
                self.turn = Some(usage);
                self.native_turn_usage = true;
                self.timing.output(usage.output, v, false);
            }
        }
        if row_type == "event_msg" && kind == "token_count" {
            if let Some(usage) = p
                .pointer("/info/total_token_usage")
                .and_then(|u| Usage::parse(u, false))
            {
                self.session = Some(usage);
                // Older CLIs expose only cumulative counters. A decreasing counter starts a new epoch.
                let base = self
                    .baseline
                    .filter(|b| b.input <= usage.input && b.output <= usage.output)
                    .unwrap_or_default();
                if !self.native_turn_usage {
                    let turn = usage.difference(base);
                    self.turn = Some(turn);
                    self.timing.output(turn.output, v, false);
                }
            }
        }
        if row_type != "response_item" {
            return;
        }
        if matches!(
            kind,
            "function_call" | "custom_tool_call" | "local_shell_call" | "web_search_call"
        ) {
            let id = p
                .get("call_id")
                .or_else(|| p.get("id"))
                .and_then(Value::as_str)
                .map(str::to_string)
                .unwrap_or_else(|| format!("line-{}", self.line_number));
            self.timing.start_tool(p.get("call_id").or_else(|| p.get("id"))
                .and_then(Value::as_str).unwrap_or(""), v);
            if !self.tools.insert(id.clone()) {
                return;
            }
            let name = p.get("name").and_then(Value::as_str).unwrap_or("");
            if name.rsplit('.').next() == Some("apply_patch") {
                let raw = p
                    .get("input")
                    .or_else(|| p.get("arguments"))
                    .and_then(Value::as_str)
                    .unwrap_or("");
                let args = serde_json::from_str::<Value>(raw).ok();
                let patch = args
                    .as_ref()
                    .and_then(|a| a.get("patch").and_then(Value::as_str))
                    .unwrap_or(raw);
                self.pending_files.insert(id, patch_paths(patch));
            }
        }
        if matches!(kind, "function_call_output" | "custom_tool_call_output" | "local_shell_call_output" | "web_search_call_output") {
            let id = p.get("call_id").and_then(Value::as_str).unwrap_or("");
            self.timing.finish_tool(id, v);
            if let Some(paths) = self.pending_files.remove(id) {
                let output = p.get("output").and_then(Value::as_str).unwrap_or("");
                if output.contains("Success. Updated the following files:")
                    || output.contains("\"exit_code\":0")
                    || output.contains("\"exit_code\": 0")
                {
                    self.files.extend(paths);
                }
            }
        }
    }
    fn snapshot(&self) -> TurnStats {
        let turn = self.turn;
        TurnStats {
            tokens: turn.map(|u| u.output),
            input_tokens: turn.map(|u| u.input),
            total_tokens: turn.map(Usage::total),
            cached_tokens: turn.filter(|u| u.cache_known).map(|u| u.cached),
            cache_hit_percent: turn
                .filter(|u| u.cache_known && u.input > 0 && u.cached <= u.input)
                .map(|u| u.cached as f64 * 100.0 / u.input as f64),
            session_tokens: self.session.map(Usage::total),
            tools_used: self.has_turn.then_some(self.tools.len() as u32),
            files_touched: self.has_turn.then_some(self.files.len() as u32),
            estimated_output_tokens_per_second: self.timing.rate(),
            reasoning_tokens: turn.and_then(|u| u.reasoning),
            ..TurnStats::default()
        }
    }
}
fn patch_paths(patch: &str) -> Vec<String> {
    patch
        .lines()
        .filter_map(|line| {
            [
                "*** Update File: ",
                "*** Add File: ",
                "*** Delete File: ",
                "*** Move to: ",
            ]
            .iter()
            .find_map(|prefix| line.strip_prefix(prefix))
        })
        .map(str::trim)
        .filter(|p| !p.is_empty())
        .map(str::to_string)
        .collect()
}

#[derive(Default)]
struct Cached {
    path: PathBuf,
    modified: Option<SystemTime>,
    offset: u64,
    accounting: Accounting,
}
static CACHE: OnceLock<Mutex<HashMap<String, Cached>>> = OnceLock::new();

/// 只累计与用量匹配的完整请求窗口，缺失任一步计时就不报告整轮吞吐率。
#[derive(Default)]
struct RequestTiming {
    elapsed_ms: u64,
    tokens: u64,
    incomplete: bool,
}
impl RequestTiming {
    fn add(&mut self, output: Option<u64>, duration: Option<u64>) {
        match (output, duration) {
            (Some(tokens), Some(ms)) if ms > 0 => {
                self.tokens = self.tokens.saturating_add(tokens);
                self.elapsed_ms = self.elapsed_ms.saturating_add(ms);
            }
            _ => self.incomplete = true,
        }
    }
    fn rate(&self) -> Option<f64> {
        (!self.incomplete && self.tokens > 0 && self.elapsed_ms >= 100)
            .then(|| self.tokens as f64 * 1000.0 / self.elapsed_ms as f64)
    }
}

fn opencode_stats_from_messages(messages: &[OpencodeMessage]) -> TurnStats {
    let input = |m: &OpencodeMessage| matches!(m.info.get("role").and_then(Value::as_str),
        Some("user" | "synthetic" | "compaction"));
    let last_input = messages.iter().rposition(input);
    let modern = last_input.is_some_and(|index| messages[index].info.get("seq").is_some());
    // 新版 idle 划分整轮，运行中追加提示不另开一轮；没有 idle 时从新版记录开始，排除未迁移的旧历史。
    let turn_start = if modern {
        last_input.map(|last_input| {
            let before = &messages[..last_input];
            before.iter().rposition(|m| m.info.get("role").and_then(Value::as_str) == Some("idle"))
                .map(|index| index + 1)
                .unwrap_or_else(|| before.iter().position(|m| m.info.get("seq").is_some()).unwrap_or(last_input))
        })
    } else {
        last_input.map(|index| index + 1)
    };
    let turn_id = (!modern).then_some(last_input).flatten()
        .and_then(|index| messages[index].info.get("id").and_then(Value::as_str));
    let mut session = Usage::default();
    let mut session_known = false;
    let mut turn = Usage::default();
    let mut turn_known = false;
    let mut turn_incomplete = false;
    let mut tools = HashSet::new();
    let mut files = HashSet::new();
    let mut timing = RequestTiming::default();
    for (index, message) in messages.iter().enumerate() {
        let info = &message.info;
        if !matches!(info.get("role").and_then(Value::as_str), Some("assistant" | "compaction")) {
            continue;
        }
        let usage = info.get("tokens").and_then(opencode_usage);
        if let Some(usage) = usage {
            session.add(usage);
            session_known = true;
        }
        let in_turn = turn_start.is_none_or(|start| index >= start)
            && info.get("parentID").and_then(Value::as_str)
                .is_none_or(|parent| turn_id.is_none_or(|id| parent == id));
        if !in_turn {
            continue;
        }
        if let Some(usage) = usage {
            turn.add(usage);
            turn_known = true;
        } else {
            turn_incomplete = true;
        }
        timing.add(usage.map(|u| u.output), opencode_step_ms(info));
        for path in info.pointer("/snapshot/files").and_then(Value::as_array)
            .into_iter().flatten().filter_map(Value::as_str)
        {
            files.insert(path.to_string());
        }
        for part in &message.parts {
            match part.get("type").and_then(Value::as_str) {
                Some("tool") => {
                    if let Some(id) = part.get("callID").or_else(|| part.get("id")).and_then(Value::as_str) {
                        tools.insert(id.to_string());
                    }
                    if part.pointer("/state/status").and_then(Value::as_str) == Some("completed") {
                        match part.get("tool").and_then(Value::as_str) {
                            Some("edit" | "write" | "multiedit") => {
                                if let Some(input) = part.pointer("/state/input") {
                                    if let Some(path) = input.get("filePath").and_then(Value::as_str)
                                        .map(str::to_string).or_else(|| super::tool_file_path(input))
                                    {
                                        files.insert(path);
                                    }
                                }
                            }
                            Some("apply_patch" | "patch") => {
                                if let Some(patch) = part.pointer("/state/input/patchText")
                                    .or_else(|| part.pointer("/state/input/patch_text")).and_then(Value::as_str)
                                {
                                    files.extend(patch_paths(patch));
                                }
                            }
                            _ => {}
                        }
                    }
                }
                Some("patch") => {
                    for path in part.get("files").and_then(Value::as_array).into_iter().flatten().filter_map(Value::as_str) {
                        files.insert(path.to_string());
                    }
                }
                _ => {}
            }
        }
    }
    let known = turn_known && !turn_incomplete;
    let rate = timing.rate();
    TurnStats {
        tokens: known.then_some(turn.output),
        input_tokens: known.then_some(turn.input),
        total_tokens: known.then_some(turn.total()),
        cached_tokens: (known && turn.cache_known).then_some(turn.cached),
        cache_hit_percent: (known && turn.cache_known && turn.input > 0 && turn.cached <= turn.input)
            .then(|| turn.cached as f64 * 100.0 / turn.input as f64),
        session_tokens: session_known.then_some(session.total()),
        tools_used: turn_start.map(|_| tools.len() as u32),
        files_touched: turn_start.map(|_| files.len() as u32),
        generation_tokens_per_second: rate,
        output_throughput_basis: rate.map(|_| ThroughputBasis::Request),
        reasoning_tokens: known.then_some(turn.reasoning).flatten(),
        ..TurnStats::default()
    }
}

fn opencode_usage(tokens: &Value) -> Option<Usage> {
    let input = tokens.get("input")?.as_u64()?;
    let output = tokens.get("output")?.as_u64()?;
    let reasoning = tokens.get("reasoning").and_then(Value::as_u64);
    let cache_read = tokens.pointer("/cache/read").and_then(Value::as_u64);
    let cached = cache_read.unwrap_or(0);
    let created = tokens.pointer("/cache/write").and_then(Value::as_u64).unwrap_or(0);
    Some(Usage {
        input: input.saturating_add(cached).saturating_add(created),
        output: output.saturating_add(reasoning.unwrap_or(0)),
        cached,
        cache_known: cache_read.is_some(),
        reasoning,
    })
}

fn opencode_step_ms(info: &Value) -> Option<u64> {
    let created = info.pointer("/time/created")?.as_u64()?;
    // completed 可能在工具执行后才写入，只有 streamed 能定义模型请求的结束。
    let streamed = info.pointer("/time/streamed")?.as_u64()?;
    streamed.checked_sub(created).filter(|ms| *ms > 0)
}

/// Pi and OMP record one `usage` object per assistant message: uncached input, output with reasoning
/// already included, and the cache read/write pair. As with Claude, input is widened to include cache
/// traffic, so the panel's input, cache-hit and turn-token figures mean the same thing for every agent.
fn pi_usage(usage: &Value) -> Option<Usage> {
    let input = usage.get("input")?.as_u64()?;
    let output = usage.get("output")?.as_u64()?;
    let cache_read = usage.get("cacheRead").and_then(Value::as_u64);
    let cached = cache_read.unwrap_or(0);
    let created = usage.get("cacheWrite").and_then(Value::as_u64).unwrap_or(0);
    Some(Usage {
        input: input.saturating_add(cached).saturating_add(created),
        output,
        cached,
        cache_known: cache_read.is_some(),
        reasoning: usage.get("reasoning").and_then(Value::as_u64).filter(|count| *count <= output),
    })
}

/// Native model-request durations, excluding tools and time between user messages.
/// Pi uses `durationMs`; older OMP recordings use `duration` (also milliseconds).
fn pi_step_ms(message: &Value) -> Option<u64> {
    let ms = message.get("durationMs").or_else(|| message.get("duration"))?.as_f64()?;
    (ms.is_finite() && ms >= 1.0 && ms < u64::MAX as f64).then_some(ms as u64)
}

/// The file-editing tools Pi and OMP expose. Both write through lowercase `edit` and `write`; OMP adds the
/// AST rewrite. Matching is by exact name, so a model inventing a differently-cased tool is not counted.
fn is_pi_edit_tool(name: &str) -> bool {
    matches!(name, "edit" | "write" | "ast_edit" | "apply_patch")
}

/// Turn accounting for one Pi or OMP recording.
///
/// A turn begins at the last user message and consists of every assistant message after it, one per model
/// step. Session totals accumulate all assistant messages seen, including ones later abandoned by a fork,
/// because those tokens were still spent.
#[derive(Default)]
struct PiAccounting {
    messages: HashMap<String, Usage>,
    turn_messages: HashSet<String>,
    tools: HashSet<String>,
    pending_files: HashMap<String, Vec<String>>,
    files: HashSet<String>,
    session: Option<Usage>,
    turn: Option<Usage>,
    timings: HashMap<String, u64>,
    missing_usage: HashSet<String>,
    has_turn: bool,
    /// Id of the last entry consumed, which the next appended entry has to name as its parent.
    tail: Option<String>,
}
impl PiAccounting {
    fn start(&mut self) {
        self.turn = None;
        self.turn_messages.clear();
        self.tools.clear();
        self.pending_files.clear();
        self.files.clear();
        self.timings.clear();
        self.missing_usage.clear();
        self.has_turn = true;
    }
    fn consume(&mut self, entry: &Value) {
        if let Some(id) = entry.get("id").and_then(Value::as_str) {
            self.tail = Some(id.to_string());
        }
        if entry.get("type").and_then(Value::as_str) != Some("message") {
            return;
        }
        let Some(id) = entry.get("id").and_then(Value::as_str) else {
            return;
        };
        let Some(message) = entry.get("message") else {
            return;
        };
        match message.get("role").and_then(Value::as_str) {
            Some("user") => self.start(),
            Some("assistant") => self.assistant(id, message),
            Some("toolResult") => self.tool_result(message),
            _ => {}
        }
    }
    fn assistant(&mut self, id: &str, message: &Value) {
        if let Some(usage) = message.get("usage").and_then(pi_usage) {
            self.missing_usage.remove(id);
            let previous = self.messages.insert(id.to_string(), usage).unwrap_or_default();
            let mut session = self.session.unwrap_or_default().difference(previous);
            session.add(usage);
            self.session = Some(session);
            if self.has_turn {
                self.turn_messages.insert(id.to_string());
                let mut turn = Usage::default();
                for message_id in &self.turn_messages {
                    if let Some(usage) = self.messages.get(message_id) {
                        turn.add(*usage);
                    }
                }
                self.turn = Some(turn);
                self.timings.remove(id);
                if let Some(ms) = pi_step_ms(message) {
                    self.timings.insert(id.to_string(), ms);
                }
            }
        } else if self.has_turn {
            self.missing_usage.insert(id.to_string());
        }
        let Some(blocks) = message.get("content").and_then(Value::as_array) else {
            return;
        };
        for block in blocks {
            if block.get("type").and_then(Value::as_str) != Some("toolCall") {
                continue;
            }
            let Some(call_id) = block.get("id").and_then(Value::as_str) else {
                continue;
            };
            if !self.tools.insert(call_id.to_string()) {
                continue;
            }
            let name = block.get("name").and_then(Value::as_str).unwrap_or("");
            if is_pi_edit_tool(name) {
                if let Some(path) = block.get("arguments").and_then(super::tool_file_path) {
                    self.pending_files.insert(call_id.to_string(), vec![path]);
                }
            }
        }
    }
    fn tool_result(&mut self, message: &Value) {
        let call_id = message
            .get("toolCallId")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let Some(paths) = self.pending_files.remove(call_id) else {
            return;
        };
        if message.get("isError").and_then(Value::as_bool) != Some(true) {
            self.files.extend(paths);
        }
    }
    fn snapshot(&self) -> TurnStats {
        let turn = self.turn.filter(|_| self.missing_usage.is_empty());
        let mut timing = RequestTiming::default();
        timing.incomplete = !self.missing_usage.is_empty();
        for id in &self.turn_messages {
            timing.add(self.messages.get(id).map(|u| u.output), self.timings.get(id).copied());
        }
        let rate = timing.rate();
        TurnStats {
            tokens: turn.map(|u| u.output),
            input_tokens: turn.map(|u| u.input),
            total_tokens: turn.map(Usage::total),
            cached_tokens: turn.filter(|u| u.cache_known).map(|u| u.cached),
            cache_hit_percent: turn
                .filter(|u| u.cache_known && u.input > 0 && u.cached <= u.input)
                .map(|u| u.cached as f64 * 100.0 / u.input as f64),
            session_tokens: self.session.map(Usage::total),
            tools_used: self.has_turn.then_some(self.tools.len() as u32),
            files_touched: self.has_turn.then_some(self.files.len() as u32),
            generation_tokens_per_second: rate,
            output_throughput_basis: rate.map(|_| ThroughputBasis::Request),
            reasoning_tokens: turn.and_then(|u| u.reasoning),
            ..TurnStats::default()
        }
    }
}

/// Whether appended lines leave the branch the accounting already followed.
///
/// A fork or `/tree` jump writes its new entries under an earlier parent, which makes every count derived
/// from file order wrong; detecting that early lets the caller rebuild from the branch instead.
fn pi_branch_broken(tail: Option<&str>, suffix: &str) -> bool {
    let mut expected = tail.map(str::to_string);
    for line in suffix.lines() {
        let Ok(entry) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        let Some(id) = entry.get("id").and_then(Value::as_str) else {
            continue;
        };
        if entry.get("parentId").and_then(Value::as_str) != expected.as_deref() {
            return true;
        }
        expected = Some(id.to_string());
    }
    false
}

#[derive(Default)]
struct PiCached {
    path: PathBuf,
    modified: Option<SystemTime>,
    offset: u64,
    accounting: PiAccounting,
}
static PI_CACHE: OnceLock<Mutex<HashMap<String, PiCached>>> = OnceLock::new();

/// Reads a Pi or OMP recording and consumes only the lines appended since the last read. A fork forces a
/// rebuild, because entries written under an earlier parent are not the continuation of the current branch.
fn pi_stats_from_file(kind: SessionKind, agent_session_id: &str) -> Result<TurnStats, String> {
    let key = format!("{kind:?}:{agent_session_id}");
    let mut cache = PI_CACHE.get_or_init(Default::default).lock().unwrap();
    if cache.len() >= 16 && !cache.contains_key(&key) {
        cache.clear();
    }
    let entry = cache.entry(key).or_default();
    if entry.path.as_os_str().is_empty() || !entry.path.is_file() {
        entry.path = resume::find_pi_session(kind, agent_session_id)
            .ok_or("The agent's session recording could not be found")?;
        entry.offset = 0;
        entry.modified = None;
        entry.accounting = PiAccounting::default();
    }
    pi_read_cached(entry)
}

fn pi_read_cached(entry: &mut PiCached) -> Result<TurnStats, String> {
    let metadata = std::fs::metadata(&entry.path).map_err(|e| e.to_string())?;
    let modified = metadata.modified().ok();
    if entry.modified == modified && entry.offset == metadata.len() {
        return Ok(entry.accounting.snapshot());
    }
    if metadata.len() <= entry.offset {
        // The recording was replaced rather than appended to; every count has to start over.
        entry.offset = 0;
        entry.accounting = PiAccounting::default();
    }
    let mut file = BufReader::new(std::fs::File::open(&entry.path).map_err(|e| e.to_string())?);
    file.seek(SeekFrom::Start(entry.offset))
        .map_err(|e| e.to_string())?;
    let mut suffix = String::new();
    let mut complete_bytes = 0;
    loop {
        let mut line = String::new();
        let n = file.read_line(&mut line).map_err(|e| e.to_string())?;
        if n == 0 || !line.ends_with('\n') {
            break;
        }
        complete_bytes += n as u64;
        suffix.push_str(&line);
    }
    if entry.offset == 0 || pi_branch_broken(entry.accounting.tail.as_deref(), &suffix) {
        let end = entry.offset + complete_bytes;
        let content = std::fs::read_to_string(&entry.path).map_err(|e| e.to_string())?;
        let content = content.get(..end as usize).ok_or("Invalid recording boundary")?;
        entry.accounting = pi_accounting_from_text(content);
    } else {
        for line in suffix.lines() {
            if let Ok(entry_value) = serde_json::from_str::<Value>(line) {
                entry.accounting.consume(&entry_value);
            }
        }
    }
    entry.offset += complete_bytes;
    entry.modified = modified;
    Ok(entry.accounting.snapshot())
}

fn pi_accounting_from_text(text: &str) -> PiAccounting {
    let mut all = PiAccounting::default();
    for line in text.lines() {
        if let Ok(entry) = serde_json::from_str::<Value>(line) {
            all.consume(&entry);
        }
    }
    let mut active = PiAccounting::default();
    for entry in resume::pi_active_branch(text) {
        active.consume(&entry);
    }
    // 切换分支只改变当前回合，已消耗的历史用量仍计入主会话。
    active.messages = all.messages;
    active.session = all.session;
    active
}

pub fn current_turn_stats(kind: SessionKind, agent_session_id: &str) -> Result<TurnStats, String> {
    if kind == SessionKind::Kiro { return crate::agent::kiro_info::turn_stats(agent_session_id); }
    if kind == SessionKind::Opencode {
        return Ok(opencode_stats_from_messages(&opencode_store::messages(agent_session_id)?));
    }
    if matches!(kind, SessionKind::Pi | SessionKind::Omp) {
        return pi_stats_from_file(kind, agent_session_id);
    }
    if kind == SessionKind::Grok { return grok::stats(agent_session_id); }
    if kind == SessionKind::Antigravity { return antigravity::stats(agent_session_id); }
    if !matches!(kind, SessionKind::Claude | SessionKind::Codex) {
        return Ok(TurnStats::default());
    }
    let key = format!("{kind:?}:{agent_session_id}");
    let mut cache = CACHE.get_or_init(Default::default).lock().unwrap();
    // Bound retained accounting state when many archived sessions are inspected.
    if cache.len() >= 16 && !cache.contains_key(&key) {
        cache.clear();
    }
    let entry = cache.entry(key).or_default();
    if entry.path.as_os_str().is_empty() || !entry.path.is_file() {
        entry.path = if kind == SessionKind::Claude {
            resume::find_claude_transcript(agent_session_id)
        } else {
            resume::find_codex_rollout(agent_session_id)
        }
        .ok_or("Agent transcript not found")?;
        entry.offset = 0;
        entry.modified = None;
    }
    read_cached(entry, kind)
}

fn read_cached(entry: &mut Cached, kind: SessionKind) -> Result<TurnStats, String> {
    let metadata = std::fs::metadata(&entry.path).map_err(|e| e.to_string())?;
    let modified = metadata.modified().ok();
    if entry.modified == modified && entry.offset == metadata.len() {
        return Ok(entry.accounting.snapshot());
    }
    if metadata.len() <= entry.offset {
        entry.offset = 0;
    }
    let mut file = BufReader::new(std::fs::File::open(&entry.path).map_err(|e| e.to_string())?);
    file.seek(SeekFrom::Start(entry.offset))
        .map_err(|e| e.to_string())?;
    let mut suffix = String::new();
    let mut complete_bytes = 0;
    loop {
        let mut line = String::new();
        let n = file.read_line(&mut line).map_err(|e| e.to_string())?;
        if n == 0 || !line.ends_with('\n') {
            break;
        }
        complete_bytes += n as u64;
        suffix.push_str(&line);
    }
    let rewound = kind == SessionKind::Claude
        && suffix.lines().any(|line| {
            serde_json::from_str::<Value>(line)
                .ok()
                .is_some_and(|v| v.get("rewound").and_then(Value::as_bool) == Some(true))
        });
    if entry.offset == 0 || rewound {
        if rewound && entry.offset != 0 {
            suffix = std::fs::read_to_string(&entry.path).map_err(|e| e.to_string())?;
            complete_bytes = suffix.rfind('\n').map_or(0, |i| i + 1) as u64;
            suffix.truncate(complete_bytes as usize);
        }
        entry.offset = 0;
        entry.accounting = Accounting::default();
        if kind == SessionKind::Claude {
            // Rewinding changes the active turn, but does not refund tokens already consumed.
            let mut all = Accounting::default();
            for line in suffix.lines() {
                all.consume(line, kind);
            }
            let active = resume::claude_active_branch(&suffix);
            for line in active.lines() {
                entry.accounting.consume(line, kind);
            }
            entry.accounting.messages = all.messages;
            entry.accounting.session = all.session;
            suffix.clear();
        }
    }
    for line in suffix.lines() {
        entry.accounting.consume(line, kind);
    }
    entry.offset += complete_bytes;
    entry.modified = modified;
    Ok(entry.accounting.snapshot())
}

#[cfg(test)]
pub(super) fn turn_stats_from_text(text: &str) -> TurnStats {
    parse(text, SessionKind::Claude)
}
#[cfg(test)]
fn parse(text: &str, kind: SessionKind) -> TurnStats {
    let mut accounting = Accounting::default();
    for line in text.lines() {
        accounting.consume(line, kind);
    }
    accounting.snapshot()
}

#[cfg(test)]
fn pi_parse(text: &str) -> TurnStats {
    pi_accounting_from_text(text).snapshot()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    fn lines(values: Vec<Value>) -> String {
        values
            .iter()
            .map(Value::to_string)
            .collect::<Vec<_>>()
            .join("\n")
    }
    #[test]
    fn claude_counts_usage_once_and_only_successful_file_edits() {
        let message = json!({"type":"assistant","message":{"id":"m","usage":{"input_tokens":10,"output_tokens":50,"cache_read_input_tokens":80,"cache_creation_input_tokens":10},"content":[{"type":"tool_use","id":"write","name":"Write","input":{"file_path":"a.ts"}}]}});
        let text = lines(vec![
            json!({"type":"user","message":{"content":"go"}}),
            message.clone(),
            message,
            json!({"type":"user","message":{"content":[{"type":"tool_result","tool_use_id":"write","content":"done"}]}}),
            json!({"type":"assistant","isSidechain":true,"message":{"id":"child","usage":{"input_tokens":1000,"output_tokens":1000}}}),
        ]);
        let stats = parse(&text, SessionKind::Claude);
        assert_eq!(stats.total_tokens, Some(150));
        assert_eq!(stats.cache_hit_percent, Some(80.0));
        assert_eq!(stats.tools_used, Some(1));
        assert_eq!(stats.files_touched, Some(1));
    }
    #[test]
    fn claude_updated_usage_replaces_previous_frames_and_resets_turn_only() {
        let text = lines(vec![
            json!({"type":"user","message":{"content":"first"}}),
            json!({"type":"assistant","message":{"id":"a","usage":{"input_tokens":100,"output_tokens":20}}}),
            json!({"type":"assistant","message":{"id":"a","usage":{"input_tokens":100,"output_tokens":40}}}),
            json!({"type":"user","message":{"content":"second"}}),
            json!({"type":"assistant","message":{"id":"b","usage":{"input_tokens":200,"output_tokens":60},"content":[{"type":"tool_use","id":"edit","name":"Edit","input":{"file_path":"a.ts"}}]}}),
            json!({"type":"user","message":{"content":[{"type":"tool_result","tool_use_id":"edit","is_error":true,"content":"failed"}]}}),
        ]);
        let stats = parse(&text, SessionKind::Claude);
        assert_eq!(stats.tokens, Some(60));
        assert_eq!(stats.total_tokens, Some(260));
        assert_eq!(stats.session_tokens, Some(400));
        assert_eq!(stats.files_touched, Some(0));
    }
    #[test]
    fn codex_prefers_native_turn_usage_and_never_adds_repeated_snapshots() {
        let usage = json!({"input_tokens":1000,"output_tokens":100,"cached_input_tokens":800});
        let record = json!({"type":"token_usage_record","payload":{"turn_token_usage":usage,"thread_token_usage":{"input_tokens":9000,"output_tokens":900,"cached_input_tokens":5000}}});
        let stats = parse(
            &lines(vec![
                json!({"type":"event_msg","payload":{"type":"task_started","turn_id":"a"}}),
                record.clone(),
                record,
                json!({"type":"event_msg","payload":{"type":"token_count","info":{"total_token_usage":{"input_tokens":9000,"output_tokens":900,"cached_input_tokens":5000}}}}),
            ]),
            SessionKind::Codex,
        );
        assert_eq!(stats.total_tokens, Some(1100));
        assert_eq!(stats.session_tokens, Some(9900));
        assert_eq!(stats.cache_hit_percent, Some(80.0));
    }
    #[test]
    fn codex_legacy_usage_uses_difference_across_turns() {
        let count = |input, output| json!({"type":"event_msg","payload":{"type":"token_count","info":{"total_token_usage":{"input_tokens":input,"output_tokens":output}}}});
        let stats = parse(
            &lines(vec![
                json!({"type":"event_msg","payload":{"type":"task_started"}}),
                count(100, 10),
                json!({"type":"event_msg","payload":{"type":"task_started"}}),
                count(400, 40),
                count(400, 40),
            ]),
            SessionKind::Codex,
        );
        assert_eq!(stats.total_tokens, Some(330));
        assert_eq!(stats.session_tokens, Some(440));
    }
    #[test]
    fn codex_counts_calls_once_and_waits_for_successful_patch() {
        let call = json!({"type":"response_item","payload":{"type":"custom_tool_call","call_id":"c","name":"apply_patch","input":"*** Update File: a.ts\n*** Add File: b.ts"}});
        let mut accounting = Accounting::default();
        for line in lines(vec![
            json!({"type":"event_msg","payload":{"type":"task_started","turn_id":"a"}}),
            call.clone(),
            call,
        ])
        .lines()
        {
            accounting.consume(line, SessionKind::Codex);
        }
        assert_eq!(accounting.snapshot().tools_used, Some(1));
        assert_eq!(accounting.snapshot().files_touched, Some(0));
        accounting.consume(&json!({"type":"response_item","payload":{"type":"custom_tool_call_output","call_id":"c","output":"Success. Updated the following files:\nM a.ts\nA b.ts"}}).to_string(), SessionKind::Codex);
        assert_eq!(accounting.snapshot().files_touched, Some(2));
    }
    fn stamp(mut row: Value, seconds: u32) -> Value {
        row["timestamp"] = json!(format!("2026-10-10T09:{:02}:{:02}.000Z", seconds / 60, seconds % 60));
        row
    }
    fn claude_usage(id: &str, output: u64, content: Value) -> Value {
        json!({"type":"assistant","message":{"id":id,"usage":{"input_tokens":100,"output_tokens":output},"content":content}})
    }
    fn codex_usage(turn: &str, output: u64) -> Value {
        json!({"type":"token_usage_record","payload":{"turn_id":turn,
            "turn_token_usage":{"input_tokens":100,"output_tokens":output},
            "thread_token_usage":{"input_tokens":100,"output_tokens":output}}})
    }
    #[test]
    fn claude_recorded_rate_deduplicates_blocks_and_excludes_parallel_tool_waits() {
        let text = lines(vec![
            stamp(json!({"type":"user","message":{"content":"go"}}), 0),
            stamp(claude_usage("a", 120, json!([{"type":"thinking","thinking":"plan"}])), 1),
            stamp(claude_usage("a", 120, json!([{"type":"tool_use","id":"one","name":"Bash"},{"type":"tool_use","id":"two","name":"Read"}])), 2),
            stamp(json!({"type":"user","message":{"content":[{"type":"tool_result","tool_use_id":"one"}]}}), 12),
            stamp(json!({"type":"assistant","isSidechain":true,"message":{"id":"child","usage":{"input_tokens":1000,"output_tokens":1000}}}), 15),
            stamp(json!({"type":"user","message":{"content":[{"type":"tool_result","tool_use_id":"two"}]}}), 22),
            stamp(claude_usage("b", 120, json!([{"type":"text","text":"done"}])), 24),
        ]);
        let stats = parse(&text, SessionKind::Claude);
        assert_eq!(stats.tokens, Some(240));
        assert_eq!(stats.generation_tokens_per_second, None);
        assert_eq!(stats.estimated_output_tokens_per_second, Some(60.0));
    }
    #[test]
    fn codex_recorded_rate_pairs_native_usage_and_excludes_tool_and_approval_waits() {
        let text = lines(vec![
            stamp(json!({"type":"event_msg","payload":{"type":"task_started","turn_id":"a"}}), 0),
            stamp(json!({"type":"response_item","payload":{"type":"function_call","call_id":"tool","name":"shell"}}), 2),
            stamp(codex_usage("a", 120), 3),
            // Duplicate usage and legacy snapshots must not extend the measured response.
            stamp(codex_usage("a", 120), 10),
            stamp(json!({"type":"event_msg","payload":{"type":"token_count","info":{"total_token_usage":{"input_tokens":100,"output_tokens":120}}}}), 12),
            stamp(json!({"type":"response_item","payload":{"type":"function_call_output","call_id":"tool","output":"done"}}), 22),
            stamp(codex_usage("other-turn", 10000), 23),
            stamp(codex_usage("a", 240), 24),
            stamp(json!({"type":"event_msg","payload":{"type":"task_complete","turn_id":"a"}}), 50),
        ]);
        let stats = parse(&text, SessionKind::Codex);
        assert_eq!(stats.tokens, Some(240));
        assert_eq!(stats.generation_tokens_per_second, None);
        assert_eq!(stats.estimated_output_tokens_per_second, Some(60.0));
    }
    #[test]
    fn codex_recorded_rate_handles_legacy_usage_without_double_counting_reasoning() {
        let count = |output| json!({"type":"event_msg","payload":{"type":"token_count","info":{
            "total_token_usage":{"input_tokens":100,"output_tokens":output,"reasoning_output_tokens":40}}}});
        let stats = parse(&lines(vec![
            stamp(json!({"type":"event_msg","payload":{"type":"task_started","turn_id":"a"}}), 0),
            stamp(count(120), 2),
            stamp(count(120), 12),
            stamp(json!({"type":"event_msg","payload":{"type":"task_started","turn_id":"b"}}), 20),
            stamp(count(240), 22),
        ]), SessionKind::Codex);
        assert_eq!(stats.tokens, Some(120));
        assert_eq!(stats.generation_tokens_per_second, None);
        assert_eq!(stats.estimated_output_tokens_per_second, Some(60.0));
    }
    #[test]
    fn recorded_rate_stays_unknown_for_missing_invalid_or_reversed_timestamps() {
        for timestamp in [Value::Null, json!("bad"), json!("2026-02-30T09:00:02Z"), json!("2026-10-10T08:59:59Z")] {
            let mut response = claude_usage("a", 120, json!([]));
            response["timestamp"] = timestamp;
            let stats = parse(&lines(vec![
                stamp(json!({"type":"user","message":{"content":"go"}}), 0), response,
            ]), SessionKind::Claude);
            assert_eq!(stats.tokens, Some(120));
            assert_eq!(stats.generation_tokens_per_second, None);
        }
        let stats = parse(&lines(vec![stamp(claude_usage("a", 120, json!([])), 2)]), SessionKind::Claude);
        assert_eq!(stats.generation_tokens_per_second, None);
        // A tool result must not accidentally create a clock when the prompt/start row is missing.
        let stats = parse(&lines(vec![
            stamp(json!({"type":"response_item","payload":{"type":"function_call","call_id":"tool","name":"shell"}}), 2),
            stamp(json!({"type":"response_item","payload":{"type":"function_call_output","call_id":"tool"}}), 22),
            stamp(codex_usage("a", 120), 24),
        ]), SessionKind::Codex);
        assert_eq!(stats.generation_tokens_per_second, None);
    }
    #[test]
    fn recorded_rate_rejects_missing_tool_completion_overlapping_generation_and_compaction() {
        for fault in ["missing-completion", "orphan-result", "compaction"] {
            let mut rows = vec![
                stamp(json!({"type":"event_msg","payload":{"type":"task_started","turn_id":"a"}}), 0),
                stamp(json!({"type":"response_item","payload":{"type":"function_call","call_id":"tool","name":"shell"}}), 2),
                stamp(codex_usage("a", 120), 3),
            ];
            if fault == "orphan-result" {
                rows.push(stamp(json!({"type":"response_item","payload":{"type":"function_call_output","call_id":"unknown"}}), 12));
            } else if fault == "compaction" {
                rows.push(stamp(json!({"type":"compacted","payload":{}}), 12));
            }
            rows.push(stamp(codex_usage("a", 240), 24));
            assert_eq!(parse(&lines(rows), SessionKind::Codex).estimated_output_tokens_per_second, None);
        }
    }
    #[test]
    fn recorded_rate_resets_at_a_new_prompt_and_understands_timezones() {
        let mut first = claude_usage("a", 120, json!([]));
        first["timestamp"] = json!("2026-10-10T17:00:02+08:00");
        let mut accounting = Accounting::default();
        for row in [stamp(json!({"type":"user","message":{"content":"first"}}), 0), first] {
            accounting.consume(&row.to_string(), SessionKind::Claude);
        }
        assert_eq!(accounting.snapshot().estimated_output_tokens_per_second, Some(60.0));
        accounting.consume(&stamp(json!({"type":"user","message":{"content":"next"}}), 30).to_string(), SessionKind::Claude);
        assert_eq!(accounting.snapshot().generation_tokens_per_second, None);
        accounting.consume(&stamp(claude_usage("b", 60, json!([])), 32).to_string(), SessionKind::Claude);
        assert_eq!(accounting.snapshot().estimated_output_tokens_per_second, Some(30.0));
    }
    #[test]
    fn recorded_rate_survives_incremental_reads_partial_lines_and_reader_restart() {
        use std::io::Write;
        let dir = std::env::temp_dir().join(format!("velaterm-turn-timing-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("transcript.jsonl");
        let start = stamp(json!({"type":"user","message":{"content":"go"}}), 0);
        std::fs::write(&path, format!("{start}\n")).unwrap();
        let mut entry = Cached { path: path.clone(), ..Cached::default() };
        assert_eq!(read_cached(&mut entry, SessionKind::Claude).unwrap().generation_tokens_per_second, None);
        let response = stamp(claude_usage("a", 120, json!([])), 2).to_string();
        let mut file = std::fs::OpenOptions::new().append(true).open(&path).unwrap();
        write!(file, "{response}").unwrap();
        assert_eq!(read_cached(&mut entry, SessionKind::Claude).unwrap().generation_tokens_per_second, None);
        writeln!(file).unwrap();
        assert_eq!(read_cached(&mut entry, SessionKind::Claude).unwrap().estimated_output_tokens_per_second, Some(60.0));
        assert_eq!(read_cached(&mut entry, SessionKind::Claude).unwrap().estimated_output_tokens_per_second, Some(60.0));
        let mut restarted = Cached { path, ..Cached::default() };
        assert_eq!(read_cached(&mut restarted, SessionKind::Claude).unwrap().estimated_output_tokens_per_second, Some(60.0));
        std::fs::remove_dir_all(dir).unwrap();
    }
    #[test]
    fn pi_native_duration_updates_are_deduplicated_and_reset() {
        let mut accounting = PiAccounting::default();
        accounting.start();
        let message = json!({"role":"assistant","usage":{"input":100,"output":200,"cacheRead":800,"reasoning":150},"durationMs":4000});
        accounting.assistant("one", &message);
        accounting.assistant("one", &message);
        assert_eq!(accounting.snapshot().generation_tokens_per_second, Some(50.0));
        accounting.assistant("two", &json!({"usage":{"input":50,"output":100},"duration":6000}));
        assert_eq!(accounting.snapshot().generation_tokens_per_second, Some(30.0));
        assert_eq!(accounting.snapshot().tokens, Some(300));
        accounting.assistant("two", &json!({"usage":{"input":50,"output":100}}));
        assert_eq!(accounting.snapshot().generation_tokens_per_second, None);
        accounting.start();
        assert_eq!(accounting.snapshot().generation_tokens_per_second, None);
        assert_eq!(pi_step_ms(&json!({"durationMs":-1})), None);
        assert_eq!(pi_step_ms(&json!({"durationMs":0})), None);
    }
    /// Full native recording -> session database -> Info command -> frontend JSON contract. Read-only
    /// provider samples are selected explicitly; only numeric statistics are retained in the report.
    #[test]
    #[ignore = "requires explicitly selected local agent recordings"]
    fn tui_turn_stats_end_to_end() {
        let output = PathBuf::from(std::env::var("VLX_TUI_STATS_OUTPUT_DIR").unwrap());
        std::fs::create_dir_all(&output).unwrap();
        let db = crate::db::Db::open(&output.join(format!("app-{}.db", uuid::Uuid::new_v4()))).unwrap();
        let ctx = crate::host::AppCtx::Headless(std::sync::Arc::new(crate::host::HeadlessHost::new(output.clone(), db)));
        let mut reports = Vec::new();
        for (kind, variable) in [(SessionKind::Claude, "VLX_TUI_STATS_CLAUDE_ID"), (SessionKind::Codex, "VLX_TUI_STATS_CODEX_ID"), (SessionKind::Pi, "VLX_TUI_STATS_PI_ID"), (SessionKind::Grok, "VLX_TUI_STATS_GROK_ID"), (SessionKind::Antigravity, "VLX_TUI_STATS_ANTIGRAVITY_ID")] {
            let agent_id = std::env::var(variable).unwrap();
            let path = match kind {
                SessionKind::Claude => resume::find_claude_transcript(&agent_id),
                SessionKind::Codex => resume::find_codex_rollout(&agent_id),
                SessionKind::Pi => resume::find_pi_session(kind, &agent_id),
                SessionKind::Grok => resume::find_grok_updates(&agent_id),
                SessionKind::Antigravity => antigravity::path(&agent_id),
                _ => unreachable!(),
            }.unwrap();
            let before = std::fs::read(&path).unwrap();
            let local_id = {
                let conn = ctx.db().conn.lock().unwrap();
                let project = crate::db::repo::create_virtual_project(&conn, &format!("{} TUI output-speed validation", kind.as_str())).unwrap();
                let session = crate::db::repo::create_session(&conn, &project.id, None, "Read-only sample", kind, None, None, None, None, None).unwrap();
                assert_eq!(session.engine, "tui");
                crate::db::repo::set_agent_session_id(&conn, &session.id, &agent_id, kind).unwrap();
                session.id
            };
            let stats = crate::command_core::agent_turn_stats(&ctx, &local_id).unwrap();
            let rate = stats.generation_tokens_per_second.or(stats.estimated_output_tokens_per_second)
                .expect("A selected recording must provide native timing or a labeled estimate");
            assert!(rate.is_finite() && rate > 0.0);
            assert!(stats.tokens.is_some_and(|tokens| tokens > 0));
            let json = serde_json::to_value(&stats).unwrap();
            if matches!(kind, SessionKind::Claude | SessionKind::Codex) {
                assert_eq!(json["generationTokensPerSecond"], Value::Null);
                assert_eq!(json["estimatedOutputTokensPerSecond"].as_f64(), Some(rate));
            } else {
                assert_eq!(json["generationTokensPerSecond"].as_f64(), Some(rate));
                assert_eq!(json["outputThroughputBasis"], "request");
            }
            let again = crate::command_core::agent_turn_stats(&ctx, &local_id).unwrap();
            assert_eq!(again, stats, "Opening Info later must not add idle/read time");
            assert_eq!(std::fs::read(&path).unwrap(), before, "Provider history must stay untouched");
            reports.push(json!({"kind":kind.as_str(),"engine":"tui","agentSessionId":agent_id,"stats":json,"repeatedReadStable":true,"sourceUnchanged":true}));
            println!("{} TUI Info: output={} rate={rate:.3} tok/s; repeated read stable; source unchanged", kind.as_str(), stats.tokens.unwrap());
        }
        std::fs::write(output.join("results.json"), serde_json::to_string_pretty(&json!({"command":"rtk proxy cargo test --manifest-path src-tauri/Cargo.toml --lib --no-default-features tui_turn_stats_end_to_end -- --ignored --nocapture","sessions":reports})).unwrap()).unwrap();
    }
    #[test]
    fn missing_usage_and_zero_input_have_no_invented_ratios() {
        let mut stats = parse("{}\nnot json", SessionKind::Claude);
        assert_eq!(stats.total_tokens, None);
        stats.with_context(&AgentContextInfo {
            model: Some("claude-sonnet-4-5".to_string()),
            context_tokens: Some(73_000),
            context_limit: 258_400,
            current_tool: None,
        });
        assert_eq!(stats.model.as_deref(), Some("claude-sonnet-4-5"));
        assert!((stats.context_percent.unwrap() - 28.25077).abs() < 0.001);
        stats.with_context(&AgentContextInfo {
            model: None,
            context_tokens: Some(73_000),
            context_limit: 0,
            current_tool: None,
        });
        assert_eq!(stats.context_percent, None);
        let stats = parse(
            &lines(vec![
                json!({"type":"user","message":{"content":"go"}}),
                json!({"type":"assistant","message":{"usage":{"input_tokens":0,"output_tokens":0}}}),
            ]),
            SessionKind::Claude,
        );
        assert_eq!(stats.total_tokens, Some(0));
        assert_eq!(stats.cache_hit_percent, None);
    }
    #[test]
    fn pi_reads_the_last_turn_and_only_counts_successful_edits() {
        let text = lines(vec![
            json!({"type":"session","id":"s"}),
            json!({"type":"model_change","id":"m1","parentId":null,"provider":"deepseek","modelId":"deepseek-flash"}),
            json!({"type":"message","id":"u1","parentId":"m1","message":{"role":"user","content":"first"}}),
            json!({"type":"message","id":"a1","parentId":"u1","message":{"role":"assistant","model":"deepseek-flash","usage":{"input":100,"output":20,"cacheRead":800,"cacheWrite":100},"content":[]}}),
            json!({"type":"message","id":"u2","parentId":"a1","message":{"role":"user","content":"second"}}),
            json!({"type":"message","id":"a2","parentId":"u2","message":{"role":"assistant","model":"deepseek-flash","usage":{"input":50,"output":10,"cacheRead":100,"cacheWrite":0},"duration":2000.0,"content":[
                {"type":"toolCall","id":"c1","name":"edit","arguments":{"path":"a.ts"}},
                {"type":"toolCall","id":"c2","name":"write","arguments":{"path":"b.ts"}}]}}),
            json!({"type":"message","id":"r1","parentId":"a2","message":{"role":"toolResult","toolCallId":"c1","toolName":"edit","isError":false}}),
            json!({"type":"message","id":"r2","parentId":"r1","message":{"role":"toolResult","toolCallId":"c2","toolName":"write","isError":true}}),
        ]);
        let stats = pi_parse(&text);
        assert_eq!(stats.input_tokens, Some(150));
        assert_eq!(stats.tokens, Some(10));
        assert_eq!(stats.total_tokens, Some(160));
        assert_eq!(stats.session_tokens, Some(1180));
        assert_eq!(stats.tools_used, Some(2));
        assert_eq!(stats.files_touched, Some(1));
        assert!((stats.cache_hit_percent.unwrap() - 100.0 / 150.0 * 100.0).abs() < 0.001);
        assert!((stats.generation_tokens_per_second.unwrap() - 5.0).abs() < 0.001);
    }
    #[test]
    fn pi_ignores_an_abandoned_branch_and_user_only_sessions_report_nothing() {
        let text = lines(vec![
            json!({"type":"message","id":"u1","parentId":null,"message":{"role":"user","content":"go"}}),
            json!({"type":"message","id":"a1","parentId":"u1","message":{"role":"assistant","usage":{"input":100,"output":10},"content":[]}}),
            json!({"type":"message","id":"a2","parentId":"u1","message":{"role":"assistant","usage":{"input":900,"output":90},"content":[]}}),
            json!({"type":"message","id":"u2","parentId":"a2","message":{"role":"user","content":"branch"}}),
            json!({"type":"message","id":"a3","parentId":"u2","message":{"role":"assistant","usage":{"input":300,"output":30},"content":[]}}),
        ]);
        let stats = pi_parse(&text);
        assert_eq!(stats.total_tokens, Some(330));
        // 活动回合排除弃用分支，但会话总量仍包含已经消耗的 a1。
        assert_eq!(stats.session_tokens, Some(1430));
        let idle = pi_parse(&lines(vec![json!({"type":"message","id":"u1","parentId":null,"message":{"role":"user","content":"go"}})]));
        assert_eq!(idle.total_tokens, None);
        assert_eq!(idle.session_tokens, None);
        assert_eq!(idle.tools_used, Some(0));
        assert_eq!(idle.generation_tokens_per_second, None);
        let empty = pi_parse("");
        assert_eq!(empty.total_tokens, None);
        assert_eq!(empty.session_tokens, None);
        assert_eq!(empty.tools_used, None);
        assert_eq!(empty.files_touched, None);
    }
    #[test]
    fn pi_branch_check_detects_a_fork_but_not_a_non_message_entry() {
        assert!(!pi_branch_broken(None, "{\"id\":\"a1\",\"parentId\":null}\n"));
        assert!(!pi_branch_broken(Some("a1"), "{\"id\":\"a2\",\"parentId\":\"a1\"}\n"));
        assert!(pi_branch_broken(Some("a2"), "{\"id\":\"a3\",\"parentId\":\"a1\"}\n"));
        assert!(!pi_branch_broken(Some("a2"), "{\"type\":\"title\"}\n{\"id\":\"a3\",\"parentId\":\"a2\"}\n"));
    }
}

#[cfg(test)]
mod file_tests {
    use super::*;
    use serde_json::json;
    use std::io::Write;
    #[test]
    fn appended_partial_lines_are_retried_and_truncation_resets_accounting() {
        let path = std::env::temp_dir().join(format!("turn-stats-{}.jsonl", uuid::Uuid::new_v4()));
        let mut cached = Cached {
            path: path.clone(),
            ..Cached::default()
        };
        let user = "{\"type\":\"user\",\"message\":{\"content\":\"go\"}}\n";
        let message = "{\"type\":\"assistant\",\"message\":{\"id\":\"a\",\"usage\":{\"input_tokens\":100,\"output_tokens\":50}}}\n";
        std::fs::write(&path, format!("{user}{}", &message[..40])).unwrap();
        assert_eq!(
            read_cached(&mut cached, SessionKind::Claude)
                .unwrap()
                .total_tokens,
            None
        );
        std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap()
            .write_all(message[40..].as_bytes())
            .unwrap();
        assert_eq!(
            read_cached(&mut cached, SessionKind::Claude)
                .unwrap()
                .total_tokens,
            Some(150)
        );
        assert_eq!(
            read_cached(&mut cached, SessionKind::Claude)
                .unwrap()
                .total_tokens,
            Some(150)
        );
        std::fs::write(&path, user).unwrap();
        assert_eq!(
            read_cached(&mut cached, SessionKind::Claude)
                .unwrap()
                .total_tokens,
            None
        );
        std::fs::remove_file(path).unwrap();
    }
    #[test]
    fn a_turn_larger_than_the_old_tail_limit_remains_complete() {
        let path =
            std::env::temp_dir().join(format!("turn-stats-long-{}.jsonl", uuid::Uuid::new_v4()));
        let mut text = String::from("{\"type\":\"user\",\"message\":{\"content\":\"go\"}}\n{\"type\":\"assistant\",\"message\":{\"id\":\"a\",\"usage\":{\"input_tokens\":100,\"output_tokens\":50}}}\n");
        text.push_str(
            &serde_json::json!({"type":"attachment","content":"x".repeat(600_000)}).to_string(),
        );
        text.push('\n');
        std::fs::write(&path, text).unwrap();
        let mut cached = Cached {
            path: path.clone(),
            ..Cached::default()
        };
        assert_eq!(
            read_cached(&mut cached, SessionKind::Claude)
                .unwrap()
                .total_tokens,
            Some(150)
        );
        std::fs::remove_file(path).unwrap();
    }
    #[test]
    fn rewind_changes_the_active_turn_without_refunding_session_usage() {
        let path =
            std::env::temp_dir().join(format!("turn-stats-rewind-{}.jsonl", uuid::Uuid::new_v4()));
        let rows = [
            serde_json::json!({"type":"user","uuid":"u1","parentUuid":null,"message":{"content":"first"}}),
            serde_json::json!({"type":"assistant","uuid":"a1","parentUuid":"u1","message":{"id":"m1","usage":{"input_tokens":100,"output_tokens":10}}}),
            serde_json::json!({"type":"user","uuid":"u2","parentUuid":"a1","message":{"content":"second"}}),
            serde_json::json!({"type":"assistant","uuid":"a2","parentUuid":"u2","message":{"id":"m2","usage":{"input_tokens":200,"output_tokens":20}}}),
        ];
        std::fs::write(
            &path,
            rows.iter().map(|r| format!("{r}\n")).collect::<String>(),
        )
        .unwrap();
        let mut cached = Cached {
            path: path.clone(),
            ..Cached::default()
        };
        assert_eq!(
            read_cached(&mut cached, SessionKind::Claude)
                .unwrap()
                .total_tokens,
            Some(220)
        );
        writeln!(
            std::fs::OpenOptions::new()
                .append(true)
                .open(&path)
                .unwrap(),
            "{}",
            serde_json::json!({"type":"last-prompt","rewound":true,"leafUuid":"a1"})
        )
        .unwrap();
        let stats = read_cached(&mut cached, SessionKind::Claude).unwrap();
        assert_eq!(stats.total_tokens, Some(110));
        assert_eq!(stats.session_tokens, Some(330));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn pi_reader_follows_appended_entries_and_rebuilds_after_a_fork() {
        let path = std::env::temp_dir().join(format!("turn-stats-pi-{}.jsonl", uuid::Uuid::new_v4()));
        let user = "{\"type\":\"message\",\"id\":\"u1\",\"parentId\":null,\"message\":{\"role\":\"user\",\"content\":\"go\"}}\n";
        std::fs::write(&path, user).unwrap();
        let mut cached = PiCached {
            path: path.clone(),
            ..PiCached::default()
        };
        assert_eq!(pi_read_cached(&mut cached).unwrap().total_tokens, None);
        let append = |text: &str| {
            std::fs::OpenOptions::new()
                .append(true)
                .open(&path)
                .unwrap()
                .write_all(text.as_bytes())
                .unwrap();
        };
        append("{\"type\":\"message\",\"id\":\"a1\",\"parentId\":\"u1\",\"message\":{\"role\":\"assistant\",\"usage\":{\"input\":100,\"output\":10},\"content\":[]}}\n");
        assert_eq!(pi_read_cached(&mut cached).unwrap().total_tokens, Some(110));
        // A fork names an earlier parent, so the incremental counts have to be rebuilt from the branch.
        append("{\"type\":\"message\",\"id\":\"a2\",\"parentId\":\"u1\",\"message\":{\"role\":\"assistant\",\"usage\":{\"input\":900,\"output\":90},\"content\":[]}}\n");
        let stats = pi_read_cached(&mut cached).unwrap();
        assert_eq!(stats.total_tokens, Some(990));
        assert_eq!(stats.session_tokens, Some(1100));
        // A recording replaced in place must not keep counts from the file it used to be.
        std::fs::write(&path, user).unwrap();
        let stats = pi_read_cached(&mut cached).unwrap();
        assert_eq!(stats.total_tokens, None);
        assert_eq!(stats.session_tokens, None);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn opencode_counts_the_last_turn_and_measures_generation() {
        fn user(id: &str) -> OpencodeMessage {
            OpencodeMessage { info: serde_json::json!({"id": id, "role": "user"}), parts: vec![] }
        }
        fn assistant(id: &str, parent: &str, created: u64, completed: u64, out: u64, reas: u64, cache: u64) -> OpencodeMessage {
            OpencodeMessage {
                info: serde_json::json!({
                    "id": id, "role": "assistant", "parentID": parent,
                    "tokens": {
                        "total": 10 + out + reas + cache, "input": 10, "output": out,
                        "reasoning": reas, "cache": {"read": cache, "write": 0}
                    },
                    "time": {"created": created, "streamed": completed, "completed": completed + 20000}
                }),
                parts: vec![
                    serde_json::json!({"id": format!("{id}-tool"), "type": "tool", "callID": format!("call-{id}")}),
                    serde_json::json!({"id": format!("{id}-patch"), "type": "patch", "files": ["/a.ts"]}),
                ],
            }
        }
        let messages = vec![
            user("u1"),
            assistant("a1", "u1", 0, 1000, 50, 10, 0),
            user("u2"),
            assistant("a2", "u2", 2000, 3000, 100, 20, 40),
            assistant("a3", "u2", 4000, 5000, 60, 0, 50),
        ];
        let stats = opencode_stats_from_messages(&messages);
        assert_eq!(stats.total_tokens, Some(290));
        assert_eq!(stats.tokens, Some(180));
        assert_eq!(stats.input_tokens, Some(110));
        assert_eq!(stats.session_tokens, Some(360));
        assert_eq!(stats.tools_used, Some(2));
        assert_eq!(stats.files_touched, Some(1));
        assert!((stats.cache_hit_percent.unwrap() - 90.0 / 110.0 * 100.0).abs() < 0.001);
        assert!((stats.generation_tokens_per_second.unwrap() - 90.0).abs() < 0.001);
    }

    #[test]
    fn opencode_idle_boundaries_include_steered_prompts_and_compaction_usage() {
        let message = |seq: u64, role: &str| OpencodeMessage {
            info: json!({"id":seq.to_string(),"seq":seq,"role":role}), parts: vec![] };
        let assistant = |seq: u64| OpencodeMessage { info: json!({"id":seq.to_string(),"seq":seq,
            "role":"assistant","tokens":{"input":100,"output":50,"reasoning":0,"cache":{"read":0}},
            "time":{"created":seq * 1000,"streamed":seq * 1000 + 1000}}), parts: vec![] };
        let mut current = assistant(5);
        current.parts = vec![json!({"id":"edit","type":"tool","tool":"edit",
            "state":{"status":"completed","input":{"filePath":"a.ts"}}}),
            json!({"id":"fail","type":"tool","tool":"write",
            "state":{"status":"error","input":{"filePath":"failed.ts"}}})];
        current.info["snapshot"] = json!({"files":["a.ts","b.ts"]});
        let mut messages = vec![message(0,"user"), assistant(1), message(2,"idle"),
            message(3,"user"), current, message(6,"user"), assistant(7), message(8,"idle")];
        let stats = opencode_stats_from_messages(&messages);
        assert_eq!(stats.total_tokens, Some(300));
        assert_eq!(stats.session_tokens, Some(450));
        assert_eq!(stats.generation_tokens_per_second, Some(50.0));
        assert_eq!(stats.files_touched, Some(2));
        let mut compact = message(9,"compaction");
        compact.info["tokens"] = json!({"input":200,"output":20,"reasoning":0,"cache":{"read":0}});
        messages.push(compact);
        messages.push(assistant(10));
        let stats = opencode_stats_from_messages(&messages);
        assert_eq!(stats.total_tokens, Some(370));
        assert_eq!(stats.session_tokens, Some(820));
        // 压缩没有完整请求计时，不能把之后一个 assistant 的速率当成本轮吞吐率。
        assert_eq!(stats.generation_tokens_per_second, None);
        messages.push(message(11,"idle"));
        messages.push(message(12,"user"));
        let stats = opencode_stats_from_messages(&messages);
        assert_eq!(stats.total_tokens, None);
        assert_eq!(stats.files_touched, Some(0));
    }

    #[test]
    fn opencode_mixed_history_does_not_merge_legacy_usage_into_the_first_modern_turn() {
        let messages = vec![
            OpencodeMessage { info: json!({"id":"old-user","role":"user"}), parts: vec![] },
            OpencodeMessage { info: json!({"id":"old-assistant","role":"assistant","parentID":"old-user",
                "tokens":{"input":1000,"output":500}}), parts: vec![] },
            OpencodeMessage { info: json!({"id":"new-user","role":"user","seq":0}), parts: vec![] },
            OpencodeMessage { info: json!({"id":"new-assistant","role":"assistant","seq":1,
                "tokens":{"input":100,"output":50,"reasoning":0,"cache":{"read":0}},
                "time":{"created":1000,"streamed":2000}}), parts: vec![] },
        ];
        let stats = opencode_stats_from_messages(&messages);
        assert_eq!(stats.total_tokens, Some(150));
        assert_eq!(stats.session_tokens, Some(1650));
        assert_eq!(stats.generation_tokens_per_second, Some(50.0));
    }

    #[test]
    fn request_throughput_requires_every_step_and_preserves_zero_usage() {
        let mut timing = RequestTiming::default();
        timing.add(Some(200), Some(1000));
        timing.add(Some(0), Some(1000));
        assert_eq!(timing.rate(), Some(100.0));
        timing.add(Some(100), None);
        assert_eq!(timing.rate(), None);
        let mut missing = RequestTiming::default();
        missing.add(None, Some(1000));
        assert_eq!(missing.rate(), None);
    }

    #[test]
    fn opencode_usage_without_total_and_missing_stream_boundaries_are_distinct() {
        let user = OpencodeMessage { info: json!({"id":"u","role":"user"}), parts: vec![] };
        let assistant = OpencodeMessage {
            info: json!({"id":"a","role":"assistant","tokens":{
                "input":100,"output":200,"reasoning":50,"cache":{"read":0,"write":0}},
                "time":{"created":1000,"streamed":2000,"completed":99000}}),
            parts: vec![],
        };
        let stats = opencode_stats_from_messages(&[user.clone(), assistant.clone()]);
        assert_eq!(stats.tokens, Some(250));
        assert_eq!(stats.total_tokens, Some(350));
        assert_eq!(stats.session_tokens, Some(350));
        assert_eq!(stats.cached_tokens, Some(0));
        assert_eq!(stats.cache_hit_percent, Some(0.0));
        assert_eq!(stats.reasoning_tokens, Some(50));
        assert_eq!(stats.generation_tokens_per_second, Some(250.0));
        assert_eq!(stats.output_throughput_basis, Some(ThroughputBasis::Request));
        let mut no_stream = assistant.clone();
        no_stream.info["time"].as_object_mut().unwrap().remove("streamed");
        let stats = opencode_stats_from_messages(&[user.clone(), assistant.clone(), no_stream]);
        assert_eq!(stats.tokens, Some(500));
        assert_eq!(stats.generation_tokens_per_second, None);
        let mut no_output = assistant;
        no_output.info["tokens"].as_object_mut().unwrap().remove("output");
        let stats = opencode_stats_from_messages(&[user, no_output]);
        assert_eq!(stats.tokens, None);
        assert_eq!(stats.total_tokens, None);
        assert_eq!(stats.cached_tokens, None);
        assert_eq!(stats.generation_tokens_per_second, None);
    }

    #[test]
    fn pi_cache_zero_reasoning_and_missing_timing_do_not_inflate_throughput() {
        let mut accounting = PiAccounting::default();
        accounting.start();
        accounting.assistant("a", &json!({"usage":{"input":100,"output":200,
            "reasoning":150,"cacheRead":0},"durationMs":1000}));
        let stats = accounting.snapshot();
        assert_eq!(stats.tokens, Some(200));
        assert_eq!(stats.reasoning_tokens, Some(150));
        assert_eq!(stats.cached_tokens, Some(0));
        assert_eq!(stats.cache_hit_percent, Some(0.0));
        assert_eq!(stats.generation_tokens_per_second, Some(200.0));
        accounting.assistant("b", &json!({"usage":{"input":100,"output":100,"cacheRead":0}}));
        assert_eq!(accounting.snapshot().tokens, Some(300));
        assert_eq!(accounting.snapshot().generation_tokens_per_second, None);
        accounting.assistant("c", &json!({"usage":{"input":100}}));
        assert_eq!(accounting.snapshot().tokens, None);
        assert_eq!(accounting.snapshot().generation_tokens_per_second, None);
    }

    /// Opt-in read-only validation against a locally selected provider transcript. Only numeric
    /// accounting leaves the parser; no message content or tool arguments are printed or exported.
    #[test]
    #[ignore]
    fn real_turn_stats_sample() {
        let id = std::env::var("VLX_TURN_STATS_SAMPLE_ID").unwrap();
        let kind = match std::env::var("VLX_TURN_STATS_SAMPLE_KIND").as_deref() {
            Ok("claude") => SessionKind::Claude,
            Ok("opencode") => SessionKind::Opencode,
            Ok("pi") => SessionKind::Pi,
            Ok("omp") => SessionKind::Omp,
            _ => SessionKind::Codex,
        };
        let mut stats = current_turn_stats(kind, &id).unwrap();
        stats.with_context(&super::super::context_info(kind, &id).unwrap());
        let output = std::env::var("VLX_TURN_STATS_SAMPLE_OUTPUT").unwrap();
        std::fs::write(output, serde_json::to_string_pretty(&stats).unwrap()).unwrap();
        assert!(stats.total_tokens.is_some());
    }
}
