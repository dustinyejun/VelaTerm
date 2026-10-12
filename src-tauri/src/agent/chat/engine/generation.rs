//! 实时流式窗口必须覆盖整轮输出；Codex 的回合减等待算法仅提供响应估算。
//! 原生请求计时优先，缺失边界或用量无法完整匹配时不覆盖录制统计。
use serde_json::Value;
use std::collections::{HashMap, HashSet};

#[derive(Clone, Debug, Default)]
struct Sample {
    start: u64,
    end: u64,
    tokens: Option<u64>,
    complete: bool,
}
#[derive(Clone, Debug, Default)]
pub(super) struct GenerationTiming {
    samples: HashMap<String, Sample>,
    claude_message: Option<String>,
    codex: CodexTiming,
}
impl GenerationTiming {
    pub fn rate(&self) -> Option<f64> {
        if self.codex.turn_id.is_some() {
            return None;
        }
        let mut tokens = 0u64;
        let mut elapsed = 0u64;
        for sample in self.samples.values() {
            if !sample.complete || sample.end <= sample.start {
                return None;
            }
            let count = sample.tokens?;
            tokens = tokens.saturating_add(count);
            elapsed = elapsed.saturating_add(sample.end - sample.start);
        }
        (elapsed >= 100 && tokens > 0).then(|| tokens as f64 * 1000.0 / elapsed as f64)
    }
    pub fn rate_for_output(&self, expected: Option<u64>) -> Option<f64> {
        let mut output = 0u64;
        for sample in self.samples.values() {
            output = output.checked_add(sample.tokens?)?;
        }
        (expected == Some(output)).then(|| self.rate()).flatten()
    }
    pub fn estimated_rate(&self) -> Option<f64> {
        self.codex.rate()
    }
    pub fn claude(&mut self, value: &Value, now: u64) {
        if value
            .get("parent_tool_use_id")
            .is_some_and(|v| !v.is_null())
        {
            return;
        }
        if value.get("type").and_then(Value::as_str) == Some("assistant") {
            if let (Some(id), Some(tokens)) = (
                value.pointer("/message/id").and_then(Value::as_str),
                value
                    .pointer("/message/usage/output_tokens")
                    .and_then(Value::as_u64),
            ) {
                if let Some(sample) = self.samples.get_mut(id) {
                    sample.tokens = Some(tokens);
                }
            }
        }
        let Some(event) = value.get("event") else {
            return;
        };
        match event.get("type").and_then(Value::as_str) {
            Some("message_start") => {
                self.claude_message = event
                    .pointer("/message/id")
                    .and_then(Value::as_str)
                    .map(str::to_string);
                if let Some(id) = &self.claude_message {
                    self.samples.insert(
                        id.clone(),
                        Sample {
                            start: now,
                            end: now,
                            ..Sample::default()
                        },
                    );
                }
            }
            Some("message_delta") => {
                if let Some(sample) = self
                    .claude_message
                    .as_ref()
                    .and_then(|id| self.samples.get_mut(id))
                {
                    sample.tokens = event
                        .pointer("/usage/output_tokens")
                        .and_then(Value::as_u64)
                        .or(sample.tokens);
                }
            }
            Some("message_stop") => {
                if let Some(id) = self.claude_message.take() {
                    if let Some(sample) = self.samples.get_mut(&id) {
                        sample.end = now;
                        sample.complete = true;
                    }
                }
            }
            _ => {}
        }
    }
    /// Pi and OMP stream one assistant message at a time, so the message itself is the measured window:
    /// `message_start` opens it and `message_end` closes it with the provider's own output-token count.
    pub fn pi_start(&mut self, key: &str, now: u64) {
        self.samples.insert(
            key.to_string(),
            Sample {
                start: now,
                end: now,
                ..Sample::default()
            },
        );
    }
    pub fn pi_end(&mut self, key: &str, tokens: Option<u64>, now: u64) {
        let Some(sample) = self.samples.get_mut(key) else {
            return;
        };
        sample.end = now;
        sample.tokens = tokens;
        sample.complete = true;
    }
    pub fn codex(&mut self, method: &str, params: &Value, now: u64) {
        self.codex.observe(method, params, now);
    }
    pub fn codex_request(&mut self, key: &str, now: u64) {
        self.codex.pause(format!("request-{key}"), now);
    }
    pub fn codex_request_resolved(&mut self, key: &str, now: u64) {
        self.codex.resume(&format!("request-{key}"), now, false);
    }
}

#[derive(Clone, Debug, Default)]
struct CodexTiming {
    turn_id: Option<String>,
    running: bool,
    active_since: Option<u64>,
    elapsed: u64,
    waits: HashSet<String>,
    total: Option<u64>,
    tokens: u64,
    measured_ms: u64,
    unreliable: bool,
}
impl CodexTiming {
    fn rate(&self) -> Option<f64> {
        (!self.unreliable && self.measured_ms >= 100 && self.tokens > 0)
            .then(|| self.tokens as f64 * 1000.0 / self.measured_ms as f64)
    }
    fn elapsed_at(&self, now: u64) -> Option<u64> {
        let active = match self.active_since {
            Some(start) => now.checked_sub(start)?,
            None => 0,
        };
        self.elapsed.checked_add(active)
    }
    fn pause(&mut self, key: String, now: u64) {
        if !self.running || !self.waits.insert(key) {
            return;
        }
        if self.waits.len() == 1 {
            match self.elapsed_at(now) {
                Some(elapsed) => self.elapsed = elapsed,
                None => self.unreliable = true,
            }
            self.active_since = None;
        }
    }
    fn resume(&mut self, key: &str, now: u64, required: bool) {
        if !self.running {
            return;
        }
        if !self.waits.remove(key) {
            self.unreliable |= required;
            return;
        }
        if self.waits.is_empty() {
            self.active_since = Some(now);
        }
    }
    fn observe(&mut self, method: &str, params: &Value, now: u64) {
        if method == "turn/started" {
            let Some(id) = params.pointer("/turn/id").and_then(Value::as_str) else {
                self.unreliable = true;
                return;
            };
            if self.turn_id.as_deref() != Some(id) {
                *self = Self { turn_id: Some(id.to_string()), running: true,
                    active_since: Some(now), total: self.total, ..Self::default() };
            }
            return;
        }
        if !self.running || params.get("turnId").and_then(Value::as_str)
            .is_some_and(|id| self.turn_id.as_deref() != Some(id))
        {
            return;
        }
        match method {
            "item/started" | "item/completed" => {
                let kind = params.pointer("/item/type").and_then(Value::as_str).unwrap_or("");
                // Compaction has separate requests whose usage cannot be paired with these boundaries.
                if kind == "contextCompaction" {
                    self.unreliable = true;
                }
                if !matches!(kind, "commandExecution" | "fileChange" | "mcpToolCall"
                    | "dynamicToolCall" | "collabToolCall" | "collabAgentToolCall"
                    | "webSearch" | "imageView" | "imageGeneration" | "sleep")
                {
                    return;
                }
                let Some(id) = params.pointer("/item/id").and_then(Value::as_str) else {
                    self.unreliable = true;
                    return;
                };
                let key = format!("item-{id}");
                if method == "item/started" { self.pause(key, now); }
                else { self.resume(&key, now, true); }
            }
            "hook/started" | "hook/completed" => {
                let Some(id) = params.pointer("/run/id").and_then(Value::as_str) else {
                    self.unreliable = true;
                    return;
                };
                let key = format!("hook-{id}");
                if method == "hook/started" { self.pause(key, now); }
                else { self.resume(&key, now, true); }
            }
            "item/agentMessage/delta" | "item/reasoning/textDelta"
                | "item/reasoning/summaryTextDelta" | "item/plan/delta" => {
                // Subtracting a tool interval while the model also generates would inflate the rate.
                if !self.waits.is_empty() { self.unreliable = true; }
            }
            "thread/tokenUsage/updated" => {
                let (Some(total), Some(last)) = (
                    params.pointer("/tokenUsage/total/outputTokens").and_then(Value::as_u64),
                    params.pointer("/tokenUsage/last/outputTokens").and_then(Value::as_u64),
                ) else {
                    self.unreliable = true;
                    return;
                };
                if self.total == Some(total) { return; }
                if self.total.is_some_and(|previous| total.checked_sub(previous) != Some(last)) {
                    self.unreliable = true;
                }
                self.total = Some(total);
                let previous_tokens = self.tokens;
                self.tokens = self.tokens.saturating_add(last);
                match self.elapsed_at(now) {
                    Some(elapsed) => {
                        // A new response without any additional active time means a wait boundary was
                        // missed, or generation overlapped the work we excluded.
                        if last > 0 && previous_tokens > 0 && elapsed <= self.measured_ms {
                            self.unreliable = true;
                        }
                        self.measured_ms = elapsed;
                    }
                    None => self.unreliable = true,
                }
            }
            "turn/completed" => {
                if params.pointer("/turn/id").and_then(Value::as_str)
                    .is_some_and(|id| self.turn_id.as_deref() != Some(id)) { return; }
                self.running = false;
                self.active_since = None;
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn claude_matches_usage_once_and_excludes_tool_wait() {
        let mut timing = GenerationTiming::default();
        timing.claude(
            &json!({"event":{"type":"message_start","message":{"id":"a"}}}),
            1000,
        );
        timing.claude(
            &json!({"event":{"type":"message_delta","usage":{"output_tokens":120}}}),
            2900,
        );
        timing.claude(&json!({"event":{"type":"message_stop"}}), 3000);
        timing.claude(
            &json!({"type":"assistant","message":{"id":"a","usage":{"output_tokens":120}}}),
            50000,
        );
        assert_eq!(timing.rate(), Some(60.0));
    }
    #[test]
    fn codex_includes_time_before_the_first_visible_fragment() {
        let mut timing = GenerationTiming::default();
        timing.codex("turn/started", &json!({"turn":{"id":"turn-a"}}), 1000);
        timing.codex("item/started", &json!({"item":{"id":"r","type":"reasoning"}}), 1100);
        timing.codex("item/reasoning/summaryTextDelta", &json!({}), 30100);
        timing.codex("item/agentMessage/delta", &json!({}), 31200);
        let usage = json!({"turnId":"turn-a","tokenUsage":{
            "total":{"outputTokens":100000},"last":{"outputTokens":2193,"reasoningOutputTokens":1900}}});
        timing.codex("thread/tokenUsage/updated", &usage, 31300);
        assert!((timing.estimated_rate().unwrap() - 2193.0 / 30.3).abs() < 0.001);
        timing.codex("thread/tokenUsage/updated", &usage, 100000);
        assert!((timing.estimated_rate().unwrap() - 2193.0 / 30.3).abs() < 0.001);
    }
    #[test]
    fn codex_excludes_the_union_of_parallel_tools_and_user_waits() {
        let mut timing = GenerationTiming::default();
        timing.codex("turn/started", &json!({"turn":{"id":"turn-a"}}), 1000);
        timing.codex("thread/tokenUsage/updated", &json!({"tokenUsage":{
            "total":{"outputTokens":500},"last":{"outputTokens":100}}}), 3000);
        let first = json!({"item":{"id":"tool-a","type":"commandExecution"}});
        let second = json!({"item":{"id":"tool-b","type":"mcpToolCall"}});
        timing.codex("item/started", &first, 3000);
        timing.codex_request("question", 4000);
        timing.codex("item/started", &second, 5000);
        timing.codex("item/completed", &first, 10000);
        timing.codex_request_resolved("question", 14000);
        timing.codex_request_resolved("child-request", 15000);
        timing.codex("item/completed", &second, 20000);
        timing.codex("thread/tokenUsage/updated", &json!({"tokenUsage":{
            "total":{"outputTokens":600},"last":{"outputTokens":100}}}), 22000);
        assert_eq!(timing.estimated_rate(), Some(50.0));
    }
    #[test]
    fn codex_without_turn_start_has_no_rate_even_with_visible_output() {
        let mut timing = GenerationTiming::default();
        timing.codex("item/agentMessage/delta", &json!({}), 1000);
        timing.codex("thread/tokenUsage/updated", &json!({"tokenUsage":{
            "total":{"outputTokens":500},"last":{"outputTokens":100}}}), 3000);
        assert_eq!(timing.estimated_rate(), None);
        assert_eq!(GenerationTiming::default().rate(), None);
    }
    #[test]
    fn codex_tool_arguments_are_counted_without_visible_text() {
        let mut timing = GenerationTiming::default();
        timing.codex("turn/started", &json!({"turn":{"id":"turn-a"}}), 1000);
        timing.codex("item/started", &json!({"item":{"id":"tool-a","type":"fileChange"}}), 3000);
        timing.codex("thread/tokenUsage/updated", &json!({"tokenUsage":{
            "total":{"outputTokens":100},"last":{"outputTokens":100}}}), 50000);
        assert_eq!(timing.estimated_rate(), Some(50.0));
    }
    #[test]
    fn codex_drops_estimates_with_overlapping_output_or_missing_wait_boundaries() {
        for missing_start in [false, true] {
            let mut timing = GenerationTiming::default();
            timing.codex("turn/started", &json!({"turn":{"id":"turn-a"}}), 1000);
            let tool = json!({"item":{"id":"tool-a","type":"commandExecution"}});
            if missing_start {
                timing.codex("item/completed", &tool, 3000);
            } else {
                timing.codex("item/started", &tool, 2000);
                timing.codex("item/reasoning/summaryTextDelta", &json!({}), 2500);
                timing.codex("item/completed", &tool, 3000);
            }
            timing.codex("thread/tokenUsage/updated", &json!({"tokenUsage":{
                "total":{"outputTokens":100},"last":{"outputTokens":100}}}), 4000);
            assert_eq!(timing.estimated_rate(), None);
        }
    }
    #[test]
    fn codex_rejects_counter_gaps_and_ignores_other_turns() {
        let mut timing = GenerationTiming::default();
        timing.codex("turn/started", &json!({"turn":{"id":"turn-a"}}), 1000);
        timing.codex("thread/tokenUsage/updated", &json!({"turnId":"older","tokenUsage":{
            "total":{"outputTokens":100},"last":{"outputTokens":100}}}), 2000);
        assert_eq!(timing.estimated_rate(), None);
        timing.codex("thread/tokenUsage/updated", &json!({"turnId":"turn-a","tokenUsage":{
            "total":{"outputTokens":200},"last":{"outputTokens":100}}}), 3000);
        assert_eq!(timing.estimated_rate(), Some(50.0));
        timing.codex("thread/tokenUsage/updated", &json!({"turnId":"turn-a","tokenUsage":{
            "total":{"outputTokens":500},"last":{"outputTokens":100}}}), 5000);
        assert_eq!(timing.estimated_rate(), None);
        timing.codex("turn/completed", &json!({"turn":{"id":"older"}}), 6000);
        assert!(timing.codex.running);
        timing.codex("turn/completed", &json!({"turn":{"id":"turn-a"}}), 7000);
        timing.codex("turn/started", &json!({"turn":{"id":"turn-b"}}), 8000);
        timing.codex("thread/tokenUsage/updated", &json!({"turnId":"turn-b","tokenUsage":{
            "total":{"outputTokens":600},"last":{"outputTokens":100}}}), 10000);
        assert_eq!(timing.estimated_rate(), Some(50.0));
    }
    #[test]
    fn codex_missing_tool_completion_cannot_hide_a_second_response() {
        let mut timing = GenerationTiming::default();
        timing.codex("turn/started", &json!({"turn":{"id":"turn-a"}}), 1000);
        timing.codex("item/started", &json!({"item":{"id":"tool-a","type":"commandExecution"}}), 3000);
        timing.codex("thread/tokenUsage/updated", &json!({"tokenUsage":{
            "total":{"outputTokens":100},"last":{"outputTokens":100}}}), 4000);
        assert_eq!(timing.estimated_rate(), Some(50.0));
        timing.codex("thread/tokenUsage/updated", &json!({"tokenUsage":{
            "total":{"outputTokens":200},"last":{"outputTokens":100}}}), 8000);
        assert_eq!(timing.estimated_rate(), None);
    }
    #[test]
    fn codex_unknown_usage_compaction_and_invalid_clock_have_no_rate() {
        for event in ["missing-usage", "contextCompaction", "clock"] {
            let mut timing = GenerationTiming::default();
            timing.codex("turn/started", &json!({"turn":{"id":"turn-a"}}), 1000);
            if event == "contextCompaction" {
                timing.codex("item/started", &json!({"item":{"id":"compact","type":event}}), 1500);
            }
            let usage = if event == "missing-usage" { json!({"tokenUsage":{"total":{"outputTokens":100}}}) }
                else { json!({"tokenUsage":{"total":{"outputTokens":100},"last":{"outputTokens":100}}}) };
            timing.codex("thread/tokenUsage/updated", &usage, if event == "clock" { 900 } else { 3000 });
            assert_eq!(timing.estimated_rate(), None);
        }
    }
    #[test]
    fn stream_rate_must_match_the_complete_turn_output() {
        let mut timing = GenerationTiming::default();
        timing.pi_start("one", 1000);
        timing.pi_end("one", Some(120), 3000);
        assert_eq!(timing.rate_for_output(Some(120)), Some(60.0));
        assert_eq!(timing.rate_for_output(Some(240)), None);
        assert_eq!(timing.rate_for_output(None), None);
        timing.pi_start("two", 4000);
        assert_eq!(timing.rate_for_output(Some(120)), None);
        timing.pi_end("two", Some(0), 6000);
        assert_eq!(timing.rate_for_output(Some(120)), Some(30.0));
    }
    #[test]
    fn codex_active_turn_is_only_a_labeled_estimate() {
        let mut timing = GenerationTiming::default();
        timing.codex("turn/started", &json!({"turn":{"id":"one"}}), 1000);
        timing.codex("thread/tokenUsage/updated", &json!({"turnId":"one","tokenUsage":{
            "total":{"outputTokens":120},"last":{"outputTokens":120}}}), 3000);
        assert_eq!(timing.rate(), None);
        assert_eq!(timing.rate_for_output(Some(120)), None);
        assert_eq!(timing.estimated_rate(), Some(60.0));
    }
    #[test]
    fn pi_requires_usage_for_every_observed_message() {
        let mut timing = GenerationTiming::default();
        timing.pi_start("pi-a0", 1000);
        timing.pi_end("pi-a0", Some(120), 3000);
        timing.pi_start("pi-a1", 4000);
        timing.pi_end("pi-a1", None, 9000);
        assert_eq!(timing.rate(), None);
        assert_eq!(timing.rate_for_output(Some(120)), None);
    }
}
