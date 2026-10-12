//! TUI recordings have response/usage timestamps, but no live protocol stream. Estimate response
//! throughput from those durable boundaries, excluding the union of tool/approval waits. Never use
//! transcript read time: opening Info later, resuming, or replaying a recording must keep the same rate.
use serde_json::Value;
use std::collections::HashSet;

#[derive(Default)]
pub(super) struct RecordedTiming {
    started: bool,
    active_since: Option<u64>,
    elapsed: u64,
    last_time: Option<u64>,
    tools: HashSet<String>,
    seen_tools: HashSet<String>,
    tokens: Option<u64>,
    measured_ms: u64,
    invalid: bool,
}

impl RecordedTiming {
    pub fn start(&mut self, row: &Value) {
        *self = Self::default();
        self.active_since = self.timestamp(row);
        self.started = self.active_since.is_some();
    }

    pub fn invalidate(&mut self) {
        self.invalid = true;
    }

    fn timestamp(&mut self, row: &Value) -> Option<u64> {
        let ms = row
            .get("timestamp")
            .and_then(Value::as_str)
            .and_then(|s| {
                time::OffsetDateTime::parse(s, &time::format_description::well_known::Rfc3339).ok()
            })
            .and_then(|t| u64::try_from(t.unix_timestamp_nanos() / 1_000_000).ok());
        if ms.is_none() || self.last_time.zip(ms).is_some_and(|(last, now)| now < last) {
            self.invalid = true;
            return None;
        }
        self.last_time = ms;
        ms
    }

    fn elapsed_at(&self, now: u64) -> Option<u64> {
        let active = self
            .active_since
            .map_or(Some(0), |start| now.checked_sub(start))?;
        self.elapsed.checked_add(active)
    }

    pub fn output(&mut self, tokens: u64, row: &Value, update_repeated: bool) {
        if !self.started {
            self.invalidate();
            return;
        }
        // Codex writes native and legacy snapshots of the same response. Claude writes multiple
        // blocks with the same message usage; its last block supplies the response's end timestamp.
        if !update_repeated && self.tokens == Some(tokens) {
            return;
        }
        let Some(now) = self.timestamp(row) else {
            return;
        };
        let Some(elapsed) = self.elapsed_at(now) else {
            self.invalidate();
            return;
        };
        if self.tokens.is_some_and(|previous| {
            tokens < previous || (tokens > previous && previous > 0 && elapsed <= self.measured_ms)
        }) {
            // Another model response while all measured time is paused means generation overlapped
            // a tool, or a completion boundary is missing. Subtracting that wait would inflate speed.
            self.invalidate();
        }
        self.tokens = Some(tokens);
        self.measured_ms = elapsed;
    }

    pub fn start_tool(&mut self, id: &str, row: &Value) {
        if !self.started {
            self.invalidate();
            return;
        }
        if id.is_empty() {
            self.invalidate();
            return;
        }
        if !self.seen_tools.insert(id.to_string()) {
            return;
        }
        let Some(now) = self.timestamp(row) else {
            return;
        };
        if self.tools.is_empty() {
            match self.elapsed_at(now) {
                Some(elapsed) => self.elapsed = elapsed,
                None => self.invalidate(),
            }
            self.active_since = None;
        }
        self.tools.insert(id.to_string());
    }

    pub fn finish_tool(&mut self, id: &str, row: &Value) {
        if !self.started {
            self.invalidate();
            return;
        }
        // A repeated result is harmless, but a result without its call cannot define a reliable wait.
        if !self.tools.contains(id) {
            if !self.seen_tools.contains(id) {
                self.invalidate();
            }
            return;
        }
        let Some(now) = self.timestamp(row) else {
            return;
        };
        self.tools.remove(id);
        if self.tools.is_empty() {
            self.active_since = Some(now);
        }
    }

    pub fn rate(&self) -> Option<f64> {
        let tokens = self.tokens?;
        (self.started && !self.invalid && self.measured_ms >= 100 && tokens > 0)
            .then(|| tokens as f64 * 1000.0 / self.measured_ms as f64)
    }
}
