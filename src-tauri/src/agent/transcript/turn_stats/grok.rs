//! Grok's completion events report turn usage and model API time (not whole-turn elapsed time).
use super::*;

#[derive(Default)]
struct Accounting {
    completed: HashMap<String, Usage>,
    prompt: Option<String>,
    turn: Option<Usage>,
    rate: Option<f64>,
    tools: HashSet<String>,
    events: HashSet<String>,
    accepting_user: bool,
}
impl Accounting {
    fn consume(&mut self, row: &Value) {
        if !matches!(
            row["method"].as_str(),
            Some("session/update" | "_x.ai/session/update")
        ) {
            return;
        }
        let params = &row["params"];
        if let Some(id) = params["_meta"]["eventId"].as_str() {
            if !self.events.insert(id.to_string()) {
                return;
            }
        }
        let update = &params["update"];
        match update["sessionUpdate"].as_str() {
            Some("user_message_chunk") => {
                let prompt = update["_meta"]["promptIndex"]
                    .as_u64()
                    .map(|n| n.to_string());
                if !self.accepting_user || (prompt.is_some() && prompt != self.prompt) {
                    self.turn = None;
                    self.rate = None;
                    self.tools.clear();
                    self.prompt = prompt;
                }
                self.accepting_user = true;
            }
            Some("tool_call") => {
                self.accepting_user = false;
                if let Some(id) = update["toolCallId"].as_str() {
                    self.tools.insert(id.to_string());
                }
            }
            Some("turn_completed") => {
                self.accepting_user = false;
                let Some(id) = update["prompt_id"].as_str() else {
                    return;
                };
                let usage = &update["usage"];
                let (Some(input), Some(output)) = (
                    usage["inputTokens"].as_u64(),
                    usage["outputTokens"].as_u64(),
                ) else {
                    return;
                };
                // Grok input includes cache reads; output already includes reasoningTokens.
                let cached = usage["cachedReadTokens"].as_u64();
                let value = Usage {
                    input,
                    output,
                    cached: cached.unwrap_or(0),
                    cache_known: cached.is_some(),
                };
                if self.completed.insert(id.to_string(), value).is_some() {
                    return;
                }
                self.turn = Some(value);
                self.rate = usage["apiDurationMs"]
                    .as_u64()
                    .filter(|ms| *ms > 0 && output > 0)
                    .map(|ms| output as f64 * 1000.0 / ms as f64);
            }
            Some("agent_message_chunk" | "agent_thought_chunk") => self.accepting_user = false,
            _ => {}
        }
    }
    fn snapshot(&self) -> TurnStats {
        let session = self
            .completed
            .values()
            .fold(0u64, |total, usage| total.saturating_add(usage.total()));
        let turn = self.turn;
        TurnStats {
            tokens: turn.map(|u| u.output),
            input_tokens: turn.map(|u| u.input),
            total_tokens: turn.map(Usage::total),
            cached_tokens: turn
                .filter(|u| u.cache_known && u.cached > 0)
                .map(|u| u.cached),
            cache_hit_percent: turn
                .filter(|u| u.cache_known && u.input > 0 && u.cached <= u.input)
                .map(|u| u.cached as f64 * 100.0 / u.input as f64),
            session_tokens: (!self.completed.is_empty()).then_some(session),
            tools_used: Some(self.tools.len() as u32),
            generation_tokens_per_second: self.rate,
            ..TurnStats::default()
        }
    }
}

#[derive(Default)]
struct Cached {
    path: PathBuf,
    modified: Option<SystemTime>,
    offset: u64,
    accounting: Accounting,
}
static CACHE: OnceLock<Mutex<HashMap<String, Cached>>> = OnceLock::new();
pub(super) fn stats(id: &str) -> Result<TurnStats, String> {
    let mut cache = CACHE.get_or_init(Default::default).lock().unwrap();
    if cache.len() >= 16 && !cache.contains_key(id) {
        cache.clear();
    }
    let entry = cache.entry(id.to_string()).or_default();
    if !entry.path.is_file() {
        *entry = Cached {
            path: resume::find_grok_updates(id).ok_or("Grok recording not found")?,
            ..Cached::default()
        };
    }
    read(entry)
}
fn read(entry: &mut Cached) -> Result<TurnStats, String> {
    let metadata = std::fs::metadata(&entry.path).map_err(|e| e.to_string())?;
    let modified = metadata.modified().ok();
    if entry.modified == modified && entry.offset == metadata.len() {
        return Ok(entry.accounting.snapshot());
    }
    if metadata.len() <= entry.offset {
        entry.offset = 0;
        entry.accounting = Accounting::default();
    }
    let mut file = BufReader::new(std::fs::File::open(&entry.path).map_err(|e| e.to_string())?);
    file.seek(SeekFrom::Start(entry.offset))
        .map_err(|e| e.to_string())?;
    loop {
        let mut line = String::new();
        let n = file.read_line(&mut line).map_err(|e| e.to_string())?;
        if n == 0 || !line.ends_with('\n') {
            break;
        }
        if let Ok(row) = serde_json::from_str(&line) {
            entry.accounting.consume(&row);
        }
        entry.offset += n as u64;
    }
    entry.modified = modified;
    Ok(entry.accounting.snapshot())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    fn update(value: Value) -> Value {
        json!({"method":"_x.ai/session/update","params":{"update":value}})
    }
    #[test]
    fn native_api_time_deduplicates_and_resets_at_next_prompt() {
        let mut a = Accounting::default();
        let done = update(
            json!({"sessionUpdate":"turn_completed","prompt_id":"first","usage":{"inputTokens":1000,"outputTokens":200,"reasoningTokens":150,"cachedReadTokens":800,"apiDurationMs":4000},"elapsed_ms":90000}),
        );
        a.consume(&done);
        a.consume(&done);
        let stats = a.snapshot();
        assert_eq!(stats.tokens, Some(200));
        assert_eq!(stats.total_tokens, Some(1200));
        assert_eq!(stats.session_tokens, Some(1200));
        assert_eq!(stats.generation_tokens_per_second, Some(50.0));
        a.consume(&update(
            json!({"sessionUpdate":"user_message_chunk","_meta":{"promptIndex":2}}),
        ));
        a.consume(&done); // A late duplicate must not resurrect the completed turn.
        assert_eq!(a.snapshot().generation_tokens_per_second, None);
        a.consume(&update(json!({"sessionUpdate":"turn_completed","prompt_id":"second","usage":{"inputTokens":100,"outputTokens":10}})));
        assert_eq!(a.snapshot().generation_tokens_per_second, None);
        assert_eq!(a.snapshot().session_tokens, Some(1310));
    }
}
