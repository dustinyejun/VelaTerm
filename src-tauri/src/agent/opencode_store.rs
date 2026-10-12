//! Read-only access to OpenCode's own conversation store.
//!
//! OpenCode keeps sessions, messages, and message parts in one SQLite database rather than in flat files.
//! The chat engine reads it to replay a conversation before its server is up, the read-only session view
//! reads it for a terminal-driven session, and the Markdown export reads it for a finished conversation.
//! Nothing here writes: OpenCode owns the file, and this module only mirrors what it recorded.
//!
//! Every row stores its JSON body without the identifying columns, so the columns are folded back into the
//! value on the way out. Callers then see the same `Session`, `Message`, and `Part` shapes the HTTP API
//! returns, which is what lets one converter serve both the store and the live server.
//!
//! 新版 OpenCode 把一条消息整体内联进 `session_message` 一行（`data` 之外只有 id/type/seq 等识别列），旧版的
//! `message`/`part` 两表可能与它共存于同一库。两种行都会被读出，并归一化成上面这套旧 HTTP 形状，消费者无需
//! 区分来源。

use std::path::PathBuf;

use rusqlite::{Connection, OpenFlags};
use serde_json::Value;

/// One message with its parts, in recording order.
#[derive(Clone, Debug)]
pub struct OpencodeMessage {
    pub info: Value,
    pub parts: Vec<Value>,
}

/// Where OpenCode keeps its data: `$XDG_DATA_HOME/opencode`, else `~/.local/share/opencode` — the same
/// answer on every platform, because OpenCode's own path resolution falls back to that directory when no
/// XDG variable is set.
pub fn data_dir() -> Option<PathBuf> {
    if let Some(xdg) = std::env::var_os("XDG_DATA_HOME") {
        let base = PathBuf::from(xdg);
        if !base.as_os_str().is_empty() {
            return Some(base.join("opencode"));
        }
    }
    crate::host::home_dir().map(|home| home.join(".local").join("share").join("opencode"))
}

pub fn db_path() -> Option<PathBuf> {
    data_dir().map(|dir| dir.join("opencode.db"))
}

fn open() -> Result<Connection, String> {
    let path = db_path().ok_or("OpenCode's data directory could not be located")?;
    if !path.exists() {
        return Err("OpenCode has not recorded any conversation on this machine".to_string());
    }
    let conn = Connection::open_with_flags(
        &path,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .map_err(|e| format!("Failed to open OpenCode's store: {e}"))?;
    // OpenCode writes to the same file while it runs; wait briefly rather than fail on a busy lock.
    conn.busy_timeout(std::time::Duration::from_secs(2))
        .map_err(|e| format!("Failed to open OpenCode's store: {e}"))?;
    Ok(conn)
}

fn parse_json(text: &str) -> Value {
    serde_json::from_str(text).unwrap_or(Value::Null)
}

/// The native model and variant used when OpenCode resumes its terminal interface.
pub fn session_model(session_id: &str) -> Result<Option<Value>, String> {
    use rusqlite::OptionalExtension;
    let conn = open()?;
    let raw: Option<Option<String>> = conn.query_row(
        "SELECT model FROM session WHERE id = ?1", [session_id], |row| row.get(0),
    ).optional().map_err(|e| format!("Failed to read OpenCode session model: {e}"))?;
    Ok(raw.flatten().map(|text| parse_json(&text)))
}

/// Every message of a session with its parts, oldest first.
///
/// Messages that OpenCode has reverted are left out, the way its own client hides them: a revert is
/// staged on the session and cleaned up at the next prompt, and until then the rows still exist.
pub fn messages(session_id: &str) -> Result<Vec<OpencodeMessage>, String> {
    let conn = open()?;
    let revert = {
        let mut statement = conn
            .prepare("SELECT revert FROM session WHERE id = ?1")
            .map_err(|e| format!("Failed to read OpenCode's store: {e}"))?;
        let mut rows = statement
            .query([session_id])
            .map_err(|e| format!("Failed to read OpenCode's store: {e}"))?;
        match rows.next().map_err(|e| format!("Failed to read OpenCode's store: {e}"))? {
            Some(row) => row
                .get::<_, Option<String>>(0)
                .map_err(|e| format!("Failed to read OpenCode's store: {e}"))?
                .map(|text| parse_json(&text)),
            None => return Err(format!("OpenCode has no session {session_id}")),
        }
    };
    let mut out = read_messages(&conn, session_id)?;
    if let Some(cut) = revert.as_ref().and_then(|value| reverted_from(value)) {
        if let Some(index) = out.iter().position(|m| m.info.get("id").and_then(Value::as_str) == Some(cut)) {
            out.truncate(index);
        }
    }
    Ok(out)
}

/// The message a staged revert cut the conversation at, when the whole message was reverted.
///
/// A revert that names a part keeps the message and drops only later parts; OpenCode's own client shows
/// such a message in full, so it is kept here too.
pub fn reverted_from(revert: &Value) -> Option<&str> {
    if revert.get("partID").and_then(Value::as_str).is_some() {
        return None;
    }
    revert.get("messageID").and_then(Value::as_str)
}

fn read_messages(conn: &Connection, session_id: &str) -> Result<Vec<OpencodeMessage>, String> {
    let mut legacy = read_legacy_messages(conn, session_id)?;
    let modern = read_session_messages(conn, session_id)?;
    let mut latest = std::collections::HashMap::new();
    for (index, message) in modern.iter().enumerate() {
        latest.insert(message_id_key(&message.info), index);
    }
    // 同 id 新版优先；新版重复记录保留最后一个 seq，避免迁移后重复计费。
    legacy.retain(|message| !latest.contains_key(&message_id_key(&message.info)));
    let modern: Vec<_> = modern.into_iter().enumerate()
        .filter(|(index, message)| latest.get(&message_id_key(&message.info)) == Some(index))
        .map(|(_, message)| message).collect();
    let mut merged = Vec::with_capacity(legacy.len() + modern.len());
    let mut legacy = legacy.into_iter().peekable();
    let mut modern = modern.into_iter().peekable();
    // 按创建时间插入剩余旧记录，但始终保留新版 seq 的因果顺序，包括相同时间戳或时钟回拨。
    while let (Some(old), Some(new)) = (legacy.peek(), modern.peek()) {
        let old_key = (created_ms(&old.info), message_id_key(&old.info));
        let new_key = (created_ms(&new.info), message_id_key(&new.info));
        merged.push(if old_key <= new_key { legacy.next().unwrap() } else { modern.next().unwrap() });
    }
    merged.extend(legacy);
    merged.extend(modern);
    Ok(merged)
}

fn created_ms(info: &Value) -> i64 {
    info.pointer("/time/created")
        .and_then(|value| value.as_i64().or_else(|| value.as_f64().map(|f| f as i64)))
        .unwrap_or(0)
}

fn message_id_key(info: &Value) -> String {
    info.get("id").and_then(Value::as_str).unwrap_or_default().to_string()
}

fn table_exists(conn: &Connection, name: &str) -> bool {
    conn.query_row(
        "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = ?1",
        [name],
        |row| row.get::<_, i64>(0),
    )
    .map(|count| count > 0)
    .unwrap_or(false)
}

/// 旧版存储：`message` 行加 `part` 行，识别列折叠回 JSON。
fn read_legacy_messages(conn: &Connection, session_id: &str) -> Result<Vec<OpencodeMessage>, String> {
    if !table_exists(conn, "message") {
        return Ok(Vec::new());
    }
    let mut statement = conn
        .prepare("SELECT id, data FROM message WHERE session_id = ?1 ORDER BY time_created, id")
        .map_err(|e| format!("Failed to read OpenCode's store: {e}"))?;
    let mut rows = statement
        .query([session_id])
        .map_err(|e| format!("Failed to read OpenCode's store: {e}"))?;
    let mut messages: Vec<OpencodeMessage> = Vec::new();
    while let Some(row) = rows.next().map_err(|e| format!("Failed to read OpenCode's store: {e}"))? {
        let id: String = row
            .get(0)
            .map_err(|e| format!("Failed to read OpenCode's store: {e}"))?;
        let data: String = row
            .get(1)
            .map_err(|e| format!("Failed to read OpenCode's store: {e}"))?;
        let mut info = parse_json(&data);
        if let Value::Object(map) = &mut info {
            map.insert("id".into(), Value::String(id));
            map.insert("sessionID".into(), Value::String(session_id.to_string()));
        }
        messages.push(OpencodeMessage { info, parts: Vec::new() });
    }
    if !table_exists(conn, "part") {
        return Ok(messages);
    }
    let mut statement = conn
        .prepare("SELECT id, message_id, data FROM part WHERE session_id = ?1 ORDER BY time_created, id")
        .map_err(|e| format!("Failed to read OpenCode's store: {e}"))?;
    let mut rows = statement
        .query([session_id])
        .map_err(|e| format!("Failed to read OpenCode's store: {e}"))?;
    let index: std::collections::HashMap<String, usize> = messages
        .iter()
        .enumerate()
        .filter_map(|(i, m)| m.info.get("id").and_then(Value::as_str).map(|id| (id.to_string(), i)))
        .collect();
    while let Some(row) = rows.next().map_err(|e| format!("Failed to read OpenCode's store: {e}"))? {
        let id: String = row
            .get(0)
            .map_err(|e| format!("Failed to read OpenCode's store: {e}"))?;
        let message_id: String = row
            .get(1)
            .map_err(|e| format!("Failed to read OpenCode's store: {e}"))?;
        let data: String = row
            .get(2)
            .map_err(|e| format!("Failed to read OpenCode's store: {e}"))?;
        let Some(&slot) = index.get(&message_id) else { continue };
        let mut part = parse_json(&data);
        if let Value::Object(map) = &mut part {
            map.insert("id".into(), Value::String(id));
            map.insert("messageID".into(), Value::String(message_id));
            map.insert("sessionID".into(), Value::String(session_id.to_string()));
        }
        messages[slot].parts.push(part);
    }
    Ok(messages)
}

/// 新版存储：`session_message` 一行一条消息，`data` 不含 id/type，识别列取自列值；行不存在时视为空表。
fn read_session_messages(conn: &Connection, session_id: &str) -> Result<Vec<OpencodeMessage>, String> {
    if !table_exists(conn, "session_message") {
        return Ok(Vec::new());
    }
    let mut statement = conn
        .prepare("SELECT id, type, seq, data FROM session_message WHERE session_id = ?1 ORDER BY seq, id")
        .map_err(|e| format!("Failed to read OpenCode's store: {e}"))?;
    let mut rows = statement
        .query([session_id])
        .map_err(|e| format!("Failed to read OpenCode's store: {e}"))?;
    let mut messages: Vec<OpencodeMessage> = Vec::new();
    while let Some(row) = rows.next().map_err(|e| format!("Failed to read OpenCode's store: {e}"))? {
        let id: String = row
            .get(0)
            .map_err(|e| format!("Failed to read OpenCode's store: {e}"))?;
        let kind: Option<String> = row
            .get(1)
            .map_err(|e| format!("Failed to read OpenCode's store: {e}"))?;
        let seq = sqlite_value(
            row.get::<_, rusqlite::types::Value>(2)
                .map_err(|e| format!("Failed to read OpenCode's store: {e}"))?,
        );
        let data: Option<String> = row
            .get(3)
            .map_err(|e| format!("Failed to read OpenCode's store: {e}"))?;
        let (Some(kind), Some(data)) = (kind, data) else { continue };
        if let Some(message) = normalize_session_message(&id, session_id, &kind, seq, parse_json(&data)) {
            messages.push(message);
        }
    }
    Ok(messages)
}

/// 新版行的 `seq` 列可能是整数或可排序文本，统一转成 JSON 值原样透传给消费者。
fn sqlite_value(value: rusqlite::types::Value) -> Value {
    match value {
        rusqlite::types::Value::Null => Value::Null,
        rusqlite::types::Value::Integer(n) => Value::from(n),
        rusqlite::types::Value::Real(f) => Value::from(f),
        rusqlite::types::Value::Text(t) => Value::from(t),
        rusqlite::types::Value::Blob(_) => Value::Null,
    }
}

/// 新版行 → 旧 HTTP 形状。role 从 type 得出；非 user/assistant 的控制行（synthetic/idle/compaction 等）
/// role 原样等于 type，让只认 user/assistant 的渲染自然跳过，而轮次统计仍能按 type/seq 找到轮次边界。
fn normalize_session_message(
    id: &str,
    session_id: &str,
    kind: &str,
    seq: Value,
    data: Value,
) -> Option<OpencodeMessage> {
    let mut info = data;
    let map = info.as_object_mut()?;
    map.insert("id".into(), Value::String(id.to_string()));
    map.insert("sessionID".into(), Value::String(session_id.to_string()));
    map.insert("type".into(), Value::String(kind.to_string()));
    if !seq.is_null() {
        map.insert("seq".into(), seq);
    }
    map.insert("role".into(), Value::String(kind.to_string()));
    let parts = match kind {
        "assistant" => {
            // 新版把模型引用收进 model{providerID,id,variant?}；旧 HTTP 形状用平铺的 modelID/providerID。
            // 完整 model 对象原样保留，variant 等新字段不丢。
            if let Some(model) = map.remove("model") {
                if let Some(model_id) = model.get("id").and_then(Value::as_str) {
                    map.insert("modelID".into(), Value::String(model_id.to_string()));
                }
                if let Some(provider_id) = model.get("providerID").and_then(Value::as_str) {
                    map.insert("providerID".into(), Value::String(provider_id.to_string()));
                }
                map.insert("model".into(), model);
            }
            let content = map.remove("content");
            assistant_parts(id, session_id, content.as_ref())
        }
        "user" => user_parts(id, session_id, map),
        _ => Vec::new(),
    };
    Some(OpencodeMessage { info, parts })
}

/// assistant 的 `content` 数组 → 旧版 parts：text/reasoning 原样补识别列，tool 项改名并转换 state。
fn assistant_parts(message_id: &str, session_id: &str, content: Option<&Value>) -> Vec<Value> {
    let Some(items) = content.and_then(Value::as_array) else {
        return Vec::new();
    };
    items
        .iter()
        .enumerate()
        .map(|(index, item)| {
            let mut part = item.clone();
            let Some(map) = part.as_object_mut() else {
                return item.clone();
            };
            let is_tool = map.get("type").and_then(Value::as_str) == Some("tool");
            // 新版 text/reasoning 项没有 id，合成一个稳定 id；tool 项自带 id 则沿用。
            let part_id = map
                .get("id")
                .and_then(Value::as_str)
                .map(str::to_string)
                .unwrap_or_else(|| format!("{message_id}_c{index}"));
            map.insert("id".into(), Value::String(part_id));
            map.insert("messageID".into(), Value::String(message_id.to_string()));
            map.insert("sessionID".into(), Value::String(session_id.to_string()));
            if let Some(time) = map.get_mut("time").and_then(Value::as_object_mut) {
                if let Some(created) = time.get("created").cloned() {
                    time.entry("start").or_insert(created);
                }
                if let Some(completed) = time.get("completed").cloned() {
                    time.entry("end").or_insert(completed);
                }
            }
            if is_tool {
                // 新版工具项叫 name；旧 HTTP part 用 tool。
                if let Some(name) = map.remove("name") {
                    map.insert("tool".into(), name);
                }
                let time = map.get("time").cloned();
                if let Some(state) = map.get_mut("state").and_then(Value::as_object_mut) {
                    if let Some(time) = time {
                        state.entry("time").or_insert(time);
                    }
                    // 旧版完成的输出是字符串 `output`；新版是 content 数组。
                    if !state.contains_key("output") {
                        if let Some(output) = content_to_output(state.get("content")) {
                            state.insert("output".into(), Value::String(output));
                        }
                    }
                    // 旧版 `state.error` 是字符串；新版是结构化对象。
                    if let Some(error) = state.get("error") {
                        if !error.is_string() {
                            let text = error_text(error);
                            state.insert("error".into(), Value::String(text));
                        }
                    }
                }
            }
            part
        })
        .collect()
}

/// 新版工具完成的 `content` 数组 → 旧版 `state.output` 字符串：文本项拼接，文件项退化为 URL。
fn content_to_output(content: Option<&Value>) -> Option<String> {
    let content = content?;
    if let Some(text) = content.as_str() {
        return (!text.is_empty()).then(|| text.to_string());
    }
    let items = content.as_array()?;
    let mut out: Vec<String> = Vec::new();
    for item in items {
        match item.get("type").and_then(Value::as_str) {
            Some("text") => {
                if let Some(text) = item.get("text").and_then(Value::as_str) {
                    out.push(text.to_string());
                }
            }
            Some("file") => {
                if let Some(url) = item.get("uri").or_else(|| item.get("url")).and_then(Value::as_str) {
                    out.push(url.to_string());
                }
            }
            _ => {}
        }
    }
    (!out.is_empty()).then(|| out.join("\n"))
}

/// 新版错误对象 → 旧版错误字符串：取 message/name，退化到原始 JSON。
fn error_text(error: &Value) -> String {
    if let Some(text) = error.get("message").and_then(Value::as_str) {
        return text.to_string();
    }
    if let Some(text) = error.get("name").and_then(Value::as_str) {
        return text.to_string();
    }
    error.to_string()
}

/// user 的 `text`/`files` → 旧版 parts：一条 text part 加每个附件一条 file part。
fn user_parts(message_id: &str, session_id: &str, map: &serde_json::Map<String, Value>) -> Vec<Value> {
    let mut parts = Vec::new();
    if let Some(text) = map.get("text").and_then(Value::as_str) {
        parts.push(serde_json::json!({
            "id": format!("{message_id}_text"),
            "type": "text",
            "text": text,
            "messageID": message_id,
            "sessionID": session_id,
        }));
    }
    if let Some(files) = map.get("files").and_then(Value::as_array) {
        for (index, file) in files.iter().enumerate() {
            let mime = file.get("mime").and_then(Value::as_str).unwrap_or("");
            // 内联附件回 data URL，uri 附件直接用其地址，与旧 HTTP file part 的 url 语义一致。
            let url = match file.pointer("/source/type").and_then(Value::as_str) {
                Some("uri") => file
                    .pointer("/source/uri")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                _ => file
                    .get("data")
                    .and_then(Value::as_str)
                    .map(|data| format!("data:{mime};base64,{data}"))
                    .unwrap_or_default(),
            };
            parts.push(serde_json::json!({
                "id": format!("{message_id}_f{index}"),
                "type": "file",
                "mime": mime,
                "url": url,
                "filename": file.get("name").and_then(Value::as_str).unwrap_or(""),
                "messageID": message_id,
                "sessionID": session_id,
            }));
        }
    }
    parts
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_whole_message_revert_names_its_cut() {
        assert_eq!(reverted_from(&serde_json::json!({"messageID":"msg_1"})), Some("msg_1"));
        assert_eq!(
            reverted_from(&serde_json::json!({"messageID":"msg_1","partID":"prt_2"})),
            None
        );
        assert_eq!(reverted_from(&Value::Null), None);
    }

    #[test]
    fn data_dir_follows_xdg_then_home() {
        // Whatever the machine has, the answer ends in OpenCode's own directory name.
        let dir = data_dir();
        assert!(dir.is_none_or(|d| d.ends_with("opencode")));
    }

    // ---- 新版 session_message 兼容的内存库测试 ----

    use rusqlite::Connection;

    fn memory_conn() -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE session (id TEXT PRIMARY KEY, revert TEXT, model TEXT);
             CREATE TABLE message (id TEXT PRIMARY KEY, session_id TEXT, data TEXT, time_created INTEGER);
             CREATE TABLE part (id TEXT PRIMARY KEY, session_id TEXT, message_id TEXT, data TEXT, time_created INTEGER);
             -- id 不设主键约束：允许插入同 id 重复行，覆盖合并去重的路径
             CREATE TABLE session_message (
                 id TEXT, session_id TEXT, type TEXT, seq TEXT,
                 time_created INTEGER, time_updated INTEGER, data TEXT
             );",
        )
        .unwrap();
        conn
    }

    fn insert_legacy_message(conn: &Connection, id: &str, created: i64, data: Value) {
        conn.execute(
            "INSERT INTO message (id, session_id, data, time_created) VALUES (?1, 'ses_1', ?2, ?3)",
            rusqlite::params![id, data.to_string(), created],
        )
        .unwrap();
    }

    fn insert_legacy_part(conn: &Connection, id: &str, message_id: &str, created: i64, data: Value) {
        conn.execute(
            "INSERT INTO part (id, session_id, message_id, data, time_created) VALUES (?1, 'ses_1', ?2, ?3, ?4)",
            rusqlite::params![id, message_id, data.to_string(), created],
        )
        .unwrap();
    }

    fn insert_modern_message(conn: &Connection, id: &str, kind: &str, seq: &str, created: i64, data: Value) {
        conn.execute(
            "INSERT INTO session_message (id, session_id, type, seq, time_created, time_updated, data)
             VALUES (?1, 'ses_1', ?2, ?3, ?4, ?4, ?5)",
            rusqlite::params![id, kind, seq, created, data.to_string()],
        )
        .unwrap();
    }

    fn role_of(message: &OpencodeMessage) -> &str {
        message.info.get("role").and_then(Value::as_str).unwrap_or("")
    }

    #[test]
    fn legacy_only_rows_keep_http_shape() {
        let conn = memory_conn();
        insert_legacy_message(
            &conn,
            "msg_u",
            10,
            serde_json::json!({"role": "user", "time": {"created": 10}}),
        );
        insert_legacy_part(&conn, "prt_1", "msg_u", 11, serde_json::json!({"type": "text", "text": "hi"}));
        insert_legacy_message(
            &conn,
            "msg_a",
            20,
            serde_json::json!({"role": "assistant", "time": {"created": 20}}),
        );
        let messages = read_messages(&conn, "ses_1").unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(role_of(&messages[0]), "user");
        assert_eq!(role_of(&messages[1]), "assistant");
        assert_eq!(messages[0].info.get("id").and_then(Value::as_str), Some("msg_u"));
        assert_eq!(
            messages[0].info.get("sessionID").and_then(Value::as_str),
            Some("ses_1")
        );
        let part = &messages[0].parts[0];
        assert_eq!(part.get("id").and_then(Value::as_str), Some("prt_1"));
        assert_eq!(part.get("messageID").and_then(Value::as_str), Some("msg_u"));
        assert_eq!(part.get("sessionID").and_then(Value::as_str), Some("ses_1"));
    }

    #[test]
    fn modern_only_rows_normalize_to_http_shape() {
        let conn = memory_conn();
        insert_modern_message(
            &conn,
            "msg_1",
            "user",
            "0001",
            5,
            serde_json::json!({
                "text": "hello",
                "files": [{"data": "QUJD", "mime": "image/png", "source": {"type": "inline"}, "name": "shot.png"}],
                "time": {"created": 5}
            }),
        );
        insert_modern_message(
            &conn,
            "msg_2",
            "assistant",
            "0002",
            6,
            serde_json::json!({
                "agent": "build",
                "model": {"providerID": "anthropic", "id": "claude-x", "variant": "fast"},
                "tokens": {"input": 1, "output": 2, "reasoning": 3, "cache": {"read": 4, "write": 5}},
                "time": {"created": 6, "streamed": 7, "completed": 8},
                "content": [
                    {"type": "text", "text": "plan"},
                    {"type": "reasoning", "text": "think"},
                    {"type": "tool", "id": "tool_1", "name": "bash",
                     "state": {"status": "completed", "input": {"command": "ls"},
                               "content": [{"type": "text", "text": "out"}], "metadata": {"sessionId": "child"}},
                     "time": {"created": 6, "completed": 8}}
                ]
            }),
        );
        let messages = read_messages(&conn, "ses_1").unwrap();
        assert_eq!(messages.len(), 2);

        let user = &messages[0];
        assert_eq!(role_of(user), "user");
        assert_eq!(user.info.get("type").and_then(Value::as_str), Some("user"));
        assert_eq!(user.info.get("seq").and_then(Value::as_str), Some("0001"));
        let parts = &user.parts;
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0].get("type").and_then(Value::as_str), Some("text"));
        assert_eq!(parts[0].get("text").and_then(Value::as_str), Some("hello"));
        assert_eq!(parts[0].get("id").and_then(Value::as_str), Some("msg_1_text"));
        assert_eq!(parts[1].get("type").and_then(Value::as_str), Some("file"));
        assert_eq!(parts[1].get("mime").and_then(Value::as_str), Some("image/png"));
        assert_eq!(parts[1].get("filename").and_then(Value::as_str), Some("shot.png"));
        assert_eq!(
            parts[1].get("url").and_then(Value::as_str),
            Some("data:image/png;base64,QUJD")
        );

        let assistant = &messages[1];
        assert_eq!(role_of(assistant), "assistant");
        assert_eq!(assistant.info.get("modelID").and_then(Value::as_str), Some("claude-x"));
        assert_eq!(
            assistant.info.get("providerID").and_then(Value::as_str),
            Some("anthropic")
        );
        assert_eq!(
            assistant.info.pointer("/model/variant").and_then(Value::as_str),
            Some("fast")
        );
        assert_eq!(
            assistant.info.pointer("/time/streamed").and_then(Value::as_i64),
            Some(7)
        );
        assert!(assistant.info.get("content").is_none());
        let parts = &assistant.parts;
        assert_eq!(parts.len(), 3);
        assert_eq!(parts[0].get("type").and_then(Value::as_str), Some("text"));
        assert_eq!(parts[0].get("id").and_then(Value::as_str), Some("msg_2_c0"));
        assert_eq!(parts[1].get("type").and_then(Value::as_str), Some("reasoning"));
        assert_eq!(parts[2].get("id").and_then(Value::as_str), Some("tool_1"));
        assert_eq!(parts[2].get("tool").and_then(Value::as_str), Some("bash"));
        assert_eq!(parts[2].pointer("/state/time/start").and_then(Value::as_u64), Some(6));
        assert_eq!(parts[2].pointer("/state/time/end").and_then(Value::as_u64), Some(8));
        assert_eq!(
            parts[2].pointer("/state/output").and_then(Value::as_str),
            Some("out")
        );
        assert_eq!(
            parts[2].pointer("/state/metadata/sessionId").and_then(Value::as_str),
            Some("child")
        );
    }

    #[test]
    fn mixed_old_and_new_rows_coexist() {
        let conn = memory_conn();
        insert_legacy_message(
            &conn,
            "msg_old",
            10,
            serde_json::json!({"role": "user", "time": {"created": 10}}),
        );
        insert_modern_message(
            &conn,
            "msg_new",
            "user",
            "0001",
            5,
            serde_json::json!({"text": "new", "time": {"created": 5}}),
        );
        let messages = read_messages(&conn, "ses_1").unwrap();
        // 旧记录没有被丢，且按创建时间交错排序。
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].info.get("id").and_then(Value::as_str), Some("msg_new"));
        assert_eq!(messages[1].info.get("id").and_then(Value::as_str), Some("msg_old"));
    }

    #[test]
    fn modern_row_wins_on_shared_id() {
        let conn = memory_conn();
        insert_legacy_message(
            &conn,
            "msg_1",
            10,
            serde_json::json!({"role": "user", "legacy": true, "time": {"created": 10}}),
        );
        insert_modern_message(
            &conn,
            "msg_1",
            "user",
            "0001",
            10,
            serde_json::json!({"text": "new", "time": {"created": 10}}),
        );
        let messages = read_messages(&conn, "ses_1").unwrap();
        // 同 id 只保留一条，且是新版形状。
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].info.get("text").and_then(Value::as_str), Some("new"));
        assert!(messages[0].info.get("legacy").is_none());
        assert_eq!(role_of(&messages[0]), "user");
    }

    #[test]
    fn modern_sequence_survives_equal_timestamps_and_clock_reversal() {
        let conn = memory_conn();
        insert_modern_message(&conn, "z_user", "user", "0001", 10,
            serde_json::json!({"text":"go","time":{"created":10}}));
        insert_modern_message(&conn, "a_assistant", "assistant", "0002", 10,
            serde_json::json!({"time":{"created":10}}));
        insert_modern_message(&conn, "b_assistant", "assistant", "0003", 9,
            serde_json::json!({"time":{"created":9}}));
        let messages = read_messages(&conn, "ses_1").unwrap();
        assert_eq!(messages.iter().map(|m| message_id_key(&m.info)).collect::<Vec<_>>(),
            ["z_user", "a_assistant", "b_assistant"]);
    }

    #[test]
    fn either_storage_schema_can_be_absent() {
        let conn = memory_conn();
        conn.execute_batch("DROP TABLE message; DROP TABLE part;").unwrap();
        insert_modern_message(&conn, "a", "user", "0001", 1,
            serde_json::json!({"text":"go","time":{"created":1}}));
        assert_eq!(read_messages(&conn, "ses_1").unwrap().len(), 1);
        let conn = memory_conn();
        conn.execute_batch("DROP TABLE session_message;").unwrap();
        insert_legacy_message(&conn, "a", 1, serde_json::json!({"role":"user","time":{"created":1}}));
        assert_eq!(read_messages(&conn, "ses_1").unwrap().len(), 1);
    }

    #[test]
    fn duplicate_modern_id_keeps_last_seq() {
        let conn = memory_conn();
        insert_modern_message(
            &conn,
            "msg_1",
            "user",
            "0001",
            5,
            serde_json::json!({"text": "first", "time": {"created": 5}}),
        );
        insert_modern_message(
            &conn,
            "msg_1",
            "user",
            "0002",
            6,
            serde_json::json!({"text": "second", "time": {"created": 6}}),
        );
        let messages = read_messages(&conn, "ses_1").unwrap();
        // 迁移重写可能留下同 id 两行，取 seq 更大的那行。
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].info.get("text").and_then(Value::as_str), Some("second"));
    }
}
