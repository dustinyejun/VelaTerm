//! Read only the numeric subset of Antigravity's native SQLite/protobuf store.
//! Field numbers come from the CLI's embedded cortex.proto and codeium_common.proto descriptors.
//! One generator can produce several steps; count its usage and duration once, never per step.
use super::*;
use rusqlite::{Connection, OpenFlags};
use std::path::Path;

pub(super) fn path(id: &str) -> Option<PathBuf> {
    if id.is_empty()
        || !id
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, b'-' | b'_'))
    {
        return None;
    }
    let path = crate::host::home_dir()?
        .join(".gemini/antigravity-cli/conversations")
        .join(format!("{id}.db"));
    path.is_file().then_some(path)
}
#[derive(Default)]
struct Cached {
    stamp: Vec<(u64, Option<SystemTime>)>,
    stats: TurnStats,
}
static CACHE: OnceLock<Mutex<HashMap<PathBuf, Cached>>> = OnceLock::new();
pub(super) fn stats(id: &str) -> Result<TurnStats, String> {
    let path = path(id).ok_or("Antigravity native conversation store not found")?;
    let stamp = [
        path.clone(),
        PathBuf::from(format!("{}-wal", path.display())),
    ]
    .iter()
    .map(|p| {
        std::fs::metadata(p)
            .map(|m| (m.len(), m.modified().ok()))
            .unwrap_or((0, None))
    })
    .collect::<Vec<_>>();
    let mut cache = CACHE.get_or_init(Default::default).lock().unwrap();
    if let Some(entry) = cache.get(&path).filter(|e| e.stamp == stamp) {
        return Ok(entry.stats.clone());
    }
    let stats = read(&path)?;
    if cache.len() >= 16 {
        cache.clear();
    }
    cache.insert(
        path,
        Cached {
            stamp,
            stats: stats.clone(),
        },
    );
    Ok(stats)
}

fn read(path: &Path) -> Result<TurnStats, String> {
    let wal = PathBuf::from(format!("{}-wal", path.display()));
    let live_wal = std::fs::metadata(wal).is_ok_and(|m| m.len() > 0);
    let mut uri = url::Url::from_file_path(path).map_err(|_| "Invalid native store path")?;
    uri.set_query(Some(if live_wal {
        "mode=ro"
    } else {
        "mode=ro&immutable=1"
    }));
    // A checkpointed WAL-mode store may have no writable -shm file. Immutable opens it without
    // creating sidecars. A live WAL must use SQLite's normal reader so recent records remain visible.
    let mut conn = Connection::open_with_flags(
        uri.as_str(),
        OpenFlags::SQLITE_OPEN_READ_ONLY
            | OpenFlags::SQLITE_OPEN_URI
            | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .map_err(|e| e.to_string())?;
    conn.busy_timeout(std::time::Duration::from_millis(250))
        .map_err(|e| e.to_string())?;
    let transaction = conn.transaction().map_err(|e| e.to_string())?;
    let last_user: Option<u64> = transaction
        .query_row("SELECT max(idx) FROM steps WHERE step_type = 14", [], |r| {
            r.get(0)
        })
        .map_err(|e| e.to_string())?;
    let mut statement = transaction
        .prepare("SELECT data FROM gen_metadata ORDER BY idx")
        .map_err(|e| e.to_string())?;
    let rows = statement
        .query_map([], |r| r.get::<_, Vec<u8>>(0))
        .map_err(|e| e.to_string())?;
    let mut session: Option<Usage> = None;
    let mut turn: Option<Usage> = None;
    let mut generated_ms = 0.0;
    let mut generated_tokens = 0u64;
    let mut timing_complete = true;
    for row in rows {
        let data = row.map_err(|e| e.to_string())?;
        let Some(sample) = sample(&data) else {
            continue;
        };
        session.get_or_insert_with(Usage::default).add(sample.usage);
        if last_user.is_some_and(|start| {
            !sample.steps.is_empty() && sample.steps.iter().all(|idx| *idx > start)
        }) {
            turn.get_or_insert_with(Usage::default).add(sample.usage);
            if let Some(ms) = sample.ms.filter(|ms| *ms > 0.0 && ms.is_finite()) {
                generated_ms += ms;
                generated_tokens = generated_tokens.saturating_add(sample.usage.output);
            } else {
                timing_complete = false;
            }
        }
    }
    let rate = (timing_complete && generated_ms >= 100.0 && generated_tokens > 0)
        .then(|| generated_tokens as f64 * 1000.0 / generated_ms);
    Ok(TurnStats {
        tokens: turn.map(|u| u.output),
        input_tokens: turn.map(|u| u.input),
        total_tokens: turn.map(Usage::total),
        cached_tokens: turn.filter(|u| u.cache_known).map(|u| u.cached),
        cache_hit_percent: turn
            .filter(|u| u.cache_known && u.input > 0 && u.cached <= u.input)
            .map(|u| u.cached as f64 * 100.0 / u.input as f64),
        session_tokens: session.map(Usage::total),
        generation_tokens_per_second: rate,
        output_throughput_basis: rate.map(|_| ThroughputBasis::Request),
        reasoning_tokens: turn.and_then(|u| u.reasoning),
        ..TurnStats::default()
    })
}

// A bounded, nonrecursive protobuf wire reader. Unknown fields are skipped without parsing prompts,
// tool arguments or response text. Malformed/unsupported records never create made-up measurements.
#[derive(Clone, Copy)]
enum Field<'a> {
    Integer(u64),
    Bytes(&'a [u8]),
    Other,
}
fn varint(data: &[u8], offset: &mut usize) -> Option<u64> {
    let mut result = 0u64;
    for shift in (0..70).step_by(7) {
        let byte = *data.get(*offset)?;
        *offset += 1;
        if shift == 63 && byte > 1 {
            return None;
        }
        result |= u64::from(byte & 127) << shift;
        if byte & 128 == 0 {
            return Some(result);
        }
    }
    None
}
fn fields(data: &[u8]) -> Option<Vec<(u64, Field<'_>)>> {
    let mut result = Vec::new();
    let mut offset = 0usize;
    while offset < data.len() {
        if result.len() >= 100_000 {
            return None;
        }
        let tag = varint(data, &mut offset)?;
        if tag >> 3 == 0 {
            return None;
        }
        let field = match tag & 7 {
            0 => Field::Integer(varint(data, &mut offset)?),
            1 | 5 => {
                offset = offset.checked_add(if tag & 7 == 1 { 8 } else { 4 })?;
                if offset > data.len() {
                    return None;
                }
                Field::Other
            }
            2 => {
                let len = usize::try_from(varint(data, &mut offset)?).ok()?;
                let end = offset.checked_add(len)?;
                let bytes = data.get(offset..end)?;
                offset = end;
                Field::Bytes(bytes)
            }
            _ => return None,
        };
        result.push((tag >> 3, field));
    }
    Some(result)
}
fn bytes<'a>(fields: &[(u64, Field<'a>)], number: u64) -> Option<&'a [u8]> {
    fields.iter().rev().find_map(|(n, f)| match f {
        Field::Bytes(b) if *n == number => Some(*b),
        _ => None,
    })
}
fn integer(fields: &[(u64, Field<'_>)], number: u64) -> u64 {
    fields
        .iter()
        .rev()
        .find_map(|(n, f)| match f {
            Field::Integer(v) if *n == number => Some(*v),
            _ => None,
        })
        .unwrap_or(0)
}
fn duration(data: &[u8]) -> Option<f64> {
    let fields = fields(data)?;
    let seconds = integer(&fields, 1);
    let nanos = integer(&fields, 2);
    if seconds > 315_576_000_000 || nanos >= 1_000_000_000 {
        return None;
    }
    Some(seconds as f64 * 1000.0 + nanos as f64 / 1_000_000.0)
}
struct Sample {
    steps: Vec<u64>,
    usage: Usage,
    ms: Option<f64>,
}
fn sample(data: &[u8]) -> Option<Sample> {
    let generator = fields(data)?;
    let chat = fields(bytes(&generator, 1)?)?;
    let usage = fields(bytes(&chat, 4)?)?;
    let mut steps = Vec::new();
    for (number, field) in &generator {
        if *number != 2 {
            continue;
        }
        match field {
            Field::Integer(step) => steps.push(*step),
            Field::Bytes(packed) => {
                let mut offset = 0;
                while offset < packed.len() {
                    steps.push(varint(packed, &mut offset)?);
                }
            }
            Field::Other => return None,
        }
    }
    // ModelUsageStats.output_tokens includes thinking_output_tokens and response_output_tokens.
    // input_tokens excludes the separate cache read/write counts.
    let cached = integer(&usage, 5);
    let value = Usage {
        input: integer(&usage, 2)
            .saturating_add(cached)
            .saturating_add(integer(&usage, 4)),
        output: integer(&usage, 3),
        cached,
        // 已知 proto3 schema 的数值字段缺省即为零，不能用 wire 字段存在性判断未知。
        cache_known: true,
        reasoning: Some(integer(&usage, 9)).filter(|count| *count <= integer(&usage, 3)),
    };
    // 零输出请求也占用模型耗时；保留已报告的零用量，避免只统计有输出的子集。
    // First-token time includes reasoning; streaming duration covers the rest of the response.
    let ms = bytes(&chat, 11)
        .and_then(duration)
        .zip(bytes(&chat, 12).and_then(duration))
        .map(|(first, stream)| first + stream);
    Some(Sample {
        steps,
        usage: value,
        ms,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn v(mut n: u64) -> Vec<u8> {
        let mut b = Vec::new();
        while n >= 128 {
            b.push((n as u8 & 127) | 128);
            n >>= 7;
        }
        b.push(n as u8);
        b
    }
    fn int(n: u64, value: u64) -> Vec<u8> {
        [v(n << 3), v(value)].concat()
    }
    fn blob(n: u64, value: &[u8]) -> Vec<u8> {
        [v(n << 3 | 2), v(value.len() as u64), value.to_vec()].concat()
    }
    fn fixture(steps: &[u64]) -> Vec<u8> {
        let usage = [
            int(2, 100),
            int(3, 200),
            int(9, 150),
            int(10, 50),
            int(5, 800),
        ]
        .concat();
        let chat = [
            blob(4, &usage),
            blob(11, &int(1, 3)),
            blob(12, &int(1, 1)),
            blob(50, b"unknown future field"),
        ]
        .concat();
        [
            blob(1, &chat),
            blob(2, &steps.iter().flat_map(|n| v(*n)).collect::<Vec<_>>()),
        ]
        .concat()
    }
    #[test]
    fn protobuf_subset_keeps_total_output_and_rejects_malformed_timing() {
        let s = sample(&fixture(&[1, 2, 3])).unwrap();
        assert_eq!(s.steps, vec![1, 2, 3]);
        assert_eq!(s.usage.output, 200);
        assert_eq!(s.usage.input, 900);
        assert_eq!(s.ms, Some(4000.0));
        assert!(sample(&[10, 255]).is_none());
        assert!(fields(&[0]).is_none());
        assert!(duration(&int(1, u64::MAX)).is_none());
        assert!(duration(&int(2, 1_000_000_000)).is_none());
        assert!(fields(&[255; 12]).is_none());
    }
    #[test]
    fn zero_usage_and_missing_timing_remain_part_of_native_request_coverage() {
        let dir = std::env::temp_dir().join(format!("vlx-agy-coverage-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("store.db");
        let conn = Connection::open(&path).unwrap();
        conn.execute_batch("CREATE TABLE steps(idx INTEGER,step_type INTEGER); CREATE TABLE gen_metadata(idx INTEGER,data BLOB); INSERT INTO steps VALUES(0,14);").unwrap();
        conn.execute("INSERT INTO gen_metadata VALUES(0,?1)", [fixture(&[1])]).unwrap();
        let zero_chat = [blob(4, &[]), blob(11, &int(1, 3)), blob(12, &int(1, 1))].concat();
        let zero = [blob(1, &zero_chat), int(2, 2)].concat();
        let usage = sample(&zero).unwrap().usage;
        assert_eq!(usage.cached, 0);
        assert!(usage.cache_known);
        assert_eq!(usage.reasoning, Some(0));
        conn.execute("INSERT INTO gen_metadata VALUES(1,?1)", [zero]).unwrap();
        let stats = read(&path).unwrap();
        assert_eq!(stats.tokens, Some(200));
        assert_eq!(stats.reasoning_tokens, Some(150));
        assert_eq!(stats.generation_tokens_per_second, Some(25.0));
        let missing = [blob(1, &blob(4, &int(3, 100))), int(2, 3)].concat();
        conn.execute("INSERT INTO gen_metadata VALUES(2,?1)", [missing]).unwrap();
        let stats = read(&path).unwrap();
        assert_eq!(stats.tokens, Some(300));
        assert_eq!(stats.generation_tokens_per_second, None);
        drop(conn);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn native_database_counts_generators_once_and_reads_live_wal() {
        let dir = std::env::temp_dir().join(format!("vlx-agy-stats-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("store.db");
        let conn = Connection::open(&path).unwrap();
        conn.execute_batch("PRAGMA journal_mode=WAL; CREATE TABLE steps(idx INTEGER,step_type INTEGER); CREATE TABLE gen_metadata(idx INTEGER,data BLOB); INSERT INTO steps VALUES(0,14);").unwrap();
        conn.execute(
            "INSERT INTO gen_metadata VALUES(0,?1)",
            [fixture(&[1, 2, 3])],
        )
        .unwrap();
        let s = read(&path).unwrap();
        assert_eq!(s.tokens, Some(200));
        assert_eq!(s.generation_tokens_per_second, Some(50.0));
        conn.execute("INSERT INTO steps VALUES(4,14)", []).unwrap();
        assert_eq!(read(&path).unwrap().tokens, None);
        conn.execute("INSERT INTO gen_metadata VALUES(1,?1)", [fixture(&[5, 6])])
            .unwrap();
        assert_eq!(read(&path).unwrap().session_tokens, Some(2200));
        drop(conn);
        let before = std::fs::read(&path).unwrap();
        assert_eq!(
            read(&path).unwrap().generation_tokens_per_second,
            Some(50.0)
        );
        assert_eq!(std::fs::read(&path).unwrap(), before);
        std::fs::remove_dir_all(dir).unwrap();
    }
}
