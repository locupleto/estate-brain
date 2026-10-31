"""Runs ON THE HOST that holds the transcript (piped over ssh, stdlib only).

argv: <transcript.jsonl>. Prints one JSON object: the first user prompt, the
final reply (last assistant text, or the last StructuredOutput input), the
tool calls with their outcome, the model, token usage (deduped per
message.id) and the cwd the turn ran in.
"""

import json
import sys


def _text(content):
    if isinstance(content, str):
        return content
    out = []
    for b in content or []:
        if isinstance(b, dict) and b.get("type") == "text":
            out.append(b.get("text") or "")
    return "\n".join(out)


def main(path):
    prompt, cwd, model, recorded_at = None, None, None, None
    final_text, structured = "", None
    calls, by_id, usage = [], {}, {}
    with open(path) as f:
        for line in f:
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            cwd = cwd or ev.get("cwd")
            recorded_at = recorded_at or ev.get("timestamp")
            kind = ev.get("type")
            msg = ev.get("message") or {}
            if kind == "user":
                content = msg.get("content")
                if prompt is None and not (isinstance(content, list) and any(
                        isinstance(b, dict) and b.get("type") == "tool_result" for b in content)):
                    prompt = _text(content)
                    continue
                for b in content if isinstance(content, list) else []:
                    if isinstance(b, dict) and b.get("type") == "tool_result":
                        c = by_id.get(b.get("tool_use_id"))
                        if c is not None:
                            c["ok"] = not b.get("is_error")
            elif kind == "assistant":
                model = model or msg.get("model")
                if msg.get("id") and msg.get("usage"):
                    usage[msg["id"]] = msg["usage"]
                for b in msg.get("content") or []:
                    if b.get("type") == "text" and (b.get("text") or "").strip():
                        final_text = b["text"]
                    elif b.get("type") == "tool_use":
                        if b.get("name") == "StructuredOutput":
                            structured = b.get("input")
                            continue
                        inp = b.get("input") or {}
                        c = {"name": b.get("name"), "ok": None,
                             "input": {k: (v[:200] if isinstance(v, str) else v)
                                       for k, v in inp.items() if k in (
                                           "file_path", "path", "pattern", "query", "url",
                                           "command", "description", "subagent_type")}}
                        by_id[b.get("id")] = c
                        calls.append(c)
    tok = {"input": 0, "output": 0}
    for u in usage.values():
        tok["input"] += (u.get("input_tokens") or 0) + (u.get("cache_read_input_tokens") or 0) \
            + (u.get("cache_creation_input_tokens") or 0)
        tok["output"] += u.get("output_tokens") or 0
    print(json.dumps({"prompt": prompt or "", "cwd": cwd, "model": model,
                      "recorded_at": recorded_at,
                      "reference_text": final_text, "reference_structured": structured,
                      "tool_calls": calls, "tokens": tok}))


if __name__ == "__main__":
    main(sys.argv[1])
