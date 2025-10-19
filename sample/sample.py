from __future__ import annotations

import json
import os
import re
import sys
import math
import unicodedata
import difflib
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np

# Hard-coded parameters (per requirements)
LLM_MODEL = "gpt-4o-mini"
EMBED_MODEL = "text-embedding-3-small"
ITERATIONS = 6
SUBJ_RADIUS = 1.5
TOPK = 12
CHUNK_MAX_CHARS = 800  # simple character-based chunking size
_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9\"\\(\\[])')

# Fuzzy mapping parameters
FUZZY_MIN_RATIO = 0.72  # lower this if you want more aggressive corrections

# Import prototype module
from rihu import KAGUniverse

# --- Optional OpenAI client import (new SDK preferred) ---
try:
    from openai import OpenAI  # new SDK
    CLIENT_MODE = "v1"
except Exception:  # pragma: no cover
    import openai as openai_legacy  # legacy SDK
    CLIENT_MODE = "legacy"


def fail(msg: str):
    print(f"[ERROR] {msg}")
    sys.exit(1)


# -----------------------------
# Seed class normalization & mapping
# -----------------------------

def _norm_name(s: str) -> str:
    """NFKC, lowercase, remove punctuation, collapse spaces."""
    s = unicodedata.normalize("NFKC", s or "")
    s = s.strip().lower()
    s = s.replace("-", " ")
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s

def _build_class_alias_index(uni: KAGUniverse) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    """
    Build:
      - alias_to_canonical: normalized alias/name -> canonical class name
      - canonical_to_aliases: canonical class name -> list of display aliases (incl. canonical)
    """
    alias_to_canonical: Dict[str, str] = {}
    canonical_to_aliases: Dict[str, List[str]] = {}
    for cname, node in uni.classes.items():
        if not cname:
            continue
        canonical_to_aliases.setdefault(cname, [])
        # include canonical
        canonical_to_aliases[cname].append(cname)
        alias_to_canonical[_norm_name(cname)] = cname
        # include aliases (if any)
        for a in (node.aliases or []):
            if not a:
                continue
            canonical_to_aliases[cname].append(a)
            alias_to_canonical[_norm_name(a)] = cname
    # remove duplicates in alias lists while preserving order
    for k, arr in canonical_to_aliases.items():
        seen = set()
        dedup = []
        for x in arr:
            if x not in seen:
                seen.add(x)
                dedup.append(x)
        canonical_to_aliases[k] = dedup
    return alias_to_canonical, canonical_to_aliases

def _best_fuzzy_match(
    query: str,
    alias_to_canonical: Dict[str, str],
    canonical_to_aliases: Dict[str, List[str]],
    min_ratio: float = FUZZY_MIN_RATIO,
) -> Tuple[str | None, str | None, float]:
    """
    Return (canonical_name, matched_alias, ratio) or (None, None, 0.0)
    Steps:
      1) exact normalized match
      2) fuzzy match against all known aliases/names
    """
    if not query:
        return None, None, 0.0
    qn = _norm_name(query)

    # 1) exact normalized match
    if qn in alias_to_canonical:
        can = alias_to_canonical[qn]
        # choose a display alias (prefer exact original if belongs to that canonical)
        aliases = canonical_to_aliases.get(can, [can])
        display = query if query in aliases else aliases[0]
        return can, display, 1.0

    # 2) fuzzy match: compare against ALL alias strings (de-normalized, but compare on normalized)
    all_aliases = []
    for can, aliases in canonical_to_aliases.items():
        for a in aliases:
            all_aliases.append(a)

    # difflib works on raw strings; we compare normalized forms but want original display alias
    # create parallel arrays
    norm_aliases = [_norm_name(a) for a in all_aliases]
    # compute ratios and find best
    best_idx = -1
    best_ratio = 0.0
    for i, na in enumerate(norm_aliases):
        r = difflib.SequenceMatcher(None, qn, na).ratio()
        if r > best_ratio:
            best_ratio = r
            best_idx = i

    if best_idx >= 0 and best_ratio >= min_ratio:
        matched_alias = all_aliases[best_idx]
        can = alias_to_canonical.get(_norm_name(matched_alias))
        return can, matched_alias, best_ratio

    return None, None, 0.0

def map_seed_classes(uni: KAGUniverse, seeds: List[str], verbose: bool = True) -> List[str]:
    """
    Map user-provided seed class strings to the closest existing canonical Class names in the universe.
    - Handles typos and variants via normalization + fuzzy matching.
    - Logs mapping results; falls back to original if no good match was found.
    """
    alias_to_canonical, canonical_to_aliases = _build_class_alias_index(uni)
    out: List[str] = []
    print("\n[Normalize] Mapping seed classes (typo/variant correction):")
    for s in seeds:
        can, alias, score = _best_fuzzy_match(s, alias_to_canonical, canonical_to_aliases, FUZZY_MIN_RATIO)
        if can:
            if verbose:
                if _norm_name(s) == _norm_name(alias):
                    print(f"  - '{s}' → {can}  (exact/alias match)")
                else:
                    print(f"  - '{s}' → {can}  (fuzzy: '{alias}', score={score:.3f})")
            out.append(can)
        else:
            # keep original if no match (and warn)
            print(f"  - '{s}' → (no close match; using original)")
            out.append(s)
    return out


# -----------------------------
# Universe I/O and building
# -----------------------------

def load_or_build_universe(uname: str) -> KAGUniverse:
    """Load universe from <uname>/universe.json if present; otherwise build from <uname>.txt."""
    apikey = os.getenv("OPENAI_API_KEY")
    if not apikey:
        fail("OPENAI_API_KEY is not set. Please `set OPENAI_API_KEY=...` (Windows) or `export OPENAI_API_KEY=...`.")

    udir = Path(uname)
    usrc = Path(f"{uname}.txt")
    ujson = udir / "universe.json"

    # Prepare OpenAI client
    if CLIENT_MODE == "v1":
        client = OpenAI(api_key=apikey)
    else:  # legacy
        openai_legacy.api_key = apikey
        client = openai_legacy  # type: ignore

    if udir.exists() and ujson.exists():
        print(f"[RIHU] Reusing existing universe: {ujson}")
        data = json.loads(ujson.read_text(encoding="utf-8"))
        return KAGUniverse.from_json(data)

    # Build path and ensure source exists
    if not usrc.exists():
        fail(f"Source file not found: {usrc}")

    print(f"[RIHU] Building new universe from {usrc} -> {udir}/")
    udir.mkdir(parents=True, exist_ok=True)

    # 1) Chunking (simple, local)
    text = usrc.read_text(encoding="utf-8")
    #chunks = simple_chunk_text(text, max_chars=CHUNK_MAX_CHARS)
    chunks = chunk_text_tokenwise(text, max_tokens=700, overlap_tokens=60)

    # 2) Build universe
    uni = KAGUniverse.build(
        chunks=chunks,
        openai_client=client,
        llm_model=LLM_MODEL,
        embedding_model=EMBED_MODEL,
        iterations=ITERATIONS,
        use_llm_hf_extraction=True,
        use_llm_class_consolidation=True,
    )

    # 3) Save
    ujson.write_text(json.dumps(uni.to_json(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[RIHU] Saved universe -> {ujson}")
    # also persist axes metadata for inspection
    (udir / "axes.json").write_text(
        json.dumps({"axes": [ax.__dict__ for ax in (uni.meta.axes if uni.meta else [])]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return uni


# -----------------------------
# Chunking utilities
# -----------------------------

def simple_chunk_text(text: str, max_chars: int = 800) -> List[Dict[str, Any]]:
    """Very simple chunker: split by paragraph, then pack to length <= max_chars."""
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[Dict[str, Any]] = []
    buf = []
    cur = 0
    cid = 1
    for p in paras:
        if cur + len(p) + (2 if buf else 0) <= max_chars:
            buf.append(p)
            cur += len(p) + (2 if buf else 0)
        else:
            if buf:
                chunks.append({"id": f"ch{cid}", "text": "\n\n".join(buf)})
                cid += 1
            buf = [p]
            cur = len(p)
    if buf:
        chunks.append({"id": f"ch{cid}", "text": "\n\n".join(buf)})
    return chunks

def _get_encoding():
    try:
        import tiktoken  # pip install tiktoken
        # For GPT-4o/4.1 series use o200k_base, for GPT-3 series use cl100k_base
        try:
            return tiktoken.get_encoding("o200k_base")
        except Exception:
            return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None  # fallback

def _count_tokens(text: str, enc=None) -> int:
    if enc is None:
        enc = _get_encoding()
    if enc is None:
        # Fallback: roughly 1 token ≈ 4 chars
        return max(1, len(text) // 4)
    return len(enc.encode(text))


def _split_into_sentences(text: str) -> list[str]:
    # Roughly tuned for English; adjust for Japanese/other languages if needed
    parts = _SENT_SPLIT.split(text.strip())
    # If a sentence is abnormally long, further split by punctuation
    out = []
    for p in parts:
        p = p.strip()
        if not p: continue
        if len(p) > 2000:
            out.extend(re.split(r'(?<=[,;:])\\s+', p))
        else:
            out.append(p)
    return out


def chunk_text_tokenwise(
    text: str,
    max_tokens: int = 700,       # Max tokens per chunk (after reserving prompt space)
    overlap_tokens: int = 60,    # Overlap tokens with next chunk
    min_sent_tokens: int = 5,    # Very short sentences below this will try to merge with neighbors
) -> list[dict]:
    """
    Pack sentences while respecting boundaries; cut near the token limit.
    Optionally add overlapping context (overlap).
    Returns: [{'id': 'ch1', 'text': '...'}, ...]
    """
    enc = _get_encoding()
    sentences = _split_into_sentences(text)
    chunks = []
    buf: list[str] = []
    buf_tokens = 0
    cid = 1

    def flush():
        nonlocal buf, buf_tokens, cid
        if not buf: return
        chunk_text = " ".join(buf).strip()
        if chunk_text:
            chunks.append({"id": f"ch{cid}", "text": chunk_text})
            cid += 1
        buf, buf_tokens = [], 0

    i = 0
    while i < len(sentences):
        s = sentences[i]
        st = _count_tokens(s, enc)
        # If a single sentence is extremely long, force a split
        if st > max_tokens:
            # Roughly split by punctuation or spaces
            pieces = re.split(r'(?:\\s{2,}|;|:|,)', s)
            for piece in pieces:
                piece = piece.strip()
                if not piece: continue
                pt = _count_tokens(piece, enc)
                if buf_tokens + pt <= max_tokens:
                    buf.append(piece)
                    buf_tokens += pt
                else:
                    flush()
                    # Still too large to fit — forcibly include it
                    if pt > max_tokens:
                        # If it's very long, roughly split by token estimate (in half for safety)
                        half = max(1, len(piece)//2)
                        part1, part2 = piece[:half], piece[half:]
                        for sub in (part1, part2):
                            if _count_tokens(sub, enc) > max_tokens:
                                sub = sub[: len(sub)//2]
                            chunks.append({"id": f"ch{cid}", "text": sub})
                            cid += 1
                        buf, buf_tokens = [], 0
                    else:
                        buf = [piece]; buf_tokens = pt
            i += 1
            continue

        # Normal sentence packing
        if buf_tokens + st <= max_tokens:
            buf.append(s); buf_tokens += st
            i += 1
        else:
            # Allow overflow if only a very short sentence remains
            if st < min_sent_tokens and not buf:
                buf.append(s); buf_tokens += st; i += 1
            flush()

            # Add overlap (copy tail from previous chunk)
            if overlap_tokens > 0 and chunks:
                overlap_buf = []
                t = 0
                # Take sentences from the end of the previous chunk
                for prev_s in reversed(_split_into_sentences(chunks[-1]["text"])):
                    tt = _count_tokens(prev_s, enc)
                    if t + tt > overlap_tokens: break
                    overlap_buf.append(prev_s); t += tt
                overlap_buf.reverse()
                if overlap_buf:
                    buf = overlap_buf[:]
                    buf_tokens = sum(_count_tokens(x, enc) for x in buf)
            # Next loop will push the next sentence

    flush()
    return chunks


# -----------------------------
# Geometry helpers
# -----------------------------

def nearest_class_to_point(uni: KAGUniverse, point: List[float]) -> str | None:
    best = None
    best_d = float("inf")
    for name, node in uni.classes.items():
        if node.coord is None:
            continue
        d = float(np.linalg.norm(np.array(node.coord) - np.array(point)))
        if d < best_d:
            best_d = d
            best = name
    return best


def centroid_of_hfs(uni: KAGUniverse, hf_ids: List[str]) -> List[float] | None:
    coords = []
    for hid in hf_ids:
        hf = uni.hfs.get(hid)
        if hf and hf.coord is not None:
            coords.append(hf.coord)
    if not coords:
        return None
    arr = np.array(coords, dtype=float)
    return arr.mean(axis=0).tolist()


# -----------------------------
# Search Orchestration
# -----------------------------

def run_searches(uname: str, classes: List[str]):
    uni = load_or_build_universe(uname)

    # --- NEW: normalize & map seed classes to canonical names in the universe ---
    mapped_classes = map_seed_classes(uni, classes, verbose=True)

    # (A) Objective search from user-specified (mapped) Classes
    print("\n[Search] Objective from seed classes (mapped):", mapped_classes)
    res_obj = uni.objective_search(class_names=mapped_classes, k=TOPK)

    # Derive centroid from the returned neighbors' HF coordinates (if any)
    hf_ids = [r["hf_id"] for r in res_obj.get("neighbors", [])]
    hf_centroid = centroid_of_hfs(uni, hf_ids)

    # Subjective search from the nearest class to that centroid
    subj_from = None
    res_sub_from_centroid = None
    if hf_centroid is not None:
        subj_from = nearest_class_to_point(uni, hf_centroid)
        if subj_from is not None:
            print(f"[Search] Subjective from centroid → nearest class: {subj_from}")
            res_sub_from_centroid = uni.subjective_search(vantage=subj_from, k=TOPK, radius=SUBJ_RADIUS)
        else:
            print("[Search] No suitable vantage class found near centroid.")
    else:
        print("[Search] No HF centroid computed (no neighbors).")

    # (B) Subjective search for each mapped Class
    res_sub_each: Dict[str, Any] = {}
    for c in mapped_classes:
        print(f"[Search] Subjective per class: {c}")
        res_sub_each[c] = uni.subjective_search(vantage=c, k=TOPK, radius=SUBJ_RADIUS)

    # Save results
    out_dir = Path(uname)
    results = {
        "seed_classes_input": classes,
        "seed_classes_mapped": mapped_classes,
        "objective": res_obj,
        "subjective_from_centroid_vantage": subj_from,
        "subjective_from_centroid": res_sub_from_centroid,
        "subjective_each": res_sub_each,
    }
    (out_dir / "searches.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[RIHU] Saved search results -> {out_dir / 'searches.json'}")

    # Pretty-print a brief digest to stdout
    dump_digest(res_obj, res_sub_from_centroid, res_sub_each)


def dump_digest(res_obj: Dict[str, Any], res_sub_from_centroid: Dict[str, Any] | None, res_sub_each: Dict[str, Any]):
    def hf_lines(neis: List[Dict[str, Any]], n=5):
        lines = []
        for r in neis[:n]:
            # Depending on your RIHU build, subjective may return 'neighbors' or 'inside'/'periphery'.
            d = r.get('distance', 0.0)
            lines.append(f"  - HF[{r['hf_id']}]: d={d:.3f} classes={', '.join(r['classes'][:4])} | {r['text'][:120]}...")
        return "\n".join(lines) if lines else "  (none)"

    print("\n===== Objective (top neighbors) =====")
    print(hf_lines(res_obj.get("neighbors", []), n=8))

    print("\n===== Subjective from centroid (nearest class) =====")
    if res_sub_from_centroid:
        # compatibility: prefer 'neighbors'; fallback to 'inside'
        neis = res_sub_from_centroid.get("neighbors") or res_sub_from_centroid.get("inside", [])
        print(hf_lines(neis, n=8))
        ctx = res_sub_from_centroid.get("context_classes", [])[:8]
        for c in ctx:
            r = c.get('radius', 0.0) or 0.0
            p = c.get('power', 0.0) or 0.0
            print(f"  * Ctx: {c['class']} (r={r:.3f}, power={p:.3g})")
    else:
        print("  (none)")

    print("\n===== Subjective per seed class =====")
    for k, v in res_sub_each.items():
        print(f"- {k}")
        neis = v.get("neighbors") or v.get("inside", [])
        print(hf_lines(neis, n=6))


if __name__ == "__main__":
    universe_name = "holmes"
    seed_classes = ["holmes", "adler"]
    run_searches(universe_name, seed_classes)