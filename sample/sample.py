import json
import os
import random
from typing import Any, Dict, List, Optional
import hashlib

# Change here if your core file has a different name
import rihu

# -----------------------------
# Hard-coded project settings
# -----------------------------
PROJECT_NAME = "holmes"               # <- hard-coded project name
SOURCE_TXT   = f"{PROJECT_NAME}.txt"        # source text path
PROJECT_DIR  = f"{PROJECT_NAME}"            # output dir
UNIVERSE_JSON = os.path.join(PROJECT_DIR, "universe.json")

# -----------------------------
# tiktoken-based chunking
# -----------------------------
def chunk_text_with_tiktoken(
    text: str,
    chunk_tokens: int = 500,
    overlap_tokens: int = 50,
    encoding_name: str = "cl100k_base",
) -> List[str]:
    """
    Chunk long natural-language text into token-aware segments using tiktoken.
    Falls back to a simple paragraph splitter if tiktoken is unavailable.
    """
    try:
        import tiktoken  # type: ignore
        enc = tiktoken.get_encoding(encoding_name)
        tokens = enc.encode(text)
        chunks = []
        start = 0
        while start < len(tokens):
            end = min(start + chunk_tokens, len(tokens))
            piece = tokens[start:end]
            chunks.append(enc.decode(piece))
            if end == len(tokens):
                break
            # move start forward with overlap
            start = max(0, end - overlap_tokens)
        return [c.strip() for c in chunks if c.strip()]
    except Exception:
        # Fallback: paragraph-based chunking (approximate)
        paras = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks: List[str] = []
        buf = []
        cur_len = 0
        target_chars = 2000
        overlap_chars = 200
        for p in paras:
            if cur_len + len(p) + 2 > target_chars and buf:
                chunk = "\n\n".join(buf)
                chunks.append(chunk)
                # overlap: keep tail of previous chunk
                tail = chunk[-overlap_chars:]
                buf = [tail, p]
                cur_len = len(tail) + len(p)
            else:
                buf.append(p)
                cur_len += len(p) + 2
        if buf:
            chunks.append("\n\n".join(buf))
        return chunks

def make_chunks_from_txt(source_path: str) -> List[Dict[str, Any]]:
    """
    Load plaintext and convert to KAG-ready chunks: [{id, name, text}].
    """
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Source text not found: {source_path}")
    with open(source_path, "r", encoding="utf-8") as f:
        text = f.read()
    segments = chunk_text_with_tiktoken(text)
    chunks: List[Dict[str, Any]] = []
    for i, seg in enumerate(segments, 1):
        chunks.append({"id": f"{PROJECT_NAME}_c{i}", "name": f"{PROJECT_NAME} - Chunk {i}", "text": seg})
    return chunks

# -----------------------------
# Embedding availability & fallback
# -----------------------------
class _OfflineEmbeddingHelper(rihu._EmbeddingHelper):
    def __init__(self, client: Any, model: str, dim: int = 256):
        super().__init__(client, model)
        self.dim = dim

    def _stable_hash(self, text: str) -> int:
        h = hashlib.md5(text.encode("utf-8")).hexdigest()
        return int(h[:8], 16)  # 先頭8バイトをint化（0〜2^32程度）

    def embed_texts(self, texts: List[str]):
        import numpy as np
        if not texts:
            return np.zeros((0, 0), dtype=float)
        vecs = []
        for t in texts:
            seed = self._stable_hash(t)
            rnd = random.Random(seed)
            v = [rnd.uniform(-1, 1) for _ in range(self.dim)]
            vecs.append(v)
        return np.array(vecs, dtype=float)

def embedding_available() -> bool:
    """
    Return True if an OpenAI embedding client is likely usable.
    """
    try:
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            return False
        return rihu._HAS_OPENAI_V1 or rihu._HAS_OPENAI_LEGACY
    except Exception:
        return False

def maybe_patch_embedding_helper():
    """
    Replace _EmbeddingHelper with offline version if embeddings are not available.
    """
    if not embedding_available():
        rihu._EmbeddingHelper = _OfflineEmbeddingHelper  # type: ignore
        print("[info] Using offline pseudo-embeddings (no OpenAI API).")

# -----------------------------
# Build or load universe
# -----------------------------
def load_universe_if_exists() -> Optional[rihu.KAGUniverse]:
    """
    If {PROJECT_NAME}/universe.json exists, load and return it.
    """
    if os.path.isdir(PROJECT_DIR) and os.path.isfile(UNIVERSE_JSON):
        with open(UNIVERSE_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        uni = rihu.KAGUniverse.from_json(data)
        print(f"[ok] Reused universe from: {UNIVERSE_JSON}")
        return uni
    return None

def build_and_save_universe() -> rihu.KAGUniverse:
    """
    Build a new universe from {PROJECT_NAME}.txt and save it under {PROJECT_NAME}/universe.json.
    """
    maybe_patch_embedding_helper()

    # Initialize OpenAI-compatible client if available (per the shim)
    if rihu._HAS_OPENAI_V1:
        from openai import OpenAI
        client = OpenAI()
    else:
        client = None  # offline embeddings do not need a client

    # Read & chunk the source text
    chunks = make_chunks_from_txt(SOURCE_TXT)

    # Choose models (env override is allowed, but not required)
    llm_model = os.getenv("RIHU_LLM_MODEL", "gpt-4o-mini")
    embed_model = os.getenv("RIHU_EMBED_MODEL", "text-embedding-3-small")

    # Build the universe (LLM extraction disabled for a lightweight, API-agnostic demo)
    uni = rihu.KAGUniverse.build(
        chunks=chunks,
        openai_client=client,
        llm_model=llm_model,
        embedding_model=embed_model,
        dims=None,                         # dims will be clamped safely in build()
        iterations=4,
        use_llm_class_consolidation=False, # ditto for class consolidation
        anchors=None,                      # optionally set known class anchors
        use_llm_anchor_coords=False,
    )

    # Ensure project dir exists and save universe.json
    os.makedirs(PROJECT_DIR, exist_ok=True)
    with open(UNIVERSE_JSON, "w", encoding="utf-8") as f:
        json.dump(uni.to_json(), f, ensure_ascii=False, indent=2)
    print(f"[ok] Built universe and saved to: {UNIVERSE_JSON}")
    return uni

# -----------------------------
# Simple search demos
# -----------------------------
def demo_objective_search(uni: rihu.KAGUniverse):
    """Run an objective search centered on a couple of existing classes."""
    seed_classes = [n for n, nd in uni.classes.items() if nd.coord is not None][:2]
    if not seed_classes:
        seed_classes = list(uni.classes.keys())[:2] or ["Example"]
    res = uni.objective_search(
        class_names=seed_classes,
        k=5,
        openai_client=None,      # hybrid text similarity disabled for speed
        embedding_model=None,
    )
    print("\n[objective_search]")
    print("  seeds:", res.get("seed_classes"))
    for n in res.get("neighbors", []):
        print(f"  - {n['hf_id']}  score={n['score']:.3f}  dist={n['distance']:.3f}  text={n['text'][:80]}...  classes={n['classes']}")

def demo_subjective_search(uni: rihu.KAGUniverse):
    """Run a subjective search from one vantage class."""
    vantage = next((n for n, nd in uni.classes.items() if nd.coord is not None), None)
    vantage = vantage or (list(uni.classes.keys())[0] if uni.classes else "Example")
    res = uni.subjective_search(
        vantage=vantage,
        k=5,
        radius=1.2,
        openai_client=None,  # hybrid off
        embedding_model=None,
    )
    print("\n[subjective_search]")
    print("  vantage:", res.get("vantage"))
    print("  inside:")
    for n in res.get("inside", []):
        print(f"  - {n['hf_id']}  score={n['score']:.3f}  dist={n['distance']:.3f}  text={n['text'][:80]}...  classes={n['classes']}")
    print("  periphery:")
    for n in res.get("periphery", []):
        print(f"  - {n['hf_id']}  score={n['score']:.3f}  dist={n['distance']:.3f}  text={n['text'][:80]}...  classes={n['classes']}")

# -----------------------------
# Entry point
# -----------------------------
def main():
    # 1) Try load existing universe
    uni = load_universe_if_exists()
    # 2) Otherwise, build from {PROJECT_NAME}.txt and save
    if uni is None:
        uni = build_and_save_universe()
    # 3) Run quick search demos
    demo_objective_search(uni)
    demo_subjective_search(uni)

if __name__ == "__main__":
    main()