from __future__ import annotations

import json
import math
import os
import re
import unicodedata
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ------------------------------
# Fixed bounds for LLM-chosen axes
# ------------------------------
MIN_AXES = 3
MAX_AXES = 10

# --- OpenAI client compatibility shim ---------------------------------------
try:
    # New-style SDK (>=1.0)
    from openai import OpenAI  # type: ignore
    _HAS_OPENAI_V1 = True
except Exception:  # pragma: no cover - runtime import guard
    OpenAI = None  # type: ignore
    _HAS_OPENAI_V1 = False

try:
    # Legacy SDK (<=0.28)
    import openai as _openai_legacy  # type: ignore
    _HAS_OPENAI_LEGACY = True
except Exception:  # pragma: no cover - runtime import guard
    _HAS_OPENAI_LEGACY = False


# =============================================================================
# Data structures (KAG primitives)
# =============================================================================

@dataclass
class HF:
    """Hypothetical Fact: minimal proposition inferred from chunk(s)."""
    id: str
    text: str
    weight: float = 1.0
    class_names: List[str] = field(default_factory=list)  # names of Classes participating
    coord: Optional[List[float]] = None  # filled after placement
    source_ids: List[str] = field(default_factory=list)  # chunk ids or URIs


@dataclass
class Instance:
    """An occurrence of a Class inside an HF."""
    id: str
    class_name: str
    hf_id: str
    coord: Optional[List[float]] = None


@dataclass
class ClassNode:
    name: str
    aliases: List[str] = field(default_factory=list)
    instances: List[str] = field(default_factory=list)  # Instance ids
    coord: Optional[List[float]] = None
    radius: Optional[float] = None  # influence range (approx)
    volume: Optional[float] = None
    power: Optional[float] = None
    is_anchor: bool = False


@dataclass
class AxisSpec:
    name: str
    description: str


@dataclass
class KAGMeta:
    axes: List[AxisSpec]
    dims: int
    embedding_model: str
    llm_model: str
    notes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KAGUniverse:
    """Geometric manifold of meaning: Classes, HFs, Instances, and metrics."""
    classes: Dict[str, ClassNode] = field(default_factory=dict)
    hfs: Dict[str, HF] = field(default_factory=dict)
    instances: Dict[str, Instance] = field(default_factory=dict)
    meta: Optional[KAGMeta] = None

    # ------------------------------
    # Construction / update pipeline
    # ------------------------------

    @staticmethod
    def from_json(data: Dict[str, Any]) -> "KAGUniverse":
        cls_map = {k: ClassNode(**v) for k, v in data["classes"].items()}
        hf_map = {k: HF(**v) for k, v in data["hfs"].items()}
        inst_map = {k: Instance(**v) for k, v in data["instances"].items()}
        meta = KAGMeta(
            axes=[AxisSpec(**ax) for ax in data["meta"]["axes"]],
            dims=data["meta"]["dims"],
            embedding_model=data["meta"]["embedding_model"],
            llm_model=data["meta"]["llm_model"],
            notes=data["meta"].get("notes", {}),
        )
        return KAGUniverse(classes=cls_map, hfs=hf_map, instances=inst_map, meta=meta)

    def to_json(self) -> Dict[str, Any]:
        return {
            "classes": {k: asdict(v) for k, v in self.classes.items()},
            "hfs": {k: asdict(v) for k, v in self.hfs.items()},
            "instances": {k: asdict(v) for k, v in self.instances.items()},
            "meta": {
                "axes": [asdict(ax) for ax in (self.meta.axes if self.meta else [])],
                "dims": self.meta.dims if self.meta else 0,
                "embedding_model": self.meta.embedding_model if self.meta else "",
                "llm_model": self.meta.llm_model if self.meta else "",
                "notes": self.meta.notes if self.meta else {},
            },
        }

    # ------------------------------
    # Public API (high-level)
    # ------------------------------

    @classmethod
    def build(
        cls,
        *,
        chunks: List[Dict[str, Any]],
        openai_client: Any,
        llm_model: str,
        embedding_model: str,
        # If dims is None, we set dims to the number of axes returned by the LLM (clamped to [MIN_AXES, MAX_AXES]).
        dims: Optional[int] = None,
        iterations: int = 6,
        alpha: float = 0.35,
        epsilon: float = 1e-4,
        use_llm_hf_extraction: bool = True,
        use_llm_class_consolidation: bool = True,
        anchors: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        # Optional (Improvement #2)
        use_llm_anchor_coords: bool = False,
        anchor_target_range: Tuple[float, float] = (-1.0, 1.0),
    ) -> "KAGUniverse":
        """Construct a KAG Universe from prepared chunks with bounded, LLM-chosen axis count."""
        if not chunks:
            raise ValueError("`chunks` cannot be empty.")

        # 1) Axis proposal: LLM chooses K in [MIN_AXES, MAX_AXES]
        axes = _propose_axes_dynamic_k(
            client=openai_client,
            llm_model=llm_model,
            sample_texts=[c.get("text", "") for c in chunks[: min(16, len(chunks))]],
            system_prompt=system_prompt,
            min_axes=MIN_AXES,
            max_axes=MAX_AXES,
        )
        axes_count = len(axes)
        # Guard against pathological responses
        if axes_count < MIN_AXES:
            axes += [AxisSpec(name=f"Axis-{i+1}", description="Ad hoc latent dimension.")
                     for i in range(MIN_AXES - axes_count)]
            axes_count = MIN_AXES
        elif axes_count > MAX_AXES:
            axes = axes[:MAX_AXES]
            axes_count = MAX_AXES

        # Decide the working dimensionality
        selected_dims = int(dims) if dims is not None else int(axes_count)

        # 2) HF & Class extraction (LLM-based, fallback heuristic)
        hfs, class_map, instances = _extract_hfs_and_classes(
            client=openai_client,
            llm_model=llm_model,
            chunks=chunks,
            use_llm=use_llm_hf_extraction,
            system_prompt=system_prompt,
        )

        # 3) Mechanical class merge (returns merged classes + mapping)
        class_map, mech_mapping = _mechanical_class_merge(class_map)
        _apply_class_mapping_to_refs(mapping=mech_mapping, hfs=hfs, instances=instances, classes=class_map)

        # 4) Optional LLM consolidation over the full Class/HF list
        llm_mapping: Dict[str, str] = {}
        if use_llm_class_consolidation:
            class_map, llm_mapping = _llm_class_consolidation(
                client=openai_client,
                llm_model=llm_model,
                classes=class_map,
                hfs=hfs,
                system_prompt=system_prompt,
            )
            _apply_class_mapping_to_refs(mapping=llm_mapping, hfs=hfs, instances=instances, classes=class_map)

        # 5) Initial coordinates via embeddings + SVD projection
        embedder = _EmbeddingHelper(openai_client, embedding_model)
        class_names_sorted = sorted(class_map.keys())
        class_texts = class_names_sorted
        hf_ids_sorted = sorted(hfs.keys())
        hf_texts = [hfs[h].text for h in hf_ids_sorted]

        class_emb = embedder.embed_texts(class_texts)
        hf_emb = embedder.embed_texts(hf_texts)

        all_emb = np.vstack([class_emb, hf_emb])
        proj = _svd_project(all_emb, dims=selected_dims)
        class_coords = proj[: len(class_emb), :]
        hf_coords = proj[len(class_emb) :, :]

        for name, coord in zip(class_names_sorted, class_coords.tolist()):
            node = class_map[name]
            node.coord = coord
            if anchors and (name in anchors):
                node.is_anchor = True
        for hf_id, coord in zip(hf_ids_sorted, hf_coords.tolist()):
            hfs[hf_id].coord = coord

        # 5.5) Optional: LLM-proposed anchor targets + one-shot affine alignment (Improvement #2)
        if use_llm_anchor_coords:
            target = _propose_anchor_coords(
                client=openai_client,
                llm_model=llm_model,
                anchor_names=[n for n, nd in class_map.items() if nd.is_anchor],
                dims=selected_dims,
                value_range=anchor_target_range,
                sample_texts=[c.get("text", "") for c in chunks[:8]],
                system_prompt=system_prompt,
            )
            # Use only anchors present on both sides
            names = [n for n in class_map if class_map[n].is_anchor and class_map[n].coord is not None and n in target]
            if len(names) >= max(2, selected_dims):
                X = np.array([class_map[n].coord for n in names], dtype=float)           # current
                Y = np.array([target[n][:selected_dims] for n in names], dtype=float)    # desired
                A, b = _affine_align(X, Y, l2=1e-3)                                      # find linear map
                # Apply to all nodes/HFs once
                for node in class_map.values():
                    if node.coord is not None:
                        node.coord = (np.array(node.coord) @ A + b).tolist()
                for hf in hfs.values():
                    if hf.coord is not None:
                        hf.coord = (np.array(hf.coord) @ A + b).tolist()

        # 6) Iterative relaxation
        max_move = float("inf")
        it = 0
        while it < iterations and max_move > epsilon:
            max_move = 0.0

            # HFs: centroid of participating classes (safe lookup)
            for hf in hfs.values():
                if not hf.class_names:
                    continue
                cls_coords: List[List[float]] = []
                for c in hf.class_names:
                    node = class_map.get(c)
                    if node and node.coord is not None:
                        cls_coords.append(node.coord)
                if not cls_coords:
                    continue
                arr = np.array(cls_coords, dtype=float)
                centroid = arr.mean(axis=0)
                if hf.coord is None:
                    hf.coord = centroid.tolist()
                else:
                    prev = np.array(hf.coord, dtype=float)
                    new = centroid
                    moved = float(np.linalg.norm(prev - new))
                    hf.coord = new.tolist()
                    max_move = max(max_move, moved)

            # Classes: move toward centroid of their instances' HFs (anchors don't move)
            for cls_name, node in class_map.items():
                if node.is_anchor:
                    continue  # anchored: don't move in the relaxation step
                inst_coords = []
                for inst in _instances_of_class(instances, cls_name):
                    hf_node = hfs.get(inst.hf_id)
                    if hf_node and hf_node.coord is not None:
                        inst_coords.append(hf_node.coord)
                if not inst_coords:
                    continue
                tgt = np.array(inst_coords, dtype=float).mean(axis=0)
                if node.coord is None:
                    node.coord = tgt.tolist()
                else:
                    prev = np.array(node.coord, dtype=float)
                    new = (1 - alpha) * prev + alpha * tgt
                    moved = float(np.linalg.norm(prev - new))
                    node.coord = new.tolist()
                    max_move = max(max_move, moved)

            # Normalize global scale (apply to all nodes), preferring anchors (Improvement #1)
            anchor_coords = np.array(
                [n.coord for n in class_map.values() if n.is_anchor and n.coord is not None],
                dtype=float,
            )
            if len(anchor_coords) >= 2:
                mean_dist = _mean_pairwise_distance(anchor_coords)
            else:
                coords = np.array([c.coord for c in class_map.values() if c.coord is not None], dtype=float)
                mean_dist = _mean_pairwise_distance(coords)

            if mean_dist > 0:
                scale = 1.0 / mean_dist
                for node in class_map.values():
                    if node.coord is not None:
                        node.coord = (np.array(node.coord) * scale).tolist()
                for hf in hfs.values():
                    if hf.coord is not None:
                        hf.coord = (np.array(hf.coord) * scale).tolist()

            it += 1

        # 7) Per-class metrics (anchors are not overwritten by centroid)
        for cls_name, node in class_map.items():
            inst_hf_coords = []
            for i in _instances_of_class(instances, cls_name):
                hf_node = hfs.get(i.hf_id)
                if hf_node and hf_node.coord is not None:
                    inst_hf_coords.append(hf_node.coord)
            if not inst_hf_coords:
                continue
            arr = np.array(inst_hf_coords, dtype=float)
            centroid = arr.mean(axis=0)
            if not node.is_anchor:
                node.coord = centroid.tolist()
            center = np.array(node.coord if node.coord is not None else centroid, dtype=float)
            dists = np.linalg.norm(arr - center, axis=1)
            r = float(dists.max()) if len(dists) > 0 else 0.0
            node.radius = r
            node.volume = _ball_volume(r, selected_dims)
            node.power = (len(inst_hf_coords) / max(node.volume, 1e-9))

        notes = {
            "iterations": it,
            "alpha": alpha,
            "mechanical_mapping": mech_mapping,
            "llm_mapping": llm_mapping,
            "axes_count": axes_count,
        }
        meta = KAGMeta(
            axes=axes,
            dims=selected_dims,
            embedding_model=embedding_model,
            llm_model=llm_model,
            notes=notes,
        )

        return KAGUniverse(classes=class_map, hfs=hfs, instances=instances, meta=meta)

    # ------------------------------
    # Incremental update
    # ------------------------------

    def extend_and_rebuild(
        self,
        *,
        new_chunks: List[Dict[str, Any]],
        openai_client: Any,
        llm_model: str,
        embedding_model: Optional[str] = None,
        dims: Optional[int] = None,
        **build_kwargs: Any,
    ) -> "KAGUniverse":
        """Extend universe with additional data and rebuild (keeping axis metadata)."""
        embedding_model = embedding_model or (self.meta.embedding_model if self.meta else None)
        #dims = dims or (self.meta.dims if self.meta else MIN_AXES)
        if not embedding_model:
            raise ValueError("embedding_model not provided and not available in meta")
        if not self.meta:
            raise ValueError("Universe meta missing; cannot reuse axis metadata")

        # Build new from combined data using previous axes as anchors (names only)
        all_chunks = _chunks_from_universe(self) + new_chunks
        anchors = [ax.name for ax in self.meta.axes]
        new_u = KAGUniverse.build(
            chunks=all_chunks,
            openai_client=openai_client,
            llm_model=llm_model,
            embedding_model=embedding_model,
            dims=dims,
            anchors=anchors,
            **build_kwargs,
        )
        return new_u

    # ------------------------------
    # Retrieval APIs
    # ------------------------------

    def objective_search(
        self,
        *,
        class_names: List[str],
        k: int = 10,
        radius: Optional[float] = None,
        # hybrid/density weights (Improvement #3/#6)
        weight_distance: float = 0.7,
        weight_density: float = 0.2,
        weight_similarity: float = 0.1,
        # optional hybrid similarity
        openai_client: Any = None,
        embedding_model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Objective retrieval: centroid of specified classes, then nearest HFs with density/similarity scoring."""
        if not class_names:
            raise ValueError("class_names cannot be empty")
        valid = [self.classes[c].coord for c in class_names if c in self.classes and self.classes[c].coord is not None]
        if not valid:
            return {"mode": "objective", "centroid": None, "neighbors": [], "note": "No valid classes with coordinates."}
        centroid = np.array(valid, dtype=float).mean(axis=0)

        # Candidate HFs: prioritize those sharing at least one class with seeds
        seed_set = set(class_names)
        candidates = [hf for hf in self.hfs.values() if hf.coord is not None and (set(hf.class_names) & seed_set)]
        if not candidates:
            candidates = [hf for hf in self.hfs.values() if hf.coord is not None]

        # Precompute features
        dists: List[float] = []
        dens: List[float] = []  # mean power of classes participating in HF
        for hf in candidates:
            d = float(np.linalg.norm(np.array(hf.coord) - centroid))
            dists.append(d)
            if hf.class_names:
                powers = [self.classes[c].power for c in hf.class_names if c in self.classes and self.classes[c].power is not None]
                dens.append(float(np.mean(powers)) if powers else 0.0)
            else:
                dens.append(0.0)

        # Optional hybrid: cosine similarity between HF text and seed query text
        sims: List[float] = [0.0] * len(candidates)
        if openai_client is not None and embedding_model:
            embedder = _EmbeddingHelper(openai_client, embedding_model)
            query_text = ", ".join(class_names)  # simple seed query text
            qv = embedder.embed_texts([query_text])[0]
            hf_vecs = embedder.embed_texts([hf.text for hf in candidates])
            sims = _cosine_sim_matrix(hf_vecs, qv.reshape(1, -1)).ravel().tolist()

        # Normalize features to [0,1] and compute convex combination
        d_scores = _invert_and_normalize(dists)          # smaller distance => higher score
        den_scores = _normalize(dens)
        sim_scores = _normalize(sims)

        scored = []
        for idx, hf in enumerate(candidates):
            score = (weight_distance * d_scores[idx]
                     + weight_density * den_scores[idx]
                     + weight_similarity * sim_scores[idx])
            if radius is not None and dists[idx] > radius:
                continue
            scored.append((score, dists[idx], hf))
        scored.sort(key=lambda x: x[0], reverse=True)

        out = []
        for score, d, hf in scored[:k]:
            out.append({
                "hf_id": hf.id,
                "score": score,
                "distance": d,
                "text": hf.text,
                "classes": hf.class_names,
                "sources": hf.source_ids,
            })
        return {
            "mode": "objective",
            "centroid": centroid.tolist(),
            "neighbors": out,
            "seed_classes": class_names,
            "weights": {"distance": weight_distance, "density": weight_density, "similarity": weight_similarity},
        }

    def subjective_search(
        self,
        *,
        vantage: str,
        k: int = 10,
        radius: float = 1.5,
        periphery_radius: Optional[float] = None,
        # optional hybrid sim weight (we keep the interface lean)
        openai_client: Any = None,
        embedding_model: Optional[str] = None,
        weight_distance: float = 0.8,
        weight_similarity: float = 0.2,
    ) -> Dict[str, Any]:
        """Subjective retrieval from an observer vantage (a Class name), with periphery (Improvement #4)."""
        if vantage not in self.classes or self.classes[vantage].coord is None:
            return {"mode": "subjective", "vantage": vantage, "inside": [], "periphery": [], "context_classes": []}
        v = np.array(self.classes[vantage].coord, dtype=float)
        periphery_radius = periphery_radius or (radius * 1.5)

        # Optional hybrid similarity to prioritize closer-about texts
        sims: Dict[str, float] = {}
        if openai_client is not None and embedding_model:
            embedder = _EmbeddingHelper(openai_client, embedding_model)
            qv = embedder.embed_texts([vantage])[0]
            hf_list = [hf for hf in self.hfs.values() if hf.coord is not None]
            hf_vecs = embedder.embed_texts([hf.text for hf in hf_list])
            sim_vals = _cosine_sim_matrix(hf_vecs, qv.reshape(1, -1)).ravel().tolist()
            sims = {hf_list[i].id: sim_vals[i] for i in range(len(hf_list))}

        inside_scored = []
        periph_scored = []
        for hf in self.hfs.values():
            if hf.coord is None:
                continue
            d = float(np.linalg.norm(np.array(hf.coord) - v))
            sim = sims.get(hf.id, 0.0)
            s = weight_distance * _inv_dist_score(d) + weight_similarity * sim
            rec = (s, d, hf)
            if d <= radius:
                inside_scored.append(rec)
            elif d <= periphery_radius:
                periph_scored.append(rec)

        inside_scored.sort(key=lambda x: x[0], reverse=True)
        periph_scored.sort(key=lambda x: x[0], reverse=True)

        inside_neighbors = [{
            "hf_id": hf.id,
            "score": s,
            "distance": d,
            "text": hf.text,
            "classes": hf.class_names,
            "sources": hf.source_ids,
        } for (s, d, hf) in inside_scored[:k]]

        periphery_neighbors = [{
            "hf_id": hf.id,
            "score": s,
            "distance": d,
            "text": hf.text,
            "classes": hf.class_names,
            "sources": hf.source_ids,
        } for (s, d, hf) in periph_scored[:k]]

        # Context classes: containment of vantage within other classes
        ctx = []
        for name, node in self.classes.items():
            if node.coord is None or node.radius is None:
                continue
            d = float(np.linalg.norm(np.array(node.coord) - v))
            if d <= (node.radius + 1e-9):
                ctx.append({
                    "class": name,
                    "distance_to_center": d,
                    "radius": node.radius,
                    "power": node.power,
                })
        ctx.sort(key=lambda x: (x["distance_to_center"]))

        return {
            "mode": "subjective",
            "vantage": vantage,
            "inside": inside_neighbors,
            "periphery": periphery_neighbors,
            "context_classes": ctx,
        }

    # ------------------------------
    # Containment / relations (Improvement #5)
    # ------------------------------

    def class_contains_hf(self, class_name: str, hf_id: str, tol: float = 1e-9) -> bool:
        """Return True if HF lies within the class influence region (center + radius)."""
        node = self.classes.get(class_name)
        hf = self.hfs.get(hf_id)
        if not node or node.coord is None or node.radius is None or not hf or hf.coord is None:
            return False
        d = float(np.linalg.norm(np.array(node.coord) - np.array(hf.coord)))
        return d <= (node.radius + tol)

    def class_contains_class(self, outer: str, inner: str, tol: float = 1e-9) -> bool:
        """Return True if Class 'inner' lies entirely within 'outer' (center distance + radius test)."""
        A = self.classes.get(outer)
        B = self.classes.get(inner)
        if not A or not B or A.coord is None or B.coord is None or A.radius is None or B.radius is None:
            return False
        d = float(np.linalg.norm(np.array(A.coord) - np.array(B.coord)))
        return d + (B.radius or 0.0) <= (A.radius or 0.0) + tol

    def containment_graph(self, classes: Optional[List[str]] = None, tol: float = 1e-9) -> Dict[str, List[str]]:
        """Return adjacency list where edges A->B mean 'B is contained in A'."""
        names = classes or list(self.classes.keys())
        names = [n for n in names if self.classes.get(n) and self.classes[n].coord is not None and self.classes[n].radius is not None]
        graph: Dict[str, List[str]] = {n: [] for n in names}
        for i, A in enumerate(names):
            for j, B in enumerate(names):
                if i == j:
                    continue
                if self.class_contains_class(A, B, tol=tol):
                    graph[A].append(B)
        return graph


# =============================================================================
# Helpers — construction
# =============================================================================

class _EmbeddingHelper:
    def __init__(self, client: Any, model: str):
        self.client = client
        self.model = model

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=float)
        if _HAS_OPENAI_V1 and isinstance(self.client, OpenAI):
            res = self.client.embeddings.create(model=self.model, input=texts)
            vecs = [d.embedding for d in res.data]
            return np.array(vecs, dtype=float)
        elif _HAS_OPENAI_LEGACY:
            _openai_legacy.api_key = getattr(self.client, "api_key", os.getenv("OPENAI_API_KEY"))
            res = _openai_legacy.Embedding.create(model=self.model, input=texts)
            vecs = [d["embedding"] for d in res["data"]]
            return np.array(vecs, dtype=float)
        else:  # pragma: no cover
            raise RuntimeError("OpenAI client not available. Install/openai and pass a client instance.")


def _propose_axes_dynamic_k(
    *, client: Any, llm_model: str, sample_texts: List[str],
    system_prompt: Optional[str], min_axes: int, max_axes: int
) -> List[AxisSpec]:
    """
    Single LLM call to choose K axes with K in [min_axes, max_axes].
    Returns a variable-length list of AxisSpec.
    """
    prompt = (
        "You are a KAG (Knowledge as Geometry) architect. "
        f"Given representative texts, choose an appropriate number K of high-level, continuous axes "
        f"in the range [{min_axes}, {max_axes}] that best organize the corpus. "
        "For each axis, return an object with fields: name, description. "
        "Names should be short (1-3 words). Descriptions must be one sentence explaining how to interpret the axis.\n\n"
        "Rules:\n"
        "- Prefer fewer axes when the corpus is narrow; choose more only if clearly beneficial.\n"
        "- Axes must be continuous and interpretable (time, location, abstraction, social hierarchy, etc.), not entity names.\n"
        "- Return strict JSON: {\"axes\": [{\"name\":..., \"description\":...}, ...]} with exactly K items."
    )
    user = {"sample_texts": sample_texts[:8]}
    raw = _llm_json(
        client=client,
        llm_model=llm_model,
        system_msg=system_prompt or "You design interpretable coordinate systems.",
        user_msg=prompt + "\n\nInput:" + json.dumps(user, ensure_ascii=False),
        # Safe fallback to 3 canonical axes
        fallback={"axes": [
            {"name": "Time", "description": "Earlier to later across events."},
            {"name": "Location", "description": "Spatial or place-related variation."},
            {"name": "Semantics", "description": "Abstract topical similarity and meaning."},
        ]},
    )

    axes_items: List[Dict[str, Any]] = []
    if isinstance(raw, dict) and isinstance(raw.get("axes"), list):
        axes_items = raw["axes"]
    elif isinstance(raw, list):
        axes_items = raw
    if not axes_items:
        axes_items = [
            {"name": "Time", "description": "Earlier to later across events."},
            {"name": "Location", "description": "Spatial or place-related variation."},
            {"name": "Semantics", "description": "Abstract topical similarity and meaning."},
        ]

    axes: List[AxisSpec] = []
    for ax in axes_items:
        name = str(ax.get("name", "Axis")).strip()
        desc = str(ax.get("description", "")).strip()
        if name:
            axes.append(AxisSpec(name=name, description=desc))

    # Clamp to [min_axes, max_axes]
    if len(axes) < min_axes:
        axes += [AxisSpec(name=f"Axis-{i+1}", description="Ad hoc latent dimension.")
                 for i in range(min_axes - len(axes))]
    elif len(axes) > max_axes:
        axes = axes[:max_axes]
    return axes


def _extract_hfs_and_classes(
    *, client: Any, llm_model: str, chunks: List[Dict[str, Any]], use_llm: bool, system_prompt: Optional[str]
) -> Tuple[Dict[str, HF], Dict[str, ClassNode], Dict[str, Instance]]:
    hfs: Dict[str, HF] = {}
    classes: Dict[str, ClassNode] = {}
    instances: Dict[str, Instance] = {}

    if use_llm:
        batch_size = 12
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            payload = [
                {"id": c.get("id", str(i + j)), "text": c.get("text", ""), "meta": {k: v for k, v in c.items() if k not in ("id", "text")}}
                for j, c in enumerate(batch)
            ]
            prompt = (
                "You are a Knowledge-as-Geometry (KAG) analyst. "
                "Your goal is to decompose text into minimal, concrete knowledge atoms called Hypothetical Facts (HFs). "
                "Each HF expresses a specific relation, event, or property that can be geometrically represented as a point in latent space.\n\n"
                "For each HF, identify the involved Classes — tangible, nameable entities that could serve as coordinate anchors. "
                "These should be specific and referential (people, organizations, works, places, dates, items, measurable values).\n\n"
                "Do NOT list general or abstract categories such as 'people', 'characters', 'concept', 'policy', or 'phenomenon'. "
                "If a term refers to an abstract dimension shared by many HFs (e.g., time, location, emotional tone, importance), "
                "it should NOT be a Class — it belongs to the latent axes of the universe.\n\n"
                "Each HF must have:\n"
                "- id: unique short id\n"
                "- text: short declarative summary (<= 2 sentences)\n"
                "- weight: float in [0,1] indicating importance or confidence\n"
                "- class_names: list of Class strings (concrete entities only)\n"
                "- source_id: the id of the input chunk\n\n"
                "Return strictly JSON: {\"hfs\": [ ... ]}."
            )
            raw = _llm_json(
                client=client,
                llm_model=llm_model,
                system_msg=system_prompt or "You are precise at information extraction.",
                user_msg=prompt + "\n\nInput:" + json.dumps(payload, ensure_ascii=False),
                fallback={"hfs": [{"id": c.get("id", str(i)), "text": c.get("text", ""), "weight": 1.0, "class_names": [], "source_id": c.get("id", str(i))} for c in batch]},
            )
            for rec in raw.get("hfs", []):
                hf_id = f"hf_{rec.get('id', str(uuid.uuid4()))}"
                text = rec.get("text", "").strip()
                weight = float(rec.get("weight", 1.0) or 1.0)
                class_names = [str(n).strip() for n in (rec.get("class_names", []) or []) if str(n).strip()]
                src = str(rec.get("source_id", "")) or None
                hfs[hf_id] = HF(id=hf_id, text=text, weight=weight, class_names=class_names, source_ids=[src] if src else [])
                for cn in class_names:
                    if cn not in classes:
                        classes[cn] = ClassNode(name=cn)
                    inst_id = f"inst_{uuid.uuid4().hex[:10]}"
                    instances[inst_id] = Instance(id=inst_id, class_name=cn, hf_id=hf_id)
                    classes[cn].instances.append(inst_id)
    else:
        for c in chunks:
            hf_id = f"hf_{c.get('id', uuid.uuid4().hex[:8])}"
            hfs[hf_id] = HF(id=hf_id, text=c.get("text", ""), weight=1.0, class_names=[], source_ids=[c.get("id", "")])

    return hfs, classes, instances


# --- Name normalization & merging -------------------------------------------

def _norm_name(s: str) -> str:
    """Stronger but simple normalization: NFKC, lower, remove punctuation, absorb hyphens/spaces."""
    s = unicodedata.normalize("NFKC", s or "")
    s = s.strip().lower()
    s = s.replace("-", " ")
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def _mechanical_class_merge(classes: Dict[str, ClassNode]) -> Tuple[Dict[str, ClassNode], Dict[str, str]]:
    """Merge classes by normalized names. Returns (merged_classes, mapping old->canonical)."""
    norm_to_canonical: Dict[str, str] = {}
    merged: Dict[str, ClassNode] = {}
    mapping: Dict[str, str] = {}

    for name, node in classes.items():
        key = _norm_name(name)
        if key in norm_to_canonical:
            can_name = norm_to_canonical[key]
            mapping[name] = can_name
            merged[can_name].aliases = sorted(list(set(merged[can_name].aliases + [name] + node.aliases)))
            merged[can_name].instances.extend(node.instances)
        else:
            norm_to_canonical[key] = name
            merged[name] = node
            mapping[name] = name
    return merged, mapping


def _apply_class_mapping_to_refs(
    *, mapping: Dict[str, str],
    hfs: Dict[str, HF],
    instances: Dict[str, Instance],
    classes: Dict[str, ClassNode],
) -> None:
    """Apply name mapping to HF.class_names / Instance.class_name and rebuild classes keys if needed."""
    if not mapping:
        return

    # HF side
    for hf in hfs.values():
        hf.class_names = [mapping.get(n, n) for n in hf.class_names]

    # Instance side
    for inst in instances.values():
        inst.class_name = mapping.get(inst.class_name, inst.class_name)

    # Classes dict: rename keys when needed and merge collisions
    for old, new in list(mapping.items()):
        if old == new or old not in classes:
            continue
        src = classes.pop(old)
        if new in classes:
            dst = classes[new]
            dst.aliases = sorted(set(dst.aliases + [old] + src.aliases))
            dst.instances.extend(src.instances)
        else:
            src.name = new
            if old not in src.aliases:
                src.aliases.append(old)
            classes[new] = src


def _llm_class_consolidation(
    *, client: Any, llm_model: str, classes: Dict[str, ClassNode], hfs: Dict[str, HF], system_prompt: Optional[str]
) -> Tuple[Dict[str, ClassNode], Dict[str, str]]:
    """Ask the LLM to propose merges/renames for remaining near-duplicate Classes. Return (canonical_classes, mapping)."""
    names = sorted(classes.keys())
    sample_hf_texts = [hfs[h].text for h in list(hfs.keys())[:100]]
    prompt = (
        "You see a list of concept class names and some sample facts. "
        "Suggest a mapping from old class names to canonical names (merge near-duplicates, spelling variants, aliases). "
        "IMPORTANT: Only merge when the meaning is clearly the same; do not over-merge. "
        "Return JSON object {mapping: {old_name: canonical_name}}."
    )
    payload = {"class_names": names, "sample_hfs": sample_hf_texts[:30]}
    raw = _llm_json(
        client=client,
        llm_model=llm_model,
        system_msg=system_prompt or "You consolidate taxonomies carefully.",
        user_msg=prompt + "\n\nInput:" + json.dumps(payload, ensure_ascii=False),
        fallback={"mapping": {}},
    )
    mapping: Dict[str, str] = raw.get("mapping", {}) or {}
    canonical: Dict[str, ClassNode] = {}
    for old, node in classes.items():
        new = mapping.get(old, old)
        if new in canonical:
            canonical[new].aliases = sorted(list(set(canonical[new].aliases + [old] + node.aliases)))
            canonical[new].instances.extend(node.instances)
        else:
            node.name = new
            if new != old:
                node.aliases = sorted(list(set(node.aliases + [old])))
            canonical[new] = node
    return canonical, mapping


# =============================================================================
# Helpers — math & projection
# =============================================================================

def _svd_project(X: np.ndarray, dims: int) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    Z = U[:, :dims] * S[:dims]
    return Z


def _mean_pairwise_distance(coords: np.ndarray) -> float:
    n = len(coords)
    if n < 2:
        return 1.0
    dsum = 0.0
    cnt = 0
    for i in range(n):
        diff = coords[i + 1 :] - coords[i]
        if diff.size == 0:
            continue
        d = np.linalg.norm(diff, axis=1)
        dsum += float(d.sum())
        cnt += len(d)
    return dsum / max(cnt, 1)


def _ball_volume(r: float, d: int) -> float:
    if r <= 0:
        return 0.0
    unit = (math.pi ** (d / 2.0)) / math.gamma(d / 2.0 + 1)
    return unit * (r ** d)


def _instances_of_class(instances: Dict[str, Instance], class_name: str) -> List[Instance]:
    return [inst for inst in instances.values() if inst.class_name == class_name]


def _chunks_from_universe(u: KAGUniverse) -> List[Dict[str, Any]]:
    out = []
    for hf in u.hfs.values():
        out.append({"id": hf.id, "text": hf.text})
    return out


def _cosine_sim_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Return cosine similarity matrix between rows of A (n,d) and rows of B (m,d)."""
    if A.size == 0 or B.size == 0:
        return np.zeros((A.shape[0], B.shape[0]), dtype=float)
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return A_norm @ B_norm.T


def _normalize(vals: List[float]) -> List[float]:
    if not vals:
        return []
    vmin, vmax = min(vals), max(vals)
    if vmax - vmin < 1e-12:
        return [0.5 for _ in vals]
    return [(v - vmin) / (vmax - vmin) for v in vals]


def _invert_and_normalize(dists: List[float]) -> List[float]:
    """Map distances to relevance: smaller distance -> higher score, normalized to [0,1]."""
    if not dists:
        return []
    inv = [1.0 / (1.0 + d) for d in dists]
    return _normalize(inv)


def _inv_dist_score(d: float) -> float:
    """Single distance to [0,1] score convenience."""
    return 1.0 / (1.0 + max(d, 0.0))


# =============================================================================
# Helpers — LLM anchor targets & alignment (Improvement #2)
# =============================================================================

def _propose_anchor_coords(
    *, client: Any, llm_model: str, anchor_names: List[str], dims: int,
    value_range: Tuple[float, float], sample_texts: List[str], system_prompt: Optional[str]
) -> Dict[str, List[float]]:
    """Ask LLM for target anchor coordinates in a bounded range (e.g., [-1,1])."""
    if not anchor_names:
        return {}
    lo, hi = float(value_range[0]), float(value_range[1])
    prompt = (
        "You are designing a geometric coordinate frame for a KAG universe. "
        "Given anchor class names, propose interpretable target coordinates for each, "
        f"with dimensionality {dims} and each coordinate in the range [{lo}, {hi}]. "
        "Prefer intuitive layouts (e.g., temporal anchors ordered along one dimension). "
        "Return strict JSON: {\"coords\": {\"<anchor_name>\": [x1,...,xD], ...}}"
    )
    payload = {"anchors": anchor_names, "dims": dims, "range": [lo, hi], "samples": sample_texts[:8]}
    raw = _llm_json(
        client=client,
        llm_model=llm_model,
        system_msg=system_prompt or "You propose compact, interpretable coordinates.",
        user_msg=prompt + "\n\nInput:" + json.dumps(payload, ensure_ascii=False),
        fallback={"coords": {}},
    )
    coords = raw.get("coords", {}) or {}
    out: Dict[str, List[float]] = {}
    for k, v in coords.items():
        arr = [float(x) for x in (v or [])][:dims]
        if len(arr) < dims:
            arr = arr + [0.0] * (dims - len(arr))
        # clamp to [lo, hi]
        arr = [max(lo, min(hi, x)) for x in arr]
        out[k] = arr
    return out


def _affine_align(X_src: np.ndarray, Y_tgt: np.ndarray, l2: float = 1e-4) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find affine transform X @ A + b ≈ Y in least squares sense.
    Returns (A: (d,d), b: (d,)).
    """
    n, d = X_src.shape
    X1 = np.hstack([X_src, np.ones((n, 1))])      # (n, d+1)
    # Solve (X1^T X1 + l2 I) W = X1^T Y
    XtX = X1.T @ X1 + l2 * np.eye(d + 1)
    XtY = X1.T @ Y_tgt
    W = np.linalg.solve(XtX, XtY)                 # (d+1, d)
    A = W[:d, :]                                   # (d, d)
    b = W[d:, :].ravel()                            # (d,)
    return A, b


# =============================================================================
# Helpers — OpenAI LLM JSON calling
# =============================================================================

def _llm_json(*, client: Any, llm_model: str, system_msg: str, user_msg: str, fallback: Any) -> Any:
    """Call the OpenAI client to get structured JSON; return fallback on any error or parse failure."""
    try:
        if _HAS_OPENAI_V1 and isinstance(client, OpenAI):
            resp = client.chat.completions.create(
                model=llm_model,
                messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content
        elif _HAS_OPENAI_LEGACY:
            _openai_legacy.api_key = getattr(client, "api_key", os.getenv("OPENAI_API_KEY"))
            resp = _openai_legacy.ChatCompletion.create(
                model=llm_model,
                messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
                temperature=0.0,
            )
            content = resp["choices"][0]["message"]["content"]
        else:  # pragma: no cover
            return fallback
        try:
            return json.loads(content)
        except Exception:
            return fallback
    except Exception:
        return fallback


# =============================================================================
# Lightweight CLI (optional)
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RIHU prototype CLI (improved, bounded axes)")
    parser.add_argument("input", help="Path to a JSONL file with objects {id, text, ...} (pre-chunked)")
    parser.add_argument("output", help="Path to save the Universe JSON")
    parser.add_argument("--model", default=os.getenv("RIHU_LLM_MODEL", "gpt-4o-mini"))
    parser.add_argument("--embed", default=os.getenv("RIHU_EMBED_MODEL", "text-embedding-3-small"))
    parser.add_argument("--dims", type=int, default=None, help="If omitted, dims=K (LLM-chosen axes within [3,10])")
    parser.add_argument("--iters", type=int, default=6)
    parser.add_argument("--no_llm_hf", action="store_true", help="Disable LLM HF extraction")
    parser.add_argument("--no_llm_merge", action="store_true", help="Disable LLM class consolidation")
    parser.add_argument("--anchors", nargs="*", help="Anchor class names (space-separated)")
    parser.add_argument("--use_llm_anchor_coords", action="store_true", help="Enable LLM-proposed anchor coords + one-shot alignment")

    args = parser.parse_args()

    # Load chunks
    chunks: List[Dict[str, Any]] = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunks.append(json.loads(line))

    # Init client
    if _HAS_OPENAI_V1:
        client = OpenAI()
    elif _HAS_OPENAI_LEGACY:
        client = _openai_legacy  # type: ignore
    else:
        raise SystemExit("OpenAI SDK not installed. pip install openai")

    uni = KAGUniverse.build(
        chunks=chunks,
        openai_client=client,
        llm_model=args.model,
        embedding_model=args.embed,
        dims=args.dims,  # None => dims = K in [3,10]
        iterations=args.iters,
        use_llm_hf_extraction=not args.no_llm_hf,
        use_llm_class_consolidation=not args.no_llm_merge,
        anchors=args.anchors,
        use_llm_anchor_coords=args.use_llm_anchor_coords,
    )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(uni.to_json(), f, ensure_ascii=False, indent=2)

    print(f"Saved Universe -> {args.output}")