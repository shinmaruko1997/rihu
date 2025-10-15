# SPDX-License-Identifier: MIT
# Copyright (c) 2025 H.Kiriyama

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Iterable

import numpy as np

# ------------------------------
# Utility geometry helpers
# ------------------------------

def unit_sphere_volume(dim: int) -> float:
    """Volume of the unit d-ball (closed ball) in R^dim.

    V_d(1) = pi^(d/2) / Gamma(d/2 + 1)
    """
    return math.pi ** (dim / 2) / math.gamma(dim / 2 + 1)


def sphere_volume(radius: float, dim: int) -> float:
    return unit_sphere_volume(dim) * (max(radius, 0.0) ** dim)


def normalize_scale(X: np.ndarray, ref_pairs: Optional[List[Tuple[int, int]]] = None,
                    target_mean_dist: float = 1.0) -> np.ndarray:
    """Scale coordinate cloud X (N x D) to keep a stable global scale.

    If ref_pairs is provided (indexes into X), we scale such that mean distance
    among those pairs is `target_mean_dist`. Otherwise, scale s.t. global mean
    pairwise distance equals target.
    """
    if len(X) == 0:
        return X

    def mean_pairwise_dist(points: np.ndarray, pairs: Optional[List[Tuple[int,int]]]):
        if pairs:
            dists = [np.linalg.norm(points[i] - points[j]) for i, j in pairs]
        else:
            # compute upper triangle mean distance (O(N^2) OK for demo)
            dists = []
            for i in range(len(points)):
                for j in range(i + 1, len(points)):
                    dists.append(np.linalg.norm(points[i] - points[j]))
        return (sum(dists) / len(dists)) if dists else 1.0

    m = mean_pairwise_dist(X, ref_pairs)
    if m <= 1e-12:
        return X
    scale = target_mean_dist / m
    return X * scale

def normalize_scale_with_factor(X: np.ndarray, ref_pairs: Optional[List[Tuple[int, int]]] = None,
                                target_mean_dist: float = 1.0) -> Tuple[np.ndarray, float]:
    """Scale coordinate cloud X (N x D) and also return the scale factor used.

    This mirrors `normalize_scale` but exposes the multiplicative factor so that
    callers can apply the *same* scale to other related coordinate sets (e.g.,
    HF and Instance vectors) to keep geometry consistent within the iteration.
    """
    if len(X) == 0:
        return X, 1.0

    def mean_pairwise_dist(points: np.ndarray, pairs: Optional[List[Tuple[int,int]]]):
        if pairs:
            dists = [np.linalg.norm(points[i] - points[j]) for i, j in pairs]
        else:
            dists = []
            for i in range(len(points)):
                for j in range(i + 1, len(points)):
                    dists.append(np.linalg.norm(points[i] - points[j]))
        return (sum(dists) / len(dists)) if dists else 1.0

    m = mean_pairwise_dist(X, ref_pairs)
    if m <= 1e-12:
        return X, 1.0
    scale = target_mean_dist / m
    return X * scale, scale


# ------------------------------
# Core data structures
# ------------------------------

@dataclass
class ClassNode:
    key: str
    name: str
    vec: Optional[np.ndarray] = None
    is_anchor: bool = False
    init_vec: Optional[np.ndarray] = None  # for regularization if desired
    # bookkeeping
    instance_ids: List[str] = field(default_factory=list)


@dataclass
class InstanceNode:
    key: str
    class_key: str
    hf_key: str
    vec: Optional[np.ndarray] = None  # mirrors HF vec for convenience


@dataclass
class HFNode:
    key: str
    name: str
    # mapping class_key -> weight (confidence / source count / etc.)
    class_weights: Dict[str, float] = field(default_factory=dict)
    vec: Optional[np.ndarray] = None


class KnowledgeSpace:
    def __init__(self, dim: int = 3, axes: Optional[List[str]] = None, seed: int = 42):
        self.dim = dim
        self.axes = axes or ["time", "space", "semantics"][:dim]
        self.random = np.random.RandomState(seed)
        self.classes: Dict[str, ClassNode] = {}
        self.instances: Dict[str, InstanceNode] = {}
        self.hfs: Dict[str, HFNode] = {}
        # Anchor set caching for scale normalization
        self._anchor_keys: List[str] = []

    # ---- Creation helpers -------------------------------------------------

    def add_class(self, key: str, name: Optional[str] = None,
                  vec: Optional[Iterable[float]] = None,
                  is_anchor: bool = False) -> ClassNode:
        if key in self.classes:
            return self.classes[key]
        v = None
        if vec is not None:
            v = np.array(list(vec), dtype=float)
            assert v.shape == (self.dim,)
        node = ClassNode(key=key, name=name or key, vec=v, is_anchor=is_anchor,
                         init_vec=(np.copy(v) if v is not None else None))
        self.classes[key] = node
        if is_anchor:
            self._anchor_keys.append(key)
        return node

    def add_hf(self, key: str, name: str,
               class_weights: Optional[Dict[str, float]] = None) -> HFNode:
        if key in self.hfs:
            return self.hfs[key]
        node = HFNode(key=key, name=name, class_weights=class_weights or {})
        self.hfs[key] = node
        # auto-create instances per (Class, HF)
        for ck in node.class_weights.keys():
            inst_key = f"{ck}::inst@{key}"
            self.instances[inst_key] = InstanceNode(key=inst_key, class_key=ck, hf_key=key)
            self.classes.setdefault(ck, ClassNode(key=ck, name=ck))
            self.classes[ck].instance_ids.append(inst_key)
        return node

    # ---- Initialization ---------------------------------------------------

    def _ensure_class_vectors(self, scale: float = 0.01) -> None:
        for c in self.classes.values():
            if c.vec is None:
                c.vec = self.random.normal(loc=0.0, scale=scale, size=self.dim)
                c.init_vec = np.copy(c.vec)

    # ---- Iterative placement ----------------------------------------------

    def _hf_position_from_classes(self, hf: HFNode) -> Optional[np.ndarray]:
        # Weighted average of known class vectors.
        num = np.zeros(self.dim)
        denom = 0.0
        for ck, w in hf.class_weights.items():
            c = self.classes.get(ck)
            if c is None or c.vec is None or w <= 0:
                continue
            num += w * c.vec
            denom += w
        if denom <= 0:
            return None
        return num / denom

    def _class_target_from_hfs(self, class_key: str) -> Optional[np.ndarray]:
        # Weighted average of HF vectors that include this Class.
        num = np.zeros(self.dim)
        denom = 0.0
        for hf in self.hfs.values():
            w = hf.class_weights.get(class_key, 0.0)
            if w <= 0 or hf.vec is None:
                continue
            num += w * hf.vec
            denom += w
        if denom <= 0:
            return None
        return num / denom

    def optimize(self, max_iters: int = 200, alpha: float = 0.3,
                 anchor_reg: float = 0.0,
                 tol: float = 1e-4,
                 scale_ref: str = "anchors",
                 target_mean_dist: float = 1.0,
                 verbose: bool = False) -> Dict[str, float]:
        """Run the alternating update procedure.

        - First compute HF vectors from current Class vectors.
        - Then update non-anchor Class vectors towards the (weighted) mean of
          their participating HFs. Anchors can optionally be softly pulled back
          to their initial positions via `anchor_reg`.
        - After each iteration, rescale coordinates to keep global scale stable.
        - Stop when max movement < tol.
        """
        self._ensure_class_vectors()

        def anchor_pairs() -> List[Tuple[int, int]]:
            if scale_ref != "anchors":
                return []
            idx = {k: i for i, k in enumerate(self.classes.keys())}
            ks = [k for k in self.classes.keys() if self.classes[k].is_anchor]
            pairs = []
            for i in range(len(ks)):
                for j in range(i + 1, len(ks)):
                    pairs.append((idx[ks[i]], idx[ks[j]]))
            return pairs

        last_max_move = float("inf")
        for it in range(max_iters):
            # (1) HF update
            for hf in self.hfs.values():
                new_pos = self._hf_position_from_classes(hf)
                hf.vec = new_pos if new_pos is not None else hf.vec

            # (2) Class update
            max_move = 0.0
            for ck, c in self.classes.items():
                if c.is_anchor:
                    # optional: soft regularization towards init
                    if anchor_reg > 0 and c.init_vec is not None:
                        c.vec = (1 - anchor_reg) * c.vec + anchor_reg * c.init_vec
                    continue
                target = self._class_target_from_hfs(ck)
                if target is None:
                    continue
                new_vec = (1 - alpha) * c.vec + alpha * target
                move = np.linalg.norm(new_vec - c.vec)
                c.vec = new_vec
                if move > max_move:
                    max_move = move

            # (3) Instances mirror their HF positions
            for inst in self.instances.values():
                hf = self.hfs.get(inst.hf_key)
                inst.vec = None if hf is None else hf.vec

            # (4) Rescale for stability
            # Build class coordinate array for scaling (order stable by key)
            keys = list(self.classes.keys())
            X = np.stack([self.classes[k].vec for k in keys], axis=0)
            ref_pairs = anchor_pairs()

            # Scale classes and capture the factor; then propagate the same factor
            # to HFs and mirrored Instance vectors to keep geometry consistent.
            X_scaled, scale = normalize_scale_with_factor(
                X,
                ref_pairs=ref_pairs if ref_pairs else None,
                target_mean_dist=target_mean_dist
            )
            for i, k in enumerate(keys):
                self.classes[k].vec = X_scaled[i]

            if abs(scale - 1.0) > 1e-12:
                # Apply the exact same multiplicative factor to HF and Instance vectors.
                for hf in self.hfs.values():
                    if hf.vec is not None:
                        hf.vec *= scale
                for inst in self.instances.values():
                    if inst.vec is not None:
                        inst.vec *= scale

        return {"iterations": it + 1, "max_move": last_max_move}

    # ---- Metrics ----------------------------------------------------------

    def class_centroid(self, class_key: str) -> Optional[np.ndarray]:
        insts = [self.instances[i] for i in self.classes[class_key].instance_ids]
        pts = [inst.vec for inst in insts if inst.vec is not None]
        if not pts:
            return None
        return np.mean(np.stack(pts, axis=0), axis=0)

    def class_influence(self, class_key: str, eps: float = 1e-6) -> Dict[str, float]:
        """Return centroid, radius (approx min enclosing), volume, power.

        power = (#instances) / volume
        """
        c = self.classes[class_key]
        insts = [self.instances[i] for i in c.instance_ids]
        pts = [inst.vec for inst in insts if inst.vec is not None]
        n = len(pts)
        if n == 0:
            return {"centroid": None, "radius": 0.0, "volume": float("inf"), "power": 0.0}
        C = np.mean(np.stack(pts, axis=0), axis=0)
        radius = max(np.linalg.norm(p - C) for p in pts) if pts else 0.0
        volume = sphere_volume(radius + eps, self.dim)
        power = n / volume
        return {"centroid": C, "radius": radius, "volume": volume, "power": power}

    # ---- Retrieval --------------------------------------------------------

    def _collect_hf_vectors(self) -> List[Tuple[str, np.ndarray]]:
        return [(hf.key, hf.vec) for hf in self.hfs.values() if hf.vec is not None]

    def objective_search(self, target_classes: List[str],
                         k: int = 10,
                         radius_factor: float = 1.0,
                         min_class_power: float = 0.0) -> List[Tuple[str, float]]:
        """Objective search (1-1 + 1-2 blended).

        - Step 1-1: rank HFs by how closely they match the target class set
          (Jaccard) and proximity to the centroid of target classes.
        - Step 1-2: use target centroid as a center and pick HFs weighted by
          distance and local class influence around the center.

        Returns list of (hf_key, score) sorted by score desc.
        """
        # Compute target centroid from classes
        valid_classes = [ck for ck in target_classes if ck in self.classes and self.classes[ck].vec is not None]
        if not valid_classes:
            return []
        target_center = np.mean(np.stack([self.classes[ck].vec for ck in valid_classes]), axis=0)

        # Influence around target: approximate by mean power of target classes
        infl = [self.class_influence(ck)["power"] for ck in valid_classes]
        local_power = float(np.mean(infl)) if infl else 0.0

        # Characteristic radius from average of target class radii
        radii = [self.class_influence(ck)["radius"] for ck in valid_classes]
        char_r = float(np.mean(radii)) * max(radius_factor, 1e-6)

        results = []
        for hf in self.hfs.values():
            if hf.vec is None:
                continue
            # (1-1) Jaccard similarity of classes
            S = set(hf.class_weights.keys())
            T = set(valid_classes)
            jaccard = len(S & T) / max(len(S | T), 1)

            # (1-1) Distance to target centroid
            d = np.linalg.norm(hf.vec - target_center)
            dist_score = math.exp(- (d / (char_r + 1e-6))**2)

            # (1-2) Local influence weight: encourage areas where target power is high
            infl_weight = 1.0 + min(local_power, 1.0)

            # filter by min_class_power if *any* target class power fails the threshold
            if min_class_power > 0.0 and any(self.class_influence(ck)["power"] < min_class_power for ck in valid_classes):
                continue

            score = 0.6 * jaccard + 0.4 * dist_score
            score *= infl_weight
            results.append((hf.key, float(score)))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    def subjective_search(self, observer_pos: np.ndarray,
                          k_near: int = 10,
                          radius: float = 1.0,
                          power_threshold: float = 0.0,
                          k_context: int = 5) -> Dict[str, List[Tuple[str, float]]]:
        """Subjective search from an observer.

        Returns dict with keys:
          - 'near_hfs': within radius
          - 'context_classes': classes whose centroids lie within radius with power >= threshold
          - 'peripheral': nearest HFs/classes outside radius but close
        """
        # 2-1: HFs inside the observer ball
        in_ball = []
        out_ball = []
        for key, v in self._collect_hf_vectors():
            d = np.linalg.norm(v - observer_pos)
            (in_ball if d <= radius else out_ball).append((key, d))
        in_ball.sort(key=lambda x: x[1])

        # 2-2: Context classes that "contain" the observer (centroid in ball) and have power
        context = []
        for ck in self.classes:
            info = self.class_influence(ck)
            C = info["centroid"]
            if C is None:
                continue
            d = np.linalg.norm(C - observer_pos)
            if d <= radius and info["power"] >= power_threshold:
                # smaller d and larger power -> higher score
                score = (1.0 / (1.0 + d)) * (1.0 + min(info["power"], 1.0))
                context.append((ck, float(score)))
        context.sort(key=lambda x: x[1], reverse=True)

        # 2-3: Peripheral infos just outside
        out_ball.sort(key=lambda x: x[1])
        peripheral_hfs = [(k, 1.0 / (1.0 + d)) for k, d in out_ball[:k_near]]

        return {
            "near_hfs": [(k, 1.0 / (1.0 + d)) for k, d in in_ball[:k_near]],
            "context_classes": context[:k_context],
            "peripheral": peripheral_hfs,
        }

    # ---- Pretty helpers ---------------------------------------------------

    def describe_class(self, class_key: str) -> str:
        c = self.classes[class_key]
        info = self.class_influence(class_key)
        pos = np.array2string(info["centroid"], precision=3) if info["centroid"] is not None else "None"
        return (
            f"Class[{class_key}] name='{c.name}'\n"
            f"  is_anchor={c.is_anchor}\n"
            f"  centroid={pos}\n"
            f"  radius~{info['radius']:.3f}, volume~{info['volume']:.3f}, power~{info['power']:.6f}\n"
            f"  #instances={len(c.instance_ids)}\n"
        )

    def describe_hf(self, hf_key: str) -> str:
        hf = self.hfs[hf_key]
        pos = np.array2string(hf.vec, precision=3) if hf.vec is not None else "None"
        return (
            f"HF[{hf.key}] '{hf.name}'\n"
            f"  classes={list(hf.class_weights.keys())}\n"
            f"  vec={pos}\n"
        )


# ------------------------------
# Demo dataset & runner
# ------------------------------

def build_demo_space() -> KnowledgeSpace:
    ks = KnowledgeSpace(dim=3, axes=["time", "space", "semantics"], seed=7)

    # Anchors (example): coarse time bins & global locations
    ks.add_class("time_18C", "18th century", vec=[-1.0, 0.0, 0.0], is_anchor=True)
    ks.add_class("time_19C", "19th century", vec=[0.0, 0.0, 0.0], is_anchor=True)
    ks.add_class("time_20C", "20th century", vec=[+1.0, 0.0, 0.0], is_anchor=True)

    ks.add_class("place_philadelphia", "Philadelphia", vec=[-0.9, +0.8, 0.0], is_anchor=True)
    ks.add_class("place_paris", "Paris", vec=[-0.8, +0.6, 0.1], is_anchor=True)
    ks.add_class("place_tokyo", "Tokyo", vec=[+0.8, -0.7, 0.0], is_anchor=True)

    # Non-anchor Classes (concepts)
    ks.add_class("class_declaration_of_independence", "U.S. Declaration of Independence")
    ks.add_class("class_united_states", "United States")
    ks.add_class("class_france", "France")
    ks.add_class("class_revolution", "Revolution")
    ks.add_class("class_person_A", "Person A")
    ks.add_class("class_person_B", "Person B")
    ks.add_class("class_person_C", "Person C")

    # Hypothetical Facts (names kept short for readability)
    ks.add_hf(
        "hf_doI_1776",
        "1776/7/4: Declaration adopted in Philadelphia",
        class_weights={
            "time_18C": 1.0,
            "place_philadelphia": 1.0,
            "class_declaration_of_independence": 1.0,
            "class_united_states": 0.7,
            "class_revolution": 0.5,
        },
    )

    ks.add_hf(
        "hf_paris_1783",
        "1783: Treaty of Paris recognizes U.S. independence",
        class_weights={
            "time_18C": 1.0,
            "place_paris": 1.0,
            "class_united_states": 0.8,
            "class_france": 0.9,
        },
    )

    ks.add_hf(
        "hf_meeting_XXm1",
        "(XX − 1 week): A and B talk at C",
        class_weights={
            # time unknown as absolute; we emulate weak prior around 19C
            "time_19C": 0.2,
            "class_person_A": 0.8,
            "class_person_B": 0.8,
            "class_person_C": 0.6,
        },
    )

    ks.add_hf(
        "hf_tokyo_meeting",
        "Modern: A and B meet in Tokyo",
        class_weights={
            "time_20C": 0.7,  # weakly modern
            "place_tokyo": 1.0,
            "class_person_A": 0.6,
            "class_person_B": 0.6,
        },
    )

    ks.add_hf(
        "hf_paris_revolutionary_ideas",
        "18C Paris: Revolutionary ideas spread",
        class_weights={
            "time_18C": 1.0,
            "place_paris": 0.9,
            "class_france": 0.8,
            "class_revolution": 0.9,
        },
    )

    return ks


def run_demo(verbose: bool = False) -> None:
    ks = build_demo_space()

    # Optimize positions
    stats = ks.optimize(max_iters=300, alpha=0.35, anchor_reg=0.0, tol=1e-4,
                        scale_ref="anchors", target_mean_dist=1.0, verbose=verbose)
    print("Optimization stats:", stats)

    # Show a few HFs
    for k in ["hf_doI_1776", "hf_paris_1783", "hf_paris_revolutionary_ideas"]:
        print(ks.describe_hf(k))

    # Show a few Classes and their influence metrics
    for ck in [
        "class_declaration_of_independence",
        "class_united_states",
        "class_france",
        "class_revolution",
        "place_paris",
        "place_philadelphia",
    ]:
        print(ks.describe_class(ck))

    # Objective search: look for HFs related to {Declaration of Independence, United States}
    print("\n[Objective search]\n")
    results = ks.objective_search([
        "class_declaration_of_independence", "class_united_states"
    ], k=5, radius_factor=1.25)
    for key, score in results:
        print(f"HF {key} score={score:.3f} -> {ks.hfs[key].name}")

    # Subjective search: place an observer near Paris/18C region
    center = np.mean(np.stack([
        ks.classes["time_18C"].vec, ks.classes["place_paris"].vec
    ]), axis=0)
    print("\n[Subjective search @ observer near 18C Paris]\n")
    subj = ks.subjective_search(observer_pos=center, k_near=5, radius=0.6,
                                power_threshold=0.001, k_context=5)

    print("Near HFs:")
    for k, s in subj["near_hfs"]:
        print(f"  {k} score={s:.3f} -> {ks.hfs[k].name}")

    print("Context Classes:")
    for ck, s in subj["context_classes"]:
        print(f"  {ck} score={s:.3f} -> {ks.classes[ck].name}")

    print("Peripheral HFs:")
    for k, s in subj["peripheral"]:
        print(f"  {k} score={s:.3f} -> {ks.hfs[k].name}")


if __name__ == "__main__":
    run_demo(verbose=False)