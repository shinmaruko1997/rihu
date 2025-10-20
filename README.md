# RIHU — Retrieval In the Hypothetical Universe

> A search paradigm built upon **KAG (Knowledge as Geometry)** — navigating meaning through geometry.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#license)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](#quickstart)

---

## Purpose and Scope

This repository introduces **RIHU (Retrieval In the Hypothetical Universe)** —
a prototype retrieval framework **operating inside the KAG Universe**, where knowledge is represented as geometry.

RIHU is **not merely a new retrieval algorithm**.
It is a practical exploration of how the **KAG** concept — *Knowledge as Geometry* — can transform retrieval into a spatial reasoning process.

**KAG** defines how knowledge exists: as points, regions, and relations in a continuous geometric space.
**RIHU** defines how we *move through* that space: how we locate, interpret, and retrieve meaning based on distance, density, and containment.

> ⚠️ This repository introduces both a **theoretical framework (KAG)** and its **first applied search system (RIHU)**.
> It serves as an open research artifact — a foundation for discussion, visualization, and experimental RAG integration.

---

## TL;DR

**KAG** models knowledge as a shared geometric space of **Hypothetical Facts (HFs)**, **Classes**, and **Instances**.
Since the totality of “all facts” is unknowable, KAG works with **Hypothetical Facts** — inferred and contextually grounded propositions.

Within this **KAG Universe**, **RIHU** performs retrieval as **wayfinding**.
Rather than lexical matching, retrieval becomes spatial navigation: **nearby = related**, **dense = influential**, **inside = contextual**.

This repository provides:

* A reference Python implementation of the **KAG core** and the **RIHU retriever**
* Both **objective (world-centric)** and **subjective (observer-centric)** search modes
* Influence and context metrics (centroid / radius / volume / power)
* A small demonstration dataset and CLI runner

---

## Relationship Between RIHU and KAG

| Concept                                           | Role                                                                                                                         |
| ------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| **KAG (Knowledge as Geometry)**                   | The *theoretical foundation* — defines how knowledge is represented as a geometric structure of HFs, Classes, and Instances. |
| **KAG Universe**                                  | The *operational space* constructed from KAG — the geometric manifold of meaning.                                            |
| **RIHU (Retrieval In the Hypothetical Universe)** | The *retrieval methodology* that moves through the KAG Universe to find, interpret, and contextualize knowledge.             |

Think of **KAG** as the *map of meaning* — and **RIHU** as the *navigator* exploring that map.

---

## Why KAG (and why RIHU)?

Traditional RAG systems retrieve by text similarity in a single embedding space.
KAG differs fundamentally by introducing **geometry-first semantics** — explicit, interpretable coordinates of meaning.

RIHU builds on that geometry to enable **structured, contextual retrieval**.

* **Explicit axes** — e.g., time / space / semantics are treated as coordinate dimensions.
* **Classes with influence** — each Class has measurable presence across regions.
* **Subjective search** — retrieval can originate from a *vantage point*, not only from a text query.
* **Interpretable geometry** — anchor Classes stabilize the space and make positions meaningful.

---

## Core Concepts (KAG Foundation)

KAG treats the world as a space **filled only with facts**.
Since the totality of facts is **unknowable**, we work with **Hypothetical Facts (HFs)**—propositions **inferred** from observations, traces, or representations. KAG builds a **knowledge space** where HFs, **Instances**, and **Classes** are projected into a **uniform geometric model**.

### 1) Entities

* **Hypothetical Fact (HF)**
  An inferred, context-grounded proposition derived from observations or media.
  *Example:* “**1776-07-04**: The **Declaration of Independence** was adopted in **Philadelphia**.”

* **Instance**
  A concrete occurrence of a Class *within a specific HF*. It identifies a **particular** concept/component *as observed inside that HF*.
  *Example:* If “Declaration of Independence” appears in 5 HFs, there are **5 Instances** of that Class.

* **Class**
  The **unique** pattern that groups similar Instances across HFs. A Class is **one per concept**, and its Instances are the many.
  *Example:* The Class **“Declaration of Independence”** is unique, even if it has many Instances.

* **Representation**
  A human-facing artifact (text, video, visualization, etc.) that we **extract HFs from**.
  *Example:* A documentary synthesized from many HFs about the Declaration is a **Representation**.

> **Uniformity Principle**
> **Time** and **Place** are modeled **as Classes and Instances** just like any other concept. This removes schema-driven bias and lets every kind of information be treated **flatly** in the same geometry.


### 2) Knowledge Space & Projection

KAG constructs a **knowledge space** where HFs and their constituent Instances are embedded. Classes are **projected** as the shared structures that tie Instances together.

* Each HF anchors a **local configuration** of Instances (people, events, artifacts, times, places…).
* Each Class summarizes **where** and **how** its Instances appear across HFs.
* Representations are **inputs** to extraction; we do not directly reason over them as truth—only via the HFs they yield.

* **Class is unique; Instances are many.**
  A Class exists **once**; each appearance **per HF** is a distinct Instance.
* **HF coordinates = Instance coordinates.**
  An Instance **inherits** the coordinates of the HF it appears in. Thus, a Class’s position emerges from the **distribution** of its Instances across HFs.

> **Why “Hypothetical”?**
> **A complete and omniscient capture of all facts is never feasible.**
> HFs make this explicit: we reason from **what is observed and inferred**, not from a claim of total knowledge.


### 3) Coordinates & Axes (Choosing the Frame)

The knowledge space is given coordinates in $\mathbb{R}^D$. Axes are **broad abstractions of Classes** that provide continuity and coverage:

1. **Axis from Class abstractions**
   Use abstractions like **Location**, **Organization**, **Time**, **Discipline** when the observer or data suggests them.
2. **Continuity & Connectivity**
   Prefer axes that relate to **many Classes** and exhibit **smooth variation** (e.g., time is continuous and widely presupposed).
3. **Inclusion**
   Favor axes whose related Classes **contain** many others (i.e., “large” Classes that organize others).

> **Anchors**
> Concrete, interpretable Classes (e.g., historical periods, cities, known entities) can be used as **anchors** to stabilize the frame. Axes can be proposed or initialized with LLM assistance, while preserving interpretability.


### 4) Examples

* **Event HF:**
  “**1776-07-04**: **Declaration of Independence** adopted in **Philadelphia**.”
  Includes Instances of Classes: *18th century*, *Declaration of Independence*, *Philadelphia*, *United States*.

* **Relational HF:**
  “One week before **XX**, **A** and **B** conversed at **C**.”
  Includes Instances of Classes: *A*, *B*, *C (place)*, *time offset*, *XX (reference event)*.

* **Representation:**
  A documentary compiled from HFs about the Declaration is a **Representation**. We extract HFs **from** it; we do not equate the narrative with ground truth.


### 5) What the Geometry Buys Us

Because all elements—including time and place—are **Classes and Instances in one flat space**, we can compute **interpretable** geometric properties:

* **Nearness** ⇒ *aboutness / relatedness*
* **Density** ⇒ *salience / influence*
* **Containment** ⇒ *context / framing*

These properties feed downstream metrics (e.g., centroids, enclosing volumes, power) and enable **RIHU** to perform retrieval as **wayfinding** inside this geometric knowledge space.

---

## The KAG Universe: Coordinate System

All knowledge is embedded into $\mathbb{R}^D$, where axes represent broad, continuous dimensions (e.g., time / space / semantics).
Good axes:

1. Are abstractions of Classes (e.g., “location”, “organization”).
2. Connect many other Classes with continuity.
3. Provide large, interpretable “containers” of meaning.

**Anchor Classes** (like time bins or known locations) stabilize the coordinate frame via regularization.
> In practice, **axes and anchor coordinates can also be suggested or initialized by large language models (LLMs)** — for example, inferring temporal, spatial, or semantic anchors from descriptive text or datasets.
> This enables partially automated construction of the KAG Universe, while keeping its geometric interpretability intact.

---

## Universe Construction

> The **Universe** in KAG corresponds to what a traditional RAG system would call an **index** —
> the structured space in which knowledge is stored and retrieved.
> However, rather than a static vector index, KAG uses the term *Universe* to emphasize its **geometric and dynamic nature**:
> a manifold of Hypothetical Facts, Classes, and Instances, whose relationships are explicitly modeled as geometry.

The following section outlines how these geometric ideas are implemented concretely in RIHU’s Python prototype.

1. **Propose axes** (LLM, single call, **K in [3,10]**).
2. **Extract HFs & Classes** (LLM-assisted with fallbacks).
3. **Mechanical merge** (name normalization + alias consolidation), optionally followed by **LLM consolidation**.
4. **Initial coordinates** via embeddings + SVD projection to (D) dimensions.
5. *(Optional)* **Anchor targets & one-shot affine alignment** (anchors then remain fixed).
6. **Iterative relaxation** (HFs → class centroids; classes → instance-HF centroids).
7. **Scale normalization** — **prefer anchors** when available; otherwise use all classes.
8. **Metrics** — centroid / radius (enclosing-ball approx) / volume / power.

---

## Geometry & Metrics (KAG Core)

For any Class (A), using the coordinates of its Instances:

1. **Position** (centroid): the mean of Instance coordinates
   $\mathbf{c}\_A = \frac{1}{n}\sum_{i=1}^{n}\mathbf{x}\_i$
2. **Influence Range**: a simply connected region containing all Instances (approximated by the **minimal enclosing ball**); $radius (r_A)$.
3. **Volume** (in (D) dimensions):
   $V_A = V_D(1)\cdot r_A^D,\quad V_D(1)=\frac{\pi^{D/2}}{\Gamma(D/2+1)}$
4. **Power**:
   $\text{power}(A)=\frac{\text{Instances}(A)}{V_A}$
  
### Contextual Relationships

* If an HF lies inside a Class’s influence region, that Class is contextual for the HF.
* If Class A lies within Class B’s influence region, A is subordinate to B (B provides the contextual frame).

---

## RIHU Retrieval in the KAG Universe

RIHU treats retrieval as **navigation through the KAG Universe**.
Instead of asking *“Which texts match?”*, we ask:
**“Where should I stand in the geometry — and what becomes near, dense, or enclosing when I do?”**

RIHU supports two complementary retrieval modes:

* **Objective Retrieval (world-centric)** — search from the global structure of meaning.
* **Subjective Retrieval (observer-centric)** — search from a specific vantage within the space.

---

### 1) Objective Retrieval — *Searching from the God’s-eye view*

**Intent:** Discover what the world “knows” about a topic, by locating regions where relevant Classes converge.

**Conceptual flow:**

1. **Define the conceptual nucleus** — choose key Classes (e.g., *Declaration of Independence*, *United States*).
2. **Compute the centroid** — find their geometric center in the Universe.
3. **Survey the neighborhood** — identify HFs near that centroid that share overlapping Class memberships.
4. **Interpret density** — dense regions imply high influence or conceptual richness.

**Philosophical intent:**

* *The world before the observer* — reveal the structure of meaning independent of perspective.
* *Geometry before language* — relationships emerge from position, not word overlap.
* *Integrative design* — RIHU can combine geometric reasoning with vector-based similarity for hybrid RAG pipelines.

---

### 2) Subjective Retrieval — *See through a pair of eyes*

**Intent:** Explore the world **from within** — from the viewpoint of an observer in the KAG Universe.

**Conceptual flow:**

1. **Pick a vantage point** — an observer (person, role, or moment).
2. **Look around** — retrieve nearby HFs within a certain radius.
3. **Identify context** — Classes enclosing the observer define the ambient context.
4. **Scan the periphery** — what lies just beyond reach, signaling emerging relevance?

**Design principles:**

* **Context-first retrieval** — the “room” around the observer matters.
* **Periphery awareness** — sense the nearby unknown.
* **Bias transparency** — by declaring a vantage, bias becomes explicit.

---

### Choosing a Mode

* Use **Objective RIHU** for top-down, phenomenon-centric exploration.
* Use **Subjective RIHU** for bottom-up, observer-centric interpretation.
* Mix both: locate the global neighborhood first, then experience it from within.

---

## Why this matters for RAG

RIHU reframes retrieval for **explainable semantic navigation**:

| Geometric property | Meaning                                 |
| ------------------ | --------------------------------------- |
| **Distance**       | Aboutness (near = relevant)             |
| **Density**        | Salience (many Instances = influential) |
| **Containment**    | Context (inside = framing meaning)      |

Retrieval becomes **legible**:
you can literally *point to the region* in the KAG Universe where meaning was found.

---

## Quickstart

### Requirements

- Python **3.9+**
- Packages: `numpy`, `openai`, `tiktoken`  
  *(install automatically via `requirements.txt`)*

### Setup

1. Clone or download this repository, and make sure the following three files are in the **same folder**.
Note that `sample.py` and `holmes.py` are located in the **sample** directory.:

```
sample.py
rihu.py
holmes.txt
```

**About `holmes.txt`**

   The file `holmes.txt` contains the full English text of *The Adventures of Sherlock Holmes* by **Arthur Conan Doyle**.  
   The source text was obtained from [Project Gutenberg](https://www.gutenberg.org/ebooks/1661)  
   (EBook #1661, Public Domain in the USA).

   The **Project Gutenberg header and license notice have been removed** for data-processing convenience.  
   The underlying literary work is in the **public domain**, and **no Project Gutenberg trademarks are used** in this repository.

2. Install dependencies:

```bash
pip install -r requirements.txt
# or manually
pip install numpy openai tiktoken
```

3. Set your OpenAI API key:

   * **macOS / Linux**

     ```bash
     export OPENAI_API_KEY="sk-..."
     ```
   * **Windows (PowerShell)**

     ```bash
     $env:OPENAI_API_KEY="sk-..."
     ```
   * **Windows (Command Prompt)**

     ```bash
     set OPENAI_API_KEY=sk-...
     ```

### Run the demo

```bash
python sample.py
```

The script will automatically:

1. Read **`holmes.txt`**
2. Build or load the geometric **KAG Universe** (`holmes/universe.json`, `axes.json`, `searches.json`)
3. Run both **Objective** and **Subjective** RIHU searches
4. Save results and print a digest to the console

You’ll see output like:

```
[RIHU] Building new universe from holmes.txt -> holmes/
[Normalize] Mapping seed classes (typo/variant correction):
  - 'holmes' → sherlock holmes  (exact/alias match)
  - 'adler' → irene adler  (fuzzy: 'Irene Adler', score=0.97)
[Search] Objective from seed classes (mapped): ['sherlock holmes', 'irene adler']
[Search] Subjective from centroid → nearest class: sherlock holmes
[RIHU] Saved search results -> holmes/searches.json

===== Objective (top neighbors) =====
  - HF[hf_012]: d=0.214 classes=holmes, watson, crime | "Holmes looked at me and laughed..."
  - HF[hf_031]: d=0.278 classes=adler, disguise, london | "Irene Adler was the woman..."

===== Subjective from centroid (nearest class) =====
  * Ctx: 19th century (r=0.58, power=0.002)
  * Ctx: London (r=0.42, power=0.005)

===== Subjective per seed class =====
- sherlock holmes
  - HF[hf_045]: d=0.192 classes=holmes, deduction, london | "He had an extraordinary gift..."
- irene adler
  - HF[hf_081]: d=0.235 classes=adler, opera, scandal | "The Woman, as Holmes called her..."
```

All generated files will appear under the new folder:

```
holmes/
├── universe.json
├── axes.json
└── searches.json
```

---

## 3D Viewer

Explore KAG/RIHU geometries interactively in your browser.
A sample dataset universe.json, created based on The Adventures of Sherlock Holmes by Arthur Conan Doyle, is stored in sample/holmes.

* **Live Demo:** [https://shinmaruko1997.github.io/rihu/viewer.html](https://shinmaruko1997.github.io/rihu/viewer.html)
* Features:

  * Upload `universe.json`
  * Choose axes and visualize 3D structure
  * Hover to inspect Hypothetical Facts (HFs) and Classes
  * Search by class seeds (objective / subjective)
* Example:

![Viewer Screenshot](https://github.com/user-attachments/assets/e59e28d3-56af-47cf-af78-697e2062bf12)

---

## Minimal Usage Example — Search Only

If you also want to perform Universe Construction, please use the full version sample code located at sample/sample.py.

```python
from __future__ import annotations

import json
import os
import re
import sys
import unicodedata
import difflib
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np

# Search parameters (adjust if needed)
SUBJ_RADIUS = 1.5
TOPK = 12

# Sentence splitting regex (kept for compatibility)
_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9\"\\(\\[])')

# Fuzzy matching threshold
FUZZY_MIN_RATIO = 0.72

# Import RIHU
from rihu import KAGUniverse


def fail(msg: str):
    print(f"[ERROR] {msg}")
    sys.exit(1)


# -----------------------------
# Seed class normalization & mapping
# -----------------------------

def _norm_name(s: str) -> str:
    """Apply NFKC normalization, lowercase, remove punctuation, and collapse spaces."""
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
      - canonical_to_aliases: canonical class name -> list of display aliases (including canonical)
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
        # include aliases
        for a in (node.aliases or []):
            if not a:
                continue
            canonical_to_aliases[cname].append(a)
            alias_to_canonical[_norm_name(a)] = cname
    # deduplicate alias lists while preserving order
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
    1) Try exact normalized match
    2) If not found, perform fuzzy matching against all aliases
    """
    if not query:
        return None, None, 0.0
    qn = _norm_name(query)

    # 1) Exact normalized match
    if qn in alias_to_canonical:
        can = alias_to_canonical[qn]
        aliases = canonical_to_aliases.get(can, [can])
        display = query if query in aliases else aliases[0]
        return can, display, 1.0

    # 2) Fuzzy match against all aliases
    all_aliases = []
    for _, aliases in canonical_to_aliases.items():
        for a in aliases:
            all_aliases.append(a)

    norm_aliases = [_norm_name(a) for a in all_aliases]
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
    Handles typos and variations via normalization + fuzzy matching.
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
            print(f"  - '{s}' → (no close match; using original)")
            out.append(s)
    return out


# -----------------------------
# Universe loading only (no build)
# -----------------------------

def load_universe(uname: str) -> KAGUniverse:
    """
    Load an existing <uname>/universe.json.
    If not found, raise an error (no build performed).
    """
    udir = Path(uname)
    ujson = udir / "universe.json"
    if not (udir.exists() and ujson.exists()):
        fail(f"Universe file not found: {ujson}\n"
             f"Please build the universe first using RIHU's build process.")
    print(f"[RIHU] Loading prebuilt universe: {ujson}")
    data = json.loads(ujson.read_text(encoding="utf-8"))
    return KAGUniverse.from_json(data)


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
# Search orchestration
# -----------------------------

def run_searches(uname: str, classes: List[str]):
    # Load prebuilt universe
    uni = load_universe(uname)

    # 1) Normalize and map seed class names
    mapped_classes = map_seed_classes(uni, classes, verbose=True)

    # 2) Objective search from mapped classes
    print("\n[Search] Objective from seed classes (mapped):", mapped_classes)
    res_obj = uni.objective_search(class_names=mapped_classes, k=TOPK)

    # 3) Compute centroid of retrieved HFs → find nearest class and run subjective search
    hf_ids = [r["hf_id"] for r in res_obj.get("neighbors", [])]
    hf_centroid = centroid_of_hfs(uni, hf_ids)

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

    # 4) Run subjective search for each mapped class individually
    res_sub_each: Dict[str, Any] = {}
    for c in mapped_classes:
        print(f"[Search] Subjective per class: {c}")
        res_sub_each[c] = uni.subjective_search(vantage=c, k=TOPK, radius=SUBJ_RADIUS)

    # 5) Save search results and print digest
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

    dump_digest(res_obj, res_sub_from_centroid, res_sub_each)


def dump_digest(res_obj: Dict[str, Any], res_sub_from_centroid: Dict[str, Any] | None, res_sub_each: Dict[str, Any]):
    """Pretty-print search summaries to stdout."""
    def hf_lines(neis: List[Dict[str, Any]], n=5):
        lines = []
        for r in neis[:n]:
            d = r.get('distance', 0.0)
            classes = r.get('classes') or []
            text = r.get('text') or ""
            lines.append(f"  - HF[{r['hf_id']}]: d={d:.3f} classes={', '.join(classes[:4])} | {text[:120]}...")
        return "\n".join(lines) if lines else "  (none)"

    print("\n===== Objective (top neighbors) =====")
    print(hf_lines(res_obj.get("neighbors", []), n=8))

    print("\n===== Subjective from centroid (nearest class) =====")
    if res_sub_from_centroid:
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
    seed_classes = ["holmes", "adler"]  # Demo input (typo mapping visible)
    run_searches(universe_name, seed_classes)
```

---

## License

This project is licensed under the **MIT License**.
Copyright (c) 2025 H.Kiriyama
See the [LICENSE](LICENSE) file for details.

SPDX-License-Identifier: MIT

---

## Citation

If you use this repository in academic or industrial work, please cite both **RIHU** and **KAG**:

```bibtex
@software{RIHU_2025,
  title   = {RIHU: Retrieval In the Hypothetical Universe (built on KAG: Knowledge as Geometry)},
  author  = {H. Kiriyama},
  year    = {2025},
  license = {MIT},
  url     = {https://github.com/shinmaruko1997/rihu}
}
```

## Discussion and Next Steps

This repository presents **KAG (Knowledge as Geometry)** as a new way to represent knowledge,
and **RIHU (Retrieval In the Hypothetical Universe)** as its first retrieval paradigm.

Future directions include:

* Integration with open-source RAG systems
* Empirical evaluation on public datasets
* Visualization and interactive exploration tools

> “Distance is relationship; density is influence.” — *KAG Universe*