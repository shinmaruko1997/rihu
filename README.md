# KAG — Knowledge as Geometry

> A geometry-first approach to RAG: position **facts**, **classes**, and **instances** in a shared coordinate space, then retrieve by distance, density, and context.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#license)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](#quickstart)
[![Status](https://img.shields.io/badge/status-prototype-orange.svg)](#roadmap)

---

## Purpose and Scope

This repository is an **experimental and conceptual prototype** introducing the idea of *Knowledge as Geometry (KAG)*.  
Its purpose is to present the theoretical foundation and provide a working demonstration for discussion and collaboration.

KAG does **not aim to be a finalized system** at this stage.  
Rather, it serves as an *open research artifact* designed to:
- Explore the possibility of representing knowledge as a geometric structure  
- Encourage theoretical and practical feedback from researchers and engineers  
- Serve as a foundation for future extensions (integration with RAG, visualization tools, etc.)

> ⚠️ Note: This is an early-stage research prototype.  
> Stability, completeness, and empirical validation are intentionally secondary to conceptual clarity.

---

## TL;DR

KAG models the world as a space densely packed with **facts**. Since exhaustive, omniscient capture of “all facts” is impossible, we work with **Hypothetical Facts (HF)** inferred from observation. We project HFs, **Classes** (patterns), and **Instances** (occurrences) into a shared coordinate system (e.g., time / space / semantics). Retrieval becomes geometric: **nearby = related**, **dense = influential**, **inside = contextual**.

This repo provides:

* A reference Python implementation of the KAG indexer & retriever
* Objective & subjective search strategies
* Influence metrics (centroid / radius / volume / power)
* A small demo dataset and CLI runner

---

## Why KAG?

Typical RAG pipelines retrieve by lexical overlap or vector similarity in a single embedding space. KAG differs by:

* Treating **time / place / semantics** (and other high-level features) as **explicit axes**.
* Making **Classes first-class citizens** with measurable **influence** over regions of the knowledge space.
* Allowing **subjective search** from an observer’s position (a person, role, or situation) rather than only query text.
* Using **geometric constraints** (anchors, regularization, scale normalization) to keep positions interpretable.

---

## Core Concepts

* **Hypothetical Fact (HF)**: A proposition inferred from observation (e.g., “1776-07-04: Declaration adopted in Philadelphia”, or “A and B talked at C one week before XX”). HFs are the atomic points we place.
* **Instance**: An occurrence of a Class **within** a specific HF (e.g., the **“Declaration of Independence”** appearing in five HFs yields **five Instances** of that Class).
* **Class**: A pattern shared by similar Instances. **There is exactly one Class** per conceptual pattern (e.g., “Declaration of Independence”, “18th century”, “Paris”).
* **Representation**: A human-facing composition built from HFs (e.g., a documentary). In practice, we often **extract HFs from Representations**.

> Time and place are also modeled as Classes/Instances so that all information is treated **flatly** and uniformly.

---

## Coordinate System

We embed everything into $\mathbb{R}^D$ with axes chosen to be **broad, continuous, and widely connected** (e.g., time / space / semantics). Good axes:

1. Are Classes or abstractions of Classes (e.g., “city”, “organization”).
2. Connect to many other Classes and exhibit continuity.
3. Are large “containers” that naturally include other Classes.

**Anchor Classes** (e.g., coarse time bins, known locations) can be pinned or softly regularized to stabilize the geometry.

---

## Geometry & Metrics

For a Class (A), using the coordinates of all its Instances:

1. **Position** (centroid): the mean of Instance coordinates
   $\mathbf{c}\_A = \frac{1}{n}\sum_{i=1}^{n}\mathbf{x}\_i$
2. **Influence Range**: a simply connected region containing all Instances (approximated by the **minimal enclosing ball**); $radius (r_A)$.
3. **Volume** (in (D) dimensions):
   $V_A = V_D(1)\cdot r_A^D,\quad V_D(1)=\frac{\pi^{D/2}}{\Gamma(D/2+1)}$
4. **Power**:
   $\text{power}(A)=\frac{\text{Instances}(A)}{V_A}$

**Context rules** (intuition):

* If an HF lies **inside** a Class’s influence region and that Class has sufficient power, the Class is **contextual** for that HF.
* If Class (A) lies inside Class (B)’s influence region with sufficient power, (A) is **subordinate** to (B) (i.e., (B) is context for (A)).

---

## Index Construction (Provisional)

1. **Initialize axes & anchors.** Give Classes optional initial vectors; anchors are fixed or softly regularized.
2. **Place HFs** as the (weighted) centroid of their participating Classes.
3. **Update Classes** toward the (weighted) centroid of the HFs that include them (use damping: ( $\text{new} = (1-\alpha)\text{old} + \alpha\cdot\text{target}$ )).
4. **Apply anchors / constraints.** Keep time/space priors stable.
5. **Normalize scale.** Keep global distances consistent (e.g., fix mean anchor distance).
6. **Convergence.** Stop when max movement ($< \varepsilon$).
7. **Output.** Final coordinates for Classes and HFs; compute distances & influence.

---

## Retrieval

KAG treats retrieval as **navigation** through a geometry of meaning. Instead of asking “which texts match these keywords?”, we ask: **“Where in the space should I stand, and what becomes near, dense, or enclosing when I do?”**
Two complementary modes emerge:

* **Objective (world-centric):** start from the **world’s structure**—a set of Classes you care about—and let the space surface HFs that sit where those Classes **converge**.
* **Subjective (observer-centric):** start from a **point of view**—a person, role, or situation—and ask what is **near**, what **contains** you, and what is **just beyond reach**.

No strict formulas are required; what matters is the **intent** behind each step.

---

### 1) Objective Retrieval — *Searching from the God's-eye view*

**Intent:**
Objective retrieval represents a **God’s-eye perspective** — an attempt to search the knowledge space **as if one could see all relationships at once**.
Rather than standing in a single observer’s position, we look down from above, seeking **globally coherent** relationships among Classes and their associated Hypothetical Facts (HFs).

---

**Conceptual flow**

1. **Define the conceptual nucleus.**
   Choose a set of Classes that together represent the phenomenon or topic of interest (e.g., *Declaration of Independence*, *United States*).

2. **Locate its gravitational center.**
   Compute the centroid of these Classes in the knowledge space. This point serves as the *anchor* for the objective viewpoint — the geometric center of meaning for your query.

3. **Survey the neighborhood.**
   Identify HFs that:

   * share overlapping Class memberships (the same conceptual DNA), and
   * physically reside near that centroid (occupy the same region of the space).

   These HFs form the **objective cluster**: what the world “knows” about this combination of ideas.

4. **Respect the geometry of relevance.**
   In KAG, *distance* is not an abstract number but a reflection of *semantic position*.
   Proximity means **shared context**, not just shared words.
   Density reveals **influence** — regions where many related Instances exist are where the world has “more to say.”

---

**Philosophical intent**

* **World before observer.**
  Objective retrieval prioritizes the *structure of the knowledge space itself* over any particular viewpoint.
  It asks: *“Where does the world naturally form meaning?”* — instead of *“What does this person see?”*

* **A neutral frame for truth-seeking.**
  It is an effort to glimpse the landscape of knowledge **without narrative bias**.
  The retrieval results are not stories told by agents, but **maps of relational coherence**.

* **Integration with conventional search.**
  While KAG’s current implementation searches **by Class-based geometry**, it is designed to integrate seamlessly with existing retrieval techniques:

  * Vector-based search (e.g., dense embeddings) can pre-select **relevant HFs**, whose participating Classes then define the subspace for precise geometric reasoning.
  * Hybrid pipelines may combine **semantic similarity**, **text embeddings**, and **KAG Class geometry**, allowing objective retrieval to evolve from *symbolic reasoning* to *semantic navigation*.

  In other words, the “God’s-eye view” is not static — it can incorporate modern RAG-style retrieval as a **lens refinement step**, blending quantitative similarity with KAG’s spatial reasoning.

---

**Design principles**

* **Structure first, language second.**
  We search *through meaning geometry*, not mere lexical overlap.
* **Convergence as evidence.**
  The more Classes meet in one region, the more likely that region reflects a genuine underlying phenomenon.
* **Scalability through integration.**
  Class-based retrieval and vector-based similarity can reinforce each other — the former offering explainability, the latter, reach.

---

### 2) Subjective Search — *Borrow a pair of eyes*

**Intent:** You care about **experience** and **context**. Place an **observer** somewhere in the space (a person, a role, a setting, a moment) and see the world from there.

**How it feels conceptually**

* **Pick a vantage.** The observer’s coordinate can be a specific Instance (e.g., *Person A at Tokyo, modern*) or a composite (e.g., *18th-century Parisian*).
* **Look around (near HFs).** What actually happens **within arm’s reach** of this observer? That’s lived reality.
* **Name the room (context Classes).** Which Classes **enclose** this observer’s position with enough presence (power)? That’s the **ambient context**—the room you’re in, not just the objects you touch.
* **Peek beyond the wall (peripheral signals).** What’s **just outside** your radius—close enough to be rumor, media, or emerging relevance?

**Design principles**

* **Context is not an afterthought.** We explicitly retrieve the **Classes that contain you**, not only the events near you. The *room* matters as much as the *things*.
* **Periphery matters.** Humans don’t just perceive the immediate; they sense **nearby possibilities**. Retrieval should too.
* **Bias made visible.** By choosing a vantage, you **declare a bias**. That’s healthy—KAG makes it *explicit* rather than hidden in a vector.

**Useful knobs**

* **Radius.** Tight radius = intimate diary; wide radius = street reportage.
* **Power threshold.** How “thick” a Class’s presence must be to count as context.
* **Balance of near vs. peripheral.** Controls how curious your observer is.

**Failure modes we accept**

* **Parochial views.** A narrow radius may miss the bigger story. That’s fine—move the observer or widen the circle and try again.

---

### Choosing a Mode

* Use **Objective** when your intent is **top-down** (“give me HFs about *this* phenomenon”).
* Use **Subjective** when your intent is **inside-out** (“what’s the world like **from here**?”).
* In practice, **mix both**: start objectively to locate the neighborhood, then switch to a subjective pass from a representative vantage within it.

---

### Why this matters for RAG

KAG reframes retrieval as **wayfinding**:

* **Distance** becomes *aboutness* (near = relevant).
* **Density** becomes *salience* (many instances = influential).
* **Containment** becomes *context* (inside = framing meaning).

This makes the retriever’s behavior **legible**: instead of opaque scores, you can point to a map and say, *“We stood here, where these Classes converge; we pulled events from this neighborhood; and these broader Classes formed the room we were in.”*

---

## Quickstart

### Requirements

* Python **3.9+**
* `numpy`

### Install

```bash
pip install -r requirements.txt
# (or) pip install numpy
```

### Run the demo

```bash
python kag.py
```

You’ll see:

* Optimization stats
* A few HF and Class summaries
* Objective search results for a small target set
* A subjective search from a synthetic “18th century Paris” observer

---

## Minimal Usage Example

```python
from kag import KnowledgeSpace

# 1) Build a space with explicit axes (time, space, semantics)
ks = KnowledgeSpace(dim=3, axes=["time", "space", "semantics"], seed=7)

# 2) Add anchor classes (stabilize the geometry)
ks.add_class("time_18C", "18th century", vec=[-1.0, 0.0, 0.0], is_anchor=True)
ks.add_class("place_paris", "Paris",       vec=[-0.8, 0.6, 0.1], is_anchor=True)

# 3) Add conceptual classes
ks.add_class("class_declaration_of_independence", "U.S. Declaration of Independence")
ks.add_class("class_france", "France")

# 4) Add Hypothetical Facts (weighted by confidence/source count/etc.)
ks.add_hf(
    "hf_doI_1776",
    "1776/7/4: Declaration adopted in Philadelphia",
    class_weights={
        "time_18C": 1.0,
        "class_declaration_of_independence": 1.0,
    },
)

ks.add_hf(
    "hf_paris_revolutionary_ideas",
    "18C Paris: Revolutionary ideas spread",
    class_weights={
        "time_18C": 1.0, "place_paris": 0.9, "class_france": 0.8,
    },
)

# 5) Optimize (alternate HF & Class updates with damping and scale control)
ks.optimize(max_iters=300, alpha=0.35, tol=1e-4,
            scale_ref="anchors", target_mean_dist=1.0, verbose=False)

# 6) Inspect
print(ks.describe_hf("hf_doI_1776"))
print(ks.describe_class("class_declaration_of_independence"))

# 7) Retrieve: Objective search by class-set and proximity
hits = ks.objective_search(
    target_classes=["class_declaration_of_independence"], k=5, radius_factor=1.25
)
for key, score in hits:
    print(key, score)
```

---

## API Sketch

> See `kag.py` for full docstrings.

### Construction

* `KnowledgeSpace(dim=3, axes=[...], seed=42)`
* `add_class(key, name=None, vec=None, is_anchor=False)`
* `add_hf(key, name, class_weights: Dict[str, float])`

### Optimization

* `optimize(max_iters=200, alpha=0.3, anchor_reg=0.0, tol=1e-4, scale_ref="anchors", target_mean_dist=1.0, verbose=False)`

### Metrics

* `class_centroid(class_key) -> np.ndarray | None`
* `class_influence(class_key, eps=1e-6) -> {centroid, radius, volume, power}`

### Retrieval

* `objective_search(target_classes: List[str], k=10, radius_factor=1.0, min_class_power=0.0) -> List[Tuple[str, float]]`
* `subjective_search(observer_pos: np.ndarray, k_near=10, radius=1.0, power_threshold=0.0, k_context=5) -> Dict`

### Introspection

* `describe_class(class_key) -> str`
* `describe_hf(hf_key) -> str`

---

## Design Notes

* **Anchors vs. flexibility.** Use anchors for stability, but keep **α** and **regularization** modest to allow HFs to pull Classes where evidence suggests.
* **Weights = trust.** Put more weight on reliable sources; set unknown/low-trust elements to small or zero.
* **Scale normalization.** After each iteration, rescale to keep distances meaningful across runs.
* **Dimensionality.** Start with (D=3); add axes only if they provide continuity and broad coverage (per the 1–3 criteria above).

---

## Extending KAG

* **Better radii:** Use true minimal enclosing balls (Welzl’s algorithm) instead of max-distance-to-centroid.
* **Axis learning:** Learn axes from data (e.g., time/place inferred from text) while still allowing anchors.
* **LLM-assisted HF extraction:** Turn Representations (docs, transcripts, media) into HFs with provenance.
* **Confidence propagation:** Let HF-level confidence affect Class updates and retrieval scores.
* **Multi-index fusion:** Hybridize KAG geometry with classic dense retrieval for production RAG.

---

## Limitations

* **Approximate geometry.** The current radius/volume approximations are simple and may over/under-estimate influence.
* **Input sensitivity.** Anchors and initial scales can affect convergence; inspect with `verbose=True`.
* **No automatic HF extraction.** This prototype assumes HFs and Class membership are already identified.

---

## Project Structure

```
.
├── kag.py                # Reference implementation & demo runner
├── requirements.txt      # numpy (and friends if you add them)
└── README.md             # This file
```

---

## Contributing

Issues and PRs are welcome! Please:

* Keep PRs small and well-documented.
* Add or update tests/demos where relevant.
* Describe design tradeoffs in the PR body.

---

## License

This project is licensed under the **MIT License**.
Copyright (c) 2025 H.Kiriyama
See the [LICENSE](LICENSE) file for details.

SPDX-License-Identifier: MIT

---

## Citation

If you use KAG in academic or industrial work, please cite this repository and concept:

```
@software{KAG_2025,
  title   = {KAG: Knowledge as Geometry},
  author  = {H.Kiriyama},
  year    = {2025},
  license = {MIT},
  url     = {https://github.com/shinmaruko1997/kag}
}
```

---

## Appendix: Demo Snippets

**Build & optimize a demo space:**

```python
from kag import build_demo_space

ks = build_demo_space()
stats = ks.optimize(max_iters=300, alpha=0.35, tol=1e-4,
                    scale_ref="anchors", target_mean_dist=1.0)
print("Optimization stats:", stats)
```

**Objective search over a class set:**

```python
hits = ks.objective_search(
    ["class_declaration_of_independence", "class_united_states"],
    k=5, radius_factor=1.25
)
for k, s in hits:
    print(f"{k}: {s:.3f} -> {ks.hfs[k].name}")
```

**Subjective search from an observer near “18C Paris”:**

```python
import numpy as np

center = np.mean(np.stack([
    ks.classes["time_18C"].vec,
    ks.classes["place_paris"].vec
]), axis=0)

view = ks.subjective_search(observer_pos=center,
                            k_near=5, radius=0.6,
                            power_threshold=0.001, k_context=5)

print("Near HFs:", view["near_hfs"])
print("Context Classes:", view["context_classes"])
print("Peripheral:", view["peripheral"])
```

---

## Discussion and Next Steps

This prototype demonstrates the core mechanics of KAG — knowledge as a geometric space.  
The next directions include:
- Formal convergence analysis and loss formulation  
- Integration with open-source RAG pipelines  
- Empirical evaluation on public datasets  
- Visualization and interaction tools  

Contributions and critiques are very welcome.

---

> “Distance is relationship; density is influence.” — KAG