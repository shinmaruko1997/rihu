# RIHU — Retrieval In the Hypothetical Universe

> A search paradigm built upon **KAG (Knowledge as Geometry)** — navigating meaning through geometry.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#license)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](#quickstart)

---

## TL;DR

Traditional text and knowledge models analyze *flat projections* — patterns that appear when the rich, three-dimensional structure of facts and meanings is cast onto the surface of language.Even when these models use thousands of dimensions to describe those surfaces, what they explore remains essentially **the geometry of the shadow**, not the form of what casts it.  
**RIHU** takes a different approach: it aims to **approximate and reconstruct the underlying semantic structures** that language projects.
Instead of comparing texts within a flattened embedding space, RIHU builds an interpretable *coordinate system* in which facts and concepts appear as **points and regions** along explicit, human-readable axes.  
This shift — from analyzing the “map of shadows” to reasoning within the “space of meaning” — enables RIHU to describe not only *how close* things are, but also *in which direction and for what reason*.

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

## Core Concepts

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
Each **HF** is located within this coordinate system according to the structure of Classes and Instances that define it.

### Good axes: choosing and defining the frame

Good axes are those that preserve interpretability and continuity, providing stable reference for HF positioning.

Good axes:

1. Are abstractions of Classes (e.g., “location”, “organization”, “time”).
2. Connect many other Classes with continuity.
3. Provide large, interpretable “containers” of meaning.

**Anchor Classes** (like time bins or known locations) stabilize the coordinate frame via regularization and alignment.

> In practice, **axes and anchors can also be suggested or initialized by large language models (LLMs)** —
> for example, by inferring temporal, spatial, or semantic structures from text or datasets.
> This allows partial automation of Universe construction while maintaining geometric interpretability.

---

## Universe Construction

> The **Universe** in KAG corresponds to what a traditional RAG system would call an **index** — the structured space in which knowledge is stored and retrieved.
> However, rather than a static vector index, KAG uses the term *Universe* to emphasize its **geometric and dynamic nature**:
> a manifold of Hypothetical Facts, Classes, and Instances, whose relationships are explicitly modeled as geometry.

The following section outlines how these geometric ideas are implemented concretely in RIHU’s Python prototype.

1. **Propose axes** (LLM, single call, **K in [3,10]**; if `dims` is unset, the working dimensionality defaults to K and is later clamped to the available embedding rank and sample count).
2. **Propose axis bins** (LLM, per axis, **3–7 discrete bins**) — short, interpretable patterns (label/description/prototypes) along each continuous axis.
3. **Extract HFs & Classes** (LLM-assisted with fallbacks).
4. **Merge near-duplicates**
   4.1 **Mechanical merge** (name normalization + alias consolidation).
   4.2 **Embedding-based blocking** to propose candidate class pairs → **auto-merge high-confidence pairs**, **LLM disambiguation** for ambiguous pairs.
   4.3 *(Optional)* **LLM consolidation** over remaining Classes.
5. **Initial coordinates** via text embeddings + **SVD projection** to **D** dimensions (D chosen per step 1 and **clamped** to embedding rank / sample count).
6. *(Optional)* **Anchor targets & one-shot affine alignment** — LLM proposes target coordinates for anchors within a bounded range; solve a single affine map; **anchors remain fixed** thereafter.
7. **Iterative relaxation** (HFs → class centroids; classes → instance-HF centroids; **anchors do not move**).
8. **Scale normalization** — **prefer anchors** when available; otherwise use all classes.
9. **Metrics** — centroid / radius (enclosing-ball approx) / volume / power.
10. **Axis-bin interpretability scoring** — embed HF/class signatures vs. bin prototypes to record per-axis soft memberships and top-bin labels for HFs and Classes.

---

## Alternative Universe Construction (Draft)

A Class can be viewed as a *continuous presence* that emerges from the distribution of factual “particles” within the data.
When many Hypothetical Facts cluster around similar semantic or structural patterns, the outline of a Class begins to appear naturally.
This suggests that Classes need not be invented by a model; rather, they can be *discovered* through the observable regularities in how facts co-occur and organize themselves.

With this perspective, we are exploring methods to identify Classes and axes in ways that are as **stable**, **reproducible**, and **LLM-independent** as possible.
Instead of relying on a generative model to propose semantic structures, we we aim to let geometric and statistical patterns to surface first — and bring in an LLM only at the final stage, for optional naming or summarization.

This is an exploratory direction and not yet integrated into the main RIHU pipeline.
The following outlines an in-progress approach toward this goal.

### Overview of a More Reproducible Class and Axis Discovery Workflow

1. **Fact–Class Geometry First**
   Begin from HF–Class incidence patterns (co-occurrence matrices, TF-IDF profiles, PMI weighting), allowing structure to arise from actual data distributions rather than model-generated abstractions.

2. **Soft Clustering of HFs**
   Use overlapping or probabilistic clustering (e.g., GMM/HDBSCAN/NMF-based profiles) so that HFs can participate in multiple semantic neighborhoods.
   This reflects the natural ambiguity and multidimensionality of real-world facts.

3. **Class Relationships via Shared Structure**
   Derive relatedness among Classes by examining how strongly they co-appear across HF groups — not through conceptual descriptions, but through measurable geometric patterns.

4. **Axes as Low-Rank Semantic Directions**
   Instead of predefined or LLM-proposed axes, infer axes directly from the latent structure (e.g., NMF or low-rank decompositions).
   These axes become interpretable by inspecting which Classes contribute most strongly to them.

5. **Multi-Aspect Classes**
   A Class can load onto several axes, reflecting its multiple facets (e.g., a person who is also a political entity, a location associated with a historical period).
   This multidimensionality emerges naturally from the factorization rather than being hand-assigned.

6. **Geometric Space Construction**
   With HFs and Classes embedded into a shared low-dimensional coordinate system, the Universe becomes navigable and inspectable without requiring generative inference at each step.

7. **Optional LLM Naming Layer**
   Only after the geometry is stable do we bring in an LLM — not to define Classes, but to *name* or summarize them using the top contributing HFs and Classes as evidence.
   This keeps the interpretability benefits of LLMs while minimizing their influence on the underlying structure.

### Intent

This direction aims to strengthen RIHU’s foundations by grounding the semantic geometry in reproducible statistical structure.
By letting Classes and axes *emerge* from factual distributions — and reserving LLM work for the final, human-facing layer — we encourage stability, interpretability, and transparency across the entire Universe Construction process.

---

## Geometry & Metrics

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
Instead of asking *“Which texts match?”*, we ask: **“Where should I stand in the geometry — and what becomes near, dense, or enclosing when I do?”**

RIHU supports two complementary retrieval modes:

* **Objective Retrieval (world-centric)** — search from the global structure of meaning.  
* **Subjective Retrieval (observer-centric)** — search from a specific vantage within the space.

### 1) Objective Retrieval

**Intent:**
Discover what the world “knows” about a topic, by locating regions where relevant Classes or HFs converge.

**Conceptual flow:**

1. **Determine the reference coordinates** — either  
   (a) identify the position (or centroid) of the target HF or Class by reference search (optionally using text-based vector retrieval within HFs), or  
   (b) estimate the position of the target information within the Universe using the same placement method applied to HFs.  
2. **Define the conceptual nucleus** — select key Classes or conceptual anchors (e.g., *Declaration of Independence*, *United States*).  
3. **Compute the centroid** — calculate the geometric center of these reference coordinates in the Universe.  
4. **Survey the neighborhood** — identify nearby HFs or regions with overlapping Class memberships.  
5. **Interpret density** — dense regions imply high influence or conceptual richness.


### 2) Subjective Retrieval

**Intent:** Explore the world **from within** — from the viewpoint of an observer in the KAG Universe.

**Conceptual flow:**

1. **Pick a vantage point** — select a coordinate in the observer space, using existing Classes (person, role, moment) as reference points.  
2. **Look around** — retrieve nearby HFs within a certain radius.  
3. **Identify context** — Classes enclosing the observer define the ambient context.  
4. **Scan the periphery** — what lies just beyond reach, signaling emerging relevance?

**Design principles:**

* **Context-first retrieval** — the “room” around the observer matters.  
* **Periphery awareness** — sense the nearby unknown.  
* **Bias transparency** — by declaring a vantage, bias becomes explicit.


### 3) Detective Search (Prototype — Not Yet Implemented)

**Intent:**
Infer **unobserved or hypothetical facts** by reasoning through the geometric relationships of known Classes — searching not for what *is*, but for what *must be*, given the structure of the Universe.

**Conceptual flow:**

1. **Identify involved Classes** — determine which Classes are likely to participate in the unseen fact (e.g., *detective*, *crime scene*, *witness*).  
2. **Estimate the geometric barycenter** — compute the centroid of these Classes’ coordinates to hypothesize the most probable region for the missing Hypothetical Fact (HF).  
3. **Survey nearby evidence** — collect related Classes, adjacent HFs, and contextual information within that region.  
4. **Invoke generative reasoning** — provide these collected elements (Classes, HFs, Context) to a generative model to infer the *most plausible unobserved fact*.

**Philosophical intent:**

* *Inference as retrieval* — the act of “searching” extends to reasoning about the unseen.  
* *Geometry as hypothesis space* — positional relationships suggest what *ought to exist*, even if not yet observed.  
* *Generative triangulation* — large language models act as geometric interpreters, connecting partial evidence into coherent hypothetical facts.


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

Retrieval becomes **legible**: you can literally *point to the region* in the KAG Universe where meaning was found.

---

## Quickstart

### Requirements

- Python **3.9+**
- Packages: `numpy`, `openai`, `tiktoken`  
  *(install automatically via `requirements.txt`)*

### Setup

1. Clone or download this repository, and make sure the following three files are in the **same folder**.  
Note that `sample.py` and `holmes.txt` are located in the **sample** directory.:

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
[objective_search]
  seeds: ['Sherlock Holmes', 'Irene Adler']
  - hf_hf1  score=0.500  dist=0.764  text=Holmes deduces that the child's disposition is abnormally cruel, which may come ...  classes=['Sherlock Holmes', 'Watson', 'Miss Hunter']

[subjective_search]
  vantage: Sherlock Holmes
  inside:
  - hf_hf13  score=0.632  dist=0.266  text=Holmes revealed that Sir George Burnwell and Mary fled together....  classes=['Sir George Burnwell', 'Mary']
  - hf_hf9  score=0.627  dist=0.275  text=Holmes expresses concern about the unusual conditions of the job offer and the p...  classes=['Winchester', 'Winchester', '£120', 'Violet Hunter', 'Jephro Rucastle']
  - hf_hf17  score=0.578  dist=0.384  text=Holmes disguised himself to gather information about Sir George Burnwell....  classes=['Holmes', 'Sir George Burnwell']
  - hf_hf14  score=0.569  dist=0.406  text=Arthur saw Mary carrying the coronet and handing it to someone....  classes=['Arthur', 'Mary', 'coronet']
  - hf_hf12  score=0.560  dist=0.429  text=Holmes and Watson arrive at the Black Swan Hotel in Winchester to meet Violet Hu...  classes=['Black Swan Hotel', 'Winchester', 'lunch', 'Violet Hunter', 'Baker Street']
  periphery:
  - hf_hf4  score=0.363  dist=1.203  text=A fat man appears at the door with a heavy stick, indicating a confrontation is ...  classes=['fat man', 'heavy stick']
  - hf_hf16  score=0.362  dist=1.211  text=Holmes identified a struggle in the snow with drops of blood....  classes=['struggle', 'snow', 'blood']
  - hf_hf8  score=0.358  dist=1.232  text=Miss Violet Hunter is now the head of a private school at Walsall, where she has...  classes=['Miss Violet Hunter', 'Miss Violet Hunter']
  - hf_hf_10  score=0.351  dist=1.281  text=The young lady described a peculiar performance where Mr. Rucastle told her funn...  classes=['funny stories', 'window']
  - hf_hf_17  score=0.346  dist=1.310  text=The young lady discovered a locked door in the deserted wing of the house, which...  classes=['locked door', 'skylight']
```

All generated files will appear under the new folder:

```
holmes/universe.json
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

## License

This project is licensed under the **MIT License**.
Copyright (c) 2025 H.Kiriyama
See the [LICENSE](LICENSE) file for details.

SPDX-License-Identifier: MIT

---

## Discussion and Next Steps

This repository presents **KAG (Knowledge as Geometry)** as a new way to represent knowledge,
and **RIHU (Retrieval In the Hypothetical Universe)** as its first retrieval paradigm.

Future directions include:

* Integration with open-source RAG systems
* Empirical evaluation on public datasets
* Visualization and interactive exploration tools