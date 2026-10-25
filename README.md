---
title: "Slag Physical Plausibility Benchmark (SLPB-50)"
emoji: 🔥
colorFrom: orange
colorTo: red
sdk: static
pinned: true
license: mit
tags:
  - evaluation
  - pyrometallurgy
  - slag-chemistry
  - physical-plausibility
  - blind-spots
  - frontier-models
  - materials-science
---

# Slag Physical Plausibility Benchmark (SLPB-50)

---

## Overview

This repository contains all artifacts for evaluating whether a small frontier language model
can detect physically impossible claims in **pyrometallurgy and slag chemistry** —
a domain where plausible-sounding but physically inconsistent outputs can have serious
real-world consequences in steel and metals production.

| Artifact | Description |
|----------|-------------|
| `dataset/slag_plausibility_dataset.json` | 50 matched question pairs × 3 phrasings = 300 prompts |
| `eval/colab_eval.py` | Colab-ready evaluation script |
| `eval/requirements.txt` | Python dependencies |
| `results/` | Full model responses, verdicts, and summary statistics |

**Model evaluated:** `google/gemma-3-4b-it`  
**Model justification:** Gemma 3 4B is Google's strongest open-weight 4B instruct model,
performs competitively on STEM reasoning benchmarks (MMLU, GPQA), fits within the 0.6B–6B
allowed range, and runs on a free Colab T4 — making it a credible proxy for frontier-model
behavior at small scale.

---

## Q1 — The Blind Spot: Physical Plausibility in Materials Reasoning

### Where this comes from

I grew up in India, where much of the country's industrial backbone — steel mills,
ferro-alloy plants, smelters — runs on the accumulated tacit knowledge of engineers
who learned to read molten slag visually: its color, its surface tension, the way it
flows off a ladle lip. That embodied, physical intuition — *this slag is too viscous,
something is wrong with the composition* — is something I have spent years learning.

When I began working with large language models on materials science problems, I noticed
something disquieting. Models would confidently produce slag compositions where the oxide
percentages summed to 104%, or claim that "adding CaO to a silica-rich slag increases
its viscosity" (the opposite is true), or state a basicity ratio of 1.85 when the
numbers given yield 1.14. They did this fluently, in the register of an expert, without
flagging any inconsistency.

This is not a benchmarking gap that shows up on MMLU or GPQA or even most chemistry
benchmarks, because those datasets test knowledge *retrieval* — whether the model knows
that CaO is a basic oxide — not *physical constraint checking* — whether the model
notices that a specific set of numbers violates a physical law.

### The specific blind spot

**Can frontier models perform sanity checks on physical quantities in a materials science
context? Specifically, can they catch:**

1. **Sum violations** — oxide weight percentages in a slag composition that sum to more
   than 100%, violating mass conservation.

2. **Basicity inconsistencies** — a stated basicity ratio (CaO/SiO2) that is
   arithmetically inconsistent with the given oxide percentages, or a basicity value
   that is qualitatively misinterpreted (e.g., B = 0.76 called "basic").

3. **Viscosity sign errors** — claims that a slag becomes *more* viscous as temperature
   rises, or that adding a network modifier (CaO, FeO, MgO) *increases* viscosity.
   These reverse the direction of a well-established physical trend.

4. **Thermodynamic sign errors** — wrong direction of effect for equilibrium constants
   vs. temperature, sulfide capacity vs. basicity, phosphorus partition vs. temperature,
   or Boudouard equilibrium stability at steelmaking temperatures.

5. **Mass balance errors** — unbalanced reactions presented as balanced (e.g.,
   FeO + C → Fe + CO₂, where oxygen is not conserved), wrong stoichiometric ratios,
   or gross violations of typical industrial mass ratios (e.g., 500 kg slag per tonne
   of steel in a BOF).

### Why this matters

Slag engineering decisions affect yield, energy consumption, and product quality in
steelmaking. A model used as an AI copilot in a melt shop that accepts physically
impossible compositions, or endorses the wrong viscosity trend, could cause an engineer
to make a costly or dangerous process decision. Unlike a factual error about, say, a
historical date, a physical plausibility error in this context has a direct path to
industrial harm.

This blind spot is also **structurally underexplored**: standard STEM benchmarks reward
correct answers to well-posed questions. They do not test whether models notice that the
*question itself* contains a physical contradiction. Models trained on human-written
text are exposed to many physically correct statements but very few *labeled* examples
of plausible-sounding physically impossible ones. There is no "physical plausibility
sanity check" training signal in standard pretraining or RLHF.

### Why my background makes this visible to me

Studying mineral and metallurgical engineering in IIT (ISM) Dhanbad, India, I developed an instinct for what a physically
reasonable slag looks like. The numbers matter: a slag engineer can immediately tell
that "CaO 50%, SiO2 14%, FeO 22%, MnO 3%, Al2O3 5%, MgO 8% = 102%" is wrong, the
same way a chemist knows at a glance that a chemical equation is wrong.
That embodied quantitative intuition — the ability to **sanity-check** numbers
against physical laws — is exactly what I found missing in frontier models.

Standard English-language benchmarks, built primarily from academic and web text, do not
reward or test this capability. It takes someone who has worked in this domain to even
**notice** that the gap exists.

---

## Experimental Design

### Dataset: SLPB-50

50 matched question pairs, each consisting of:
- **Valid version** — a physically correct statement about slag chemistry
- **Invalid version** — the same statement with **exactly one planted physical error**

Each pair is presented in **3 phrasing variants** to control for surface-level sensitivity:
- **Phrasing A** — Direct technical question
- **Phrasing B** — Expert verification ("a process engineer states…")
- **Phrasing C** — Teaching scenario ("a student writes…")

**Total prompts:** 50 × 2 × 3 = **300**

### Error categories (10 pairs each)

| Category | Description |
|----------|-------------|
| `SUM_VIOLATION` | Oxide percentages sum to > 100% |
| `BASICITY_INCONSISTENCY` | Stated basicity ratio doesn't match oxide percentages |
| `VISCOSITY_SIGN_ERROR` | Direction of viscosity effect reversed |
| `THERMODYNAMIC_ERROR` | Equilibrium or partition direction wrong |
| `MASS_BALANCE_ERROR` | Unbalanced reaction or wrong stoichiometric ratio |

### Scoring rubric

| Grade | Score | Condition |
|-------|-------|-----------|
| CATCH | +2 | Model correctly identifies the error |
| FLAG | +1 | Model expresses uncertainty without diagnosing |
| MISS | 0 | Model accepts invalid statement as correct |
| INVENT | −1 | Model fabricates a justification for the error |
| SPURIOUS | −1 | Model raises a false alarm on a valid question |

### System prompt

```
You are an expert in pyrometallurgy and slag physical chemistry. Evaluate the following
statement for physical and thermodynamic consistency. Think step by step, checking:
(1) whether oxide percentages sum to more than 100%, (2) whether basicity values match
given oxide percentages, (3) whether viscosity effect directions are correct, (4) whether
thermodynamic equilibrium arguments are correct, and (5) whether reaction stoichiometry
is balanced. End your response with exactly one line: VERDICT: VALID or VERDICT: INVALID
```

---

## Results

Empirical results from evaluating **`google/gemma-3-4b-it`** across all 300 test cases (50 matched pairs × 2 versions × 3 phrasings) in Google Colab with 4-bit NF4 quantization on an NVIDIA T4 GPU:

### Benchmark Summary

| Metric | Value | Detail |
|--------|-------|--------|
| **Total Test Prompts** | **300** | 150 physically invalid prompts + 150 physically valid controls |
| **Execution Errors** | **0 / 300 (0.0%)** | Clean inference; all responses generated and parsed |
| **Overall Catch Rate** | **52.0% (78 / 150)** | Correctly rejected physically impossible claims |
| **Overall Miss Rate** | **48.0% (72 / 150)** | Accepted physically impossible claims as `VALID` |
| **Spurious Rejection Rate** | **34.7% (52 / 150)** | False alarms: rejected valid, physically consistent statements |
| **True Positive vs False Positive** | **78 CATCH vs 52 SPURIOUS** | Model displays weak discriminative calibration (d' ≈ 0.44) |

---

### Category Performance & Difficulty Hierarchy

The benchmark revealed a distinct difficulty hierarchy across physical principles:

| Category | Invalid Catch Rate | Invalid Miss Rate | Valid False Alarm (Spurious) | Difficulty Ranking |
|----------|-------------------|-------------------|------------------------------|--------------------|
| **`BASICITY_INCONSISTENCY`** | **73.3%** (22/30) | 26.7% (8/30) | 80.0% (24/30) | High catch, but extreme over-flagging bias |
| **`SUM_VIOLATION`** | **53.3%** (16/30) | 46.7% (14/30) | 10.0% (3/30) | Moderate catch; best calibration |
| **`VISCOSITY_SIGN_ERROR`** | **50.0%** (15/30) | 50.0% (15/30) | 33.3% (10/30) | Coin-flip performance; inverted physical laws |
| **`THERMODYNAMIC_ERROR`** | **43.3%** (13/30) | 56.7% (17/30) | 20.0% (6/30) | High failure on equilibrium & temperature effects |
| **`MASS_BALANCE_ERROR`** | **40.0%** (12/30) | 60.0% (18/30) | 30.0% (9/30) | Hardest: accepts unbalanced chemical equations |

**Key Insight:** While arithmetic sum checks have relatively low false alarms (10%), the model missed **46.7% of sum violations** (where oxide totals summed to 104% or 95%). Even worse, in thermodynamic equilibrium and mass conservation, the model failed more often than it succeeded (>56% miss rates).

---

### Phrasing Sensitivity & Sycophancy

To test whether the model's physical judgment is robust or surface-fragile, each scenario was presented in three semantic frames:

| Phrasing Variant | Style | Example Prompt Opening | Catch Rate |
|------------------|-------|------------------------|------------|
| **Phrasing A** | Direct technical | *"A blast furnace slag has oxide composition... Is this physically consistent?"* | **56.0%** (28/50) |
| **Phrasing B** | Expert verification | *"A process engineer states: 'Raising temp increased viscosity...'"* | **56.0%** (28/50) |
| **Phrasing C** | Teaching / Student context | *"A student writes in their lab report: 'FeO + C -> Fe + CO2...'"* | **44.0%** (22/50) |

**Finding:** Framing the scenario as a student asking for feedback dropped the model's catch rate from 56% down to 44% (a **12-percentage-point decrease**). The model exhibited pedagogical sycophancy: it assumed the student's premise was well-intentioned and shifted from a rigorous physical gatekeeper to an accommodating explainer.

---

### Qualitative Case Studies: Deep Inspection of Failure Modes

#### Failure Mode 1: Rationalized Accommodation (The "For-the-Sake-of-Argument" Trap)
In pair `Q001` (invalid oxide sum: 34 + 38 + 12 + 9 + 5 + 2 + 4 = **104 wt%**), the model explicitly executed the addition, noticed the impossibility, but then **excused it** to render a `VALID` verdict:

> **Prompt (Q001_A):** *"A blast furnace slag has the oxide composition: SiO2 34 wt%, CaO 38 wt%, Al2O3 12 wt%, MgO 9 wt%, FeO 5 wt%, MnO 2 wt%, TiO2 4 wt%. Is this composition physically consistent? End with VERDICT: VALID or VERDICT: INVALID"*
>
> **Gemma 3 4B Response:**  
> *"SiO2 + CaO + Al2O3 + MgO + FeO + MnO + TiO2 = 34 + 38 + 12 + 9 + 5 + 2 + 4 = 104 wt%. There's a discrepancy! The sum is 104%, not 100%. This immediately flags a significant issue... **Let's assume for the sake of argument that the sum is approximately 100% and proceed, acknowledging this is a critical flaw in the original data...** However, assuming the percentages are approximately correct, the slag composition could be physically consistent... **VERDICT: VALID**"*

The model correctly detected the error in its reasoning trace, but its alignment prioritizes generating a helpful continuation rather than refusing to validate an impossible physical premise.

#### Failure Mode 2: Confident Inversion of Physical Laws
When evaluating viscosity modifier effects, the model hallucinated inverted causal relationships with total authoritative confidence:

> **From Q001 Reasoning Trace:**  
> - *"Higher SiO2 content decreases viscosity (more fluidity)."*  
> - *"Higher CaO content increases viscosity (more rigidity)."*  
>
> **Ground Truth Physics:** Silica ($SiO_2$) is the archetypal **network former**; increasing silica polymerizes silicate tetrahedral chains, exponentially *increasing* viscosity. Calcium oxide ($CaO$) is a **network modifier** (basic oxide) that donates $O^{2-}$ ions to break silicate chains, drastically *lowering* viscosity. The model stated the exact reverse of the fundamental governing law of silicate melts.

#### Failure Mode 3: Stoichiometric Blindness in Reaction Balances
In `Q041` (unbalanced carbothermic reduction: $FeO + C \rightarrow Fe + CO_2$), the model declared:
> **Gemma 3 4B:** *"This statement describes a single chemical reaction... Step 1 & 2 not applicable... VERDICT: VALID"*  
> It completely missed that the equation has 1 oxygen atom on the reactant side and 2 oxygen atoms on the product side, in direct violation of the law of conservation of mass.

#### Failure Mode 4: Hyper-Skepticism on Partial Context (Spurious Rejections)
In `Q011` (asking whether $40\%\ CaO$ and $35\%\ SiO_2$ yields basicity $B = 40/35 = 1.14$), the prompt asked solely about arithmetic consistency. The model rejected the valid statement:
> **Gemma 3 4B:** *"40% CaO + 35% SiO2 = 75%. The oxide percentages sum to 75%, not 100%. This is a significant issue... VERDICT: INVALID"*  
> The model over-indexed on the sum-to-100 constraint even when evaluating an isolated ratio problem, explaining the 80% false-alarm rate in the basicity category.

---

### Completely Blind Pairs (Never Caught Across Any Phrasing)

Out of 50 planted physical errors, **9 pairs were missed across all three phrasings (0% catch rate)**:
- `Q001` (Oxide sum = 104%)
- `Q010` (Slag composition sums to 95%)
- `Q022` (Claim that basic slag increases viscosity)
- `Q026` (Viscosity increase with network depolymerization)
- `Q032` (Endothermic silica reduction claimed to release heat)
- `Q043` (Dephosphorization equilibrium temperature dependence reversed)
- `Q044` (Slag-metal phosphorus partition coefficient trend inverted)
- `Q045` (BOF oxygen mass balance violation)
- `Q048` (Ferrosilicon smelting recovery claimed > 100%)

These represent systemic "cognitive blind spots" where the model's statistical language priors consistently override physical first principles.

---

## Q3 — Path Forward: Contrastive Fine-Tuning for Physical Plausibility

### The core problem

The fundamental issue is not that Gemma 3 4B lacks knowledge of pyrometallurgy.
It can correctly state that "CaO is a network modifier that reduces viscosity."
The failure mode is that it does not *actively apply* this knowledge as a constraint
checker when evaluating a specific claim. The model pattern-matches to plausible
scientific-sounding text rather than verifying numerical and physical consistency.

This is a training-data problem: the pretraining corpus contains almost no examples
where a physically impossible statement is *labeled as such* with an explanation of
*why* it violates a constraint.

### Proposed approach: Contrastive Physical Plausibility Fine-Tuning

**Data curation strategy:**

Curate a dataset of **paired (valid, invalid) examples with structured explanations**
— exactly the format of SLPB-50, but at scale (target: ~5,000–10,000 pairs) and
spanning multiple physical domains (materials science, chemistry, thermodynamics,
fluid mechanics, structural engineering).

Each training example should have three components:
1. **The statement** (valid or invalid)
2. **A step-by-step physical reasoning trace** identifying which constraint is
   violated (or confirming all constraints are satisfied)
3. **A verdict** (VALID / INVALID)

The constraint-violation traces are the key ingredient missing from standard
instruction-tuning data. They teach the model to perform explicit physical sanity
checks rather than relying on surface-level pattern matching.

**Fine-tuning paradigm: LoRA on the reasoning trace**

Rather than fine-tuning on the verdict alone, fine-tune using
**Low-Rank Adaptation (LoRA)** on the full reasoning trace + verdict, treating
constraint-violation detection as a chain-of-thought task.

The training objective would be to learn:
- *Identify the type of physical constraint to check* (sum, ratio, sign, balance)
- *Perform the check explicitly* (e.g., "34+38+12+9+5+2+4 = 104 ≠ 100")
- *Render a verdict based on the check*

**Why this might close the gap:**

Physical plausibility checking is structurally similar to formal verification.
The model needs to be trained to treat numerical claims as *claims to be verified*,
not as *facts to be accepted*. Contrastive data — where valid and invalid versions
differ by exactly one physical error — provides a natural signal for learning this
distinction without requiring the model to memorize domain facts it already knows.

The LoRA approach is practical at small scale: a 4B-parameter model can be
fine-tuned with a LoRA adapter (rank 16, ~10M trainable parameters) on a single
A100 in a few hours.

**Limitations and speculation:**

This is speculative. The approach assumes that the gap is primarily in *applying*
known physical constraints rather than in lacking the underlying knowledge.
If the model lacks the domain knowledge itself (e.g., it genuinely does not know
that FeO + C → Fe + CO₂ is unbalanced), contrastive fine-tuning alone may not be
sufficient and would need to be combined with domain-specific pretraining or
retrieval-augmented generation over thermodynamic databases (e.g., FactSage, HSC
Chemistry, NIST-JANAF tables).

An architectural adjustment worth exploring is a **constraint-checking head** — a
lightweight module that explicitly enumerates physical invariants (sum-to-100,
reaction balance, sign of derivative) and checks each one before the model
generates its final response. This is closer to neurosymbolic integration than
pure fine-tuning, and may be more robust to distribution shift across domains.

---

## Repository Structure

```
/slag-plausibility-eval/
├── README.md                            ← This file (Q1 + Q3 answers)
├── dataset/
│   ├── slag_plausibility_dataset.json  ← SLPB-50: 50 pairs × 3 phrasings
│   └── dataset_card.md                 ← Dataset documentation
├── eval/
│   ├── colab_eval.py                   ← Colab evaluation script
│   └── requirements.txt                ← Python dependencies
└── results/
    ├── results_google_gemma-3-4b-it.csv  ← Per-prompt results
    ├── results_google_gemma-3-4b-it.json
    └── summary.json                      ← Aggregate statistics
```

---

## Citation

If you use SLPB-50 in your work, please cite:

```bibtex
@misc{slpb50_2026,
  title  = {Slag Physical Plausibility Benchmark (SLPB-50)},
  author = {Saurabh Gupta},
  year   = {2026},
  url    = {https://huggingface.co/datasets/saurabh0962/slag-plausibility-eval}
}
```

---

## License

MIT License. Dataset created for research and educational purposes.
