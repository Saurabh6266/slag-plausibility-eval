# Dataset Card — Slag Physical Plausibility Benchmark (SLPB-50)

## Dataset Summary

SLPB-50 is an adversarial evaluation dataset for testing physical plausibility reasoning
in language models, specifically in the domain of **pyrometallurgy and slag chemistry**.

It consists of **50 matched question pairs** (valid + invalid with one planted error each),
each presented in **3 phrasing variants**, yielding **300 total prompts**.

## Dataset Details

- **Languages:** English
- **Domain:** Pyrometallurgy, Slag Chemistry, Physical Chemistry of Melts
- **Task:** Binary physical plausibility detection (VALID / INVALID)
- **Size:** 50 pairs × 2 versions × 3 phrasings = 300 prompts

## Error Categories

| Category | Count | Description |
|----------|-------|-------------|
| `SUM_VIOLATION` | 10 | Oxide percentages sum to > 100% |
| `BASICITY_INCONSISTENCY` | 10 | Stated basicity ratio doesn't match oxide data |
| `VISCOSITY_SIGN_ERROR` | 10 | Viscosity direction claimed to be opposite to reality |
| `THERMODYNAMIC_ERROR` | 10 | Equilibrium or partition relationship reversed |
| `MASS_BALANCE_ERROR` | 10 | Unbalanced reaction or incorrect stoichiometry |

## Phrasing Variants

Each pair is presented in three styles:
- **Phrasing A** — Direct technical question
- **Phrasing B** — Expert verification framing ("a process engineer states…")
- **Phrasing C** — Teaching/mentorship scenario ("a student writes…")

## Data Schema

```json
{
  "id": "Q001",
  "category": "SUM_VIOLATION",
  "domain": "blast furnace slag",
  "valid": {
    "id": "Q001V",
    "core": "...",
    "ground_truth_verdict": "VALID",
    "ground_truth_note": "...",
    "phrasing_A": "...",
    "phrasing_B": "...",
    "phrasing_C": "..."
  },
  "invalid": {
    "id": "Q001I",
    "core": "...",
    "planted_error": "...",
    "planted_error_type": "...",
    "ground_truth_verdict": "INVALID",
    "ground_truth_note": "...",
    "phrasing_A": "...",
    "phrasing_B": "...",
    "phrasing_C": "..."
  }
}
```

## Intended Use

- **Primary use:** Evaluating physical plausibility reasoning in LLMs
- **Secondary use:** Data for contrastive fine-tuning to improve constraint-checking

## Limitations

- Covers only five error types in one sub-domain of materials science
- Questions are in English only
- All errors are synthetically planted (not drawn from real industrial reports)

## Creation Process

Questions were designed by a materials engineering researcher with expertise in
pyrometallurgy. Every valid question was verified against standard slag chemistry
references (SLAG ATLAS, Turkdogan's "Physical Chemistry of High Temperature Technology",
Gaskell's "Introduction to the Thermodynamics of Materials"). Every invalid question
was checked to ensure exactly one clearly identifiable physical error.
