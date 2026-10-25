# ============================================================
#  SLAG PHYSICAL PLAUSIBILITY BENCHMARK — COLAB EVALUATION
#  Model: google/gemma-3-4b-it
#  Dataset: SLPB-50 (50 pairs x 2 versions x 3 phrasings = 300 prompts)
# ============================================================

# ---------------------------------------------------------
# CELL 1 -- Install / upgrade dependencies
# ---------------------------------------------------------
# !pip install -q -U transformers accelerate bitsandbytes huggingface_hub
# NOTE: Requires transformers >= 4.50 for Gemma3ForConditionalGeneration.

# ---------------------------------------------------------
# CELL 2 -- HuggingFace login (Gemma 3 is a gated model)
# ---------------------------------------------------------
# from google.colab import userdata
# from huggingface_hub import login
# login(token=userdata.get('HF_TOKEN'))

# ---------------------------------------------------------
# CELL 3 -- Imports and configuration
# ---------------------------------------------------------
import json, re, time, csv, os, traceback
from datetime import datetime
import torch

# FIX #1 — WRONG MODEL CLASS
# google/gemma-3-4b-it is Gemma 3 — a multimodal model (text + SigLIP vision
# encoder). Its config.json stores vocab_size/hidden_size in a nested
# "text_config" key, NOT at the root. AutoModelForCausalLM.from_pretrained()
# cannot parse that nested structure and raises RuntimeError("") — an
# EMPTY-MESSAGE exception. That is exactly why all 300 prompts returned
# grade=ERROR with full_response="" in the previous run.
#
# Correct classes: Gemma3ForConditionalGeneration + AutoProcessor
from transformers import (
    AutoProcessor,
    Gemma3ForConditionalGeneration,
    BitsAndBytesConfig,
)

MODEL_ID       = "google/gemma-3-4b-it"
DATASET_PATH   = "slag_plausibility_dataset.json"  # upload to Colab Files panel
RESULTS_DIR    = "results"
MAX_NEW_TOKENS = 1536   # was 512 — Gemma 3 4B averages ~600 tokens per response;
                        # 512 cut off 83% of responses before the VERDICT: line

os.makedirs(RESULTS_DIR, exist_ok=True)

# FIX #2 — bfloat16 IS NOT SUPPORTED ON T4 (TURING, SM 7.5)
# Colab free tier allocates a Tesla T4 (Turing, Compute Capability 7.5).
# T4 does NOT have native bfloat16 hardware; bfloat16 requires SM >= 8.0
# (Ampere, Ada, Hopper). Setting bnb_4bit_compute_dtype=torch.bfloat16 on
# a T4 causes BitsAndBytes CUDA kernels to throw RuntimeError("") — again
# an empty-message exception — which was silently caught and stored as ERROR.
#
# Fix: detect GPU compute capability and select the dtype at runtime.
print(f"PyTorch : {torch.__version__}")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device  : {DEVICE}")

if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability(0)
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU     : {gpu_name}  (SM {major}.{minor})")
    if major >= 8:
        COMPUTE_DTYPE = torch.bfloat16
        print("Using bfloat16 (Ampere+ native support)")
    else:
        # FIX #5 — T4/float16 causes NaN logits -> model generates EOS immediately -> empty output
        # float16 is NOT stable enough for Gemma 3 sliding-window attention on Turing hardware.
        # Use float32 for the BnB compute dtype instead. The weights remain 4-bit NF4 (~4.5 GB);
        # float32 compute adds ~1.5 GB VRAM but prevents NaN collapse. T4 has 15 GB, fine.
        COMPUTE_DTYPE = torch.float32
        print("Using float32 compute dtype (T4/Turing — float16 causes NaN/EOS, float32 stable)")
else:
    COMPUTE_DTYPE = torch.float32
    print("CPU mode — using float32")

SYSTEM_PROMPT = (
    "You are an expert in pyrometallurgy and slag physical chemistry with deep "
    "knowledge of thermodynamics, reaction stoichiometry, and slag viscosity. "
    "Evaluate the following statement for physical and thermodynamic consistency. "
    "Think step by step, checking: (1) whether any oxide percentages sum to more "
    "than 100%, (2) whether any stated basicity values match the given oxide "
    "percentages, (3) whether the direction of any viscosity effect is correct, "
    "(4) whether any thermodynamic equilibrium arguments are correct, and (5) "
    "whether any reaction stoichiometry is balanced. "
    "End your response with exactly one line containing only: "
    "VERDICT: VALID\nor\nVERDICT: INVALID"
)

# ---------------------------------------------------------
# CELL 4 -- Load model (4-bit NF4) and processor
# ---------------------------------------------------------
# FIX #3 — DEPRECATED 'torch_dtype' KWARG
# transformers >= 4.50 deprecates torch_dtype= in from_pretrained().
# Use dtype= instead.

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=COMPUTE_DTYPE,   # float32 on T4 (float16 causes NaN/EOS), bfloat16 on Ampere+
    bnb_4bit_use_double_quant=True,
)

print("\nLoading processor ...")
processor = AutoProcessor.from_pretrained(MODEL_ID)

print("Loading model (4-bit NF4, ~3 GB VRAM on T4) ...")
model = Gemma3ForConditionalGeneration.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    dtype=COMPUTE_DTYPE,     # FIX: use 'dtype=' not deprecated 'torch_dtype='
)
model.eval()
print("Model loaded.\n")

# ---------------------------------------------------------
# CELL 5 -- Inference helper
# ---------------------------------------------------------
# FIX #4 — apply_chat_template MUST USE return_dict=True
# The previous code called apply_chat_template(..., return_tensors="pt") without
# return_dict=True. This returns a raw LongTensor (input_ids only). Passing only
# input_ids to model.generate() means attention_mask defaults to all-ones, which
# breaks Gemma 3's sliding-window attention — output is empty or garbage.
# return_dict=True returns {"input_ids": ..., "attention_mask": ...}.
# Unpacking with **inputs passes both correctly.

def run_prompt(user_text: str) -> str:
    """Run one prompt through Gemma 3 instruct chat template; return text response."""
    messages = [{"role": "user", "content": SYSTEM_PROMPT + "\n\n" + user_text}]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,       # returns {input_ids, attention_mask}
        return_tensors="pt",
    ).to(model.device)

    input_len = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,               # passes input_ids + attention_mask
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,        # greedy — fully deterministic, no temperature
        )

    # Slice off the prompt tokens; decode only the newly generated part
    generated = output_ids[0][input_len:]
    return processor.decode(generated, skip_special_tokens=True).strip()


# ---------------------------------------------------------
# CELL 6 -- DIAGNOSTIC: test ONE prompt before the full loop
#           If this cell fails, fix the error BEFORE running Cell 7.
# ---------------------------------------------------------
print("=" * 62)
print("  DIAGNOSTIC TEST")
print("=" * 62)

_test_prompt = (
    "A blast furnace slag has: SiO2 34 wt%, CaO 38 wt%, Al2O3 12 wt%, "
    "MgO 9 wt%, FeO 5 wt%, MnO 2 wt%, TiO2 4 wt%. "
    "Is this composition physically consistent? Explain step by step. "
    "End with VERDICT: VALID or VERDICT: INVALID"
)

try:
    t0 = time.time()
    _resp = run_prompt(_test_prompt)
    elapsed = round(time.time() - t0, 1)
    print(f"Passed in {elapsed}s\n")
    print("Model response:\n")
    print(_resp)
    print("\n" + "-" * 62)
    print("EXPECTED: VERDICT: INVALID  (34+38+12+9+5+2+4 = 104 > 100)")
    print("=" * 62)
except Exception:
    print("DIAGNOSTIC FAILED — full traceback:")
    traceback.print_exc()
    print("\nDo NOT run Cell 7 until the diagnostic passes.")
    raise


# ---------------------------------------------------------
# CELL 7 -- Verdict extraction and scoring helpers
# ---------------------------------------------------------
def extract_verdict(response: str) -> str:
    """Extract VERDICT: VALID or VERDICT: INVALID from the model's response."""
    hits = re.findall(r"VERDICT:\s*(VALID|INVALID)", response, re.IGNORECASE)
    if hits:
        return hits[-1].upper()
    lo = response.lower()
    li, lv = lo.rfind("invalid"), lo.rfind("valid")
    if li == -1 and lv == -1:
        return "NO_VERDICT"
    return "INVALID" if li > lv else "VALID"


def score_response(verdict: str, expected: str, is_valid_q: bool) -> dict:
    """
    Rubric
    ------
    INVALID questions (one error planted):
        CATCH   +2  model correctly says INVALID
        FLAG    +1  model says NO_VERDICT (expressed uncertainty)
        MISS     0  model wrongly says VALID (accepted the planted error)

    VALID questions (no error):
        CORRECT  +2  model correctly says VALID
        UNCERTAIN 0  model says NO_VERDICT
        SPURIOUS -1  model wrongly says INVALID (false alarm)
    """
    if is_valid_q:
        if verdict == "VALID":      return {"grade": "CORRECT",   "score": 2}
        if verdict == "NO_VERDICT": return {"grade": "UNCERTAIN",  "score": 0}
        return                             {"grade": "SPURIOUS",   "score": -1}
    else:
        if verdict == "INVALID":    return {"grade": "CATCH",      "score": 2}
        if verdict == "NO_VERDICT": return {"grade": "FLAG",       "score": 1}
        return                             {"grade": "MISS",       "score": 0}


# ---------------------------------------------------------
# CELL 8 -- Load dataset
# ---------------------------------------------------------
with open(DATASET_PATH) as f:
    dataset = json.load(f)

pairs = dataset["pairs"]
print(f"Loaded {len(pairs)} pairs -> {len(pairs)*6} total prompts")

# ---------------------------------------------------------
# CELL 9 -- Main evaluation loop
# ---------------------------------------------------------
results   = []
phrasings = ["phrasing_A", "phrasing_B", "phrasing_C"]

for idx, pair in enumerate(pairs):
    pid, cat, dom = pair["id"], pair["category"], pair["domain"]
    print(f"\n[{idx+1:02d}/{len(pairs)}] {pid} -- {cat} -- {dom}")

    for vkey, is_valid in [("valid", True), ("invalid", False)]:
        vdata    = pair[vkey]
        expected = vdata["ground_truth_verdict"]

        for pkey in phrasings:
            prompt_text = vdata[pkey]
            prompt_id   = f"{vdata['id']}_{pkey[-1]}"
            print(f"  {prompt_id} ... ", end="", flush=True)
            t0 = time.time()

            try:
                response = run_prompt(prompt_text)
                elapsed  = round(time.time() - t0, 1)
                mv       = extract_verdict(response)
                sc       = score_response(mv, expected, is_valid)
                print(f"model={mv:8s} | expected={expected:8s} | "
                      f"grade={sc['grade']:10s} | {elapsed}s")

                results.append({
                    "prompt_id":          prompt_id,
                    "pair_id":            pid,
                    "category":           cat,
                    "domain":             dom,
                    "version":            vkey,
                    "phrasing":           pkey[-1],
                    "is_valid_question":  is_valid,
                    "planted_error":      ("" if is_valid
                                          else pair["invalid"].get("planted_error", "")),
                    "planted_error_type": ("" if is_valid
                                          else pair["invalid"].get("planted_error_type", "")),
                    "expected_verdict":   expected,
                    "model_verdict":      mv,
                    "grade":              sc["grade"],
                    "score":              sc["score"],
                    "response_snippet":   response[:300].replace("\n", " "),
                    "full_response":      response,
                })

            except Exception as e:
                tb = traceback.format_exc()
                # repr(e) always yields a non-empty string, even for RuntimeError("")
                print(f"ERROR: {e!r}")
                results.append({
                    "prompt_id": prompt_id, "pair_id": pid,
                    "category": cat, "domain": dom,
                    "version": vkey, "phrasing": pkey[-1],
                    "is_valid_question": is_valid,
                    "planted_error": "", "planted_error_type": "",
                    "expected_verdict": expected,
                    "model_verdict": "ERROR", "grade": "ERROR", "score": 0,
                    "response_snippet": tb[:300], "full_response": tb,
                })

print(f"\nDone. {len(results)} results collected.")

# ---------------------------------------------------------
# CELL 10 -- Save raw results (CSV + JSON)
# ---------------------------------------------------------
ts           = datetime.now().strftime("%Y%m%d_%H%M%S")
results_csv  = f"{RESULTS_DIR}/results_{MODEL_ID.replace('/', '_')}_{ts}.csv"
results_json = results_csv.replace(".csv", ".json")

fieldnames = [k for k in results[0] if k != "full_response"]
with open(results_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for row in results:
        w.writerow({k: row[k] for k in fieldnames})

with open(results_json, "w") as f:
    json.dump(results, f, indent=2)

print(f"Saved:\n  CSV  -> {results_csv}\n  JSON -> {results_json}")

# ---------------------------------------------------------
# CELL 11 -- Score analysis
# ---------------------------------------------------------
from collections import defaultdict

inv   = [r for r in results if not r["is_valid_question"] and r["grade"] != "ERROR"]
val   = [r for r in results if     r["is_valid_question"] and r["grade"] != "ERROR"]
n_err = sum(1 for r in results if r["grade"] == "ERROR")

if n_err:
    print(f"WARNING: {n_err} ERROR prompts excluded from rate calculations.")

if not inv:
    print("No scoreable results -- check errors above.")
    raise SystemExit

catch_r = sum(1 for r in inv if r["grade"] == "CATCH")    / len(inv)
miss_r  = sum(1 for r in inv if r["grade"] == "MISS")     / len(inv)
flag_r  = sum(1 for r in inv if r["grade"] == "FLAG")     / len(inv)
spur_r  = sum(1 for r in val if r["grade"] == "SPURIOUS") / max(len(val), 1)

print("\n" + "="*62)
print("  OVERALL RESULTS")
print("="*62)
print(f"  INVALID prompts evaluated : {len(inv)}")
print(f"  VALID   prompts evaluated : {len(val)}")
print(f"  CATCH    rate (errors detected)  : {catch_r:.1%}")
print(f"  MISS     rate (errors accepted)  : {miss_r:.1%}")
print(f"  FLAG     rate (uncertainty)      : {flag_r:.1%}")
print(f"  SPURIOUS rate (false alarms)     : {spur_r:.1%}")

# -- By error category --
cat_s = defaultdict(lambda: {"catch": 0, "miss": 0, "flag": 0, "total": 0})
for r in inv:
    cat_s[r["category"]]["total"] += 1
    if   r["grade"] == "CATCH": cat_s[r["category"]]["catch"] += 1
    elif r["grade"] == "MISS":  cat_s[r["category"]]["miss"]  += 1
    elif r["grade"] == "FLAG":  cat_s[r["category"]]["flag"]  += 1

print("\n" + "="*62)
print("  CATCH RATE BY ERROR CATEGORY")
print("="*62)
print(f"  {'Category':<30} {'Catch':>6} {'Miss':>6} {'Flag':>6} {'Rate':>7}")
print(f"  {'-'*58}")
for cat, s in sorted(cat_s.items()):
    rate = s["catch"] / s["total"] if s["total"] else 0
    print(f"  {cat:<30} {s['catch']:>6} {s['miss']:>6} {s['flag']:>6} {rate:>7.1%}")

# -- By phrasing --
ph_s = defaultdict(lambda: {"catch": 0, "total": 0})
for r in inv:
    ph_s[r["phrasing"]]["total"] += 1
    if r["grade"] == "CATCH":
        ph_s[r["phrasing"]]["catch"] += 1

print("\n" + "="*62)
print("  CATCH RATE BY PHRASING")
print("="*62)
labels = {"A": "Direct technical", "B": "Expert verification", "C": "Teaching scenario"}
for ph in ["A", "B", "C"]:
    s = ph_s[ph]
    rate = s["catch"] / s["total"] if s["total"] else 0
    print(f"  Phrasing {ph} ({labels[ph]:<22}): {s['catch']}/{s['total']} = {rate:.1%}")

# -- Never-caught pairs --
pc = defaultdict(lambda: {"catch": 0, "total": 0, "cat": ""})
for r in inv:
    pc[r["pair_id"]]["total"] += 1
    pc[r["pair_id"]]["cat"]    = r["category"]
    if r["grade"] == "CATCH":
        pc[r["pair_id"]]["catch"] += 1

never = [(p, s) for p, s in pc.items() if s["catch"] == 0]
print(f"\n  Pairs NEVER caught ({len(never)} / {len(pc)}):")
for p, s in sorted(never):
    print(f"    {p} [{s['cat']}]")

# -- Save summary JSON --
summary_path = f"{RESULTS_DIR}/summary_{ts}.json"
with open(summary_path, "w") as f:
    json.dump({
        "model": MODEL_ID, "timestamp": ts, "dataset": DATASET_PATH,
        "num_pairs": len(pairs), "error_count": n_err,
        "total_invalid_prompts": len(inv),
        "total_valid_prompts":   len(val),
        "overall_catch_rate":    round(catch_r, 4),
        "overall_miss_rate":     round(miss_r,  4),
        "overall_flag_rate":     round(flag_r,  4),
        "overall_spurious_rate": round(spur_r,  4),
        "catch_by_category": {
            cat: {
                "catch": s["catch"], "total": s["total"],
                "catch_rate": round(s["catch"]/s["total"], 4) if s["total"] else 0,
            }
            for cat, s in cat_s.items()
        },
        "catch_by_phrasing": {
            ph: {
                "catch": ph_s[ph]["catch"], "total": ph_s[ph]["total"],
                "catch_rate": (round(ph_s[ph]["catch"]/ph_s[ph]["total"], 4)
                               if ph_s[ph]["total"] else 0),
            }
            for ph in ["A", "B", "C"]
        },
        "never_caught_pairs": [p for p, _ in never],
    }, f, indent=2)
print(f"\n  Summary -> {summary_path}")

# ---------------------------------------------------------
# CELL 12 -- Failure transcripts (first 5 MISSes)
# ---------------------------------------------------------
print("\n" + "="*62)
print("  FAILURE TRANSCRIPTS -- first 5 MISSes")
print("="*62)
misses = [r for r in results if r["grade"] == "MISS"][:5]
if not misses:
    print("  No MISSes -- model caught every planted error.")
for i, r in enumerate(misses, 1):
    print(f"\n--- Miss #{i} ---")
    print(f"  Pair      : {r['pair_id']} [{r['category']}]  Phrasing {r['phrasing']}")
    print(f"  Domain    : {r['domain']}")
    print(f"  Error type: {r['planted_error_type']}")
    print(f"  Planted   : {r['planted_error'][:120]}")
    print(f"  Verdict   : model={r['model_verdict']}  expected=INVALID")
    print(f"  Response  :\n{r['full_response'][:600]}")

# ---------------------------------------------------------
# CELL 13 -- Download results from Colab
# ---------------------------------------------------------
# from google.colab import files
# files.download(results_csv)
# files.download(results_json)
# files.download(summary_path)
