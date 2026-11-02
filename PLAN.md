# Replication plan: mask-based ground truth, cross_line at scale, left/right probe

Target sections of `Perspective Pilot Reproduction.md`:

1. **Ground truth, and which examples count**: mask-based labels, the clean rule, and the audit of centroid labels against mask labels (50 of 177 B views disagree).
2. **The cross_line test at scale**: about 177 rendered, 92 clean. Settings 1, 1-identity, 2 and 4 for Qwen3-VL-8B-Instruct and BAGEL-7B-MoT.
3. **Inside the model**: per-layer left/right readers, Check 1, Check 2, the B score, the permutation band, bootstrap CIs, and the right/wrong split.

Out of scope: the person/SMPL-X variant, setting 3 (BAGEL drawings) and steering. The small 4-mode table is optional (see Phase 5).

**Dataset policy:** the PhD student confirmed we don't need the report's exact dataset. All data-generation code is written from scratch here. Our own choices for camera settings, vocabulary, sentence templates and rooms are fine, as long as the task definition (four placement modes, mask-based labels, clean rule, prompt wording) matches the report. Record each such choice in `common/config.py` and state it in the write-up.

---

## 0. Where things run

| Machine | Role | Why |
|---|---|---|
| **Local Mac** | Writing code, git, looking at outputs | habitat-sim headless needs EGL, which macOS lacks. No CUDA for the models. **No environment setup locally** beyond an editor and `rsync`/`git`. |
| **Great Lakes (primary)** | Rendering + all model inference + probing | Slurm array jobs per room for rendering. A40 48 GB (`spgpu`) fits BAGEL-7B-MoT in bf16 (~30 GB) and Qwen3-VL-8B (~17 GB). Turbo for hidden states. |
| **Whistler (fallback)** | Same as Great Lakes if the queue is slow | Has been used for VLM inference before. Keep scripts free of Slurm-only assumptions: plain `python` entry points, with Slurm wrappers on top. |

Recommendation: do everything on Great Lakes. Code lives in this git repo and is pulled on the cluster. All large files (datasets, renders, weights, results, conda envs) live on Turbo at `/nfs/turbo/coe-chaijy-unreplicated/dannyyu/` (`$TURBO`), which both Great Lakes and Whistler mount. Download once, reuse on both machines. Nothing large goes in git.

```
$TURBO/
  datasets/replica_cad/          # ReplicaCAD assets
  datasets/perspective_pilot/    # generated A/B images, masks, labels, metadata
  models/                        # HF_HOME for model weights
  results/perspective_pilot/     # inference outputs, hidden states, probe results
  envs/                          # conda envs (shared across machines)
```

### Environments (on Great Lakes)

Provisional: three separate conda envs. This is my assumption, not something the report says. BAGEL pins its own torch, transformers and flash-attn versions, which may conflict with the recent transformers that Qwen3-VL needs. Check `Bagel/requirements.txt` in Phase 0; if it's compatible with Qwen3-VL, merge `vlm` and `bagel` into one env.

| Env | Contents | Used by |
|---|---|---|
| `habitat` | python 3.10, `habitat-sim=0.3.4 withbullet headless` (conda-forge + aihabitat), numpy, opencv, matplotlib | rendering, labels, filter |
| `vlm` | torch (CUDA 12.x), `transformers>=4.57` (Qwen3-VL support), accelerate, qwen-vl-utils, scikit-learn, pandas | Qwen3-VL behaviour + hidden states, probe analysis |
| `bagel` | BAGEL repo (`ByteDance-Seed/Bagel`) requirements, flash-attn, weights `ByteDance-Seed/BAGEL-7B-MoT` | BAGEL behaviour + hidden states |

Model weights go to `HF_HOME=$TURBO/models`, downloaded once.

---

## 1. Data download

```bash
python -m habitat_sim.utils.datasets_download \
  --uids replica_cad_dataset replica_cad_baked_lighting --data-path $TURBO/datasets/replica_cad
```

About 700 MB, CC BY 4.0. No humanoid assets are needed.

**Step-0 check (do first):** find out whether `v3_sc*_staging_*` in the baked-lighting set keeps furniture as separate objects. Mask labels need per-object `semantic_id`. If furniture is baked into the stage, render the v3 layouts from `replica_cad_dataset` instead, which has no baked lighting but keeps separate objects.

---

## 2. Rooms and splits

ReplicaCAD scenes available: `apt_0`–`apt_5`, plus 84 `v3_sc{0-4}_staging_XX` layouts.

- **Test rooms (15)**, matching the report's large set. Proposal: `apt_0`–`apt_5`, plus 9 v3 layouts (two per macro variation `sc0`–`sc4`, minus one).
- **Probe-training rooms (~10)**: other v3 layouts, disjoint from the test rooms.
- Caveat to note in the write-up: all ReplicaCAD scenes share one apartment shell, so "disjoint rooms" means different furniture layouts, not different architecture. The report has the same limitation.

---

## 3. Files to write

```
repro/
  common/
    config.py            # RES, HFOV, EYE height, thresholds, paths, room splits
    vocab.py             # object-handle → name map (sofa, fridge, kitchen counter, tv screen, ...)
  render/
    scene_index.py       # per scene: candidate objects, AABBs, semantic_id assignment, de-dup names
    sample_poses.py      # A pose + B pose for cross_line (and other modes, optional)
    render_examples.py   # RGB + semantic render of A and B; centroid (3D bearing) labels; describe() sentences
    relabel_visible.py   # mask-based labels: centre of X's visible pixels vs Y's
    clean_rule.py        # overlap / ≥40 px / ≥1.5 m / visibility filter → clean flag + reasons
    audit_labels.py      # Section-1 audit: centroid vs mask disagreement, distance-to-object stats
    make_map.py          # top-down map PNG (blue A, red B, X/Y dots); for inspection
  slurm/
    render_array.sbatch  # one array task per room
    qwen.sbatch, bagel.sbatch, probe_extract_*.sbatch
  eval/
    prompts.py           # templates for settings 1, 1-identity, 2, 4 (verbatim from the report)
    run_qwen.py          # behaviour: one-word answers → jsonl
    run_bagel.py         # behaviour via BAGEL understanding mode
    summarize.py         # Section-2 table: correct / answered-from-A per setting × model
  probing/
    PLAN.md
    probe_extract_qwen.py   # hidden state of last prompt token, all 37 layer outputs (emb + 36)
    probe_extract_bagel.py  # same at BAGEL's first generation step, all 29 (emb + 28)
    probe_analyze.py        # readers, Check 1/2, B score, permutation, bootstrap, right/wrong, 2×2
    plot_probe.py
```

### Key specifications

**Sampling (cross_line, large set)**
- A stands on the navmesh and sees X and Y. Each object covers ≥1.5% of the frame, and the two are ≥20° apart in bearing.
- B goes to the far side of the X–Y line, looks back, and sees both objects with the same thresholds.
- B's distance and turn angle are drawn so that sentences like "about 6.0 metres ahead of you and to your left, turned about 120 degrees to the right" come out naturally.
- Store full poses (position + quaternion) for A and B in `examples.jsonl`. This fixes the report's "no stored poses" gap.

**Labels and clean rule (Section 1)**
- Mask label: compare the horizontal centres of the visible pixels of the X and Y instance masks.
- Also store `relA_centroid` / `relB_centroid` from 3D-centre bearings.
- An example is clean only if all of these hold in **both** views:
  - the objects' horizontal pixel extents don't overlap;
  - their centres are ≥40 px apart;
  - the observer is ≥1.5 m from both objects;
  - for cross_line, the mask labels for A and B differ.

**Prompts.** Copy them verbatim from the report:
- Setting 1: "This image shows what you (observer A) currently see. … From B's point of view, is the X to the left or to the right of the Y? Answer with exactly one word: left or right."
- Identity: same as setting 1, with B "standing where you stand, facing the same direction".
- Settings 2 and 4: the direct question on B's photo and on A's photo.

Use greedy decoding. Parse the answer as left or right; anything else counts as wrong.

**Probe (Section 3)**
- **Features**: last prompt-token hidden state at every layer (for BAGEL, the start token of the first generation step). Save as fp16 `.npy` per (model, setting) under `$TURBO/results/perspective_pilot/`.
- **Reader training data**: direct questions (setting 2/4 format) on the training rooms, half A photos and half B photos. The label is **the model's own answer**, not the ground truth.
- **Second reader**: trained on identity-prompt states from the training rooms.
- **Reader per layer**: standardize, PCA to 256 components, then class-weighted logistic regression. Ceiling = held-out agreement with the model's own answers.
- **Check 1**: at the last layer, class-balanced agreement with the model's actual setting-1 answer must be ≥95% of the ceiling.
- **Check 2**: identity-control B score must be ≥80% at the layer. Layers that fail are greyed out and not read.
- **B score**: class-balanced over B's answer (left/right) on the clean test cross_line examples.
- **Uncertainty**: permutation band (≥100 readers trained on shuffled labels) and 1000-sample bootstrap 95% CIs over examples.
- **Extras**: left-rate per layer, the right/wrong split (balanced within group) and the 2×2 table at selected layers.

---

## 4. Target sizes

| Set | Rooms | Rendered | Expected clean | Used for |
|---|---|---|---|---|
| Large cross_line (test) | 15 test | ~180–200 | ~90+ | Section 1 audit, Section 2 table, Section 3 test |
| Probe-training | ~10 train | ~300 A/B pairs (any mode, balanced left/right) plus ~150 cross_line for identity states | — | reader training |

The report doesn't give the probe-training size. 300 pairs (600 direct questions) is my choice and can be changed.

---

## 5. Run order

| Phase | Where | What | Done when |
|---|---|---|---|
| 0 | GL login node | create envs, download data + weights, Step-0 check | one test render of `apt_0` with sensible masks |
| 1 | GL CPU/GPU array | `scene_index` → `render_array` (test + train rooms) | `examples.jsonl` + images per room |
| 2 | GL CPU | `relabel_visible`, `clean_rule`, `audit_labels` | Section-1 numbers: clean count, centroid-vs-mask disagreement, median distance of disagreeing B views |
| 3 | GL A40 | `run_qwen`, `run_bagel` on settings 1, identity, 2 and 4 | Section-2 table |
| 4 | GL A40 | `probe_extract_*` on test (settings 1, identity, direct) + train (direct, identity) | hidden-state arrays on Turbo |
| 5 | GL CPU | `probe_analyze`, `plot_probe` | Check 1/2, B-score curves, 2×2 tables |
| (opt) | GL | small 4-mode set (4 rooms, ≥10° / ≥0.3%, 10 per mode) + settings 1/2/4 | small table in Section 1 (no setting 3) |

Rough compute: rendering takes under 1 GPU-hour. Behaviour runs: about 4 prompts × ~200 examples × 2 models, under 1 GPU-hour on an A40. Hidden-state extraction takes about the same. Storage: about 1,000 prompts × 37 layers × 4096 dims × fp16 ≈ 300 MB per model.

---

## 6. Comparison to the report (what counts as "replicated")

- **Section 1**: the clean filter keeps roughly half of the cross_line renders, and centroid labels disagree with mask labels mostly when B stands close to a large object.
- **Section 2**: setting 1 accuracy is well below 50%, and most answers match A's answer. Identity, 2 and 4 are all ≥90%.
- **Section 3**: the B score stays below 50% (or inside the chance band) at every trusted layer, i.e. Outcome A. Also report which checks pass. In the report, Check 1 fails for the direct reader and passes for Qwen's identity-trained reader.
