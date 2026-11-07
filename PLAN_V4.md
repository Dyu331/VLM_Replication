# Replication plan: the v4 probe ("Three Random Objects, HSSD pool")

Source: the PhD's v4 report, https://claude.ai/artifact/Ljb9PBXewzPvhY4DihJB8a?sk=Knm45viCXxHIyYavJrQWfw.
Its code (their `repro/v4/`) is not available, so we write our own, as for the first replication (`PLAN.md`).

Labels used below:
- **[theirs]**: copied from the v4 report.
- **[ours]**: a detail the report leaves open, chosen by us and recorded in `config_v4.py`.
- **[violet]**: an addition.
- **[amber]**: a change from their method.

In the report, violet marks additions and amber marks changes, as before.

## Decisions (2026-11-07)

| Question | Decision |
|---|---|
| Objects | **HSSD** object models, the same 13 categories and the same train/test category split as theirs |
| Model input | **Plain photos** (floor grid, no outlines). The earlier outlined-photo probe showed outlines changed little. |
| Models | **Qwen3-VL-8B, Gemma 4 12B, BAGEL-7B-MoT**. Qwen is theirs; Gemma and BAGEL are violet. Qwen passes Check 3 first, then all three run in full. |
| Report | A **new artifact** for v4, linked from the old report |
| Added control | Vision-encoder features averaged **per object** (B, X, Y side by side), plus Qwen's DeepStack features [violet] |
| Follow-ups (after the replication) | Logit lens, then steering (Phase 6) |
| Not done | Asking the model about "the X–Y line" (too abstract a concept for the prompt) |

## What v4 tests

Three objects sit at random spots on the floor of an emptied room. A photo is taken from a random spot A, and every pixel that isn't one of the objects is replaced by a floor grid. One object is named as B: "Imagine you are standing where the {B} is, looking toward the {X} and the {Y}. From there, is the {X} to the left or to the right of the {Y}?" B's answer equals A's answer when B is on A's side of the X–Y line, and is the opposite when B is across it.

Their result for Qwen:
- The model gives A's answer 91% of the time and B's answer 49%.
- A linear reader trained on the task prompt reads B's answer at **86% (layer 32)**, above every control: the model "computes B's answer inside but doesn't say it".

---

## Phase 0: assets and setup

**0.1 HSSD objects**
- Accept the license for the gated Hugging Face dataset `hssd/hssd-hab`.
- `setup/download_hssd.sh` downloads only the object metadata table and the object files for the 13 categories, into `$TURBO/datasets/hssd`.
- Train categories: bench, cabinet, chair, couch, dresser, potted plant, stool, table. **[theirs]**
- Test categories: bed, bookshelf, office chair, refrigerator, trash can. **[theirs]**
- Keep 3–5 models per category **[ours]**. Drop models that are tiny, mounted on a wall, or hard to name. Make a contact sheet so the choice can be checked by eye.

**0.2 Code layout**

```
repro/v4/
  config_v4.py    room split, categories, thresholds, camera, grid settings
  assets.py       HSSD model list per category, sizes, names used in prompts
  scene.py        empty a room, place objects, draw the floor grid
  render.py       photos from A, photos from B's spot, masks, labels, bookkeeping
  select.py       validity rules, balancing, shortcut check, bookkeeping table
  prompts.py      task prompt + direct questions
  extract.py      answers + hidden states (Qwen3-VL, Gemma 4)
  extract_bagel.py  the same for BAGEL (hooks)
  analyze.py      readers, controls, token sweep, checks
repro/report/build_v4_report.py
slurm/v4_render.sbatch, v4_extract.sbatch, v4_extract_bagel.sbatch, v4_analyze.sbatch
```

What we reuse:
- `common/sim.py`: make_sim, the navmesh code, world_aabb, set_pose.
- The `Reader` class from `probing/analyze.py`.
- The model loaders in `eval/run_vlm.py` and `eval/run_bagel.py`.
- The BAGEL hook approach from `probing/extract_bagel.py`.

**Check 0.** habitat-sim 0.3.3 loads an HSSD object into an emptied ReplicaCAD room and renders it with correct textures and masks.

Known risk: HSSD files use compressed textures, which our habitat-sim build may not support. If it doesn't, convert the textures once at download time.

---

## Phase 1: scene generation

**1.1 Rooms.** 34 rooms in total. **[count theirs, choice ours]**
- Test: 9 rooms, apt_0–5 plus 3 v3 layouts.
- Train: 25 v3 layouts.

Because the background is replaced, the room mainly decides where objects and observers can stand.

**1.2 Emptying a room.**
- Remove every movable object, but keep the doors. **[theirs]**
- Rebuild the walkable floor area (navmesh) from the bare room.

**1.3 Placing objects.**
- Choose 3 different categories from the split's pool, and one random model of each.
- Give each object a random rotation about the vertical axis **[ours]**.
- Put each at a random walkable spot with clearance, at least about 1 m from the other objects **[ours]**, resting on the floor.

**1.4 A's photo.**
- A stands at a random walkable spot 1.5–7 m from the objects and looks at their centre. **[theirs]**
- Camera: 512×512, 90° horizontal field of view, eye height 1.5 m, no tilt **[ours, same as the first replication]**.
- Render a colour image and an object mask.

**1.5 Floor grid.**
- Every pixel that isn't one of the three objects is replaced by a floor grid, computed exactly from the camera pose so that lines shrink with distance. **[theirs]**
- Grid lines every 0.5 m, and plain grey above the horizon **[ours]**.

**1.6 Questions from each photo.** For each object as B (3 choices), with a random X/Y order:
- Render the view from B's spot with object B hidden, at eye height 1.5 m, looking at the midpoint of X and Y **[ours]**.
- The label is left/right from the masks in A's photo and in the view from B's spot. **[theirs]**
- Check each label against the floor-plan geometry, and drop questions where they disagree. **[theirs]**
- Store for bookkeeping: all positions, distances and angles; B's depth rank (nearest, middle or farthest from A); B's image side; each object's mask box and the pixel where it touches the floor.
- Save the photo from B's spot, with the grid, for the direct-B question.

**1.7 Validity rules.**
- From A, X and Y at least 10° apart; from B's spot, at least 15°. **[theirs]**
- B at least 0.5 m from the X–Y line, and at least 0.8 m from X and from Y. **[theirs]**
- Visibility **[amber; they state no rule]**: each object covers at least ~1.5% of the frame, no object is cut off at the image edge, and the X and Y masks don't overlap left-to-right.
- B is never between X and Y in the picture **[theirs, inferred: their data has 0 "between" cases]**.

**Compute.** One GPU job that loops over all rooms, running several renderer processes on that one GPU. Oversample by roughly 3–5×, because balancing discards many questions.

**Check 1 (smoke test).** Render one training room and one test room, about 20 photos each. Make a contact sheet of A's photo, the view from B's spot, the masks and a top-down sketch, and look at it before the full render.

---

## Phase 2: selection and data checks

**2.1 Balancing.** **[theirs]**
- Group candidate questions by depth rank × image side.
- Within each group, keep equal numbers of the four A's answer × B's answer combinations. That gives 50% flips in every group and balanced combinations overall.
- Target: about 1,500 training and 1,100 test questions.
- **[violet, optional]** Oversample "B nearest" questions. They had only 24 test questions in that group, too few to judge.

**2.2 Metadata shortcut check.** **[theirs]**
- A logistic regression predicts B's answer from bookkeeping facts: depth rank, image side, distances and angles, categories, and their combinations.
- It is trained on the training split and scored on the test split, within A's answer. It must be ≤ 60% with all features.
- The exact list of "distances and angles" is ours, and uses unsigned values only. **[ours]**

**2.3 Bookkeeping table**, as in their report:
- questions, photos, rooms and categories per split;
- the four answer combinations;
- flip rate by depth rank and by image side.

**Check 2.** All balance checks and the shortcut check pass.

---

## Phase 3: model answers and hidden states (all three models)

There is one GPU job per model, covering both behaviour and extraction.

**Order.**
1. Run a Qwen smoke test, then the full Qwen run.
2. Once Check 3 passes for Qwen, run Gemma and BAGEL in full. This avoids spending GPU time on two more models if the data has a problem.

**3.1 Prompts.** **[theirs]**
- Task: "This image shows what you currently see. Imagine you are standing where the {B} is, looking toward the {X} and the {Y}. From there, is the {X} to the left or to the right of the {Y}? Answer with exactly one word: left or right."
- Direct-A: "In this image, is the {X} to the left or to the right of the {Y}? Answer with exactly one word: left or right." asked on A's photo.
- Direct-B: the same question on the view from B's spot.
- Greedy decoding; the answer is parsed as left or right, anything else counts as invalid.

**3.2 What we save for every task question**

Hidden states at every layer, at these positions:
- the token before the answer **[theirs]**;
- the last token of B's, X's and Y's names **[theirs]**;
- the image tokens covering B's, X's and Y's masks, averaged per object **[theirs; pooling ours]**.

Plus:
- **text only**: the same prompt without the picture, at the token before the answer, every layer **[theirs]**;
- **vision encoder output**: averaged over the whole image **[theirs]**, and averaged per object for B, X and Y **[violet]**;
- for Qwen, also the DeepStack features that go into the first language-model layers **[violet]**;
- for the logit lens later **[violet, nearly free]**: the "left"/"right" scores at the answer position, the final-norm weights, and the vocabulary rows for left/Left/right/Right.

**3.3 Per-model notes.**
- **Qwen3-VL.** Each image token covers 32×32 px, so a 512×512 photo gives a 16×16 grid. Check the DeepStack code path in transformers.
- **Gemma 4.** It uses a different image-token grid and pooling, so mask→token mapping needs its own code. There is no DeepStack.
- **BAGEL.**
  - The current extraction records only the first generation step. v4 also needs the prefill states at the name and image-token positions, captured with hooks during prefill.
  - Its understanding branch uses a SigLIP encoder with its own patch grid.
  - Text-only is a prompt with no image.

Storage: about 6 GB per model on Turbo.

**Check 3 (behaviour).**
- Qwen gives A's answer about 90% of the time and B's answer about 50%.
- Both direct questions score at least 90%.
- Answers aren't one constant word (more than 95% of one word blocks; more than 80% gets a mark).

If Qwen's behaviour differs a lot from theirs, stop and look at rendering differences before probing. For Gemma and BAGEL there is no reference to match; we just record their behaviour. BAGEL's earlier "right" bias may show up again.

---

## Phase 4: readers and controls (CPU job per model)

**4.1 Reader.** **[theirs]**
- Standardize, then PCA(256), then balanced logistic regression.
- C is chosen by 5-fold cross-validation grouped by training room, from {0.001, 0.01, 0.1, 1, 10} **[grid ours]**.
- Trained on the training rooms, scored on the test rooms.

**4.2 Six targets** at the token before the answer, every layer:
- A's answer;
- the model's answer;
- **B's answer, scored within A's answer** (the class-balanced score is computed separately where A's answer is left and where it is right, then averaged);
- flips?;
- B's depth rank (3 classes);
- B's image side (effectively 2 classes, with chance at 50%; labelled in the report) **[amber]**.

**4.3 Uncertainty.**
- 1,000 bootstrap resamples, by photo **[amber if theirs was by question]**, and a room-level version as well **[violet]**.
- Permutation band: 100 readers trained on shuffled labels **[count ours]**.

**4.4 Controls for B's answer**, with their pass marks **[theirs]**:
- permutation band ≤ 60%;
- labels scrambled within room ≤ 62% (trained on scrambled training labels, scored on true test labels **[ours]**);
- whole-image vision average ≤ 55%;
- text only ≤ 55% at every layer;
- layer 0 ≤ 55%.

Added **[violet]**:
- **Per-object vision control**: vision-encoder features for B, X and Y side by side (plus DeepStack for Qwen), with the same linear reader. This tests whether the picture features already hold B's answer before the language model.
- The same three-object input from the language-model layers, at the objects' image tokens, layer by layer.

How to read the per-object vision control:
- ~50% while layer 32 is ~86%: the language model computes B's answer.
- ~80% or more: the vision encoder did most of the work, and the language model carries it.
- In between: both contribute.

**4.5 Token sweep.** The B-answer reader at each saved position, at every layer. **[theirs]**

**4.6 Group breakdown.** B's answer at its best layer, within each depth rank and each image side. **[theirs]**

**4.7 Sanity-check table.** The same rows as theirs, each marked pass, mark or fail.

**Check 4.** The A's-answer and model's-answer readers reach ≥ 90% in the mid and late layers, which shows the extraction works. Then compare B's answer with their 86% at layer 32 (Qwen).

---

## Phase 5: report (new artifact)

`repro/report/build_v4_report.py`, built from `local_data/v4/`, with this layout:
1. Introduction and example cards: A's photo, the view from B's spot, a top-down sketch, the prompt and the answers.
2. Bookkeeping table and the metadata shortcut check.
3. What each model does: behaviour per model, and split by depth rank and image side.
4. Six readers per layer, for each model.
5. B's answer against the controls, including the per-object vision control.
6. Token sweep.
7. Group breakdown.
8. Sanity checks.
9. Details, and a "differences from the PhD's v4" box (violet for additions, amber for changes).

Link the new report from the old one (https://claude.ai/artifact/TosyTwZEV6LRyVZfNyCXak).

---

## Phase 6: follow-ups (after the replication)

- **Logit lens** (CPU, from data saved in Phase 3):
  - For each layer, compute the "left" score minus the "right" score at the token before the answer.
  - Find the angle between the B-answer reader's direction and the model's own left–right output direction.
- **Steering** (one GPU job per model). Edit the hidden state at the token before the answer, at layers around the reader's peak:
  - **amplify**: scale up the component along the reader's direction;
  - **set to typical**: set that component to the value typical of the true B's answer;
  - **fixed push**: add the same vector to every question.

  Directions come from the logistic regression weights and from the difference of class means. Controls are a random direction, the A's-answer direction, and checking that the direct questions still get ≥ 90%. The measure is how often the model now gives B's answer.

---

## Rough compute

| Phase | Resource | Size |
|---|---|---|
| Rendering | 1 GPU job | a few hours with oversampling |
| Answers + extraction | 1 GPU job per model (3 jobs) | ~2,650 questions × (task + 2 direct + text only) ≈ 10k forward passes each |
| Analysis | 1 CPU job per model | 6 targets × ~37 layers × CV + controls, ~1–2 h |

## Open questions for the PhD

1. The image resolution given to the model.
2. Which features are in "all distances and angles" in the shortcut check.
3. Whether the bootstrap resamples questions, photos or rooms.
4. How "patches of B" were pooled, and how "scrambled within room" was scored.
5. Whether "B between X and Y in the picture" was excluded on purpose.
