Spatial Perspectives · Reproduction · 2026-10-25 · ReplicaCAD in Habitat-Sim · 22 clean of 33 rendered examples · 4 rooms

# Can the model see from B's spot? Reproduced.

The setup is the pilot's: person A takes a photo of a room, person B stands somewhere else in the same room. The model gets A's photo and a sentence that says where B stands, and is asked: "From where B stands, is object X on the left or on the right of object Y?" We rebuilt the whole pipeline on this machine (ReplicaCAD scenes rendered in Habitat, B's view drawn by BAGEL, questions answered by Qwen3-VL and by BAGEL itself) on new examples; additions to the pilot carry a violet rule.

B is placed in four ways. In three of them the answer is the same for A and B. In one of them (cross_line) it is different, which shows whether the model really imagines B's view or just answers from A's photo.

**Headline.** On 92 clean cross_line examples, Qwen3-VL is right 20% of the time and BAGEL 34%, far below the 50% of guessing, while both are above 94% when B is described as standing where A stands and above 93% when simply shown B's photo. Both systematically answer from A's spot. The bottleneck of the generate-then-read route is the drawing itself: on cross_line BAGEL's drawings showed both objects in 0 of 8 cases.

ABobject Xobject YQwen3-VL Qwen3-VL-8B-InstructBAGEL BAGEL-7B-MoTadded in this reproduction

## Four ways to place B

In the small drawings: blue arrow = A (the photo the model gets), red arrow = B (the view we ask about), orange and blue dots = objects X and Y. B must also see both objects here, so setting 2 is answerable (the pilot did not require this).

|  | Mode | Where B stands | Answer differs for A and B? | Examples | One example: A's photo → B's photo | 1 correct Qwen3-VL | 1 correct BAGEL | 3 BAGEL's drawing |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | rotate_in_placeTurn in place | B stands on the same spot as A and only turns (15–60° here; the pilot allowed up to 180°). | no, never 0 of 3 different | 3 | A A   B B | 2/3 | 2/3 | 3/3 show both objects judged correct for B: 3/3 · looks like A 0.95, like B 0.92 (A vs B 0.93) · closer to B: 1/3 |
|  | generalMove anywhere | B walks 0.5 to 3 m away and looks in a random direction, staying on A's side of the X–Y line. | no, almost never 0 of 7 different | 7 | A A   B B | 5/7 | 4/7 | 5/7 show both objects judged correct for B: 4/7 · looks like A 0.93, like B 0.91 (A vs B 0.95) · closer to B: 1/7 |
|  | shared_targetLook at the same thing | B walks 0.5 to 3 m away but looks at the same spot as A. | no, almost never 0 of 7 different | 7 | A A   B B | 6/7 | 6/7 | 5/7 show both objects judged correct for B: 4/7 · looks like A 0.95, like B 0.93 (A vs B 0.95) · closer to B: 1/7 |
|  | cross_lineCross to the other side | B goes to the far side of the two objects and looks back at them. Left and right are swapped compared with A. | yes, always 5 of 5 different | 5 | A A   B B | 2/5 | 1/5 | 3/5 show both objects judged correct for B: 2/5 · looks like A 0.93, like B 0.90 (A vs B 0.92) · closer to B: 0/5 |

## Experimental settings

Four settings, numbered so that the tables refer to them by code. Diagrams: the filled wedge is the view the model is given; dashed = described in words only; hatched = drawn by BAGEL; ? marks the viewpoint the question is about. In the pipeline column, grey boxes are images the model receives, the dashed box is text, the hatched box is a generated image, and the outlined box is what the model predicts (an answer, not an image). Templates use placeholders: X and Y are the two objects, where B stands and looks is the sentence about B (distance and direction from A, B's turn, what B is next to and facing). In settings 2 and 3 the model is first asked whether both objects are visible; "no" counts as a wrong answer.

| Setting | Diagram | Pipeline: what the model gets → what it predicts | What we test | Question template | Correct when |
| --- | --- | --- | --- | --- | --- |
| 1**Original** |  | A's photo+B: where it stands, where it lookspredict →B's view: X left or right of Y? | **Direct understanding.** The model must imagine B's view internally and answer from it. | This image shows what you (observer A) currently see. Another observer, B, is in the same room. where B stands and looks From B's point of view, is the X to the left or to the right of the Y? Answer with exactly one word: left or right. | the answer matches the truth from **B's** spot. |
| 2**Ground-truth target** |  | B's photopredict →B's view: X left or right of Y? | **Control.** Understanding of B's view when it is simply given. Also the ceiling that setting 3's drawings are measured against. | In this image, is the X to the left or to the right of the Y? Answer with exactly one word: left or right. | the answer matches the truth from **B's** spot. |
| 3**Generated target** |  | A's photo+B: where it stands, where it looksBAGEL generates →B's view, drawn by BAGELpredict →B's view: X left or right of Y? | **Generation + understanding.** One explicit step in between: BAGEL first draws B's view, then the answer is read off the drawing. | In this image, is the X to the left or to the right of the Y? Answer with exactly one word: left or right. | the answer matches the truth from **B's** spot (a drawing in which the model cannot see both objects counts as wrong). |
| 4**Control (own view)** |  | A's photopredict →A's view: X left or right of Y? | **Control.** Whether the model understands this kind of left/right question at all, with no perspective change. | In this image, is the X to the left or to the right of the Y? Answer with exactly one word: left or right. | the answer matches the truth from **A's** spot. |
| 1S**Original, B shown** |  | A's photo with a person standing at B's spot+B is the person in the photo; looks where the person facespredict →B's view: X left or right of Y? | **Does showing B help?** B's position and heading are given visually instead of in words. 3S and 4S are settings 3 and 4 with this photo; 2 is unchanged. | This image shows what you (observer A) currently see. Another observer, B, is in the same room. Observer B is the person you can see in the photo. B stands where the person stands and looks in the direction the person faces. From B's point of view, is the X to the left or to the right of the Y? Answer with exactly one word: left or right. | the answer matches the truth from **B's** spot. |

## Experimental results

### Ground truth, and which examples count

The correct answer for a view is read from the rendered instance masks of that view: X is "left of" Y when the centre of X's visible pixels is left of the centre of Y's. An earlier version used the bearing of each object's 3D centre instead. An audit of the large cross_line set found that this disagreed with what the photo shows in 50 of 177 B views (and 1 of 177 A views), almost always because B had been placed right beside a large object (median 0.2 m from its centre), where "left or right" has no visible answer. On those views the models scored at chance against the old label, and 94–97% wherever the old and new labels agreed.

An example is therefore kept only when, in both views, the two objects' outlines do not overlap horizontally, their centres are at least 40 px apart, and each observer stands at least 1.5 m from both objects. Of the 33 examples rendered for the tables below, 22 pass (rotate_in_place 3 of 9, general 7 of 8, shared_target 7 of 8, cross_line 5 of 8); of the 177 large-set cross_line examples, 92 pass, and 51 turned out not to flip the answer between A and B at all once read from the masks. Rejected examples still appear in the gallery, marked.

Correct left/right answers for B's view (for setting 4, for A's view), by setting and model, on the clean examples. Grey text: in how many photos the model said it could see both objects. Rows with a violet rule were not in the pilot.

Reading setting 3 against setting 2: setting 2 is the ceiling for any "generate B's view, then read it" route, and on clean examples that ceiling is high (93–97% on the large set below). The bottleneck is the drawing: on cross_line BAGEL's drawings showed both objects in 0 of 8 cases and looked more like A's photo than B's.

| Setting | Model | rotate_in_place (3) | general (7) | shared_target (7) | cross_line (5) | all (22) |
| --- | --- | --- | --- | --- | --- | --- |
| 1Original | Qwen3-VL | 2/3 | 5/7 | 6/7 | 2/5 | 15/22 |
| 1Original | BAGEL | 2/3 | 4/7 | 6/7 | 1/5 | 13/22 |
| 2Ground-truth target | Qwen3-VL | 3/3 saw both: 3 of 3 | 4/7 saw both: 6 of 7 | 6/7 saw both: 7 of 7 | 4/5 saw both: 4 of 5 | 17/22 saw both: 20 of 22 |
| 2Ground-truth target | BAGEL | 2/3 saw both: 2 of 3 | 5/7 saw both: 5 of 7 | 5/7 saw both: 7 of 7 | 4/5 saw both: 4 of 5 | 16/22 saw both: 18 of 22 |
| 3Generated target | Qwen3-VL | 3/3 saw both: 3 of 3 | 4/7 saw both: 5 of 7 | 4/7 saw both: 5 of 7 | 2/5 saw both: 3 of 5 | 13/22 saw both: 16 of 22 |
| 3Generated target | BAGEL | 2/3 saw both: 3 of 3 | 4/7 saw both: 4 of 7 | 4/7 saw both: 5 of 7 | 0/5 saw both: 0 of 5 | 10/22 saw both: 12 of 22 |
| 4Control (own view) | Qwen3-VL | 2/3 | 5/7 | 7/7 | 5/5 | 19/22 |
| 4Control (own view) | BAGEL | 2/3 | 7/7 | 6/7 | 5/5 | 20/22 |
| Diagnostic for setting 1: how often the answer equals the correct answer from **A's** spot, i.e. the model answered as if it were standing at A. Where A and B agree this coincides with being correct; on cross_line it is the opposite of being correct. |  |  |  |  |  |  |
| 1answered from A's spot | Qwen3-VL | 2/3 | 5/7 | 6/7 | 3/5 | 16/22 |
| 1answered from A's spot | BAGEL | 2/3 | 4/7 | 6/7 | 4/5 | 16/22 |

### The cross_line test at scale: 92 clean examples

cross_line is the one placement where A and B disagree, so it is the placement that decides the question in the title, and the table above has only 5 such examples. We therefore rendered 177 new cross_line examples in 15 rooms with the pilot's stricter sampling (the two objects at least 20° apart as seen from each observer and each covering at least 1.5% of the photo). After reading the labels from the masks and applying the filter above, 92 remain, every one an "answer differs for A and B" case. Setting 3 was not run on this set: that many BAGEL drawings would take about two hours of GPU time, and the small-set rows already show where that route stands.

Four things were asked, for both models:

- 1**Original**, exactly as in the table above: A's photo plus the sentence about B, "from B's point of view, is X left or right of Y?".
- 1**Identity variant**: the same prompt, but the sentence says B is standing where A stands and facing the same way, so the right answer is A's own. This checks that it is the change of viewpoint that hurts, not the extra sentence about B.
- 2**Ground-truth target**: B's real photo with the direct question. Here B always sees both objects by construction, so the "can you see both?" pre-question was not needed and every answer is left or right.
- 4**Control (own view)**: A's photo with the direct question.

| Setting | Model | correct | answered from A's spot | what it tells us |
| --- | --- | --- | --- | --- |
| 1Original | Qwen3-VL | 18/92 20% | 74/92 80% | the real task: far below the 50% of a coin flip for both models |
| 1Original | BAGEL | 31/92 34% | 61/92 66% |  |
| 1Identity variant (B stands where A stands) | Qwen3-VL | 84/92 91% | 84/92 91% | the same prompt is answered almost perfectly when no viewpoint change is needed |
| 1Identity variant (B stands where A stands) | BAGEL | 88/92 96% | 88/92 96% |  |
| 2Ground-truth target | Qwen3-VL | 88/92 96% | 4/92 4% | B's photo, labelled from what it shows, is read about as well as A's |
| 2Ground-truth target | BAGEL | 89/92 97% | 3/92 3% |  |
| 4Control (own view) | Qwen3-VL | 86/92 93% | 86/92 93% | left/right on A's own photo is easy |
| 4Control (own view) | BAGEL | 87/92 95% | 87/92 95% |  |

**How to read it.** Every answer is one of two words, so a model that guessed would be right about half the time. In the real task (1) Qwen3-VL is right 20% of the time and BAGEL 34%: not random, but systematically wrong. The second column says why: 80% and 66% of their answers are the correct answer *for A's spot*. The models read the left/right relation off the photo they are holding and report it as B's, even though the sentence tells them B is on the far side looking back.

Two controls rule out the easy excuses. The identity variant uses the very same prompt with a sentence about B, and both models are right 91% and 96% of the time, so the sentence itself is not what confuses them; it is the requirement to move. And on A's own photo (4) they are right 93% and 95% of the time, so they can read left/right in a picture.

2 sets the ceiling for the generate-then-read route. Handed B's real photo, the models are right 96% and 97% of the time, about as often as on A's photo. So reading a correct picture of B's view is not the hard part; producing one is. This is the cleaner version of the pilot's headline: the pilot had 16 cross_line examples; with 92 clean ones the same picture holds and the error bars shrink. The next section asks whether the models ever compute B's answer internally on these same examples.

### B shown as a person in A's photo: all settings, 100 sampled examples

Same four settings, but the person stands at B's spot in A's photo and the sentence only says that B is the person and looks where the person faces (1S, 3S, 4S; 2 is B's own photo as before). These examples were sampled afresh so that the figure is clearly visible from A; turn-in-place is impossible here because B would stand where A stands.

| Setting | Model | general (28) | shared_target (28) | cross_line (44) | all (100) |
| --- | --- | --- | --- | --- | --- |
| 1SOriginal | Qwen3-VL | 28/28 | 28/28 | 4/44 | 60/100 |
| 1SOriginal | BAGEL | 27/28 | 28/28 | 4/44 | 59/100 |
| 2Ground-truth target | Qwen3-VL | 26/28 saw both: 26 of 28 | 26/28 saw both: 26 of 28 | 36/44 saw both: 38 of 44 | 88/100 saw both: 90 of 100 |
| 2Ground-truth target | BAGEL | 24/28 saw both: 24 of 28 | 27/28 saw both: 27 of 28 | 34/44 saw both: 36 of 44 | 85/100 saw both: 87 of 100 |
| 3SGenerated target | Qwen3-VL | 8/28 saw both: 12 of 28 | 11/28 saw both: 13 of 28 | 4/44 saw both: 13 of 44 | 23/100 saw both: 38 of 100 |
| 3SGenerated target | BAGEL | 10/28 saw both: 14 of 28 | 11/28 saw both: 14 of 28 | 3/44 saw both: 10 of 44 | 24/100 saw both: 38 of 100 |
| 4SControl (own view) | Qwen3-VL | 28/28 | 28/28 | 40/44 | 96/100 |
| 4SControl (own view) | BAGEL | 27/28 | 28/28 | 41/44 | 96/100 |
| 1Sanswered from A's spot | Qwen3-VL | 28/28 | 28/28 | 40/44 | 96/100 |
| 1Sanswered from A's spot | BAGEL | 27/28 | 28/28 | 40/44 | 95/100 |

**Reading it.** With the person in view, the pattern is the text version's: near-perfect where A and B agree, and on cross_line Qwen3-VL is right 4 of 44 times and BAGEL 4, answering from A's spot in 40 and 40 cases. 3S collapses because BAGEL's drawings from "move to where the person stands" rarely show both objects.

### Described vs shown on the same examples: 47 paired cross_line examples

The direct test. Each of these clean cross_line examples was scored twice: once with the sentence describing B (1, 4, from the runs above) and once with the person placed at B's saved position and heading and A's photo re-rendered (1S, 3S, 4S). B's own photo, the map, the objects and the correct answers are identical, so 2 is not rerun. 47 of the 97 clean cross_line examples qualify: the person has to be inside A's field of view, unoccluded (at least 0.4% of the frame), within 7 m, and must not hide X or Y.

| Setting | Model | B described in words | B shown as a person | changed wrong → right | changed right → wrong |
| --- | --- | --- | --- | --- | --- |
| 1 vs 1S Original | Qwen3-VL | 9/47 | 3/47 | 1 | 7 |
| 1 vs 1S Original | BAGEL | 16/47 | 1/47 | 0 | 15 |
| 2Ground-truth target | Qwen3-VL | 46/47 | same photo, same question: identical | – | – |
| 2Ground-truth target | BAGEL | 46/47 | same photo, same question: identical | – | – |
| 4 vs 4S Control (own view) | Qwen3-VL | 46/47 | 45/47 | 0 | 1 |
| 4 vs 4S Control (own view) | BAGEL | 45/47 | 45/47 | 0 | 0 |
| 1 vs 1S answered from A's spot | Qwen3-VL | 38/47 | 44/47 | – | – |
| 1 vs 1S answered from A's spot | BAGEL | 31/47 | 46/47 | – | – |
| 3S Generated target | Qwen3-VL | not run on the large text set | 0/47 saw both: 18 of 47 | – | – |
| 3S Generated target | BAGEL | not run on the large text set | 2/47 saw both: 15 of 47 | – | – |

**Reading it.** On these 47 examples, seeing B in the picture moves Qwen3-VL from 9 to 3 correct and BAGEL from 16 to 1. Both remain far below the 23 of a coin flip, so the failure is not about locating B from words: given B's exact spot and heading in the picture, the models still answer from A's view, and do so even more consistently. 4 is unchanged by the figure, so the person does not disturb plain left/right reading. The "changed" columns count the per-example flips.

Compare with the pilot's numbers (Qwen2.5-VL-7B and BAGEL, 40 examples)

Same settings where the pilot ran them; the pilot's examples are different renders with centroid-based labels, so read this as indicative. Cells: this run (clean examples) on top, pilot below in grey.

| Setting | Model | rotate_in_place ours 3 · pilot 8 | general ours 7 · pilot 8 | shared_target ours 7 · pilot 8 | cross_line ours 5 · pilot 16 | all ours 22 · pilot 40 |
| --- | --- | --- | --- | --- | --- | --- |
| 1Original | Qwen3-VL | 2/3 7/8 | 5/7 7/8 | 6/7 6/8 | 2/5 6/16 | 15/22 26/40 |
| 1Original | BAGEL | 2/3 6/8 | 4/7 7/8 | 6/7 7/8 | 1/5 6/16 | 13/22 26/40 |
| 2Ground-truth target | Qwen3-VL | 3/3 5/8 | 4/7 6/8 | 6/7 6/8 | 4/5 15/16 | 17/22 32/40 |
| 3Generated target | Qwen3-VL | 3/3 4/8 | 4/7 5/8 | 4/7 6/8 | 2/5 4/16 | 13/22 19/40 |
| 4Control (own view) | Qwen3-VL | 2/3 8/8 | 5/7 7/8 | 7/7 7/8 | 5/5 15/16 | 19/22 37/40 |
| 4Control (own view) | BAGEL | 2/3 8/8 | 7/7 8/8 | 6/7 7/8 | 5/5 14/16 | 20/22 37/40 |
| 1answered from A's spot | Qwen3-VL | 2/3 7/8 | 5/7 7/8 | 6/7 6/8 | 3/5 10/16 | 16/22 30/40 |

The pilot's shown example (apt_0, basket vs kitchen counter, B turns 45° left) next to our re-render

### Pilot

pilot A's photo

A's photo

pilot B's real photo

B's real photo

pilot BAGEL's drawing

BAGEL's drawing

pilot map

map

### This run

ours A's photo

A's photo

ours B's real photo

B's real photo

ours BAGEL's drawing

BAGEL's drawing

ours map

map

The artifact stores no camera poses, so the pose was searched for from the same description. Objects cover 3.5% and 10.3% of A's image (pilot 2.6% and 10.9%). This pair is intrinsically ambiguous: the basket sits on the counter, and the example is excluded from the tables by the filter above.

## Inside the model: does it ever work out B's answer?

### Why look inside

In setting 1 both models mostly answer from A's spot: on the 92 cross_line examples below, Qwen3-VL gives A's answer 80% of the time and BAGEL 66%. Two different stories fit that behaviour:

1. **Never imagined.** The model reads left/right off A's photo and never works out what B would see.
2. **Imagined, then overwritten.** Somewhere in the middle layers the model does work out B's answer, but by the last layer A's answer wins and gets spoken.

Behaviour cannot separate the two. Reading the hidden states layer by layer can, because story 2 predicts that B's answer is present at some depth and story 1 predicts that it never is.

### The tool: a left/right reader

A **reader** is a linear classifier that looks at one layer's hidden state, at the position where the model is about to produce its one-word answer, and says whether that state means "left" or "right". We train one reader per layer on setting-4-style direct questions (half A photos, half B photos), where no perspective change is involved. The label is **the model's own answer** to that direct question, not the ground truth: we want to read what the model takes to be left or right, and the model itself is only 96% right on B photos. Then we apply the reader, unchanged, to the hidden states of setting 1 at the same layer, on 92 cross_line examples it has never seen (training and test rooms are disjoint). Readers are class-weighted logistic regressions on 256 principal components.

Why not train the reader on setting 1 directly: on cross_line the correct answer from B is always the opposite of the correct answer from A, so a reader trained to predict B's answer there would be exactly the same classifier as one trained to predict A's, with the sign flipped. It could not tell the two stories apart.

### The measurement: the B score

At a given layer, take the reader's verdict for each of the 92 cross_line examples and compare it with the two candidate answers. Because B's answer is always the opposite of A's here, each verdict matches exactly one of them. The **B score** is how often the verdict matches B's answer, **averaged over the two answer classes**: the match rate among examples where B's answer is "left" and the rate where it is "right", averaged. The averaging matters because the examples are not balanced: B's answer is "left" in 56 of 92 and "right" in 36. A reader that always says "right" would match B's answer in 39% of examples, which looks like information but is none; averaged over the classes it scores exactly 50%. So 50% is the no-information line, 0% means every verdict is A's answer, 100% every verdict is B's.

**Concrete example** (Qwen3-VL): at layer 24 the reader's verdict matches B's correct answer in 10 of 92 examples and A's in the other 82; class-averaged, the B score is 12%. That layer is holding A's answer. At the last layer (layer 36) the verdict matches B's answer in 21 of 92, B score 23%.

### What we would want to see, and what each outcome means

Outcome A · stays low

B score well below 50% at every layer: the state carries A's answer from start to finish. **Never imagined.**

Outcome B · hump, then drop

B score rises above 50% in middle layers and falls back below 50% by the end. **Imagined, then overwritten.** The interesting case.

Outcome C · flat near 50%

The reader's verdicts are unrelated to either answer: the layer holds no readable left/right, or the reader does not apply to this prompt. Nothing can be concluded.

Outcome D · high at the end

B score above 50% at the last layer: the model answers from B's view. This would already show up as correct answers in setting 1, so we do not expect it here.

### Two checks before trusting the curve

- **Check 1, the reader transfers.** At the last layer the reader's verdict must match what the model actually said in setting 1. The bar is relative: at least 95% of the reader's own ceiling, its held-out agreement with the model's direct answers. If it fails, the left/right direction learned on direct questions is not the one the model uses under the setting-1 prompt, and the curve is Outcome C by construction.
- **Check 2, the identity control.** Run setting 1 again with B described as standing where A stands and facing the same way, so B's answer equals A's. The B score should then be close to 100% at every layer we read. A layer where it is not is a layer where the reader cannot be trusted, whatever the task curve says there.
- A **permutation band** shows what readers trained on shuffled labels produce (chance), and shaded bands are bootstrap 95% intervals over examples. The reader's **left-rate** (how often it says "left") is in every tooltip: a value near 0% or 100% means the reader is outputting a constant at that layer.
- **The correct/wrong split is read class-balanced too.** The examples a model gets right are almost all one word (see the 2×2 tables), so a raw correct-vs-wrong comparison would mostly compare "left" examples with "right" examples.

Method details and the plan text: `repro/probing/PLAN.md`. Features are the last prompt token's hidden state; for BAGEL this is the start token of its first generation step.

The behavioural numbers for these 92 examples (settings 1, 2, 4 and the identity variant) are in the results section above, under "The cross_line test at scale".

### What we found

Qwen3-VL Qwen3-VL-8B-Instruct

**Check 1**: at the last layer the direct-question reader agrees with the model's actual setting-1 answer in 85% of examples (class-balanced) against its own ceiling of 99%; the bar is 94%, so it fails. A reader trained on the identity prompt instead (same task format, B at A's spot) agrees in 96%: it passes.

**Check 2**: the identity control is 91% at the last layer and at least 80% over layers 20–36 (17 layers); only those layers are read below.

**B score** over the trusted layers: 12–47% with the direct reader, 13–50% with the identity-trained reader; at the last layer 23% (95% CI 15–33%) and 17% respectively. No layer rises above the permutation band. In words: wherever a reader can be trusted, the state holds A's answer or no readable answer, never B's. **Outcome A, with 1 of 17 trusted layers near chance (Outcome C) and none above the permutation band; the other 16 hold A's answer. Never imagined**, as far as these readers can see.

**Where the readers can and cannot read.** Up to layer 15, no reader finds left/right in any format: the variable has not been computed yet. It becomes readable at layer 16 in the direct-question format and at layer 16 under the task prompt (no delay). In layers 16–19 it is readable under the task prompt (identity-trained reader 82–92%) but the direct-question reader does not transfer there (identity control 50–65%): the same variable, encoded along a format-specific direction. Even the format-matched reader finds no B there: its B score in that band is 26–34%. From layer 20 on, the two directions agree and the direct reader can be trusted.

**Right vs wrong examples.** Class-balanced within each group, the 18 examples the model answers correctly (B = left in 17, right in 1) score 82% at the last layer (raw 67%) and the 74 it gets wrong 13% (raw 12%). Caution: the correct group has only 1 example in one of the two classes, so its balanced figure rests on that example; the raw figure and the 2×2 table below are the safer reading. The two groups separate at layer 20 and stay apart: for the minority of examples the model gets right, B's answer is present in the late layers; for the rest the state holds A's answer throughout. The default is A, and the correction, when it happens, happens late.

- **Ceiling**: how often the reader matches the model's own answer on held-out direct questions, where it was trained. The best it can do at that layer.
- **Check 1**: how often it matches the model's actual setting-1 answer. Where this touches the ceiling, the reader transfers to the task prompt.
- **Check 2**: the identity control, setting 1 with B standing at A's spot. Should be near 100%; layers below 80% are greyed out in the next chart.
- Dashed line: 95% of the ceiling, the bar Check 1 has to clear.

- **B score**: how often the reader's verdict is B's answer rather than A's, class-balanced. 50% means no information; below it the layer holds A's answer, above it B's. Light blue: bootstrap 95% interval.
- **Chance band**: what readers trained on shuffled labels give. A curve inside it says nothing.
- **Grey columns**: layers where Check 2 failed. Ignore the curve there.

More curves for Qwen3-VL-8B-Instruct: identity-trained reader, right vs wrong examples

- **Identity-trained**: a reader trained on the identity-prompt states instead of direct questions, applied to setting 1. A second opinion that shares the task format.
- **Right**: B score on the 18 examples the model answered correctly, class-balanced within the group (undefined when the group has a single answer word).
- **Wrong**: the same for the 74 examples it got wrong.

2×2 breakdown for Qwen3-VL-8B-Instruct: reader matches B's answer, by (model correct / wrong) × (B's answer left / right)

Counts: correct&left 17, correct&right 1, wrong&left 39, wrong&right 35. The model's correct answers are almost all one word, which is why the correct/wrong curves must be class-balanced.

| layer | correct, B=left | correct, B=right | wrong, B=left | wrong, B=right | reader left-rate | B score (balanced) |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0% | 100% | 0% | 100% | 0% | 50% |
| 9 | 53% | 100% | 13% | 77% | 24% | 51% |
| 18 | 100% | 0% | 100% | 0% | 100% | 50% |
| 20 | 65% | 100% | 23% | 3% | 59% | 21% |
| 21 | 71% | 100% | 51% | 34% | 60% | 47% |
| 27 | 59% | 100% | 5% | 14% | 46% | 19% |
| 36 | 65% | 100% | 3% | 23% | 42% | 23% |

BAGEL BAGEL-7B-MoT

**Check 1**: at the last layer the direct-question reader agrees with the model's actual setting-1 answer in 71% of examples (class-balanced) against its own ceiling of 97%; the bar is 92%, so it fails. A reader trained on the identity prompt instead (same task format, B at A's spot) agrees in 89%: it fails.

**Check 2**: the identity control is 93% at the last layer and at least 80% over layers 20–28 (9 layers); only those layers are read below.

**B score** over the trusted layers: 29–51% with the direct reader, 12–34% with the identity-trained reader; at the last layer 48% (95% CI 37–58%) and 19% respectively. No layer rises above the permutation band. In words: wherever a reader can be trusted, the state holds A's answer or no readable answer, never B's. **Outcome A, with 5 of 9 trusted layers near chance (Outcome C) and none above the permutation band; the other 4 hold A's answer. Never imagined**, as far as these readers can see.

**Where the readers can and cannot read.** Up to layer 9, no reader finds left/right in any format: the variable has not been computed yet. It becomes readable at layer 10 in the direct-question format and at layer 15 under the task prompt (5 layers later). In layers 15–19 it is readable under the task prompt (identity-trained reader 83–96%) but the direct-question reader does not transfer there (identity control 59–73%): the same variable, encoded along a format-specific direction. Even the format-matched reader finds no B there: its B score in that band is 6–30%. From layer 20 on, the two directions agree and the direct reader can be trusted.

**Right vs wrong examples.** The 31 examples this model answers correctly all have the same B answer (left 0, right 31), so a class-balanced score is undefined for that group and the split cannot be read for this model.

- **Ceiling**: how often the reader matches the model's own answer on held-out direct questions, where it was trained. The best it can do at that layer.
- **Check 1**: how often it matches the model's actual setting-1 answer. Where this touches the ceiling, the reader transfers to the task prompt.
- **Check 2**: the identity control, setting 1 with B standing at A's spot. Should be near 100%; layers below 80% are greyed out in the next chart.
- Dashed line: 95% of the ceiling, the bar Check 1 has to clear.

- **B score**: how often the reader's verdict is B's answer rather than A's, class-balanced. 50% means no information; below it the layer holds A's answer, above it B's. Light blue: bootstrap 95% interval.
- **Chance band**: what readers trained on shuffled labels give. A curve inside it says nothing.
- **Grey columns**: layers where Check 2 failed. Ignore the curve there.

More curves for BAGEL-7B-MoT: identity-trained reader, right vs wrong examples

- **Identity-trained**: a reader trained on the identity-prompt states instead of direct questions, applied to setting 1. A second opinion that shares the task format.
- **Right**: B score on the 31 examples the model answered correctly, class-balanced within the group (undefined when the group has a single answer word).
- **Wrong**: the same for the 61 examples it got wrong.

2×2 breakdown for BAGEL-7B-MoT: reader matches B's answer, by (model correct / wrong) × (B's answer left / right)

Counts: correct&left 0, correct&right 31, wrong&left 56, wrong&right 5. The model's correct answers are almost all one word, which is why the correct/wrong curves must be class-balanced.

| layer | correct, B=left | correct, B=right | wrong, B=left | wrong, B=right | reader left-rate | B score (balanced) |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | – | 100% | 0% | 100% | 0% | 50% |
| 7 | – | 55% | 48% | 20% | 49% | 49% |
| 14 | – | 0% | 100% | 0% | 100% | 50% |
| 20 | – | 3% | 61% | 0% | 75% | 32% |
| 21 | – | 0% | 57% | 0% | 74% | 29% |
| 27 | – | 48% | 57% | 20% | 57% | 51% |
| 28 | – | 42% | 59% | 0% | 61% | 48% |

Not done: steering (adding the reader's direction to the residual stream to see whether the answer flips). Scripts: `probing/probe_extract_qwen.py`, `probe_extract_bagel.py`, `probe_analyze.py`.

## How B is shown: the stand-in, and the paired examples

The cleanest test of "does showing B help" keeps everything fixed except one thing. We took the clean cross_line examples already scored above, placed a human figure at B's saved position facing B's saved way, and re-rendered A's photo only. B's own photo, the map, the objects and the correct answers are unchanged, so 2 is identical by construction and is not rerun; 1, 3 and 4 are rerun with the new A photo and the new sentence: "Observer B is the person you can see in the photo. B stands where the person stands and looks in the direction the person faces."

47 of the clean cross_line examples qualify: the person has to be inside A's field of view, unoccluded (at least 0.4% of the frame), within 7 m, and must not hide X or Y. In the others B stands behind A or out of frame, where a stand-in cannot be shown.

First 8 paired examples; A's photo now has the person outlined green; the text-only sentence it replaces is shown below each.

**apt_1_cross_line_s101_00** · apt_1 · person covers 0.5% of A's photo

Question: is the sofa left or right of the kitchen counter?\
Correct answer from A's spot: **left** · from B's spot: **right**

Shown: Observer B is the person you can see in the photo. B stands where the person stands and looks in the direction the person faces.

Described (text version): Observer B is standing about 6.0 metres ahead of you and to your left, turned about 120 degrees to the right compared with your heading.

| setting 1 | Qwen3-VL | BAGEL |
| --- | --- | --- |
| described | left | right |
| shown | left | left |
| 3 shown, drawing | can't see both | can't see both |

A

**A's photo with the person** · 1 4

B

**B's real photo** · 2, unchanged

drawing

**BAGEL's drawing of B's view** · 3

map

**Map from above**

**apt_1_cross_line_s102_00** · apt_1 · person covers 0.4% of A's photo

Question: is the sofa left or right of the kitchen counter?\
Correct answer from A's spot: **right** · from B's spot: **left**

Shown: Observer B is the person you can see in the photo. B stands where the person stands and looks in the direction the person faces.

Described (text version): Observer B is standing about 6.5 metres ahead of you and to your right, turned about 120 degrees to the left compared with your heading.

| setting 1 | Qwen3-VL | BAGEL |
| --- | --- | --- |
| described | right | right |
| shown | right | right |
| 3 shown, drawing | can't see both | can't see both |

A

**A's photo with the person** · 1 4

B

**B's real photo** · 2, unchanged

drawing

**BAGEL's drawing of B's view** · 3

map

**Map from above**

**apt_1_cross_line_s102_01** · apt_1 · person covers 0.45% of A's photo

Question: is the kitchen counter left or right of the sofa?\
Correct answer from A's spot: **left** · from B's spot: **right**

Shown: Observer B is the person you can see in the photo. B stands where the person stands and looks in the direction the person faces.

Described (text version): Observer B is standing about 6.0 metres ahead of you and to your right, turned about 120 degrees to the left compared with your heading.

| setting 1 | Qwen3-VL | BAGEL |
| --- | --- | --- |
| described | left | right |
| shown | left | left |
| 3 shown, drawing | left | left |

A

**A's photo with the person** · 1 4

B

**B's real photo** · 2, unchanged

drawing

**BAGEL's drawing of B's view** · 3

map

**Map from above**

**apt_1_cross_line_s102_02** · apt_1 · person covers 0.46% of A's photo

Question: is the kitchen counter left or right of the sofa?\
Correct answer from A's spot: **left** · from B's spot: **right**

Shown: Observer B is the person you can see in the photo. B stands where the person stands and looks in the direction the person faces.

Described (text version): Observer B is standing about 6.0 metres ahead of you and to your right, turned about 120 degrees to the left compared with your heading.

| setting 1 | Qwen3-VL | BAGEL |
| --- | --- | --- |
| described | left | right |
| shown | left | left |
| 3 shown, drawing | can't see both | can't see both |

A

**A's photo with the person** · 1 4

B

**B's real photo** · 2, unchanged

drawing

**BAGEL's drawing of B's view** · 3

map

**Map from above**

**apt_2_cross_line_s102_00** · apt_2 · person covers 0.45% of A's photo

Question: is the sofa left or right of the kitchen counter?\
Correct answer from A's spot: **right** · from B's spot: **left**

Shown: Observer B is the person you can see in the photo. B stands where the person stands and looks in the direction the person faces.

Described (text version): Observer B is standing about 6.0 metres ahead of you and to your right, turned about 120 degrees to the left compared with your heading.

| setting 1 | Qwen3-VL | BAGEL |
| --- | --- | --- |
| described | right | right |
| shown | right | right |
| 3 shown, drawing | right | right |

A

**A's photo with the person** · 1 4

B

**B's real photo** · 2, unchanged

drawing

**BAGEL's drawing of B's view** · 3

map

**Map from above**

**v3_sc0_staging_00_cross_line_s106_02** · v3_sc0_staging_00 · person covers 0.49% of A's photo

Question: is the tv stand left or right of the fridge?\
Correct answer from A's spot: **left** · from B's spot: **right**

Shown: Observer B is the person you can see in the photo. B stands where the person stands and looks in the direction the person faces.

Described (text version): Observer B is standing about 5.0 metres ahead of you, turned about 165 degrees to the right compared with your heading.

| setting 1 | Qwen3-VL | BAGEL |
| --- | --- | --- |
| described | left | right |
| shown | left | left |
| 3 shown, drawing | left | can't see both |

A

**A's photo with the person** · 1 4

B

**B's real photo** · 2, unchanged

drawing

**BAGEL's drawing of B's view** · 3

map

**Map from above**

**v3_sc0_staging_00_cross_line_s106_04** · v3_sc0_staging_00 · person covers 0.44% of A's photo

Question: is the fridge left or right of the tv stand?\
Correct answer from A's spot: **right** · from B's spot: **left**

Shown: Observer B is the person you can see in the photo. B stands where the person stands and looks in the direction the person faces. B is facing the tv stand.

Described (text version): Observer B is standing about 6.0 metres ahead of you and to your left, turned about 135 degrees to the right compared with your heading. B is facing the tv stand.

| setting 1 | Qwen3-VL | BAGEL |
| --- | --- | --- |
| described | right | right |
| shown | right | right |
| 3 shown, drawing | can't see both | can't see both |

A

**A's photo with the person** · 1 4

B

**B's real photo** · 2, unchanged

drawing

**BAGEL's drawing of B's view** · 3

map

**Map from above**

**v3_sc0_staging_00_cross_line_s106_05** · v3_sc0_staging_00 · person covers 0.56% of A's photo

Question: is the fridge left or right of the tv stand?\
Correct answer from A's spot: **left** · from B's spot: **right**

Shown: Observer B is the person you can see in the photo. B stands where the person stands and looks in the direction the person faces.

Described (text version): Observer B is standing about 4.5 metres ahead of you, turned about 165 degrees to the left compared with your heading.

| setting 1 | Qwen3-VL | BAGEL |
| --- | --- | --- |
| described | left | right |
| shown | left | left |
| 3 shown, drawing | can't see both | right |

A

**A's photo with the person** · 1 4

B

**B's real photo** · 2, unchanged

drawing

**BAGEL's drawing of B's view** · 3

map

**Map from above**

## Variant, additional examples sampled with the person in view

Same task, one change: a stand-in for B, a human figure (Habitat's SMPL-X avatar, standing, arms down), is placed in the room at B's exact position, turned to face the way B faces, and rendered into A's photo. The sentence about B no longer gives a distance or a turn angle; it says that B is the person in the photo and looks where the person faces. If the models fail setting 1 because they cannot build B's position from words, this should help; if they fail because they do not take another viewpoint at all, it should not.

- **What the model gets in setting 1**: A's photo with the person in it, plus "Observer B is the person you can see in the photo. B stands where the person stands and looks in the direction the person faces." Then the usual question.
- **Setting 2** is unchanged: B's own photo, taken from B's eyes, without the person in it.
- **Setting 3**: BAGEL is told "the camera moves to where the person stands and turns to face the direction the person faces".
- **Setting 4**: the direct question on A's photo with the person in it.
- **Sampling**: the person must be clearly visible from A (at least 0.5% of the frame, within 5 m) without hiding X or Y; the mask-based clean rules apply while sampling, so every example passes the filter above. Turn-in-place is not possible in this variant, since B would stand where A stands.
- **Why a human figure**: a Fetch robot was tried first, but its front is hard to read at a distance; a person's face and stance give the direction at a glance. Spot (a robot dog) also loads but is low and small from across a room.

### 100 examples scored; the tables are in the results section above

In A's photo the person is outlined green; X and Y as before. The text-only description that the earlier version would have used is shown for comparison.

Showing the first 12 of 100 examples; all are in `repro/out_marker/` and in the report there.

**apt_0_general_s11_00** · apt_0 · general · person covers 13.38% of A's photo

Question: is the sofa left or right of the tv screen?\
Correct answer from A's spot: **left** · from B's spot: **left**

Sentence about B (this variant): Observer B is the person you can see in the photo. B stands where the person stands and looks in the direction the person faces.

Text-only version, for comparison: Observer B is standing about 1.0 metres ahead of you and to your right, turned about 15 degrees to the left compared with your heading.

| setting | Qwen3-VL | BAGEL |
| --- | --- | --- |
| 1Original | left | left |
| 2Ground-truth target | left | left |
| 3Generated target | can't see both | can't see both |
| 4Control (own view) | left | left |

A

**A's photo with the person** · input for 1 and 4

B

**B's real photo** · 2

drawing

**BAGEL's drawing of B's view** · 3

map

**Map from above** (blue A, red B)

**apt_0_general_s11_01** · apt_0 · general · person covers 14.61% of A's photo

Question: is the fridge left or right of the kitchen counter?\
Correct answer from A's spot: **left** · from B's spot: **left**

Sentence about B (this variant): Observer B is the person you can see in the photo. B stands where the person stands and looks in the direction the person faces.

Text-only version, for comparison: Observer B is standing about 1.0 metres ahead of you, facing the same direction as you.

| setting | Qwen3-VL | BAGEL |
| --- | --- | --- |
| 1Original | left | right |
| 2Ground-truth target | left | left |
| 3Generated target | right | right |
| 4Control (own view) | left | right |

A

**A's photo with the person** · input for 1 and 4

B

**B's real photo** · 2

drawing

**BAGEL's drawing of B's view** · 3

map

**Map from above** (blue A, red B)

**apt_0_shared_target_s11_00** · apt_0 · shared_target · person covers 17.86% of A's photo

Question: is the sofa left or right of the tv screen?\
Correct answer from A's spot: **left** · from B's spot: **left**

Sentence about B (this variant): Observer B is the person you can see in the photo. B stands where the person stands and looks in the direction the person faces.

Text-only version, for comparison: Observer B is standing about 0.5 metres ahead of you and to your right, facing the same direction as you.

| setting | Qwen3-VL | BAGEL |
| --- | --- | --- |
| 1Original | left | left |
| 2Ground-truth target | left | left |
| 3Generated target | can't see both | can't see both |
| 4Control (own view) | left | left |

A

**A's photo with the person** · input for 1 and 4

B

**B's real photo** · 2

drawing

**BAGEL's drawing of B's view** · 3

map

**Map from above** (blue A, red B)

**apt_0_shared_target_s11_01** · apt_0 · shared_target · person covers 5.95% of A's photo

Question: is the sofa left or right of the tv screen?\
Correct answer from A's spot: **left** · from B's spot: **left**

Sentence about B (this variant): Observer B is the person you can see in the photo. B stands where the person stands and looks in the direction the person faces.

Text-only version, for comparison: Observer B is standing about 1.0 metres ahead of you and to your right, turned about 30 degrees to the left compared with your heading.

| setting | Qwen3-VL | BAGEL |
| --- | --- | --- |
| 1Original | left | left |
| 2Ground-truth target | left | left |
| 3Generated target | right | right |
| 4Control (own view) | left | left |

A

**A's photo with the person** · input for 1 and 4

B

**B's real photo** · 2

drawing

**BAGEL's drawing of B's view** · 3

map

**Map from above** (blue A, red B)

**apt_1_general_s11_00** · apt_1 · general · person covers 9.3% of A's photo

Question: is the tv screen left or right of the fridge?\
Correct answer from A's spot: **left** · from B's spot: **left**

Sentence about B (this variant): Observer B is the person you can see in the photo. B stands where the person stands and looks in the direction the person faces.

Text-only version, for comparison: Observer B is standing about 1.0 metres ahead of you, facing the same direction as you. B stands next to the sofa.

| setting | Qwen3-VL | BAGEL |
| --- | --- | --- |
| 1Original | left | left |
| 2Ground-truth target | left | left |
| 3Generated target | can't see both | can't see both |
| 4Control (own view) | left | left |

A

**A's photo with the person** · input for 1 and 4

B

**B's real photo** · 2

drawing

**BAGEL's drawing of B's view** · 3

map

**Map from above** (blue A, red B)

**apt_1_general_s11_01** · apt_1 · general · person covers 2.74% of A's photo

Question: is the chest of drawers left or right of the fridge?\
Correct answer from A's spot: **right** · from B's spot: **right**

Sentence about B (this variant): Observer B is the person you can see in the photo. B stands where the person stands and looks in the direction the person faces. B is facing the kitchen counter.

Text-only version, for comparison: Observer B is standing about 1.5 metres ahead of you and to your right, facing the same direction as you. B is facing the kitchen counter.

| setting | Qwen3-VL | BAGEL |
| --- | --- | --- |
| 1Original | right | right |
| 2Ground-truth target | right | right |
| 3Generated target | left | left |
| 4Control (own view) | right | right |

A

**A's photo with the person** · input for 1 and 4

B

**B's real photo** · 2

drawing

**BAGEL's drawing of B's view** · 3

map

**Map from above** (blue A, red B)

**apt_1_shared_target_s11_00** · apt_1 · shared_target · person covers 5.02% of A's photo

Question: is the fridge left or right of the chest of drawers?\
Correct answer from A's spot: **left** · from B's spot: **left**

Sentence about B (this variant): Observer B is the person you can see in the photo. B stands where the person stands and looks in the direction the person faces. B is facing the kitchen counter.

Text-only version, for comparison: Observer B is standing about 1.5 metres ahead of you, turned about 15 degrees to the right compared with your heading. B is facing the kitchen counter.

| setting | Qwen3-VL | BAGEL |
| --- | --- | --- |
| 1Original | left | left |
| 2Ground-truth target | left | left |
| 3Generated target | can't see both | can't see both |
| 4Control (own view) | left | left |

A

**A's photo with the person** · input for 1 and 4

B

**B's real photo** · 2

drawing

**BAGEL's drawing of B's view** · 3

map

**Map from above** (blue A, red B)

**apt_1_shared_target_s11_01** · apt_1 · shared_target · person covers 8.62% of A's photo

Question: is the fridge left or right of the chest of drawers?\
Correct answer from A's spot: **left** · from B's spot: **left**

Sentence about B (this variant): Observer B is the person you can see in the photo. B stands where the person stands and looks in the direction the person faces. B is facing the kitchen counter.

Text-only version, for comparison: Observer B is standing about 0.5 metres ahead of you and to your right, facing the same direction as you. B is facing the kitchen counter.

| setting | Qwen3-VL | BAGEL |
| --- | --- | --- |
| 1Original | left | left |
| 2Ground-truth target | left | left |
| 3Generated target | left | right |
| 4Control (own view) | left | left |

A

**A's photo with the person** · input for 1 and 4

B

**B's real photo** · 2

drawing

**BAGEL's drawing of B's view** · 3

map

**Map from above** (blue A, red B)

**apt_4_general_s28_00** · apt_4 · general · person covers 3.05% of A's photo

Question: is the tv screen left or right of the sofa?\
Correct answer from A's spot: **left** · from B's spot: **left**

Sentence about B (this variant): Observer B is the person you can see in the photo. B stands where the person stands and looks in the direction the person faces.

Text-only version, for comparison: Observer B is standing about 2.5 metres ahead of you and to your right, turned about 45 degrees to the left compared with your heading. B stands next to the sofa.

| setting | Qwen3-VL | BAGEL |
| --- | --- | --- |
| 1Original | left | left |
| 2Ground-truth target | left | left |
| 3Generated target | can't see both | can't see both |
| 4Control (own view) | left | left |

A

**A's photo with the person** · input for 1 and 4

B

**B's real photo** · 2

drawing

**BAGEL's drawing of B's view** · 3

map

**Map from above** (blue A, red B)

**apt_4_general_s28_01** · apt_4 · general · person covers 3.08% of A's photo

Question: is the chest of drawers left or right of the kitchen counter?\
Correct answer from A's spot: **right** · from B's spot: **right**

Sentence about B (this variant): Observer B is the person you can see in the photo. B stands where the person stands and looks in the direction the person faces. B is facing the basket.

Text-only version, for comparison: Observer B is standing about 2.5 metres ahead of you and to your left, turned about 45 degrees to the right compared with your heading. B is facing the basket.

| setting | Qwen3-VL | BAGEL |
| --- | --- | --- |
| 1Original | right | right |
| 2Ground-truth target | right | right |
| 3Generated target | left | can't see both |
| 4Control (own view) | right | right |

A

**A's photo with the person** · input for 1 and 4

B

**B's real photo** · 2

drawing

**BAGEL's drawing of B's view** · 3

map

**Map from above** (blue A, red B)

**apt_4_shared_target_s28_00** · apt_4 · shared_target · person covers 14.94% of A's photo

Question: is the tv screen left or right of the sofa?\
Correct answer from A's spot: **left** · from B's spot: **left**

Sentence about B (this variant): Observer B is the person you can see in the photo. B stands where the person stands and looks in the direction the person faces.

Text-only version, for comparison: Observer B is standing about 1.0 metres ahead of you, facing the same direction as you.

| setting | Qwen3-VL | BAGEL |
| --- | --- | --- |
| 1Original | left | left |
| 2Ground-truth target | left | left |
| 3Generated target | left | left |
| 4Control (own view) | left | left |

A

**A's photo with the person** · input for 1 and 4

B

**B's real photo** · 2

drawing

**BAGEL's drawing of B's view** · 3

map

**Map from above** (blue A, red B)

**apt_4_shared_target_s28_01** · apt_4 · shared_target · person covers 0.74% of A's photo

Question: is the fridge left or right of the tv screen?\
Correct answer from A's spot: **right** · from B's spot: **right**

Sentence about B (this variant): Observer B is the person you can see in the photo. B stands where the person stands and looks in the direction the person faces. B is facing the sofa.

Text-only version, for comparison: Observer B is standing about 1.5 metres ahead of you and to your right, turned about 15 degrees to the left compared with your heading. B is facing the sofa.

| setting | Qwen3-VL | BAGEL |
| --- | --- | --- |
| 1Original | right | right |
| 2Ground-truth target | can't see both | right |
| 3Generated target | can't see both | can't see both |
| 4Control (own view) | right | right |

A

**A's photo with the person** · input for 1 and 4

B

**B's real photo** · 2

drawing

**BAGEL's drawing of B's view** · 3

map

**Map from above** (blue A, red B)

## Every example, by mode

For each example: the question and the sentence the model gets, both models' answers in every setting, the three views (A's photo, B's real photo, BAGEL's drawing for setting 3) and a map from above. In A's and B's photos the two objects are outlined: X Y. "Looks like" is CLIP cosine similarity between two images, from 0 (different) to 1 (identical).

[rotate_in_place (3)](#rotate_in_place)[general (7)](#general)[shared_target (7)](#shared_target)[cross_line (5)](#cross_line)

### rotate_in_place · Turn in place3 clean of 9 examples · 1 correct: Qwen3-VL 2/3 · BAGEL 2/3

**apt_0_rotate_in_place_s1_00** · apt_0 · same answer for A and B · excluded from the tables: the objects overlap in A's photo; the objects overlap in B's photo

Question: is the kitchen counter left or right of the fridge?\
Correct answer from A's spot: **right** · from B's spot: **right**

Sentence about B: Observer B is standing exactly where you are, but turned about 60 degrees to the right compared with your heading.

| setting | Qwen3-VL | BAGEL |
| --- | --- | --- |
| 1Original | right | right |
| 2Ground-truth target | right | right |
| 3Generated target | right | left |
| 4Control (own view) | right | right |

Looks like: A's photo ↔ B's photo 0.928 · drawing ↔ A 0.949 · drawing ↔ B 0.896

A

**A's photo** · input for 1 and 4

B

**B's real photo** · 2 (hidden in 1)

drawing

**BAGEL's drawing of B's view** · 3

map

**Map from above** (blue A, red B)

**apt_0_rotate_in_place_s1_01** · apt_0 · same answer for A and B · excluded from the tables: the objects overlap in A's photo; the objects overlap in B's photo

Question: is the basket left or right of the kitchen counter?\
Correct answer from A's spot: **right** · from B's spot: **right**

Sentence about B: Observer B is standing exactly where you are, but turned about 45 degrees to the left compared with your heading. B is facing the kitchen counter.

| setting | Qwen3-VL | BAGEL |
| --- | --- | --- |
| 1Original | right | right |
| 2Ground-truth target | right | right |
| 3Generated target | left | left |
| 4Control (own view) | right | right |

Looks like: A's photo ↔ B's photo 0.917 · drawing ↔ A 0.896 · drawing ↔ B 0.869

A

**A's photo** · input for 1 and 4

B

**B's real photo** · 2 (hidden in 1)

drawing

**BAGEL's drawing of B's view** · 3

map

**Map from above** (blue A, red B)

**apt_1_rotate_in_place_s1_00** · apt_1 · same answer for A and B · excluded from the tables: A stands 1.5 m from an object; B stands 1.5 m from an object

Question: is the basket left or right of the fridge?\
Correct answer from A's spot: **right** · from B's spot: **right**

Sentence about B: Observer B is standing exactly where you are, but turned about 60 degrees to the left compared with your heading. B stands next to the chest of drawers.

| setting | Qwen3-VL | BAGEL |
| --- | --- | --- |
| 1Original | right | right |
| 2Ground-truth target | right | right |
| 3Generated target | left | left |
| 4Control (own view) | right | right |

Looks like: A's photo ↔ B's photo 0.848 · drawing ↔ A 0.813 · drawing ↔ B 0.78

A

**A's photo** · input for 1 and 4

B

**B's real photo** · 2 (hidden in 1)

drawing

**BAGEL's drawing of B's view** · 3

map

**Map from above** (blue A, red B)

**apt_1_rotate_in_place_s1_01** · apt_1 · same answer for A and B · excluded from the tables: the objects overlap in A's photo; A stands 1.3 m from an object; the objects overlap in B's photo; B stands 1.3 m from an object

Question: is the tv screen left or right of the tv stand?\
Correct answer from A's spot: **right** · from B's spot: **right**

Sentence about B: Observer B is standing exactly where you are, but turned about 15 degrees to the left compared with your heading. B stands next to the tv stand, B is facing the tv stand.

| setting | Qwen3-VL | BAGEL |
| --- | --- | --- |
| 1Original | left | right |
| 2Ground-truth target | right | right |
| 3Generated target | right | right |
| 4Control (own view) | right | right |

Looks like: A's photo ↔ B's photo 0.941 · drawing ↔ A 0.959 · drawing ↔ B 0.94

A

**A's photo** · input for 1 and 4

B

**B's real photo** · 2 (hidden in 1)

drawing

**BAGEL's drawing of B's view** · 3

map

**Map from above** (blue A, red B)

**pilot0001_rotate_in_place_s0_00** · apt_0 · same answer for A and B · excluded from the tables: the objects overlap in A's photo; their centres are only 5 px apart for A; A stands 1.0 m from an object; the objects overlap in B's photo; their centres are only 26 px apart for B; B stands 1.0 m from an object

Question: is the basket left or right of the kitchen counter?\
Correct answer from A's spot: **right** · from B's spot: **right**

Sentence about B: Observer B is standing exactly where you are, but turned about 45 degrees to the left compared with your heading. B stands next to the chest of drawers, B is facing the kitchen counter.

| setting | Qwen3-VL | BAGEL |
| --- | --- | --- |
| 1Original | left | right |
| 2Ground-truth target | right | right |
| 3Generated target | left | left |
| 4Control (own view) | left | right |

Looks like: A's photo ↔ B's photo 0.924 · drawing ↔ A 0.86 · drawing ↔ B 0.879 (closer to B)

A

**A's photo** · input for 1 and 4

B

**B's real photo** · 2 (hidden in 1)

drawing

**BAGEL's drawing of B's view** · 3

map

**Map from above** (blue A, red B)

**v3_sc0_staging_00_rotate_in_place_s1_00** · v3_sc0_staging_00 · same answer for A and B

Question: is the fridge left or right of the tv stand?\
Correct answer from A's spot: **right** · from B's spot: **right**

Sentence about B: Observer B is standing exactly where you are, but turned about 45 degrees to the right compared with your heading.

| setting | Qwen3-VL | BAGEL |
| --- | --- | --- |
| 1Original | right | right |
| 2Ground-truth target | right | can't see both |
| 3Generated target | right | right |
| 4Control (own view) | right | right |

Looks like: A's photo ↔ B's photo 0.921 · drawing ↔ A 0.935 · drawing ↔ B 0.937 (closer to B)

A

**A's photo** · input for 1 and 4

B

**B's real photo** · 2 (hidden in 1)

drawing

**BAGEL's drawing of B's view** · 3

map

**Map from above** (blue A, red B)

**v3_sc0_staging_00_rotate_in_place_s1_01** · v3_sc0_staging_00 · same answer for A and B

Question: is the tv screen left or right of the kitchen counter?\
Correct answer from A's spot: **left** · from B's spot: **left**

Sentence about B: Observer B is standing exactly where you are, but turned about 30 degrees to the left compared with your heading.

| setting | Qwen3-VL | BAGEL |
| --- | --- | --- |
| 1Original | left | left |
| 2Ground-truth target | left | left |
| 3Generated target | left | left |
| 4Control (own view) | left | left |

Looks like: A's photo ↔ B's photo 0.938 · drawing ↔ A 0.948 · drawing ↔ B 0.94

A

**A's photo** · input for 1 and 4

B

**B's real photo** · 2 (hidden in 1)

drawing

**BAGEL's drawing of B's view** · 3

map

**Map from above** (blue A, red B)

**v3_sc1_staging_03_rotate_in_place_s1_00** · v3_sc1_staging_03 · same answer for A and B · excluded from the tables: the objects overlap in A's photo; the objects overlap in B's photo

Question: is the tv screen left or right of the tv stand?\
Correct answer from A's spot: **right** · from B's spot: **right**

Sentence about B: Observer B is standing exactly where you are, but turned about 45 degrees to the left compared with your heading.

| setting | Qwen3-VL | BAGEL |
| --- | --- | --- |
| 1Original | left | right |
| 2Ground-truth target | right | right |
| 3Generated target | left | right |
| 4Control (own view) | left | right |

Looks like: A's photo ↔ B's photo 0.948 · drawing ↔ A 0.968 · drawing ↔ B 0.939

A

**A's photo** · input for 1 and 4

B

**B's real photo** · 2 (hidden in 1)

drawing

**BAGEL's drawing of B's view** · 3

map

**Map from above** (blue A, red B)

**v3_sc1_staging_03_rotate_in_place_s1_01** · v3_sc1_staging_03 · same answer for A and B

Question: is the chest of drawers left or right of the tv screen?\
Correct answer from A's spot: **left** · from B's spot: **left**

Sentence about B: Observer B is standing exactly where you are, but turned about 45 degrees to the left compared with your heading.

| setting | Qwen3-VL | BAGEL |
| --- | --- | --- |
| 1Original | right | right |
| 2Ground-truth target | left | left |
| 3Generated target | left | right |
| 4Control (own view) | right | right |

Looks like: A's photo ↔ B's photo 0.919 · drawing ↔ A 0.969 · drawing ↔ B 0.895

A

**A's photo** · input for 1 and 4

B

**B's real photo** · 2 (hidden in 1)

drawing

**BAGEL's drawing of B's view** · 3

map

**Map from above** (blue A, red B)

### general · Move anywhere7 clean of 8 examples · 1 correct: Qwen3-VL 5/7 · BAGEL 4/7

**apt_0_general_s1_00** · apt_0 · same answer for A and B

Question: is the fridge left or right of the tv screen?\
Correct answer from A's spot: **right** · from B's spot: **right**

Sentence about B: Observer B is standing about 3.0 metres to your right, turned about 15 degrees to the left compared with your heading.

| setting | Qwen3-VL | BAGEL |
| --- | --- | --- |
| 1Original | right | right |
| 2Ground-truth target | right | right |
| 3Generated target | can't see both | can't see both |
| 4Control (own view) | right | right |

Looks like: A's photo ↔ B's photo 0.952 · drawing ↔ A 0.942 · drawing ↔ B 0.932

A

**A's photo** · input for 1 and 4

B

**B's real photo** · 2 (hidden in 1)

drawing

**BAGEL's drawing of B's view** · 3

map

**Map from above** (blue A, red B)

**apt_0_general_s1_01** · apt_0 · same answer for A and B

Question: is the basket left or right of the chest of drawers?\
Correct answer from A's spot: **left** · from B's spot: **left**

Sentence about B: Observer B is standing about 0.5 metres behind you and to your left, turned about 15 degrees to the right compared with your heading. B is facing the basket.

| setting | Qwen3-VL | BAGEL |
| --- | --- | --- |
| 1Original | right | right |
| 2Ground-truth target | right | can't see both |
| 3Generated target | left | left |
| 4Control (own view) | right | left |

Looks like: A's photo ↔ B's photo 0.932 · drawing ↔ A 0.906 · drawing ↔ B 0.868

A

**A's photo** · input for 1 and 4

B

**B's real photo** · 2 (hidden in 1)

drawing

**BAGEL's drawing of B's view** · 3

map

**Map from above** (blue A, red B)

**apt_1_general_s1_00** · apt_1 · same answer for A and B

Question: is the fridge left or right of the tv screen?\
Correct answer from A's spot: **right** · from B's spot: **right**

Sentence about B: Observer B is standing about 3.0 metres to your right, turned about 15 degrees to the left compared with your heading. B is facing the fridge.

| setting | Qwen3-VL | BAGEL |
| --- | --- | --- |
| 1Original | right | right |
| 2Ground-truth target | can't see both | right |
| 3Generated target | can't see both | can't see both |
| 4Control (own view) | right | right |

Looks like: A's photo ↔ B's photo 0.9 · drawing ↔ A 0.942 · drawing ↔ B 0.865

A

**A's photo** · input for 1 and 4

B

**B's real photo** · 2 (hidden in 1)

drawing

**BAGEL's drawing of B's view** · 3

map

**Map from above** (blue A, red B)

**apt_1_general_s1_01** · apt_1 · same answer for A and B

Question: is the basket left or right of the chest of drawers?\
Correct answer from A's spot: **left** · from B's spot: **left**

Sentence about B: Observer B is standing about 1.5 metres to your right, turned about 30 degrees to the left compared with your heading. B is facing the chest of drawers.

| setting | Qwen3-VL | BAGEL |
| --- | --- | --- |
| 1Original | left | right |
| 2Ground-truth target | right | left |
| 3Generated target | right | left |
| 4Control (own view) | right | left |

Looks like: A's photo ↔ B's photo 0.949 · drawing ↔ A 0.859 · drawing ↔ B 0.87 (closer to B)

A

**A's photo** · input for 1 and 4

B

**B's real photo** · 2 (hidden in 1)

drawing

**BAGEL's drawing of B's view** · 3

map

**Map from above** (blue A, red B)

**v3_sc0_staging_00_general_s1_00** · v3_sc0_staging_00 · same answer for A and B

Question: is the fridge left or right of the tv stand?\
Correct answer from A's spot: **right** · from B's spot: **right**

Sentence about B: Observer B is standing about 2.5 metres ahead of you and to your right, turned about 30 degrees to the left compared with your heading.

| setting | Qwen3-VL | BAGEL |
| --- | --- | --- |
| 1Original | right | right |
| 2Ground-truth target | right | right |
| 3Generated target | right | right |
| 4Control (own view) | right | right |

Looks like: A's photo ↔ B's photo 0.957 · drawing ↔ A 0.963 · drawing ↔ B 0.951

A

**A's photo** · input for 1 and 4

B

**B's real photo** · 2 (hidden in 1)

drawing

**BAGEL's drawing of B's view** · 3

map

**Map from above** (blue A, red B)

**v3_sc0_staging_00_general_s1_01** · v3_sc0_staging_00 · same answer for A and B

Question: is the tv screen left or right of the kitchen counter?\
Correct answer from A's spot: **left** · from B's spot: **left**

Sentence about B: Observer B is standing about 0.5 metres ahead of you and to your left, turned about 15 degrees to the right compared with your heading.

| setting | Qwen3-VL | BAGEL |
| --- | --- | --- |
| 1Original | left | right |
| 2Ground-truth target | left | left |
| 3Generated target | left | left |
| 4Control (own view) | left | left |

Looks like: A's photo ↔ B's photo 0.961 · drawing ↔ A 0.963 · drawing ↔ B 0.949

A

**A's photo** · input for 1 and 4

B

**B's real photo** · 2 (hidden in 1)

drawing

**BAGEL's drawing of B's view** · 3

map

**Map from above** (blue A, red B)

**v3_sc1_staging_03_general_s1_00** · v3_sc1_staging_03 · same answer for A and B

Question: is the tv screen left or right of the chest of drawers?\
Correct answer from A's spot: **right** · from B's spot: **right**

Sentence about B: Observer B is standing about 1.0 metres ahead of you and to your left, turned about 15 degrees to the right compared with your heading.

| setting | Qwen3-VL | BAGEL |
| --- | --- | --- |
| 1Original | left | right |
| 2Ground-truth target | right | can't see both |
| 3Generated target | right | can't see both |
| 4Control (own view) | right | right |

Looks like: A's photo ↔ B's photo 0.966 · drawing ↔ A 0.965 · drawing ↔ B 0.946

A

**A's photo** · input for 1 and 4

B

**B's real photo** · 2 (hidden in 1)

drawing

**BAGEL's drawing of B's view** · 3

map

**Map from above** (blue A, red B)

**v3_sc1_staging_03_general_s1_01** · v3_sc1_staging_03 · same answer for A and B · excluded from the tables: B stands 1.5 m from an object

Question: is the chest of drawers left or right of the tv stand?\
Correct answer from A's spot: **left** · from B's spot: **left**

Sentence about B: Observer B is standing about 2.5 metres ahead of you and to your left, turned about 30 degrees to the right compared with your heading. B stands next to the chest of drawers.

| setting | Qwen3-VL | BAGEL |
| --- | --- | --- |
| 1Original | left | left |
| 2Ground-truth target | left | left |
| 3Generated target | can't see both | can't see both |
| 4Control (own view) | left | left |

Looks like: A's photo ↔ B's photo 0.948 · drawing ↔ A 0.925 · drawing ↔ B 0.918

A

**A's photo** · input for 1 and 4

B

**B's real photo** · 2 (hidden in 1)

drawing

**BAGEL's drawing of B's view** · 3

map

**Map from above** (blue A, red B)

### shared_target · Look at the same thing7 clean of 8 examples · 1 correct: Qwen3-VL 6/7 · BAGEL 6/7

**apt_0_shared_target_s1_00** · apt_0 · same answer for A and B

Question: is the fridge left or right of the tv screen?\
Correct answer from A's spot: **right** · from B's spot: **right**

Sentence about B: Observer B is standing about 2.0 metres to your right, turned about 15 degrees to the left compared with your heading.

| setting | Qwen3-VL | BAGEL |
| --- | --- | --- |
| 1Original | right | right |
| 2Ground-truth target | right | right |
| 3Generated target | can't see both | can't see both |
| 4Control (own view) | right | right |

Looks like: A's photo ↔ B's photo 0.961 · drawing ↔ A 0.94 · drawing ↔ B 0.961 (closer to B)

A

**A's photo** · input for 1 and 4

B

**B's real photo** · 2 (hidden in 1)

drawing

**BAGEL's drawing of B's view** · 3

map

**Map from above** (blue A, red B)

**apt_0_shared_target_s1_01** · apt_0 · same answer for A and B

Question: is the kitchen counter left or right of the chest of drawers?\
Correct answer from A's spot: **left** · from B's spot: **left**

Sentence about B: Observer B is standing about 1.0 metres behind you and to your left, facing the same direction as you.

| setting | Qwen3-VL | BAGEL |
| --- | --- | --- |
| 1Original | left | right |
| 2Ground-truth target | left | right |
| 3Generated target | left | right |
| 4Control (own view) | left | left |

Looks like: A's photo ↔ B's photo 0.964 · drawing ↔ A 0.977 · drawing ↔ B 0.946

A

**A's photo** · input for 1 and 4

B

**B's real photo** · 2 (hidden in 1)

drawing

**BAGEL's drawing of B's view** · 3

map

**Map from above** (blue A, red B)

**apt_1_shared_target_s1_00** · apt_1 · same answer for A and B

Question: is the fridge left or right of the tv screen?\
Correct answer from A's spot: **right** · from B's spot: **right**

Sentence about B: Observer B is standing about 2.5 metres to your right, turned about 30 degrees to the left compared with your heading.

| setting | Qwen3-VL | BAGEL |
| --- | --- | --- |
| 1Original | right | right |
| 2Ground-truth target | right | right |
| 3Generated target | can't see both | can't see both |
| 4Control (own view) | right | right |

Looks like: A's photo ↔ B's photo 0.923 · drawing ↔ A 0.945 · drawing ↔ B 0.883

A

**A's photo** · input for 1 and 4

B

**B's real photo** · 2 (hidden in 1)

drawing

**BAGEL's drawing of B's view** · 3

map

**Map from above** (blue A, red B)

**apt_1_shared_target_s1_01** · apt_1 · same answer for A and B

Question: is the kitchen counter left or right of the fridge?\
Correct answer from A's spot: **right** · from B's spot: **right**

Sentence about B: Observer B is standing about 1.0 metres ahead of you, turned about 15 degrees to the right compared with your heading.

| setting | Qwen3-VL | BAGEL |
| --- | --- | --- |
| 1Original | right | right |
| 2Ground-truth target | left | left |
| 3Generated target | right | right |
| 4Control (own view) | right | left |

Looks like: A's photo ↔ B's photo 0.922 · drawing ↔ A 0.942 · drawing ↔ B 0.898

A

**A's photo** · input for 1 and 4

B

**B's real photo** · 2 (hidden in 1)

drawing

**BAGEL's drawing of B's view** · 3

map

**Map from above** (blue A, red B)

**v3_sc0_staging_00_shared_target_s1_00** · v3_sc0_staging_00 · same answer for A and B

Question: is the fridge left or right of the tv stand?\
Correct answer from A's spot: **right** · from B's spot: **right**

Sentence about B: Observer B is standing about 3.0 metres ahead of you, turned about 30 degrees to the left compared with your heading.

| setting | Qwen3-VL | BAGEL |
| --- | --- | --- |
| 1Original | right | right |
| 2Ground-truth target | right | right |
| 3Generated target | right | right |
| 4Control (own view) | right | right |

Looks like: A's photo ↔ B's photo 0.939 · drawing ↔ A 0.967 · drawing ↔ B 0.951

A

**A's photo** · input for 1 and 4

B

**B's real photo** · 2 (hidden in 1)

drawing

**BAGEL's drawing of B's view** · 3

map

**Map from above** (blue A, red B)

**v3_sc0_staging_00_shared_target_s1_01** · v3_sc0_staging_00 · same answer for A and B

Question: is the chest of drawers left or right of the sofa?\
Correct answer from A's spot: **right** · from B's spot: **right**

Sentence about B: Observer B is standing about 1.0 metres to your left, turned about 15 degrees to the right compared with your heading.

| setting | Qwen3-VL | BAGEL |
| --- | --- | --- |
| 1Original | right | right |
| 2Ground-truth target | right | right |
| 3Generated target | right | right |
| 4Control (own view) | right | right |

Looks like: A's photo ↔ B's photo 0.958 · drawing ↔ A 0.946 · drawing ↔ B 0.938

A

**A's photo** · input for 1 and 4

B

**B's real photo** · 2 (hidden in 1)

drawing

**BAGEL's drawing of B's view** · 3

map

**Map from above** (blue A, red B)

**v3_sc1_staging_03_shared_target_s1_00** · v3_sc1_staging_03 · same answer for A and B

Question: is the tv screen left or right of the chest of drawers?\
Correct answer from A's spot: **right** · from B's spot: **right**

Sentence about B: Observer B is standing about 1.0 metres to your right, turned about 15 degrees to the left compared with your heading. B stands next to the sofa.

| setting | Qwen3-VL | BAGEL |
| --- | --- | --- |
| 1Original | left | right |
| 2Ground-truth target | right | right |
| 3Generated target | left | right |
| 4Control (own view) | right | right |

Looks like: A's photo ↔ B's photo 0.963 · drawing ↔ A 0.932 · drawing ↔ B 0.923

A

**A's photo** · input for 1 and 4

B

**B's real photo** · 2 (hidden in 1)

drawing

**BAGEL's drawing of B's view** · 3

map

**Map from above** (blue A, red B)

**v3_sc1_staging_03_shared_target_s1_01** · v3_sc1_staging_03 · same answer for A and B · excluded from the tables: the objects overlap in A's photo; the objects overlap in B's photo

Question: is the kitchen counter left or right of the sofa?\
Correct answer from A's spot: **left** · from B's spot: **left**

Sentence about B: Observer B is standing about 2.5 metres behind you and to your right, turned about 15 degrees to the left compared with your heading. B is facing the sofa.

| setting | Qwen3-VL | BAGEL |
| --- | --- | --- |
| 1Original | left | left |
| 2Ground-truth target | left | left |
| 3Generated target | left | left |
| 4Control (own view) | left | left |

Looks like: A's photo ↔ B's photo 0.879 · drawing ↔ A 0.934 · drawing ↔ B 0.892

A

**A's photo** · input for 1 and 4

B

**B's real photo** · 2 (hidden in 1)

drawing

**BAGEL's drawing of B's view** · 3

map

**Map from above** (blue A, red B)

### cross_line · Cross to the other side5 clean of 8 examples · 1 correct: Qwen3-VL 2/5 · BAGEL 1/5

**apt_0_cross_line_s1_00** · apt_0 · answer differs for A and B · excluded from the tables: B stands 0.2 m from an object

Question: is the kitchen counter left or right of the sofa?\
Correct answer from A's spot: **right** · from B's spot: **left**

Sentence about B: Observer B is standing about 7.0 metres ahead of you and to your left, turned about 90 degrees to the right compared with your heading. B stands next to the sofa, B is facing the sofa.

| setting | Qwen3-VL | BAGEL |
| --- | --- | --- |
| 1Original | left | right |
| 2Ground-truth target | right | left |
| 3Generated target | left | can't see both |
| 4Control (own view) | left | right |

Looks like: A's photo ↔ B's photo 0.919 · drawing ↔ A 0.929 · drawing ↔ B 0.892

A

**A's photo** · input for 1 and 4

B

**B's real photo** · 2 (hidden in 1)

drawing

**BAGEL's drawing of B's view** · 3

map

**Map from above** (blue A, red B)

**apt_0_cross_line_s1_01** · apt_0 · answer differs for A and B · excluded from the tables: B stands 0.2 m from an object

Question: is the fridge left or right of the sofa?\
Correct answer from A's spot: **right** · from B's spot: **left**

Sentence about B: Observer B is standing about 7.0 metres ahead of you, turned about 105 degrees to the right compared with your heading. B stands next to the sofa.

| setting | Qwen3-VL | BAGEL |
| --- | --- | --- |
| 1Original | right | right |
| 2Ground-truth target | can't see both | can't see both |
| 3Generated target | left | can't see both |
| 4Control (own view) | right | right |

Looks like: A's photo ↔ B's photo 0.927 · drawing ↔ A 0.94 · drawing ↔ B 0.886

A

**A's photo** · input for 1 and 4

B

**B's real photo** · 2 (hidden in 1)

drawing

**BAGEL's drawing of B's view** · 3

map

**Map from above** (blue A, red B)

**apt_1_cross_line_s1_00** · apt_1 · answer differs for A and B

Question: is the sofa left or right of the kitchen counter?\
Correct answer from A's spot: **right** · from B's spot: **left**

Sentence about B: Observer B is standing about 8.0 metres ahead of you, turned about 135 degrees to the left compared with your heading.

| setting | Qwen3-VL | BAGEL |
| --- | --- | --- |
| 1Original | left | right |
| 2Ground-truth target | left | left |
| 3Generated target | left | can't see both |
| 4Control (own view) | right | right |

Looks like: A's photo ↔ B's photo 0.927 · drawing ↔ A 0.945 · drawing ↔ B 0.922

A

**A's photo** · input for 1 and 4

B

**B's real photo** · 2 (hidden in 1)

drawing

**BAGEL's drawing of B's view** · 3

map

**Map from above** (blue A, red B)

**apt_1_cross_line_s1_01** · apt_1 · same answer for A and B · excluded from the tables: the objects overlap in B's photo; their centres are only 19 px apart for B; B stands 0.3 m from an object

Question: is the tv screen left or right of the sofa?\
Correct answer from A's spot: **right** · from B's spot: **right**

Sentence about B: Observer B is standing about 4.0 metres ahead of you and to your left, turned about 60 degrees to the right compared with your heading. B stands next to the sofa.

| setting | Qwen3-VL | BAGEL |
| --- | --- | --- |
| 1Original | left | right |
| 2Ground-truth target | right | right |
| 3Generated target | left | can't see both |
| 4Control (own view) | left | right |

Looks like: A's photo ↔ B's photo 0.923 · drawing ↔ A 0.829 · drawing ↔ B 0.852 (closer to B)

A

**A's photo** · input for 1 and 4

B

**B's real photo** · 2 (hidden in 1)

drawing

**BAGEL's drawing of B's view** · 3

map

**Map from above** (blue A, red B)

**v3_sc0_staging_00_cross_line_s1_00** · v3_sc0_staging_00 · answer differs for A and B

Question: is the fridge left or right of the tv stand?\
Correct answer from A's spot: **right** · from B's spot: **left**

Sentence about B: Observer B is standing about 6.5 metres ahead of you and to your left, turned about 120 degrees to the right compared with your heading.

| setting | Qwen3-VL | BAGEL |
| --- | --- | --- |
| 1Original | right | right |
| 2Ground-truth target | left | left |
| 3Generated target | left | can't see both |
| 4Control (own view) | right | right |

Looks like: A's photo ↔ B's photo 0.953 · drawing ↔ A 0.933 · drawing ↔ B 0.897

A

**A's photo** · input for 1 and 4

B

**B's real photo** · 2 (hidden in 1)

drawing

**BAGEL's drawing of B's view** · 3

map

**Map from above** (blue A, red B)

**v3_sc0_staging_00_cross_line_s1_01** · v3_sc0_staging_00 · answer differs for A and B

Question: is the tv stand left or right of the kitchen counter?\
Correct answer from A's spot: **left** · from B's spot: **right**

Sentence about B: Observer B is standing about 7.5 metres ahead of you, turned about 120 degrees to the right compared with your heading. B stands next to the tv screen.

| setting | Qwen3-VL | BAGEL |
| --- | --- | --- |
| 1Original | left | right |
| 2Ground-truth target | right | right |
| 3Generated target | can't see both | can't see both |
| 4Control (own view) | left | left |

Looks like: A's photo ↔ B's photo 0.91 · drawing ↔ A 0.923 · drawing ↔ B 0.868

A

**A's photo** · input for 1 and 4

B

**B's real photo** · 2 (hidden in 1)

drawing

**BAGEL's drawing of B's view** · 3

map

**Map from above** (blue A, red B)

**v3_sc1_staging_03_cross_line_s1_00** · v3_sc1_staging_03 · answer differs for A and B

Question: is the tv screen left or right of the chest of drawers?\
Correct answer from A's spot: **right** · from B's spot: **left**

Sentence about B: Observer B is standing about 7.5 metres ahead of you, turned about 165 degrees to the right compared with your heading. B stands next to the tv stand.

| setting | Qwen3-VL | BAGEL |
| --- | --- | --- |
| 1Original | left | right |
| 2Ground-truth target | left | left |
| 3Generated target | right | can't see both |
| 4Control (own view) | right | right |

Looks like: A's photo ↔ B's photo 0.929 · drawing ↔ A 0.926 · drawing ↔ B 0.898

A

**A's photo** · input for 1 and 4

B

**B's real photo** · 2 (hidden in 1)

drawing

**BAGEL's drawing of B's view** · 3

map

**Map from above** (blue A, red B)

**v3_sc1_staging_03_cross_line_s1_01** · v3_sc1_staging_03 · answer differs for A and B

Question: is the sofa left or right of the kitchen counter?\
Correct answer from A's spot: **right** · from B's spot: **left**

Sentence about B: Observer B is standing about 7.5 metres ahead of you and to your right, turned about 135 degrees to the left compared with your heading. B is facing the fridge.

| setting | Qwen3-VL | BAGEL |
| --- | --- | --- |
| 1Original | right | right |
| 2Ground-truth target | can't see both | can't see both |
| 3Generated target | can't see both | can't see both |
| 4Control (own view) | right | right |

Looks like: A's photo ↔ B's photo 0.888 · drawing ↔ A 0.898 · drawing ↔ B 0.894

A

**A's photo** · input for 1 and 4

B

**B's real photo** · 2 (hidden in 1)

drawing

**BAGEL's drawing of B's view** · 3

map

**Map from above** (blue A, red B)

## Notes for the next iteration

What differs from the pilot, and why

- **VLM.** Qwen3-VL Qwen3-VL-8B-Instruct instead of the pilot's Qwen2.5-VL-7B (already on turbo). BAGEL BAGEL-7B-MoT is the pilot's model.
- **Examples are new.** The artifact stores no camera poses, so every example is re-sampled. Rooms: apt_0, apt_1, v3_sc0_staging_00, v3_sc1_staging_03.
- **Sampling.** This set used bearing margin ≥ 10° and visibility ≥ 0.3%; the pilot's rows imply ≥ 21° and ≥ 1.5%. The probing set uses the pilot's values (`MIN_MARGIN=20 MIN_VIS=1.5`).
- **B must see both objects**, so setting 2 is answerable; turn-in-place is capped at 60°.
- **Ground truth comes from the rendered masks** (`relabel_visible.py`) and examples are filtered by `clean_rule.py`; the earlier centroid-based labels are kept in the data as `relA_centroid` / `relB_centroid`.
- **GPU.** One A6000 for most of the run; the second was held by another user's server.

Knobs and files

- Render: `render_examples.py` (constants `RES, HFOV, EYE`; sentence templates in `describe()`; vocabulary in `VOCAB`), `render_batch.sh`, `render_probe_parallel.sh`.
- Models: `bagel_edit.py` (edit + VQA, `--steps`), `vlm_judge.py`, `similarity.py`; driver `run_all.sh` with `REPRO_OUT`.
- Tables and this page: `run_pipeline.py summarize`, `compare.py`, `build_artifact.py`. Probing: `probing/`.