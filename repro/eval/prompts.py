"""Prompts for the four settings, verbatim from the report, plus a marked twin of each.

Each setting returns (view whose photo the model gets, prompt text, view whose answer is correct).
Marked settings ("<setting>_marked", an extension) use photos with X outlined orange and Y blue,
and name the colours in the prompt; everything else is identical.
"""

ANSWER = "Answer with exactly one word: left or right."

TASK = ("This image shows what you (observer A) currently see. Another observer, B, is in the "
        "same room. {sentence} From B's point of view, is the {X} to the left or to the right "
        "of the {Y}? " + ANSWER)

DIRECT = "In this image, is the {X} to the left or to the right of the {Y}? " + ANSWER


def build(rec, setting):
    X, Y = rec["X"], rec["Y"]
    if is_marked(setting):
        X, Y = f"{X} (outlined in orange)", f"{Y} (outlined in blue)"
        setting = setting.removesuffix("_marked")
    if setting == "s1":        # original: A's photo + sentence about B, answer from B
        return "A", TASK.format(sentence=rec["sentence_B"], X=X, Y=Y), "B"
    if setting == "s1_identity":  # same prompt, B stands where A stands: answer from A
        return "A", TASK.format(sentence=rec["sentence_identity"], X=X, Y=Y), "A"
    if setting == "s2":        # ground-truth target: B's photo, direct question
        return "B", DIRECT.format(X=X, Y=Y), "B"
    if setting == "s4":        # control: A's photo, direct question
        return "A", DIRECT.format(X=X, Y=Y), "A"
    raise ValueError(setting)


BASE_SETTINGS = ["s1", "s1_identity", "s2", "s4"]
SETTINGS = BASE_SETTINGS + [s + "_marked" for s in BASE_SETTINGS]


def is_marked(setting):
    return setting.endswith("_marked")


def image_name(rec, setting, view):
    return f"{rec['id']}_{view}_rgb{'_marked' if is_marked(setting) else ''}.png"


def parse(text):
    t = text.strip().lower()
    for word in ("left", "right"):
        if t.startswith(word):
            return word
    has_l, has_r = "left" in t, "right" in t
    return "left" if has_l and not has_r else "right" if has_r and not has_l else "invalid"
