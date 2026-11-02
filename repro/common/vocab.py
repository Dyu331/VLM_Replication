"""X/Y object vocabulary: ReplicaCAD template name -> name used in questions.

Agreed 2026-11-02; after the smoke test tvstand was renamed "blue shelf" and cabinet added
as "sideboard". Only solid, clearly nameable furniture; tables, plants, rugs, wall cabinets,
bikes, doors and clutter are excluded. A template is used in a scene only if
it occurs exactly once there (e.g. beanbag appears twice in some scenes).
"""

VOCAB = {
    "frl_apartment_sofa": "sofa",
    "fridge": "fridge",
    "kitchen_counter": "kitchen counter",
    "frl_apartment_tv_screen": "tv screen",
    "frl_apartment_tvstand": "blue shelf",  # under the TV only in apt_*; moved in v3 layouts
    "cabinet": "sideboard",  # brown low sideboard; under the TV in v3 layouts
    "chestOfDrawers_01": "chest of drawers",
    "frl_apartment_beanbag": "beanbag",
    "frl_apartment_basket": "basket",  # apt_* only
}

# Pairs never asked about: one sits on/under the other, so left/right is not meaningful
EXCLUDED_PAIRS = {frozenset({"tv screen", "blue shelf"}), frozenset({"tv screen", "sideboard"})}
