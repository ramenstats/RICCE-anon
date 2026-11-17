"""Shared constants for the image-grounded human study (schema, axes, categories)."""
AXES = ["object", "action", "source_target", "location", "quantity",
        "change_nochange", "specificity", "completeness", "hallucination", "overall"]
# axes where 'NA' (not applicable) is allowed
NA_AXES = ["object", "action", "source_target", "location", "quantity", "specificity"]
PREF_VALUES = ["A", "B", "Tie", "CannotDetermine"]
CSV_COLUMNS = ["annotator_id", "task", "item_id"] + AXES + ["preference", "notes", "timestamp_utc", "seconds_spent"]
# Higher is always better for every axis (hallucination: 5 = none, 1 = severe).
PAIR_CATEGORIES = ["action_reversal", "coarse_but_correct", "unsupported_specificity", "omission",
                   "hallucination", "wrong_location", "wrong_quantity", "no_change_error",
                   "multi_event_binding"]
