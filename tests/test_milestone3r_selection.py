import json

from actmask.experiments.milestone3r_selection import select


def test_selection_uses_validation_and_fixed_variants(tmp_path):
    records = []
    for split, score in (("val", 1.0), ("test", 0.0)):
        records.append(dict(split=split, variant="gaussian_severe", condition="clean", method="gru", pair_order_accuracy=score))
    records.append(dict(split="val", variant="gaussian_severe", condition="mixed", method="tcn", pair_order_accuracy=0.5))
    # The remaining fixed variants make the expected method's aggregate valid.
    for variant in ("heteroscedastic_medium", "timestamp_jitter", "random_dropout", "last_frame_dropout", "burst_occlusion", "observation_latency", "partial_coordinates"):
        records.extend((dict(split="val", variant=variant, condition="clean", method="gru", pair_order_accuracy=1.0), dict(split="val", variant=variant, condition="mixed", method="tcn", pair_order_accuracy=0.5)))
    source = tmp_path / "report.json"; source.write_text(json.dumps({"records": records}))
    assert select(source)["selected"]["method"] == "gru"
