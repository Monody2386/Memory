import pathlib
import sys

import torch

if __package__ is None or __package__ == "":
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

from Consciousness import Consciousness


SENTENCE_SPECS = [
    {
        "sentence": "I cut red apple",
        "noun_relation_type": 1,
        "adjective_relation_types": {"red": "color"},
    },
    {
        "sentence": "cat eat sweet apple",
        "noun_relation_type": 1,
        "adjective_relation_types": {"sweet": "taste"},
    },
]


def apply_online_language_updates(consciousness: Consciousness, sentence_specs):
    print("Online knowledge updates")
    update_logs = []
    for spec in sentence_specs:
        result = consciousness.learn_from_sentence(
            spec["sentence"],
            noun_relation_type=spec.get("noun_relation_type"),
            adjective_relation_types=spec.get("adjective_relation_types"),
            infer_missing=False,
            save=False,
        )
        samples = result["samples"]
        training = result["results"]
        summary = {
            "sentence": spec["sentence"],
            "noun_noun_updates": len(samples.noun_noun_samples),
            "adj_noun_updates": len(samples.adj_noun_samples),
            "noun_losses": [round(item["loss"], 8) for item in training["noun_noun"]],
            "adj_losses": [round(item["loss"], 8) for item in training["adj_noun"]],
        }
        update_logs.append(summary)
        print(summary)
    return update_logs


def print_memory(consciousness: Consciousness):
    summary = [
        {
            "time": entry["time_position"],
            "pair": entry["pair_index"],
            "noun_type": entry["noun_type"],
            "action_type": entry["action_type"],
            "score": round(entry["score"], 3),
        }
        for entry in consciousness.inspect_memory(kind="event", order_by="time")
    ]
    print("Memory entries")
    print(summary)


def run_demo(num_epochs: int = 100, print_every: int = 20):
    torch.manual_seed(11)

    consciousness = Consciousness()
    apply_online_language_updates(consciousness, SENTENCE_SPECS)
    consciousness.observe_many(
        [spec["sentence"] for spec in SENTENCE_SPECS],
        start_time_position=0,
        base_score=1.0,
        adjective_relation_types=[spec.get("adjective_relation_types") for spec in SENTENCE_SPECS],
    )

    input_action_type = 1
    target_action_type = 4

    print_memory(consciousness)

    before = consciousness.evaluate_next_event(
        action_type=input_action_type,
        target_action_type=target_action_type,
    )
    print("Before training")
    print(before)

    for epoch in range(1, num_epochs + 1):
        result = consciousness.train_next_event(
            action_type=input_action_type,
            target_action_type=target_action_type,
        )

        if epoch % print_every == 0 or epoch == 1 or epoch == num_epochs:
            current = consciousness.evaluate_next_event(
                action_type=input_action_type,
                target_action_type=target_action_type,
            )
            print(
                f"Epoch {epoch:03d} | "
                f"train_loss={result['loss']:.6f} | "
                f"eval_loss={current['loss']:.6f} | "
                f"pred_action_type={current['pred_action_type']} | "
                f"top={current['top_indices']} | "
                f"scores={current['top_scores']}"
            )

    after = consciousness.evaluate_next_event(
        action_type=input_action_type,
        target_action_type=target_action_type,
    )
    print("After training")
    print(after)

    prediction = consciousness.predict_next_event(action_type=input_action_type, score=0.5)
    predicted_event = prediction["predicted_event"]
    predicted_event["noun_embedding"] = torch.linspace(-0.5, 0.5, consciousness.world_model.noun_dim)
    predicted_event["noun_type"] = 42
    predicted_event["time_position"] = 2
    stored_event = consciousness.append_predicted_event(predicted_event)
    latest_event = consciousness.inspect_focus()
    print("Predicted event append")
    print(
        {
            "pred_action_type": prediction["pred_action_type"],
            "memory_size": len(consciousness.short_memory),
            "latest_time": latest_event["time_position"] if latest_event is not None else None,
            "noun_instance_id": stored_event["noun_instance_id"],
            "action_instance_id": stored_event["action_instance_id"],
        }
    )

    return consciousness


if __name__ == "__main__":
    run_demo()
