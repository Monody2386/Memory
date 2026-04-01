import pathlib
import sys

import torch

if __package__ is None or __package__ == "":
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from Grammar import sentences_to_short_memory
from knowledge.training import train_sentence_online
from world.shortmemory import ShortMemory
from world.world_model import (
    WorldModel,
    action_dim,
    attention_dim,
    hidden_dim,
    noun_dim,
    value_dim,
)


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


def build_demo_model() -> WorldModel:
    return WorldModel(
        noun_dim=noun_dim,
        action_dim=action_dim,
        attention_dim=attention_dim,
        value_dim=value_dim,
        hidden_dim=hidden_dim,
        action_names=["cut", "cutted", "eat", "eaten"],
    )


def apply_online_language_updates(sentence_specs):
    print("Online knowledge updates")
    update_logs = []
    for spec in sentence_specs:
        samples, results = train_sentence_online(
            spec["sentence"],
            noun_relation_type=spec.get("noun_relation_type"),
            adjective_relation_types=spec.get("adjective_relation_types"),
            infer_missing=False,
            save=False,
        )
        summary = {
            "sentence": spec["sentence"],
            "noun_noun_updates": len(samples.noun_noun_samples),
            "adj_noun_updates": len(samples.adj_noun_samples),
            "noun_losses": [round(item["loss"], 8) for item in results["noun_noun"]],
            "adj_losses": [round(item["loss"], 8) for item in results["adj_noun"]],
        }
        update_logs.append(summary)
        print(summary)
    return update_logs


def build_demo_memory(world_model: WorldModel, sentence_specs) -> ShortMemory:
    memory = ShortMemory()
    sentences_to_short_memory(
        [spec["sentence"] for spec in sentence_specs],
        short_memory=memory,
        world_model=world_model,
        start_time_position=0,
        base_score=1.0,
        adjective_relation_types=[spec.get("adjective_relation_types") for spec in sentence_specs],
    )
    return memory


def evaluate_prediction(world_model: WorldModel, memory: ShortMemory, action_type: int, target_action_type: int):
    with torch.no_grad():
        pred_action, pred_action_type = world_model.predict_from_short_memory(
            memory, action_type=action_type
        )
        target_embedding = world_model.get_action_embedding(target_action_type).detach()
        loss = torch.nn.functional.mse_loss(pred_action, target_embedding).item()
        top_indices, top_scores = world_model.infer_action_type(
            pred_action, top_k=world_model.model_count
        )

    return {
        "loss": loss,
        "pred_action_type": int(pred_action_type),
        "top_indices": top_indices.tolist(),
        "top_scores": [round(float(score.item()), 4) for score in top_scores],
    }


def print_memory(memory: ShortMemory):
    summary = [
        {
            "time": entry.time_position,
            "pair": entry.pair_index,
            "noun_type": entry.noun_type,
            "action_type": entry.action_type,
            "score": round(entry.score, 3),
        }
        for entry in memory.entries
    ]
    print("Memory entries")
    print(summary)


def run_demo(num_epochs: int = 100, print_every: int = 20):
    torch.manual_seed(11)

    apply_online_language_updates(SENTENCE_SPECS)
    world_model = build_demo_model()
    memory = build_demo_memory(world_model, SENTENCE_SPECS)
    optimizer = torch.optim.Adam(
        [
            {"params": world_model.action_models.parameters(), "lr": 1e-2},
            {"params": world_model.action_embeddings.parameters(), "lr": 1e-5},
        ]
    )

    input_action_type = 1
    target_action_type = 4

    print_memory(memory)

    before = evaluate_prediction(
        world_model, memory, input_action_type, target_action_type
    )
    print("Before training")
    print(before)

    for epoch in range(1, num_epochs + 1):
        result = world_model.training_step_from_short_memory(
            short_memory=memory,
            action_type=input_action_type,
            target_action_type=target_action_type,
            optimizer=optimizer,
        )

        if epoch % print_every == 0 or epoch == 1 or epoch == num_epochs:
            current = evaluate_prediction(
                world_model, memory, input_action_type, target_action_type
            )
            print(
                f"Epoch {epoch:03d} | "
                f"train_loss={result['loss']:.6f} | "
                f"eval_loss={current['loss']:.6f} | "
                f"pred_action_type={current['pred_action_type']} | "
                f"top={current['top_indices']} | "
                f"scores={current['top_scores']}"
            )

    after = evaluate_prediction(
        world_model, memory, input_action_type, target_action_type
    )
    print("After training")
    print(after)

    rollout_noun = torch.linspace(-0.5, 0.5, noun_dim)
    rollout = world_model.autoregressive_step(
        short_memory=memory,
        noun_embedding=rollout_noun,
        action_type=input_action_type,
        noun_type=42,
        time_position=2,
        score=1.0,
    )
    print("Autoregressive step")
    print(
        {
            "pred_action_type": rollout["pred_action_type"],
            "memory_size": len(memory),
            "latest_time": memory.entries[-1].time_position,
        }
    )

    return world_model, memory


if __name__ == "__main__":
    run_demo()
