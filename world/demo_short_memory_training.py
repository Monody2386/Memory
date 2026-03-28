import pathlib
import sys

import torch

if __package__ is None or __package__ == "":
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from world.shortmemory import ScoredTensorQueue
from world.world_model import (
    WorldModel,
    action_dim,
    attention_dim,
    hidden_dim,
    noun_dim,
    value_dim,
)


def build_demo_memory(model: WorldModel) -> ScoredTensorQueue:
    memory = ScoredTensorQueue(maxlen=10, device="cpu")

    noun_embeddings = [
        torch.linspace(-1.0, 1.0, noun_dim),
        torch.linspace(1.0, -1.0, noun_dim),
        torch.sin(torch.linspace(0.0, 3.14159, noun_dim)),
    ]
    action_types = [1, 2, 1]

    for noun_embedding, action_type in zip(noun_embeddings, action_types):
        memory.append_state(
            noun_embedding=noun_embedding,
            action_embedding=model.get_action_embedding(action_type).detach().clone(),
            score=1.0,
            noun_type=11,
            action_type=action_type,
        )

    return memory


def evaluate_model(model: WorldModel, memory: ScoredTensorQueue, action_type: int, target_type: int):
    with torch.no_grad():
        pred_action, pred_type = model.predict_from_short_memory(memory, action_type=action_type)
        target_embedding = model.get_action_embedding(target_type).detach().clone()
        loss = torch.nn.functional.mse_loss(pred_action, target_embedding).item()
        top_indices, top_scores = model.infer_action_type(pred_action, top_k=model.model_count)

    return {
        "loss": loss,
        "pred_type": int(pred_type),
        "top_indices": top_indices.tolist(),
        "top_scores": [round(float(score.detach().cpu().item()), 4) for score in top_scores],
    }


def build_demo_optimizer(
    model: WorldModel,
    action_model_lr: float = 1e-2,
    action_embedding_lr: float = 1e-5,
):
    return torch.optim.Adam(
        [
            {"params": model.action_models.parameters(), "lr": action_model_lr},
            {"params": model.action_embeddings.parameters(), "lr": action_embedding_lr},
        ]
    )


def run_demo(
    num_epochs: int = 120,
    print_every: int = 20,
    action_model_lr: float = 1e-2,
    action_embedding_lr: float = 1e-5,
):
    torch.manual_seed(7)

    model = WorldModel(
        noun_dim=noun_dim,
        action_dim=action_dim,
        attention_dim=attention_dim,
        value_dim=value_dim,
        hidden_dim=hidden_dim,
    )
    memory = build_demo_memory(model)

    train_action_type = 1
    target_action_type = 3
    initial_embedding_weights = model.action_embeddings.weight.detach().clone()
    initial_target_embedding = model.get_action_embedding(target_action_type).detach().clone()

    optimizer = build_demo_optimizer(
        model,
        action_model_lr=action_model_lr,
        action_embedding_lr=action_embedding_lr,
    )

    before = evaluate_model(model, memory, train_action_type, target_action_type)
    print("Before training")
    print(before)
    print(
        {
            "action_model_lr": action_model_lr,
            "action_embedding_lr": action_embedding_lr,
        }
    )

    for epoch in range(1, num_epochs + 1):
        result = model.training_step_from_short_memory(
            short_memory=memory,
            action_type=train_action_type,
            target_action_type=target_action_type,
            optimizer=optimizer,
        )

        if epoch % print_every == 0 or epoch == 1 or epoch == num_epochs:
            current = evaluate_model(model, memory, train_action_type, target_action_type)
            embedding_shift = (
                model.action_embeddings.weight.detach() - initial_embedding_weights
            ).norm().item()
            target_shift = (
                model.get_action_embedding(target_action_type).detach() - initial_target_embedding
            ).norm().item()
            print(
                f"Epoch {epoch:03d} | "
                f"train_loss={result['loss']:.6f} | "
                f"eval_loss={current['loss']:.6f} | "
                f"pred_type={current['pred_type']} | "
                f"top={current['top_indices']} | "
                f"scores={current['top_scores']} | "
                f"embedding_shift={embedding_shift:.8f} | "
                f"target_shift={target_shift:.8f}"
            )

    after = evaluate_model(model, memory, train_action_type, target_action_type)
    final_embedding_shift = (
        model.action_embeddings.weight.detach() - initial_embedding_weights
    ).norm().item()
    final_target_shift = (
        model.get_action_embedding(target_action_type).detach() - initial_target_embedding
    ).norm().item()
    print("After training")
    print(after)
    print(
        {
            "embedding_shift": round(final_embedding_shift, 8),
            "target_embedding_shift": round(final_target_shift, 8),
        }
    )

    next_noun_embedding = torch.cos(torch.linspace(0.0, 3.14159, noun_dim))
    rollout = model.autoregressive_step(
        short_memory=memory,
        noun_embedding=next_noun_embedding,
        action_type=train_action_type,
        score=1.0,
        noun_type=19,
    )
    print("Autoregressive step")
    print(
        {
            "pred_action_type": rollout["pred_action_type"],
            "memory_size": len(memory),
        }
    )

    return model, memory


if __name__ == "__main__":
    run_demo()

