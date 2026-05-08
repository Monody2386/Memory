import importlib
import os
from typing import Iterable, Optional, Tuple

import torch

from short_memory import ScoredTensorQueue


RESET_TRAINING_MESSAGE = (
    "Legacy world-model training has been disabled because the old world-model "
    "logic was intentionally cleared. Implement the new world_state/event_state "
    "pipeline before re-enabling training."
)


def _wm():
    return importlib.import_module("world.world_model")


def _create_world_model(device: str):
    wm_module = _wm()
    world_model = wm_module.WorldModel(
        noun_dim=wm_module.noun_dim,
        action_dim=wm_module.action_dim,
        attention_dim=wm_module.attention_dim,
        value_dim=wm_module.value_dim,
        hidden_dim=wm_module.hidden_dim,
    ).to(device)
    return wm_module, world_model


def _load_weights_if_available(world_model, load_dir: Optional[str], device: str):
    if load_dir is None:
        return
    model_path = os.path.join(load_dir, "world_model.pt")
    if os.path.exists(model_path):
        world_model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded pretrained model: {model_path}")


def _is_memory_sample(sample) -> bool:
    return isinstance(sample[0], ScoredTensorQueue)


def _train_single_sample(world_model, optimizer, sample, device: str) -> float:
    if _is_memory_sample(sample):
        memory, action_type, target_action_embedding = sample[:3]
        steps = sample[3] if len(sample) > 3 else None
        result = world_model.training_step_from_short_memory(
            short_memory=memory,
            action_type=action_type,
            target_action_embedding=target_action_embedding.to(device),
            optimizer=optimizer,
            steps=steps,
        )
        return float(result["loss"])

    input_2d, action_type, target_action = sample[:3]
    input_2d = input_2d.to(device)
    target_action = target_action.to(device)

    optimizer.zero_grad()
    pred_action = world_model(input_2d, action_type)
    loss = torch.nn.functional.mse_loss(pred_action, target_action)
    loss.backward()
    optimizer.step()
    return float(loss.item())


def _eval_single_sample(world_model, sample, device: str) -> float:
    if _is_memory_sample(sample):
        memory, action_type, target_action_embedding = sample[:3]
        steps = sample[3] if len(sample) > 3 else None
        prediction = world_model.predict_from_short_memory(
            short_memory=memory,
            action_type=action_type,
            steps=steps,
        )
        pred_action = prediction["pred_action"]
        target_action_embedding = target_action_embedding.to(device)
        loss = torch.nn.functional.mse_loss(pred_action, target_action_embedding)
        return float(loss.item())

    input_2d, action_type, target_action = sample[:3]
    input_2d = input_2d.to(device)
    target_action = target_action.to(device)
    pred_action = world_model(input_2d, action_type)
    loss = torch.nn.functional.mse_loss(pred_action, target_action)
    return float(loss.item())


def train_world_model(
    samples: Iterable[Tuple],
    save_dir: str = "world_models",
    load_dir: Optional[str] = None,
    num_epochs: int = 1,
    device: str = "cpu",
):
    raise RuntimeError(RESET_TRAINING_MESSAGE)
    os.makedirs(save_dir, exist_ok=True)
    _, world_model = _create_world_model(device)
    _load_weights_if_available(world_model, load_dir, device)

    optimizer = world_model.build_optimizer()
    samples_list = list(samples)

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        batch_count = 0

        for sample in samples_list:
            loss_value = _train_single_sample(world_model, optimizer, sample, device)
            epoch_loss += loss_value
            batch_count += 1

        avg_loss = epoch_loss / batch_count if batch_count > 0 else 0.0
        print(f"Epoch [{epoch + 1}/{num_epochs}] Loss: {avg_loss:.6f}")

    model_path = os.path.join(save_dir, "world_model.pt")
    torch.save(world_model.state_dict(), model_path)
    print(f"Saved model: {model_path}")
    return world_model


def train_world_model_with_validation(
    train_samples: Iterable[Tuple],
    val_samples: Optional[Iterable[Tuple]] = None,
    save_dir: str = "world_models",
    load_dir: Optional[str] = None,
    num_epochs: int = 10,
    device: str = "cpu",
    patience: int = 5,
) -> dict:
    raise RuntimeError(RESET_TRAINING_MESSAGE)
    os.makedirs(save_dir, exist_ok=True)
    _, world_model = _create_world_model(device)
    _load_weights_if_available(world_model, load_dir, device)

    optimizer = world_model.build_optimizer()
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    patience_counter = 0
    best_epoch = 0

    train_samples_list = list(train_samples)
    val_samples_list = list(val_samples) if val_samples is not None else []

    for epoch in range(num_epochs):
        world_model.train()
        epoch_loss = 0.0
        batch_count = 0

        for sample in train_samples_list:
            loss_value = _train_single_sample(world_model, optimizer, sample, device)
            epoch_loss += loss_value
            batch_count += 1

        avg_train_loss = epoch_loss / batch_count if batch_count > 0 else 0.0
        train_losses.append(avg_train_loss)

        if val_samples_list:
            world_model.eval()
            val_epoch_loss = 0.0
            val_batch_count = 0

            with torch.no_grad():
                for sample in val_samples_list:
                    loss_value = _eval_single_sample(world_model, sample, device)
                    val_epoch_loss += loss_value
                    val_batch_count += 1

            avg_val_loss = val_epoch_loss / val_batch_count if val_batch_count > 0 else 0.0
            val_losses.append(avg_val_loss)
            print(
                f"Epoch [{epoch + 1}/{num_epochs}] "
                f"Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}"
            )

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_epoch = epoch + 1
                best_model_path = os.path.join(save_dir, "world_model_best.pt")
                torch.save(world_model.state_dict(), best_model_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(
                        f"Early stopping at epoch {epoch + 1}; best epoch: {best_epoch}"
                    )
                    break
        else:
            print(f"Epoch [{epoch + 1}/{num_epochs}] Train Loss: {avg_train_loss:.6f}")

    final_model_path = os.path.join(save_dir, "world_model.pt")
    torch.save(world_model.state_dict(), final_model_path)
    print(f"Saved model: {final_model_path}")

    return {
        "model": world_model,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_epoch": best_epoch,
    }


def autoregressive_rollout_from_memory(
    world_model,
    memory: ScoredTensorQueue,
    noun_embeddings,
    initial_action_type: int,
    score: float = 0.0,
    noun_types=None,
    steps=None,
):
    return world_model.autoregressive_rollout(
        short_memory=memory,
        noun_embeddings=noun_embeddings,
        initial_action_type=initial_action_type,
        score=score,
        noun_types=noun_types,
        steps=steps,
    )


if __name__ == "__main__":
    short_memory = ScoredTensorQueue()
