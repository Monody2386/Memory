import importlib
import os
from typing import Iterable, Optional, Tuple

import torch

from .shortmemory import ScoredTensorQueue, short_memory


def _wm():
    return importlib.import_module("world.world_model")


def train_world_model(
    samples: Iterable[Tuple[torch.Tensor, int, torch.Tensor]],
    save_dir: str = "world_models",
    load_dir: Optional[str] = None,
    num_epochs: int = 1,
    device: str = "cpu",
):
    wm_module = _wm()
    os.makedirs(save_dir, exist_ok=True)

    world_model = wm_module.WorldModel(
        noun_dim=wm_module.noun_dim,
        action_dim=wm_module.action_dim,
        attention_dim=wm_module.attention_dim,
        value_dim=wm_module.value_dim,
        hidden_dim=wm_module.hidden_dim,
    ).to(device)

    if load_dir is not None:
        model_path = os.path.join(load_dir, "world_model.pt")
        if os.path.exists(model_path):
            world_model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Loaded pretrained model: {model_path}")

    optimizer = torch.optim.Adam(world_model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()
    samples_list = list(samples)

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        batch_count = 0

        for input_2d, action_type, target_action in samples_list:
            input_2d = input_2d.to(device)
            target_action = target_action.to(device)

            optimizer.zero_grad()
            pred_action = world_model(input_2d, action_type)
            loss = loss_fn(pred_action, target_action)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1

        avg_loss = epoch_loss / batch_count if batch_count > 0 else 0.0
        print(f"Epoch [{epoch + 1}/{num_epochs}] Loss: {avg_loss:.6f}")

    model_path = os.path.join(save_dir, "world_model.pt")
    torch.save(world_model.state_dict(), model_path)
    print(f"Saved model: {model_path}")
    return world_model


def train_world_model_with_validation(
    train_samples: Iterable[Tuple[torch.Tensor, int, torch.Tensor]],
    val_samples: Optional[Iterable[Tuple[torch.Tensor, int, torch.Tensor]]] = None,
    save_dir: str = "world_models",
    load_dir: Optional[str] = None,
    num_epochs: int = 10,
    device: str = "cpu",
    patience: int = 5,
) -> dict:
    wm_module = _wm()
    os.makedirs(save_dir, exist_ok=True)

    world_model = wm_module.WorldModel(
        noun_dim=wm_module.noun_dim,
        action_dim=wm_module.action_dim,
        attention_dim=wm_module.attention_dim,
        value_dim=wm_module.value_dim,
        hidden_dim=wm_module.hidden_dim,
    ).to(device)

    if load_dir is not None:
        model_path = os.path.join(load_dir, "world_model.pt")
        if os.path.exists(model_path):
            world_model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Loaded pretrained model: {model_path}")

    optimizer = torch.optim.Adam(world_model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

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

        for input_2d, action_type, target_action in train_samples_list:
            input_2d = input_2d.to(device)
            target_action = target_action.to(device)

            optimizer.zero_grad()
            pred_action = world_model(input_2d, action_type)
            loss = loss_fn(pred_action, target_action)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1

        avg_train_loss = epoch_loss / batch_count if batch_count > 0 else 0.0
        train_losses.append(avg_train_loss)

        if val_samples_list:
            world_model.eval()
            val_epoch_loss = 0.0
            val_batch_count = 0

            with torch.no_grad():
                for input_2d, action_type, target_action in val_samples_list:
                    input_2d = input_2d.to(device)
                    target_action = target_action.to(device)

                    pred_action = world_model(input_2d, action_type)
                    loss = loss_fn(pred_action, target_action)
                    val_epoch_loss += loss.item()
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


if __name__ == "__main__":
    short_memory = ScoredTensorQueue()
