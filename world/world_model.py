import torch
import torch.nn as nn
import torch.nn.functional as F

from .action_vocab import (
    DEFAULT_ACTION_LIST,
    NO_ACTION_NAME,
    action_list as shared_action_list,
    get_full_action_type_list,
    set_action_list,
)

attention_dim = 100
action_dim = 80
noun_dim = 50
value_dim = 100
hidden_dim = 50


class FFN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FFN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Event_Attention(nn.Module):
    def __init__(self, noun_dim, action_dim, attention_dim, value_dim):
        super(Event_Attention, self).__init__()
        self.Wq = nn.Linear(noun_dim + action_dim, attention_dim)
        self.Wk = nn.Linear(noun_dim + action_dim, attention_dim)
        self.Wv = nn.Linear(noun_dim + action_dim, value_dim)
        self.scale = attention_dim ** 0.5

    def forward(self, input_tensor):
        input_prefix = input_tensor[:, :-1, :]
        input_suffix = input_tensor[:, -1, :]

        q = self.Wq(input_suffix)
        k = self.Wk(input_prefix)
        v = self.Wv(input_prefix)

        scores = torch.matmul(q.unsqueeze(1), k.transpose(-2, -1)) / self.scale
        attention_weight = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weight, v).squeeze(1)
        output = output + self.Wv(input_suffix)
        return output


class ActionModel(nn.Module):
    def __init__(self, noun_dim, action_dim, attention_dim, value_dim, hidden_dim):
        super(ActionModel, self).__init__()
        self.Wq = nn.Linear(noun_dim + action_dim, attention_dim)
        self.Wk = nn.Linear(noun_dim + action_dim, attention_dim)
        self.Wv = nn.Linear(noun_dim + action_dim, value_dim)
        self.scale = attention_dim ** 0.5
        self.action_predict = nn.Sequential(
            nn.Linear(value_dim, hidden_dim, bias=False),
            nn.Linear(hidden_dim, action_dim, bias=False),
        )

    def forward(self, x):
        input_prefix = x[:, :-1, :]
        input_suffix = x[:, -1, :]

        q = self.Wq(input_suffix)
        k = self.Wk(input_prefix)
        v = self.Wv(input_prefix)

        scores = torch.matmul(q.unsqueeze(1), k.transpose(-2, -1)) / self.scale
        attention_weight = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weight, v).squeeze(1)
        output = output + self.Wv(input_suffix)
        output = self.action_predict(output)
        return output


class WorldModel(nn.Module):
    def __init__(
        self,
        noun_dim,
        action_dim,
        attention_dim,
        value_dim,
        hidden_dim,
        action_names=None,
        action_lrs=None,
    ):
        super(WorldModel, self).__init__()
        self.noun_dim = noun_dim
        self.action_dim = action_dim
        self.attention_dim = attention_dim
        self.value_dim = value_dim
        self.hidden_dim = hidden_dim

        if action_names is not None:
            set_action_list(action_names)

        self.action_list = get_full_action_type_list()
        self.model_count = len(self.action_list) - 1
        self.action_models = nn.ModuleList(
            [
                ActionModel(noun_dim, action_dim, attention_dim, value_dim, hidden_dim)
                for _ in range(self.model_count)
            ]
        )
        self.action_embeddings = nn.Embedding(len(self.action_list), self.action_dim)

        with torch.no_grad():
            self.action_embeddings.weight[0].zero_()

        if action_lrs is None:
            self.action_learning_rates = [1e-3] + [1e-4] * max(0, self.model_count - 1)
        else:
            if len(action_lrs) != self.model_count:
                raise ValueError("action_lrs length must match the number of trainable action models")
            self.action_learning_rates = [float(lr) for lr in action_lrs]

    def action_type_to_idx(self, action_type: int) -> int:
        action_type = int(action_type)
        if action_type < 1 or action_type > self.model_count:
            raise ValueError(
                f"action_type {action_type} out of range [1, {self.model_count}] with 0 reserved for no_action"
            )
        return action_type - 1

    def action_idx_to_type(self, action_idx: int) -> int:
        action_idx = int(action_idx)
        if action_idx < 0 or action_idx >= self.model_count:
            raise ValueError(f"action_idx {action_idx} out of range [0, {self.model_count - 1}]")
        return action_idx + 1

    def _normalize_embedding_action_type(self, action_type: int) -> int:
        action_type = int(action_type)
        if action_type < 0 or action_type > self.model_count:
            raise ValueError(
                f"action_type {action_type} out of range [0, {self.model_count}]"
            )
        return action_type

    def build_optimizer(self):
        action_param_groups = [
            {"params": action_model.parameters(), "lr": lr}
            for action_model, lr in zip(self.action_models, self.action_learning_rates)
        ]
        action_param_groups.append({"params": self.action_embeddings.parameters(), "lr": 1e-3})
        return torch.optim.Adam(action_param_groups)

    def get_action_embedding(self, action_type: int) -> torch.Tensor:
        action_type = self._normalize_embedding_action_type(action_type)
        device = self.action_embeddings.weight.device
        action_type_tensor = torch.tensor(action_type, dtype=torch.long, device=device)
        return self.action_embeddings(action_type_tensor)

    def infer_action_type(self, action_tensor: torch.Tensor, top_k: int = 1, include_no_action: bool = False):
        if action_tensor.dim() > 1:
            action_tensor = action_tensor.view(-1)

        if action_tensor.numel() != self.action_dim:
            raise ValueError(
                f"action_tensor must have {self.action_dim} values, got {action_tensor.numel()}"
            )

        all_embeddings = self.action_embeddings.weight
        if include_no_action:
            embeddings = all_embeddings
            index_offset = 0
        else:
            embeddings = all_embeddings[1:]
            index_offset = 1

        action_tensor = action_tensor.to(embeddings.device, dtype=embeddings.dtype)
        query = F.normalize(action_tensor.unsqueeze(0), dim=1)
        normalized_embeddings = F.normalize(embeddings, dim=1)

        top_k = min(int(top_k), embeddings.shape[0])
        similarity = torch.matmul(query, normalized_embeddings.t()).squeeze(0)
        top_scores, top_indices = torch.topk(similarity, k=top_k)
        top_indices = top_indices + index_offset
        return top_indices, top_scores

    def nearest_action_type(self, action_tensor: torch.Tensor, include_no_action: bool = False) -> int:
        top_indices, _ = self.infer_action_type(
            action_tensor, top_k=1, include_no_action=include_no_action
        )
        return int(top_indices[0].item())

    def _prepare_model_input(self, input_tensor: torch.Tensor) -> torch.Tensor:
        if input_tensor.dim() == 2:
            model_input = input_tensor.t().unsqueeze(0)
        elif input_tensor.dim() == 3:
            model_input = input_tensor
        else:
            raise ValueError("input tensor must be 2D or 3D")
        return model_input

    def forward(self, input_2d: torch.Tensor, action_type: int) -> torch.Tensor:
        action_idx = self.action_type_to_idx(action_type)
        model_input = self._prepare_model_input(input_2d)

        device = next(self.action_models[action_idx].parameters()).device
        model_input = model_input.to(device)
        new_action = self.action_models[action_idx](model_input)
        return new_action.squeeze(0)

    def predict_from_short_memory(self, short_memory, action_type: int, steps=None):
        memory_input = short_memory.build_world_model_input(steps=steps)
        if memory_input.numel() == 0:
            raise ValueError("short_memory is empty")
        pred_action = self.forward(memory_input, action_type)
        pred_action_type = self.nearest_action_type(pred_action)
        return pred_action, pred_action_type

    def training_step_from_short_memory(
        self,
        short_memory,
        action_type: int,
        target_action_embedding=None,
        target_action_type=None,
        optimizer=None,
        steps=None,
    ):
        optimizer = optimizer or self.build_optimizer()
        pred_action, pred_action_type = self.predict_from_short_memory(
            short_memory, action_type, steps=steps
        )

        if target_action_embedding is None:
            if target_action_type is None:
                raise ValueError("Provide target_action_embedding or target_action_type")
            target_action_embedding = self.get_action_embedding(target_action_type)
        else:
            target_action_embedding = target_action_embedding.to(
                pred_action.device, dtype=pred_action.dtype
            )

        target_action_embedding = target_action_embedding.view(-1)
        loss = F.mse_loss(pred_action, target_action_embedding)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            self.action_embeddings.weight[0].zero_()

        return {
            "loss": float(loss.item()),
            "pred_action": pred_action.detach(),
            "pred_action_type": int(pred_action_type),
        }

    def autoregressive_step(
        self,
        short_memory,
        noun_embedding: torch.Tensor,
        action_type: int,
        score=0.0,
        noun_type=None,
        time_position: int = 0,
        steps=None,
    ):
        pred_action, pred_action_type = self.predict_from_short_memory(
            short_memory, action_type, steps=steps
        )
        next_action_embedding = self.get_action_embedding(pred_action_type).detach()
        short_memory.append_state(
            noun_embedding=noun_embedding,
            action_embedding=next_action_embedding,
            score=score,
            noun_type=noun_type,
            action_type=pred_action_type,
            time_position=time_position,
        )
        return {
            "pred_action": pred_action.detach(),
            "pred_action_type": int(pred_action_type),
            "stored_action_embedding": next_action_embedding.detach(),
        }

    def autoregressive_rollout(
        self,
        short_memory,
        noun_embeddings,
        initial_action_type: int,
        score=0.0,
        noun_types=None,
        time_positions=None,
        steps=None,
    ):
        outputs = []
        current_action_type = int(initial_action_type)
        noun_types = noun_types or [None] * len(noun_embeddings)
        time_positions = time_positions or list(range(len(noun_embeddings)))

        for noun_embedding, noun_type, time_position in zip(
            noun_embeddings, noun_types, time_positions
        ):
            result = self.autoregressive_step(
                short_memory=short_memory,
                noun_embedding=noun_embedding,
                action_type=current_action_type,
                score=score,
                noun_type=noun_type,
                time_position=time_position,
                steps=steps,
            )
            outputs.append(result)
            current_action_type = result["pred_action_type"]

        return outputs

    def train_action_model(
        self,
        input_data: torch.Tensor,
        target_action: torch.Tensor,
        action_type: int,
        optimizer=None,
        steps: int = 100,
    ):
        optimizer = optimizer or self.build_optimizer()
        device = next(self.parameters()).device
        input_data = input_data.to(device)
        target_action = target_action.to(device).view(-1)

        last_loss = None
        for _ in range(steps):
            optimizer.zero_grad()
            action_pred = self.forward(input_data, action_type)
            last_loss = F.mse_loss(action_pred, target_action)
            last_loss.backward()
            optimizer.step()
            with torch.no_grad():
                self.action_embeddings.weight[0].zero_()
        return float(last_loss.item()) if last_loss is not None else 0.0

    def forward_with_sequence_build(self, input_2d: torch.Tensor, action_type: int, steps: int = 1) -> dict:
        current_seq = input_2d.clone()
        predictions = []
        action_types = []
        current_action_type = int(action_type)

        for step in range(steps):
            pred_action = self.forward(current_seq, current_action_type)
            predictions.append(pred_action.detach().cpu())
            action_types.append(current_action_type)

            last_col = current_seq[:, -1]
            noun_last = last_col[: self.noun_dim]
            next_action_type = self.nearest_action_type(pred_action)
            next_action_embedding = self.get_action_embedding(next_action_type).detach().to(pred_action.device)
            new_col = torch.cat([noun_last, next_action_embedding], dim=0).unsqueeze(1)
            current_seq = torch.cat([current_seq, new_col], dim=1)
            current_action_type = next_action_type

        return {
            "final_sequence": current_seq,
            "predictions": predictions,
            "action_types": action_types,
        }


Action_list = get_full_action_type_list()


def train_action_model(world_model, input_data, target_action, action_index, optimizer=None, steps=100):
    return world_model.train_action_model(
        input_data=input_data,
        target_action=target_action,
        action_type=action_index,
        optimizer=optimizer,
        steps=steps,
    )
