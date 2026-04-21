import torch
import torch.nn as nn
import torch.nn.functional as F


class ActionModel(nn.Module):
    """Two-stage action model used by WorldModel.

    Stage 1:
        event sequence -> context vector
    Stage 2:
        context vector -> predicted action embedding
    """

    def __init__(self, noun_dim, action_dim, attention_dim, value_dim, hidden_dim):
        super().__init__()
        self.Wq = nn.Linear(noun_dim + action_dim, attention_dim)
        self.Wk = nn.Linear(noun_dim + action_dim, attention_dim)
        self.Wv = nn.Linear(noun_dim + action_dim, value_dim)
        self.scale = attention_dim ** 0.5
        self.action_predict = nn.Sequential(
            nn.Linear(value_dim, hidden_dim, bias=False),
            nn.Linear(hidden_dim, action_dim, bias=False),
        )

    def encode_sequence_context(self, x):
        """Compress short-memory events into one context vector.

        Input:
            x: event sequence shaped [batch, event_count, noun_dim + action_dim].
        Output:
            tensor: context vector shaped [batch, value_dim].
        """
        input_prefix = x[:, :-1, :]
        input_suffix = x[:, -1, :]

        q = self.Wq(input_suffix)
        k = self.Wk(input_prefix)
        v = self.Wv(input_prefix)

        scores = torch.matmul(q.unsqueeze(1), k.transpose(-2, -1)) / self.scale
        attention_weight = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weight, v).squeeze(1)
        output = output + self.Wv(input_suffix)
        return output

    def integrate_attention(self, x):
        """Backward-compatible alias for the sequence encoder stage."""
        return self.encode_sequence_context(x)

    def predict_action_from_context(self, context):
        """Predict the next action embedding from a context vector."""
        return self.action_predict(context)

    def forward(self, x):
        context = self.encode_sequence_context(x)
        return self.predict_action_from_context(context)

