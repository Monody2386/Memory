from pathlib import Path
import sys

if __package__ in {None, ''}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prototype.consciousness import Consciousness
from reward import (
    RewardEncoder,
    RewardEngine,
    RewardInput,
    RewardMemory,
    RewardNet,
    RewardSample,
    RewardTrainer,
)
from world.world_model import action_dim, noun_dim


def _print_prediction(engine: RewardEngine, title: str, *, noun_text=None, action_text=None):
    prediction = engine.predict(RewardInput(noun_text=noun_text, action_text=action_text))
    print(title)
    print({
        "noun_text": noun_text,
        "action_text": action_text,
        "score": round(prediction.score, 4),
        "label": prediction.label,
    })


def main():
    consciousness = Consciousness()
    encoder = RewardEncoder(consciousness)
    model = RewardNet(noun_dim=noun_dim, action_dim=action_dim, hidden_dim=64)
    engine = RewardEngine(model, encoder)
    trainer = RewardTrainer(model, encoder, lr=1e-3)

    reward_memory = RewardMemory(
        [
            RewardSample(noun_text="apple", reward_value=80.0),
            RewardSample(noun_text="banana", reward_value=40.0),
            RewardSample(noun_text="stone", reward_value=-80.0),
            RewardSample(action_text="eat", reward_value=10.0),
            RewardSample(action_text="sleep", reward_value=50.0),
            RewardSample(noun_text="apple", action_text="eat", reward_value=90.0),
            RewardSample(noun_text="stone", action_text="eat", reward_value=-100.0),
            RewardSample(noun_text="banana", action_text="eat", reward_value=55.0),
        ]
    )

    print("Reward demo")
    print("This demo shows standalone reward prediction and training in the range [-100, 100].")

    print()
    print("1. Predictions Before Training")
    _print_prediction(engine, "noun only", noun_text="apple")
    _print_prediction(engine, "action only", action_text="eat")
    _print_prediction(engine, "action + noun", noun_text="apple", action_text="eat")
    _print_prediction(engine, "negative pair", noun_text="stone", action_text="eat")

    print()
    print("2. Train Reward Model")
    history = trainer.train_epochs(reward_memory.all_samples(), epochs=20)
    print(history[-1])

    print()
    print("3. Predictions After Training")
    _print_prediction(engine, "noun only", noun_text="apple")
    _print_prediction(engine, "action only", action_text="eat")
    _print_prediction(engine, "action + noun", noun_text="apple", action_text="eat")
    _print_prediction(engine, "negative pair", noun_text="stone", action_text="eat")

    print()
    print("4. Observed Reward From Self Perspective")
    observed = engine.predict_observed_reward(
        RewardInput(noun_text="apple", action_text="eat"),
        observer_instance_id="speaker#core",
        target_instance_id="tom#new:1",
        empathy=0.8,
        relation=1.0,
    )
    print({
        "base_score": round(observed.base_score, 4),
        "empathy": observed.empathy,
        "relation": observed.relation,
        "final_score": round(observed.final_score, 4),
        "label": observed.label,
    })


if __name__ == "__main__":
    main()
