from pathlib import Path
import sys

if __package__ in {None, ''}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prototype.high_level_commands import HighLevelCommands
from reward import (
    RewardEncoder,
    SubjectEventRewardEngine,
    SubjectEventRewardInput,
    SubjectEventRewardNet,
    SubjectEventRewardSample,
    SubjectEventRewardTrainer,
    reward_input_from_subject_event,
    subject_events_from_short_memory,
)
from world.world_model import action_dim, noun_dim


def _prediction_view(prediction):
    return {
        "subject": prediction.subject_text,
        "action": prediction.action_text,
        "object": prediction.object_text,
        "score": round(prediction.score, 4),
        "label": prediction.label,
    }


def main():
    cmd = HighLevelCommands()
    cmd.understand("tom eat apple")
    cmd.understand("cat eat banana")
    cmd.understand("tom eat")

    events = subject_events_from_short_memory(cmd.consciousness.short_memory)
    print("Subject-event reward demo")
    print("1. Reconstructed Subject Events")
    for event in events:
        print(event)

    encoder = RewardEncoder(cmd.consciousness)
    model = SubjectEventRewardNet(noun_dim=noun_dim, action_dim=action_dim, hidden_dim=64)
    engine = SubjectEventRewardEngine(model, encoder)
    trainer = SubjectEventRewardTrainer(model, encoder, lr=1e-3)

    samples = [
        SubjectEventRewardSample(subject_text="tom", action_text="eat", object_text="apple", reward_value=90.0),
        SubjectEventRewardSample(subject_text="cat", action_text="eat", object_text="banana", reward_value=30.0),
        SubjectEventRewardSample(subject_text="tom", action_text="eat", object_text=None, reward_value=10.0),
    ]

    print()
    print("2. Prediction Before Training")
    for event in events:
        print(_prediction_view(engine.predict(reward_input_from_subject_event(event))))

    print()
    print("3. Train Subject-Event Reward Model")
    history = trainer.train_epochs(samples, epochs=30)
    print(history[-1])

    print()
    print("4. Prediction After Training")
    for event in events:
        print(_prediction_view(engine.predict(reward_input_from_subject_event(event))))

    print()
    print("5. Direct Input Prediction")
    direct = engine.predict(SubjectEventRewardInput(subject_text="tom", action_text="eat", object_text="apple"))
    print(_prediction_view(direct))


if __name__ == "__main__":
    main()
