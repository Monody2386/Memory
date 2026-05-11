from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from prototype.high_level_commands import HighLevelCommands
from surprise import (
    SubjectEventSurpriseEngine,
    SubjectEventSurpriseNet,
    SubjectEventSurpriseSample,
    SubjectEventSurpriseTrainer,
    SurpriseEncoder,
    subject_events_from_surprise_memory,
    surprise_input_from_subject_event,
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
    cmd.understand("dog eat book")

    events = subject_events_from_surprise_memory(cmd.consciousness.short_memory)
    print("Subject-event surprise demo")
    print("1. Reconstructed Subject Events")
    for event in events:
        print(event)

    encoder = SurpriseEncoder(cmd.consciousness)
    model = SubjectEventSurpriseNet(noun_dim=noun_dim, action_dim=action_dim, hidden_dim=64)
    engine = SubjectEventSurpriseEngine(model, encoder)
    trainer = SubjectEventSurpriseTrainer(model, encoder, lr=1e-3)

    samples = [
        SubjectEventSurpriseSample(subject_text="tom", action_text="eat", object_text="apple", surprise_value=10.0),
        SubjectEventSurpriseSample(subject_text="cat", action_text="eat", object_text="banana", surprise_value=35.0),
        SubjectEventSurpriseSample(subject_text="dog", action_text="eat", object_text="book", surprise_value=90.0),
    ]

    print()
    print("2. Prediction Before Training")
    for event in events:
        print(_prediction_view(engine.predict(surprise_input_from_subject_event(event))))

    print()
    print("3. Train Subject-Event Surprise Model")
    history = trainer.train_epochs(samples, epochs=30)
    print(history[-1])

    print()
    print("4. Prediction After Training")
    for event in events:
        print(_prediction_view(engine.predict(surprise_input_from_subject_event(event))))


if __name__ == "__main__":
    main()
