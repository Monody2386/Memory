from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prototype.high_level_commands import HighLevelCommands
from reward import SubjectEventRewardSample


def _emotion_view(result):
    event = result.get("event") or {}
    return {
        "status": result["status"],
        "noun": result["noun"],
        "selected_instance_id": result.get("selected_instance_id"),
        "event_role": result.get("event_role"),
        "latest_event": {
            "subject": event.get("subject_text"),
            "action": event.get("action_text"),
            "object": event.get("object_text"),
            "time_position": event.get("time_position"),
        },
        "reward_score": None if result["reward_score"] is None else round(result["reward_score"], 4),
        "reward_label": result["reward_label"],
    }


def main():
    cmd = HighLevelCommands()

    print("Predict emotion high-level command demo")
    print()
    print("1. Understand events")
    for sentence in ["tom eat apple", "cat eat banana"]:
        result = cmd.understand(sentence)
        print({"sentence": sentence, "type": result["sentence_type"], "events_added": result["event_entries_added"]})

    print()
    print("2. Understand reward preferences")
    for sentence in ["tom love eat apple", "cat hate eat banana"]:
        result = cmd.understand(sentence)
        print({"sentence": sentence, "type": result["sentence_type"], "rewards_added": result["reward_entries_added"]})

    print()
    print("3. Train subject emotion model from memory.reward_list")
    train_result = cmd.train_emotion_reward(epochs=40)
    print({
        "role": train_result["role"],
        "sample_count": train_result["sample_count"],
        "epochs": train_result["epochs"],
        "last_result": train_result["last_result"],
    })

    print()
    print("4. Train object emotion model from explicit object-reward samples")
    object_train_result = cmd.train_object_emotion_reward(
        [
            SubjectEventRewardSample(
                subject_text="tom",
                action_text="eat",
                object_text="apple",
                reward_value=-80.0,
                source="manual_object_reward",
            ),
            SubjectEventRewardSample(
                subject_text="cat",
                action_text="eat",
                object_text="banana",
                reward_value=20.0,
                source="manual_object_reward",
            ),
        ],
        epochs=40,
    )
    print({
        "role": object_train_result["role"],
        "sample_count": object_train_result["sample_count"],
        "epochs": object_train_result["epochs"],
        "last_result": object_train_result["last_result"],
    })

    print()
    print("5. predict_emotion(noun)")
    for noun in ["tom", "cat", "apple", "banana"]:
        print(_emotion_view(cmd.predict_emotion(noun)))


if __name__ == "__main__":
    main()
