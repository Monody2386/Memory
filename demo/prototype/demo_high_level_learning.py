import pathlib
import sys

if __package__ is None or __package__ == "":
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

from prototype.high_level_commands import HighLevelCommands


SENTENCES = [
    {"sentence": "cat eat apple"},
    {"sentence": "dog chase ball"},
    {"sentence": "apple fall"},
]


def print_section(title: str):
    print()
    print(title)


def run_demo(num_epochs: int = 30, print_every: int = 5):
    commands = HighLevelCommands()

    print("High-level learning demo")
    print("This demo uses only HighLevelCommands interfaces.")

    print_section("1. Understand sentences")
    understand_results = []
    for spec in SENTENCES:
        result = commands.understand(spec["sentence"])
        understand_results.append(result)
        print(
            {
                "sentence": result["sentence"],
                "sentence_type": result["sentence_type"],
                "event_entries_added": result["event_entries_added"],
                "relation_entries_added": result["relation_entries_added"],
                "focus": result["focus"],
            }
        )

    target_instance_id = understand_results[-1]["states"][0].noun_instance_id

    print_section("2. Training target")
    print(
        {
            "instance_id": target_instance_id,
            "target_sentence": understand_results[-1]["sentence"],
            "target_event_action": understand_results[-1]["states"][0].action,
        }
    )

    print_section("3. Learn event repeatedly")
    losses = []
    last_result = None
    for epoch in range(1, num_epochs + 1):
        last_result = commands.learn_event(target_instance_id, target_score=50.0)
        loss = float(last_result["train_result"]["loss"])
        losses.append(loss)
        if epoch == 1 or epoch % print_every == 0 or epoch == num_epochs:
            print(
                {
                    "epoch": epoch,
                    "loss": round(loss, 6),
                    "input_focus": last_result["input_focus"],
                    "target_event": last_result["target_event"],
                    "pred_action_type": last_result["train_result"]["pred_action_type"],
                }
            )

    print_section("4. Training summary")
    print(
        {
            "instance_id": target_instance_id,
            "initial_loss": round(losses[0], 6),
            "final_loss": round(losses[-1], 6),
            "best_loss": round(min(losses), 6),
            "epochs": num_epochs,
        }
    )

    return {
        "commands": commands,
        "losses": losses,
        "last_result": last_result,
    }


if __name__ == "__main__":
    run_demo()
