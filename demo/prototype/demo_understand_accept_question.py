from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from prototype.high_level_commands import HighLevelCommands


def _compact_pairs(pairs):
    compact = []
    for pair in pairs:
        compact.append(
            {
                "kind": pair.get("kind"),
                "source": pair.get("source_noun"),
                "target": pair.get("target"),
                "relation": pair.get("relation_name"),
                "accept_label": pair.get("accept_label"),
                "diff_value": pair.get("diff_value"),
                "method": pair.get("accept_method"),
                "prediction_label": pair.get("prediction_label"),
            }
        )
    return compact


def _compact_questions(items):
    compact = []
    for item in items:
        entry = item.get("memory_entry", {})
        compact.append(
            {
                "type": item.get("type"),
                "answer": item.get("answer"),
                "reason": item.get("reason"),
                "memory_kind": item.get("memory_kind") or entry.get("memory_kind"),
                "kind": item.get("kind") or entry.get("pair_kind"),
                "question_label": entry.get("question_label"),
                "accept_label": item.get("accept_label") or entry.get("accept_label"),
                "diff_value": item.get("diff_value") or entry.get("diff_value"),
                "source": entry.get("source_text") or entry.get("subject_text") or entry.get("noun_text"),
                "target": entry.get("target_text") or entry.get("object_text") or entry.get("action_text"),
            }
        )
    return compact


def _compact_updates(items):
    compact = []
    for item in items:
        entry = item.get("question", {}).get("memory_entry", {})
        compact.append(
            {
                "accepted": item.get("accepted"),
                "action": item.get("action"),
                "kind": item.get("kind"),
                "source": entry.get("source_text") or entry.get("subject_text") or entry.get("noun_text"),
                "target": entry.get("target_text") or entry.get("object_text") or entry.get("action_text"),
                "result_count": item.get("result", {}).get("count") if isinstance(item.get("result"), dict) else None,
                "result_reason": item.get("result", {}).get("reason") if isinstance(item.get("result"), dict) else None,
            }
        )
    return compact


def _compact_skips(items):
    compact = []
    for item in items:
        entry = item.get("memory_entry", {})
        compact.append(
            {
                "reason": item.get("reason"),
                "memory_kind": item.get("memory_kind"),
                "abs_diff_value": item.get("abs_diff_value"),
                "source": entry.get("source_text") or entry.get("subject_text") or entry.get("noun_text"),
                "target": entry.get("target_text") or entry.get("object_text") or entry.get("action_text"),
            }
        )
    return compact


def main():
    cmd = HighLevelCommands()
    sentences = [
        "tom love eat apple",
        "tom not love eat apple",
        "does cat eat apple",
        "cat can eat apple",
        "cat can not eat apple",
        "can cat eat apple",
        "a cat belong to animal",
        "is tom teacher",
    ]

    print("Demo: understand -> accept -> question")
    print("Each sentence is written into short memory with one sentence_label, then accepted and questioned.")

    for sentence in sentences:
        print("\n" + "=" * 80)
        print(f"Sentence: {sentence}")

        understand_result = cmd.understand(sentence)
        sentence_label = understand_result["sentence_label"]
        print("UNDERSTAND")
        print(
            {
                "sentence_label": sentence_label,
                "sentence_type": understand_result["sentence_type"],
                "structure": understand_result["structure"],
                "event_entries_added": understand_result["event_entries_added"],
                "relation_entries_added": understand_result["relation_entries_added"],
                "reward_entries_added": understand_result["reward_entries_added"],
                "surprise_entries_added": understand_result["surprise_entries_added"],
            }
        )

        accept_result = cmd.accept(sentence_label=sentence_label)
        print("ACCEPT")
        print(
            {
                "status": accept_result["status"],
                "labeled_pair_count": accept_result["labeled_pair_count"],
                "issue_count": accept_result["issue_count"],
                "labeled_pairs": _compact_pairs(accept_result["labeled_pairs"]),
            }
        )

        question_result = cmd.question(
            sentence_label=sentence_label,
            confirm_threshold=50.0,
            yes_threshold=30.0,
            no_threshold=70.0,
        )
        print("QUESTION")
        print(
            {
                "question_count": question_result["question_count"],
                "answer_count": question_result["answer_count"],
                "auto_update_count": question_result["auto_update_count"],
                "skipped_update_count": question_result["skipped_update_count"],
                "questions": _compact_questions(question_result["questions"]),
                "answers": _compact_questions(question_result["answers"]),
                "auto_updates": _compact_updates(question_result["auto_updates"]),
                "skipped_updates": _compact_skips(question_result["skipped_updates"]),
            }
        )

        if sentence == "tom love eat apple":
            non_interactive = cmd.question(
                sentence_label=sentence_label,
                confirm_threshold=0.0,
                interact=False,
                auto_accept=False,
            )
            print("QUESTION interact=False, threshold=0")
            print(
                {
                    "auto_update_count": non_interactive["auto_update_count"],
                    "skipped_update_count": non_interactive["skipped_update_count"],
                    "auto_updates": _compact_updates(non_interactive["auto_updates"]),
                    "skipped_updates": _compact_skips(non_interactive["skipped_updates"]),
                }
            )

    print("\n" + "=" * 80)
    print("Final short memory content")
    memory = cmd.consciousness.short_memory.get_content_view(order_by="time")
    for kind in ["event", "relation", "reward", "surprise"]:
        print(f"\n{kind.upper()}")
        for entry in memory[kind]:
            print(
                {
                    "sentence_label": entry.get("sentence_label"),
                    "question_label": entry.get("question_label"),
                    "accept_label": entry.get("accept_label"),
                    "diff_value": entry.get("diff_value"),
                    "source": entry.get("source_text") or entry.get("subject_text") or entry.get("noun_text"),
                    "relation": entry.get("relation_name") or entry.get("reward_word") or entry.get("surprise_word"),
                    "target": entry.get("target_text") or entry.get("object_text") or entry.get("action_text"),
                    "reward_value": entry.get("reward_value"),
                    "surprise_value": entry.get("surprise_value"),
                }
            )


if __name__ == "__main__":
    main()
