import pathlib
import sys

if __package__ is None or __package__ == "":
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from prototype.high_level_commands import HighLevelCommands


SENTENCE_PLANS = [
    {
        "sentence": "cat glorp apple",
        "what_answers": {
            "glorp": {"pos": "action"},
        },
    },
    {
        "sentence": "zesty apple",
        "what_answers": {
            "zesty": {"pos": "adj", "relation_name": "taste"},
        },
    },
    {
        "sentence": "apple include galaxy",
        "memory_answers": {
            ("noun_noun_relation", "apple", "galaxy", "include"): "y",
        },
    },
    {
        "sentence": "apple fall",
        "what_answers": {
            "fall": {"pos": "action"},
        },
    },
]


def print_section(title: str):
    print()
    print(title)


def answer_unknown_words(commands: HighLevelCommands, result: dict):
    answered = []
    plan_answers = result.get("what_answers", {})
    for question in result["questions"]:
        answer_info = plan_answers.get(question["word"])
        if answer_info is None:
            continue
        answered.append(
            commands.answer_what(
                question,
                answer_info,
                save=False,
                re_understand=False,
                write_facts=True,
            )
        )
    return answered


def answer_memory_questions(commands: HighLevelCommands, memory_result: dict, planned_answers: dict):
    answers = []
    for question in memory_result["questions"]:
        entry = question.get("memory_entry", {})
        source_noun = question.get("source_noun") or entry.get("source_text") or entry.get("subject_text") or entry.get("noun_text")
        target = question.get("target") or entry.get("target_text") or entry.get("object_text") or entry.get("action_text")
        relation_name = question.get("relation_name") or entry.get("relation_name") or entry.get("reward_word") or entry.get("surprise_word")
        key = (
            question["kind"],
            source_noun,
            target,
            relation_name,
        )
        answer_text = planned_answers.get(key)
        if answer_text is None:
            continue
        answers.append(
            commands.answer_memory_question(
                question,
                answer_text,
                save=False,
            )
        )
    return answers


def describe_memory_question(question: dict) -> str:
    prompt = question.get("prompt")
    if prompt:
        return str(prompt)
    entry = question.get("memory_entry", {})
    source = entry.get("source_text") or entry.get("subject_text") or entry.get("noun_text") or question.get("source_noun")
    relation = entry.get("relation_name") or entry.get("reward_word") or entry.get("surprise_word") or question.get("relation_name")
    target = entry.get("target_text") or entry.get("object_text") or entry.get("action_text") or question.get("target")
    reason = question.get("reason") or "review"
    return f"{question.get('kind')} | {source} -> {relation} -> {target} | reason={reason}"


def run_demo(event_epochs: int = 10):
    commands = HighLevelCommands()

    print("High-level full pipeline demo")
    print("This demo uses only HighLevelCommands interfaces.")

    understand_results = []

    print_section("1. Sentence Intake Through question_what")
    for plan in SENTENCE_PLANS:
        sentence = plan["sentence"]
        result = commands.question_what(sentence)
        result["what_answers"] = plan.get("what_answers", {})
        print({
            "sentence": sentence,
            "status": result["status"],
            "unknown_count": result["unknown_count"],
        })

        if result["status"] == "question":
            print("Unknown-word questions")
            for item in result["questions"]:
                print({
                    "word": item["word"],
                    "predicted_pos": item["predicted_pos"],
                    "prompt": item["prompt"],
                })
            answers = answer_unknown_words(commands, result)
            print("Applied what-answers")
            for item in answers:
                print({
                    "answer_info": item["answer_info"],
                    "fact_writes": item["fact_writes"],
                })
            understood = commands.question_what(sentence)
            print({
                "sentence": sentence,
                "status": understood["status"],
                "unknown_count": understood["unknown_count"],
            })
            result = understood

        if result["status"] == "understood":
            understand_result = result["understand_result"]
            understand_results.append(understand_result)
            print({
                "sentence_type": understand_result["sentence_type"],
                "event_entries_added": understand_result["event_entries_added"],
                "relation_entries_added": understand_result["relation_entries_added"],
                "focus": understand_result["focus"],
            })

            memory_questions = commands.question()
            planned_answers = plan.get("memory_answers", {})
            if memory_questions["question_count"]:
                print("Memory questions")
                for item in memory_questions["questions"]:
                    print({
                        "kind": item["kind"],
                        "prompt": describe_memory_question(item),
                    })
                applied = answer_memory_questions(commands, memory_questions, planned_answers)
                if applied:
                    print("Applied memory answers")
                    for item in applied:
                        print({
                            "kind": item["kind"],
                            "written": item["written"],
                            "result": item["result"],
                        })

    print_section("2. Inspect Current Memory")
    print({
        "event_memory": commands.consciousness.inspect_memory(kind="event", order_by="time"),
        "relation_memory": commands.consciousness.inspect_memory(kind="relation", order_by="time"),
    })

    print_section("3. Learn Event For Reused Apple Instance")
    target_instance_id = None
    if understand_results:
        last_understand = understand_results[-1]
        for state in last_understand["states"]:
            if getattr(state, "noun", None) == "apple":
                target_instance_id = state.noun_instance_id
                break
    if target_instance_id is None:
        raise RuntimeError("Could not find apple instance for event learning")

    learn_result = commands.learn_event(target_instance_id, target_score=50.0, epochs=event_epochs)
    print({
        "instance_id": target_instance_id,
        "input_focus": learn_result["input_focus"],
        "target_event": learn_result["target_event"],
        "final_loss": learn_result["train_result"]["loss"],
        "final_pred_action_type_on_training_context": learn_result["train_result"]["pred_action_type"],
        "target_action_type": learn_result["target_event"]["action_type"],
        "loss_history": [round(float(x), 6) for x in learn_result["train_result"]["loss_history"]],
    })

    print_section("4. Predict Next Event After Current Focus")
    focus = commands.consciousness.inspect_focus()
    prediction = commands.predict_event(int(focus["action_type"]))
    print({
        "note": "This prediction uses the full current short memory, so it predicts the event after the current focus rather than re-evaluating the training target.",
        "focus": focus,
        "prediction": prediction,
    })

    print_section("5. Sleep")
    sleep_result = commands.sleep(save=False)
    print(sleep_result)

    return {
        "commands": commands,
        "understand_results": understand_results,
        "learn_result": learn_result,
        "prediction": prediction,
        "sleep_result": sleep_result,
    }


if __name__ == "__main__":
    run_demo()



