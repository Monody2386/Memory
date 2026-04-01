import importlib
import pathlib
import random
import sys

if __package__ is None or __package__ == "":
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from Consciousness import Consciousness


def print_section(title: str):
    print()
    print(title)


def in_memory_recall(noun: str):
    rm = importlib.import_module("knowledge.relation_map")
    arm = importlib.import_module("knowledge.adj_relation_map")
    noun = noun.lower()

    noun_relations = []
    adj_relations = []

    if noun in rm.noun_list:
        noun_idx = rm.noun_list.index(noun)
        for target_idx, raw_relation_type in enumerate(rm.relation_map[noun_idx]):
            rt = int(raw_relation_type)
            if rt == 0:
                continue
            relation_name = rm.relation_list[rt - 1] if rt - 1 < len(rm.relation_list) else f"relation_{rt}"
            target_noun = rm.noun_list[target_idx] if target_idx < len(rm.noun_list) else f"noun_{target_idx}"
            noun_relations.append(
                {
                    "source_noun": noun,
                    "target_noun": target_noun,
                    "relation_type": rt,
                    "relation_name": relation_name,
                }
            )

        for adj_idx, raw_relation_type in enumerate(arm.adj_relation_map[noun_idx]):
            rt = int(raw_relation_type)
            if rt == 0:
                continue
            relation_name = arm.adj_relation_list[rt - 1] if rt - 1 < len(arm.adj_relation_list) else f"relation_{rt}"
            adjective = arm.adjective_list[adj_idx] if adj_idx < len(arm.adjective_list) else f"adj_{adj_idx}"
            adj_relations.append(
                {
                    "noun": noun,
                    "adjective": adjective,
                    "relation_type": rt,
                    "relation_name": relation_name,
                }
            )

    return {
        "noun_noun": noun_relations,
        "adj_noun": adj_relations,
    }


def run_demo():
    consciousness = Consciousness(
        ask_confidence_threshold=0.92,
        min_ask_confidence_threshold=0.15,
    )

    print("Consciousness command demo")
    print("This demo shows the current behavior of recall() and predict().")

    print_section("1. Recall existing memory")
    recall_result = consciousness.recall(noun="apple")
    print("Input: recall(noun='apple')")
    print(recall_result)

    print_section("2. Predict with a confident result")
    confident_prediction = consciousness.predict(
        kind="adj_noun",
        noun="apple",
        relation_type=1,
    )
    print("Input: predict(kind='adj_noun', noun='apple', relation_type=1)")
    print(confident_prediction)

    print_section("3. Predict with a question-worthy result")
    question_prediction = consciousness.predict(
        kind="sample",
        rng=random.Random(7),
    )
    print("Input: predict(kind='sample', rng=random.Random(7))")
    print(question_prediction)

    if question_prediction["status"] == "question":
        question = question_prediction["question"]
        print_section("4. Provide feedback and learn")
        print("Question returned by predict():")
        print(question)

        feedback = consciousness.predict(
            kind="sample",
            rng=random.Random(7),
            answer_text="no",
            corrected_target="demo_feedback_token",
            save=False,
        )
        print("Input: same predict(...) + answer_text='no' + corrected_target='demo_feedback_token'")
        print(feedback)

        print_section("5. Recall after feedback (in-memory demo view)")
        print(
            "This view inspects the current in-memory state after learning with save=False, "
            "so the demo does not persist changes to your files."
        )
        print(in_memory_recall(question.source_noun))
    else:
        print_section("4. No question was triggered in this run")
        print("The sampled prediction was not in the uncertainty band, so no feedback step was run.")


if __name__ == "__main__":
    run_demo()
