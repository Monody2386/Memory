import pathlib
import random
import sys

if __package__ is None or __package__ == "":
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from Consciousness import Consciousness


def print_section(title: str):
    print()
    print(title)


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

        print_section("5. Inspect vocabulary snapshot")
        print(
            "This high-level view inspects the currently available vocabulary through consciousness interfaces."
        )
        print(consciousness.inspect_vocab())
    else:
        print_section("4. No question was triggered in this run")
        print("The sampled prediction was not in the uncertainty band, so no feedback step was run.")


if __name__ == "__main__":
    run_demo()
