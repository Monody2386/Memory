import pathlib
import random
import sys
from typing import Optional

if __package__ is None or __package__ == "":
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from Consciousness import Consciousness
from prototype.question import ProposedQuestion

STOP_TOKEN = "stop"
THINKING_MODE = "thinking"
LEARNING_MODE = "learning"


def describe_question(question: ProposedQuestion):
    return {
        "kind": question.kind,
        "confidence": round(float(question.confidence), 4),
        "relation": question.relation_name,
        "prediction": question.predicted_target,
        "candidates": question.top_candidates,
    }


def print_defined_vocab(consciousness: Consciousness):
    print()
    print("Defined vocabulary")
    print(consciousness.inspect_vocab())


def prompt_mode() -> str:
    while True:
        raw = input(f"Choose mode [{THINKING_MODE}/{LEARNING_MODE}/{STOP_TOKEN}]: ").strip().lower()
        if raw in {THINKING_MODE, LEARNING_MODE, STOP_TOKEN}:
            return raw
        print("Please enter 'thinking', 'learning', or 'stop'.")


def prompt_yes_no(prompt: str) -> str:
    while True:
        answer = input(f"{prompt} [yes/no/{STOP_TOKEN}]: ").strip().lower()
        if answer in {"yes", "no", STOP_TOKEN}:
            return answer
        print("Please answer with 'yes', 'no', or 'stop'.")


def prompt_text(prompt: str) -> Optional[str]:
    value = input(f"{prompt} (or '{STOP_TOKEN}'): ").strip().lower()
    if value == STOP_TOKEN:
        return None
    return value


def prompt_relation_type(prompt: str) -> Optional[int]:
    while True:
        raw_value = input(f"{prompt} (int or '{STOP_TOKEN}'): ").strip().lower()
        if raw_value == STOP_TOKEN:
            return None
        if raw_value.isdigit():
            return int(raw_value)
        print("Please enter an integer relation_type or 'stop'.")


def prompt_relation_kind() -> Optional[str]:
    while True:
        raw = input(f"Choose relation kind [noun_noun/adj_noun/{STOP_TOKEN}]: ").strip().lower()
        if raw in {"noun_noun", "adj_noun", STOP_TOKEN}:
            return None if raw == STOP_TOKEN else raw
        print("Please enter 'noun_noun', 'adj_noun', or 'stop'.")


def handle_question(consciousness: Consciousness, question: ProposedQuestion):
    print()
    print("Model question")
    print(question.prompt)
    print(describe_question(question))

    answer = prompt_yes_no("Your answer")
    if answer == STOP_TOKEN:
        return None

    corrected_target = None
    corrected_relation_type = None
    if answer == "no":
        if question.kind == "adj_noun":
            corrected_target = prompt_text(
                "Please provide the correct adjective. Existing or new adjectives are both allowed"
            )
        elif question.question_target == "noun":
            corrected_target = prompt_text(
                "Please provide the correct noun. Existing or new nouns are both allowed"
            )
        else:
            corrected_relation_type = prompt_relation_type("Please provide the correct relation_type")
        if corrected_target is None and corrected_relation_type is None:
            return None

    return consciousness.answer_question(
        question,
        answer,
        corrected_target=corrected_target,
        corrected_relation_type=corrected_relation_type,
        save=True,
    )


def direct_learning_step(consciousness: Consciousness):
    kind = prompt_relation_kind()
    if kind is None:
        return None

    if kind == "noun_noun":
        source_noun = prompt_text("Source noun")
        if source_noun is None:
            return None
        target_noun = prompt_text("Target noun")
        if target_noun is None:
            return None
        relation_type = prompt_relation_type("relation_type")
        if relation_type is None:
            return None
        return consciousness.learn_noun_relation(source_noun, target_noun, relation_type, save=True)

    noun = prompt_text("Noun")
    if noun is None:
        return None
    adjective = prompt_text("Adjective")
    if adjective is None:
        return None
    relation_type = prompt_relation_type("relation_type")
    if relation_type is None:
        return None
    return consciousness.learn_adj_relation(noun, adjective, relation_type, save=True)


def run_demo(seed: int = 7):
    consciousness = Consciousness(
        ask_confidence_threshold=0.92,
        min_ask_confidence_threshold=0.15,
    )
    rng = random.Random(seed)
    thinking_count = 0
    learning_count = 0

    print("Interactive online-learning demo")
    print(
        "Thinking mode: the model predicts first, checks confidence, asks one question, then learns from your answer."
    )
    print(
        "Learning mode: you directly provide a noun_noun or adj_noun relation for immediate training."
    )
    print(f"Type '{STOP_TOKEN}' at any prompt to stop and inspect the learned vocabulary.")

    while True:
        mode = prompt_mode()
        if mode == STOP_TOKEN:
            print()
            print("Demo stopped by user.")
            print({"thinking_steps": thinking_count, "learning_steps": learning_count})
            print_defined_vocab(consciousness)
            return

        if mode == THINKING_MODE:
            question = consciousness.think(rng)
            if question is None:
                print()
                print("No suitable question was found in the current uncertainty band.")
                continue

            thinking_count += 1
            print()
            print(f"Thinking step #{thinking_count}")
            print("Before learning")
            print(describe_question(question))

            result = handle_question(consciousness, question)
            if result is None:
                print()
                print("Demo stopped by user.")
                print({"thinking_steps": thinking_count - 1, "learning_steps": learning_count})
                print_defined_vocab(consciousness)
                return

            updated_question = consciousness.re_predict_question(question)
            print()
            print("Learning result")
            print(result)
            print()
            print("After learning")
            print(updated_question.prompt)
            print(describe_question(updated_question))
            continue

        learning_result = direct_learning_step(consciousness)
        if learning_result is None:
            print()
            print("Demo stopped by user.")
            print({"thinking_steps": thinking_count, "learning_steps": learning_count})
            print_defined_vocab(consciousness)
            return

        learning_count += 1
        print()
        print(f"Learning step #{learning_count}")
        print(learning_result)


if __name__ == "__main__":
    run_demo()
