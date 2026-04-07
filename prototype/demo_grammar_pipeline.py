from pathlib import Path
import sys

if __package__ in {None, ''}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prototype.high_level_commands import HighLevelCommands
from prototype import grammar


def _print_section(title: str):
    print()
    print(title)

def main():
    cmd = HighLevelCommands()

    context_sentence = "cat eat apple"
    test_sentence = "that cat eat sweet fruit"

    print("Grammar pipeline demo")
    print("This demo shows each grammar layer on a sentence with short-memory context.")

    _print_section("1. Build Short-Memory Context")
    context_result = cmd.understand(context_sentence)
    print(f"context sentence: {context_sentence}")
    print(context_result)

    short_memory = cmd.consciousness.short_memory
    instance_context = grammar.build_instance_context_from_memory(short_memory)
    print("instance_context:")
    print(instance_context)

    _print_section("2. Layer 1: Tokenizer")
    tokens = grammar.tokenize_sentence(test_sentence)
    print(tokens)

    _print_section("3. Layer 2: Part-of-Speech Analysis")
    tagged_tokens = grammar.tag_tokens(test_sentence, short_memory=short_memory)
    print(tagged_tokens)

    _print_section("4. Layer 3a: adj+noun Reduction")
    reduced_tokens_1, reduced_tags_1, reduced_relations = grammar._reduce_adj_noun_phrases(
        tokens,
        tagged_tokens,
        adjective_relation_types={"sweet": "taste"},
        infer_missing=True,
    )
    print("tokens:")
    print(reduced_tokens_1)
    print("tags:")
    print(reduced_tags_1)
    print("extracted_relations:")
    print(reduced_relations)

    _print_section("5. Layer 3b: pronoun+noun Reduction")
    reduced_tokens_2, reduced_tags_2 = grammar._reduce_pronoun_noun_phrases(
        reduced_tokens_1,
        reduced_tags_1,
        short_memory=short_memory,
    )
    print("tokens:")
    print(reduced_tokens_2)
    print("tags:")
    print(reduced_tags_2)

    _print_section("6. Layer 3c: Existing-Noun Binding")
    reduced_tokens_3, reduced_tags_3 = grammar._bind_existing_noun_instances(
        reduced_tokens_2,
        reduced_tags_2,
        short_memory=short_memory,
    )
    print("tokens:")
    print(reduced_tokens_3)
    print("tags:")
    print(reduced_tags_3)

    _print_section("7. Layer 4: Information Extraction")
    extractor = grammar._select_extractor(reduced_tokens_3, reduced_tags_3)
    parsed_before_instance = extractor(
        reduced_tokens_3,
        reduced_tags_3,
        adjective_relation_types={"sweet": "taste"},
        infer_missing=True,
    )
    parsed_before_instance.relation_tuples = list(reduced_relations) + list(parsed_before_instance.relation_tuples)
    print(f"extractor: {extractor.__name__}")
    print("action_tuples:")
    print(parsed_before_instance.action_tuples)
    print("relation_tuples:")
    print(parsed_before_instance.relation_tuples)

    _print_section("8. Layer 5: Instance Decision Rules")
    parsed_after_instance = grammar.resolve_instances_for_parsed_sentence(
        parsed_before_instance,
        short_memory=short_memory,
    )
    print("action_tuples:")
    print(parsed_after_instance.action_tuples)
    print("relation_tuples:")
    print(parsed_after_instance.relation_tuples)

    _print_section("9. Layer 6: Time-Step Rules")
    resolved_time_position = grammar.determine_time_position(
        parsed_after_instance,
        short_memory=short_memory,
    )
    print({"time_position": resolved_time_position})

    _print_section("10. Final parse_sentence Output")
    parsed_final = grammar.parse_sentence(
        test_sentence,
        adjective_relation_types={"sweet": "taste"},
        short_memory=short_memory,
    )
    print(parsed_final)
    print({"sentence_type": grammar.classify_sentence_type_from_parsed(parsed_final)})

    _print_section("11. Understand The Test Sentence Into Memory")
    understand_result = cmd.understand(
        test_sentence,
        adjective_relation_types={"sweet": "taste"},
    )
    print(understand_result)
    print("event_memory:")
    print(cmd.consciousness.inspect_memory(kind="event", order_by="time"))
    print("relation_memory:")
    print(cmd.consciousness.inspect_memory(kind="relation", order_by="time"))
    print("focus:")
    print(cmd.consciousness.inspect_focus())


if __name__ == "__main__":
    main()
