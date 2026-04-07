"""High-level command wrappers built on top of the Consciousness API."""

from dataclasses import dataclass, field
from typing import Any, Optional

from .consciousness import Consciousness
from .grammar import parse_sentence


@dataclass
class HighLevelCommands:
    """Task-level commands composed from low-level consciousness interfaces."""

    consciousness: Consciousness = field(default_factory=Consciousness)

    def _write_supervised_facts_from_sentence(
        self,
        sentence: str,
        *,
        relation_name_override: Optional[str] = None,
        relation_token_override: Optional[str] = None,
        save: bool = True,
    ):
        parsed = parse_sentence(sentence, short_memory=self.consciousness.short_memory)
        writes = []

        for relation_tuple in parsed.relation_tuples:
            relation_label = relation_tuple.relation
            if relation_name_override and relation_token_override and relation_label == relation_token_override:
                relation_label = relation_name_override

            relation_type = self.consciousness.resolve_relation_type(relation_tuple.kind, relation_label)
            if relation_type is None:
                continue

            if relation_tuple.kind == "noun_noun_relation":
                writes.append(
                    self.consciousness.remember_noun_relation(
                        relation_tuple.source,
                        relation_tuple.target,
                        relation_type,
                        save=save,
                    )
                )
            elif relation_tuple.kind == "adj_noun_relation":
                writes.append(
                    self.consciousness.remember_adj_relation(
                        relation_tuple.source,
                        relation_tuple.target,
                        relation_type,
                        save=save,
                    )
                )

        for action_tuple in parsed.action_tuples:
            writes.append(
                self.consciousness.remember_noun_action(
                    action_tuple.noun,
                    action_tuple.action,
                    save=save,
                )
            )

        return writes
    def understand(
        self,
        sentence: str,
        *,
        time_position: Optional[int] = None,
        base_score: float = 1.0,
        adjective_relation_types: Optional[Any] = None,
    ):
        """Understand one sentence by parsing it and storing the extracted information in memory.

        Input:
            sentence: raw sentence text.
            time_position: optional explicit time step for memory insertion.
            base_score: initial attention score for created memory entries.
            adjective_relation_types: optional adjective->relation mapping for grammar.
        Output:
            dict: high-level summary of the understanding step, including grammar output
            and memory write results.
        """
        parsed = parse_sentence(
            sentence,
            adjective_relation_types=adjective_relation_types,
            short_memory=self.consciousness.short_memory,
        )
        observation = self.consciousness.observe(
            sentence,
            time_position=time_position,
            base_score=base_score,
            adjective_relation_types=adjective_relation_types,
        )

        instance_ids = []
        seen_instance_ids = set()
        for relation_tuple in parsed.relation_tuples:
            for instance_id in (relation_tuple.source_instance_id, relation_tuple.target_instance_id):
                if instance_id is None or instance_id in seen_instance_ids:
                    continue
                seen_instance_ids.add(instance_id)
                instance_ids.append(instance_id)
        for action_tuple in parsed.action_tuples:
            instance_id = action_tuple.noun_instance_id
            if instance_id is None or instance_id in seen_instance_ids:
                continue
            seen_instance_ids.add(instance_id)
            instance_ids.append(instance_id)

        rebuilt_instances = [
            self.consciousness.rebuild_instance(instance_id)
            for instance_id in instance_ids
        ]

        focus = self.consciousness.inspect_focus()
        return {
            "command": "understand",
            "sentence": sentence,
            "sentence_type": observation["sentence_type"],
            "structure": observation["structure"],
            "tokens": observation["tokens"],
            "action_count": observation["action_count"],
            "relation_count": observation["relation_count"],
            "event_entries_added": observation["event_entries_added"],
            "relation_entries_added": observation["relation_entries_added"],
            "states": observation["states"],
            "updated_instances": rebuilt_instances,
            "focus": focus,
        }

    def learn_relation(
        self,
        relation_kind: str,
        relation_name: str,
        *,
        step_scale: Optional[float] = None,
        save: bool = True,
    ):
        """Learn one relation clone from short memory, then sync it into long-term knowledge.

        Input:
            relation_kind: relation family, such as noun_noun_relation or adj_noun_relation.
            relation_name: relation label to update.
            step_scale: optional scaling factor for clone update.
            save: whether to persist the synchronized long-term weights.
        Output:
            dict: high-level summary combining clone update and knowledge synchronization.
        """
        clone_before = self.consciousness.inspect_relation_clone(relation_kind, relation_name)
        update_summary = self.consciousness.update_relation_clone(
            relation_kind,
            relation_name,
            step_scale=step_scale,
        )
        clone_after = self.consciousness.inspect_relation_clone(relation_kind, relation_name)
        sync_summary = self.consciousness.sync_relation_clone_to_knowledge(
            relation_kind,
            relation_name,
            save=save,
        )
        return {
            "command": "learn_relation",
            "relation_kind": relation_kind,
            "relation_name": relation_name,
            "before": clone_before,
            "update": update_summary,
            "after": clone_after,
            "sync": sync_summary,
        }

    def predict_event(
        self,
        action_type: int,
        *,
        steps: Optional[int] = None,
        score: float = 0.5,
    ):
        """Predict the next event from the current short-memory event sequence.

        Input:
            action_type: action selector used by the world model.
            steps: optional event window size.
            score: score assigned to the predicted event.
        Output:
            dict: high-level summary containing the current focus, input shape, and predicted event.
        """
        world_input = self.consciousness.build_world_input(steps=steps)
        focus = self.consciousness.inspect_focus(steps=steps)
        prediction = self.consciousness.predict_next_event(
            action_type,
            steps=steps,
            score=score,
        )
        return {
            "command": "predict_event",
            "action_type": int(action_type),
            "input_shape": world_input["shape"],
            "focus": focus,
            "predicted_event": prediction["predicted_event"],
            "pred_action_type": int(prediction["pred_action_type"]),
        }

    def focus_instance(self, instance_id: str, *, target_score: float = 100.0):
        """Focus one instance by boosting all related memory-entry scores.

        Input:
            instance_id: noun instance identifier.
            target_score: high score assigned to matching memory entries.
        Output:
            dict: high-level summary of score updates and the refreshed focus state.
        """
        result = self.consciousness.focus_instance(instance_id, target_score=target_score)
        return {
            "command": "focus_instance",
            "instance_id": instance_id,
            "target_score": float(target_score),
            "event_count": result["event_count"],
            "relation_count": result["relation_count"],
            "focus": result["focus"],
            "instance": result["instance"],
        }

    def learn_event(self, instance_id: str, *, target_score: float = 100.0, epochs: int = 10):
        """Train the world model using earlier events to predict the latest event of one instance.

        Input:
            instance_id: noun instance identifier.
            target_score: score floor applied before selecting the target event.
            epochs: number of optimizer steps executed inside this learning call.
        Output:
            dict: high-level summary of the focus step, selected target, input view, and training result.
        """
        result = self.consciousness.train_event_from_instance(
            instance_id,
            target_score=target_score,
            epochs=epochs,
        )
        return {
            "command": "learn_event",
            "instance_id": instance_id,
            "target_score": float(target_score),
            "focus": result["focus"],
            "input_shape": result["input_shape"],
            "input_focus": result["input_focus"],
            "target_event": result["target_event"],
            "train_result": result["train_result"],
        }


    def question(self, *, order_by: str = "time"):
        """Scan current short memory and ask about entries that disagree with long-term memory.

        Input:
            order_by: time or attention ordering used when scanning short memory.
        Output:
            dict: question summary containing relation/action mismatches found in memory.
        """
        relation_entries = self.consciousness.inspect_memory(kind="relation", order_by=order_by)
        event_entries = self.consciousness.inspect_memory(kind="event", order_by=order_by)

        questions = []
        seen_keys = set()

        for entry in relation_entries:
            relation_kind = entry["pair_kind"]
            relation_name = entry["relation_name"]
            source_text = entry.get("source_text")
            target_text = entry.get("target_text")

            if not source_text or not target_text:
                continue

            relation_type = self.consciousness.resolve_relation_type(relation_kind, relation_name)
            if relation_type is None:
                continue

            stored_value = self.consciousness.recall_relation_value(
                relation_kind,
                source_text,
                target_text,
            )
            if stored_value != 0:
                continue

            key = (relation_kind, source_text, relation_name, target_text)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            questions.append(
                {
                    "kind": relation_kind,
                    "prompt": f"I observed '{source_text}' with relation '{relation_name}' -> '{target_text}' in short memory, and long-term memory does not store this relation yet. Is this relation correct?",
                    "source_noun": source_text,
                    "target": target_text,
                    "relation_name": relation_name,
                    "relation_type": relation_type,
                    "existing_memory": [],
                    "memory_entry": entry,
                }
            )

        for entry in event_entries:
            noun_text = entry.get("noun_text")
            action_text = entry.get("action_text")
            if not noun_text or not action_text:
                continue
            recall = self.consciousness.recall_noun_action(noun=noun_text, action=action_text)
            stored_value = 0 if not recall else int(recall[0]["value"])
            if stored_value != 0:
                continue
            key = ("noun_action_relation", noun_text, action_text)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            questions.append(
                {
                    "kind": "noun_action_relation",
                    "prompt": f"I observed the event '{noun_text} -> {action_text}' in short memory, and long-term memory does not store this noun-action relation yet. Should it be remembered?",
                    "source_noun": noun_text,
                    "target": action_text,
                    "relation_name": "noun_action",
                    "relation_type": None,
                    "existing_memory": [],
                    "memory_entry": entry,
                }
            )

        return {
            "command": "question",
            "order_by": order_by,
            "question_count": len(questions),
            "questions": questions,
        }


    def answer_memory_question(
        self,
        question: dict,
        answer_text: str,
        *,
        save: bool = True,
    ):
        """Apply a supervised y/n answer to one memory-derived question.

        Input:
            question: one question item returned by question().
            answer_text: yes/y or no/n.
            save: whether to persist approved long-term map updates.
        Output:
            dict: answer summary plus optional long-term write result.
        """
        normalized_answer = str(answer_text).strip().lower()
        if normalized_answer not in {"y", "yes", "n", "no"}:
            raise ValueError("answer_text must be one of: y, yes, n, no")

        accepted = normalized_answer in {"y", "yes"}
        if not accepted:
            return {
                "command": "answer_memory_question",
                "accepted": False,
                "written": False,
                "kind": question.get("kind"),
                "question": question,
                "result": None,
            }

        kind = question.get("kind")
        if kind == "noun_noun_relation":
            result = self.consciousness.remember_noun_relation(
                question["source_noun"],
                question["target"],
                int(question["relation_type"]),
                save=save,
            )
        elif kind == "adj_noun_relation":
            result = self.consciousness.remember_adj_relation(
                question["source_noun"],
                question["target"],
                int(question["relation_type"]),
                save=save,
            )
        elif kind == "noun_action_relation":
            result = self.consciousness.remember_noun_action(
                question["source_noun"],
                question["target"],
                save=save,
            )
        else:
            raise ValueError("Unsupported memory question kind")

        return {
            "command": "answer_memory_question",
            "accepted": True,
            "written": True,
            "kind": kind,
            "question": question,
            "result": result,
        }

    def sleep(self, *, save: bool = True):
        """Clear short memory and run one joint-average consolidation step on long-term knowledge.

        Input:
            save: whether to persist the post-training long-term state.
        Output:
            dict: short-memory cleanup summary and joint-training summary.
        """
        cleared = self.consciousness.clear_short_memory()
        training = self.consciousness.train_joint_knowledge(save=save)
        return {
            "command": "sleep",
            "cleared_memory": cleared,
            "joint_training": training,
        }

    def question_what(
        self,
        sentence: str,
        *,
        time_position: Optional[int] = None,
        base_score: float = 1.0,
        adjective_relation_types: Optional[Any] = None,
    ):
        """Gate sentence understanding by first checking whether the sentence contains unknown words.

        Input:
            sentence: raw sentence text.
            time_position: optional explicit time step if the sentence can be understood directly.
            base_score: initial attention score used by understand().
            adjective_relation_types: optional adjective->relation mapping for grammar.
        Output:
            dict: either a list of what-questions or a direct understand() result when no unknown words remain.
        """
        grammar = self.consciousness._grammar()
        tokens = grammar.tokenize_sentence(sentence)
        questions = []
        for position, token in enumerate(tokens):
            result = self.consciousness.what(token, position=position, tokens=tokens)
            if result.status == "known":
                continue
            questions.append(
                {
                    "kind": "what",
                    "sentence": sentence,
                    "word": token,
                    "position": int(position),
                    "predicted_pos": result.predicted_pos,
                    "prompt": result.prompt,
                    "candidates": list(result.candidates),
                    "source": result.source,
                    "context_tokens": list(tokens),
                }
            )

        if questions:
            return {
                "command": "question_what",
                "status": "question",
                "sentence": sentence,
                "unknown_count": len(questions),
                "questions": questions,
            }

        understand_result = self.understand(
            sentence,
            time_position=time_position,
            base_score=base_score,
            adjective_relation_types=adjective_relation_types,
        )
        return {
            "command": "question_what",
            "status": "understood",
            "sentence": sentence,
            "unknown_count": 0,
            "questions": [],
            "understand_result": understand_result,
        }

    def answer_what(
        self,
        question: dict,
        answer_info: dict,
        *,
        save: bool = True,
        re_understand: bool = True,
        write_facts: bool = True,
        base_score: float = 1.0,
    ):
        """Apply a user-provided answer to one unknown-word question and optionally re-understand the sentence.

        Input:
            question: one question item returned by question_what().
            answer_info: dict containing at least pos, and optionally relation_name/relation_family.
            save: whether to persist vocabulary updates.
            re_understand: whether to re-run understand(sentence) after registration.
            write_facts: whether supervised facts from the answered sentence should be written directly.
            base_score: memory score used if re-understanding is requested.
        Output:
            dict: registration summary plus optional understanding result.
        """
        token = str(question["word"]).lower()
        sentence = question.get("sentence")
        pos = str(answer_info["pos"]).strip().lower()
        relation_name = answer_info.get("relation_name")
        relation_family = answer_info.get("relation_family")

        updates = []
        updates.append(self.consciousness.register_token_pos(token, pos, save=save))

        if pos == "adj" and relation_name:
            updates.append(
                self.consciousness.register_adjective_relation_hint(
                    token,
                    str(relation_name).lower(),
                    save=save,
                )
            )
        elif pos == "relation":
            updates.append(
                self.consciousness.register_relation_name(
                    relation_name or token,
                    relation_family=(relation_family or "noun_noun_relation"),
                    save=save,
                )
            )

        fact_writes = []
        if write_facts and sentence:
            if pos == "adj" and relation_name:
                grammar = self.consciousness._grammar()
                tokens = grammar.tokenize_sentence(sentence)
                relation_type = self.consciousness.resolve_relation_type(
                    "adj_noun_relation",
                    str(relation_name).lower(),
                )
                if relation_type is not None:
                    for candidate in tokens:
                        if candidate == token:
                            continue
                        fact_writes.append(
                            self.consciousness.remember_adj_relation(
                                candidate,
                                token,
                                relation_type,
                                save=save,
                            )
                        )
            else:
                fact_writes = self._write_supervised_facts_from_sentence(
                    sentence,
                    relation_name_override=(None if relation_name is None else str(relation_name).lower()),
                    relation_token_override=(token if pos == "relation" else None),
                    save=save,
                )

        re_understand_result = None
        if re_understand and sentence:
            re_understand_result = self.understand(sentence, base_score=base_score)

        return {
            "command": "answer_what",
            "question": question,
            "answer_info": dict(answer_info),
            "updates": updates,
            "fact_writes": fact_writes,
            "re_understand": re_understand_result,
        }
__all__ = ["HighLevelCommands"]












