"""High-level command wrappers built on top of the Consciousness API."""

from dataclasses import dataclass, field
from typing import Any, Optional

from .consciousness import Consciousness
from .grammar import parse_sentence


@dataclass
class HighLevelCommands:
    """Task-level commands composed from low-level consciousness interfaces."""

    consciousness: Consciousness = field(default_factory=Consciousness)

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
            "epochs": int(epochs),
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


__all__ = ["HighLevelCommands"]
