"""High-level orchestration for retrieval, memory control, and awareness functions."""

from dataclasses import dataclass, field
import random
from typing import Any, Optional, Sequence

import torch

from knowledge.relation_map import relation_map
from world.shortmemory import ShortMemory
from world.world_model import (
    WorldModel,
    action_dim,
    attention_dim,
    hidden_dim,
    noun_dim,
    value_dim,
)
from .question import AnswerResult, ProposedQuestion, QuestionEngine, TokenWhatResult


@dataclass
class Consciousness:
    ask_confidence_threshold: float = 0.92
    min_ask_confidence_threshold: float = 0.15
    bind_slot_confidence_threshold: float = 0.35
    short_memory_maxlen: int = 100
    short_memory_device: str = "cpu"
    question: QuestionEngine = field(init=False)
    short_memory: ShortMemory = field(init=False)
    world_model: WorldModel = field(init=False)
    world_optimizer: torch.optim.Optimizer = field(init=False)

    def __post_init__(self):
        self.question = QuestionEngine(
            ask_confidence_threshold=self.ask_confidence_threshold,
            min_ask_confidence_threshold=self.min_ask_confidence_threshold,
            bind_slot_confidence_threshold=self.bind_slot_confidence_threshold,
        )
        self.short_memory = ShortMemory(
            maxlen=self.short_memory_maxlen,
            device=self.short_memory_device,
        )
        self.world_model = WorldModel(
            noun_dim=noun_dim,
            action_dim=action_dim,
            attention_dim=attention_dim,
            value_dim=value_dim,
            hidden_dim=hidden_dim,
        )
        self.world_optimizer = self.world_model.build_optimizer()

    def available_functions(self):
        """Return a summary of the high-level command interfaces.

        Input:
            None.
        Output:
            dict[str, str]: interface name -> short description.
        """
        return {
            "observe": "parse sentence input and write relation/event information into short memory",
            "inspect": "inspect current short memory content, focus entry, instances, and relation clones",
            "update": "rebuild noun instances or explicitly update short-memory relation clones",
            "imagine": "predict the next event from short memory and optionally write it back",
            "train": "train the world model from current short memory against a target action",
            "learn_from_sentence": "apply online knowledge learning from a sentence through the language-learning path",
            "recall": "retrieve stored noun_noun and adj_noun relations from long-term memory maps",
            "predict": "run question-based knowledge prediction and optionally learn from feedback",
            "thinking": "sample one uncertainty-driven question to ask",
            "learning": "directly train noun_noun or adj_noun relations",
            "question": "handle yes/no correction for a proposed question",
            "what": "inspect whether a token is already known and ask what part-of-speech it should be",
        }

    def _prediction_status(self, confidence: float) -> str:
        confidence = float(confidence)
        if confidence >= self.ask_confidence_threshold:
            return "confident"
        if confidence <= self.min_ask_confidence_threshold:
            return "unreasonable"
        return "question"

    def _grammar(self):
        import importlib

        return importlib.import_module("prototype.grammar")

    def _sync_world_model_with_actions(self, required_actions: Optional[Sequence[str]] = None):
        import importlib

        action_vocab = importlib.import_module("world.action_vocab")
        required = [action.lower() for action in (required_actions or [])]
        for action in required:
            action_vocab.ensure_action(action)

        current_actions = list(action_vocab.get_action_list())
        model_actions = [action.lower() for action in self.world_model.action_list[1:]]
        if model_actions == current_actions:
            return

        old_model = self.world_model
        new_model = WorldModel(
            noun_dim=noun_dim,
            action_dim=action_dim,
            attention_dim=attention_dim,
            value_dim=value_dim,
            hidden_dim=hidden_dim,
            action_names=current_actions,
        )

        old_action_to_type = {name.lower(): idx for idx, name in enumerate(old_model.action_list) if idx != 0}
        new_action_to_type = {name.lower(): idx for idx, name in enumerate(new_model.action_list) if idx != 0}

        with torch.no_grad():
            for action_name, new_type in new_action_to_type.items():
                old_type = old_action_to_type.get(action_name)
                if old_type is None:
                    continue
                new_model.action_models[new_type - 1].load_state_dict(
                    old_model.action_models[old_type - 1].state_dict()
                )
                new_model.action_embeddings.weight[new_type].copy_(
                    old_model.action_embeddings.weight[old_type]
                )
            new_model.action_embeddings.weight[0].zero_()

        self.world_model = new_model
        self.world_optimizer = self.world_model.build_optimizer()

    def _event_entry_to_dict(self, entry) -> dict:
        return {
            "pair_kind": entry.pair_kind,
            "noun_instance_id": entry.noun_instance_id,
            "action_instance_id": entry.action_instance_id,
            "noun_type": entry.noun_type,
            "action_type": entry.action_type,
            "noun_text": entry.noun_text,
            "action_text": entry.action_text,
            "role": entry.role,
            "score": float(entry.score),
            "time_position": int(entry.time_position),
            "pair_index": int(entry.pair_index),
            "event_index": getattr(entry, "event_index", None),
            "polarity": int(getattr(entry, "polarity", 1)),
            "sentence_label": getattr(entry, "sentence_label", "none"),
        }

    def _relation_entry_to_dict(self, entry) -> dict:
        return {
            "relation_kind": entry.relation_kind,
            "relation_name": entry.relation_name,
            "source_text": entry.source_text,
            "target_text": entry.target_text,
            "source_instance_id": entry.source_instance_id,
            "target_instance_id": entry.target_instance_id,
            "source_type": entry.source_type,
            "target_type": entry.target_type,
            "score": float(entry.score),
            "time_position": int(entry.time_position),
            "pair_index": int(entry.pair_index),
            "polarity": int(getattr(entry, "polarity", 1)),
            "sentence_label": getattr(entry, "sentence_label", "none"),
        }

    def observe(
        self,
        sentence: str,
        *,
        time_position: Optional[int] = None,
        base_score: float = 1.0,
        adjective_relation_types=None,
    ):
        """Parse one sentence and write its relation/event groups into short memory.

        Input:
            sentence: raw sentence text.
            time_position: optional explicit time step.
            base_score: initial attention score.
            adjective_relation_types: optional adjective->relation mapping.
        Output:
            dict: sentence type, structure, tokens, counts, and created states.
        """
        grammar = self._grammar()
        before_event_count = len(self.short_memory.short_memory_event)
        before_relation_count = len(self.short_memory.short_memory_relation)
        before_reward_count = len(self.short_memory.short_memory_reward)
        before_surprise_count = len(self.short_memory.short_memory_surprise)
        parsed = grammar.parse_sentence(
            sentence,
            adjective_relation_types=adjective_relation_types,
            short_memory=self.short_memory,
        )
        sentence_label = self.short_memory.next_sentence_label()
        self._sync_world_model_with_actions([action_tuple.action for action_tuple in parsed.action_tuples])
        states = grammar.append_sentence_to_short_memory(
            sentence=sentence,
            short_memory=self.short_memory,
            world_model=self.world_model,
            time_position=time_position,
            base_score=base_score,
            adjective_relation_types=adjective_relation_types,
            sentence_label=sentence_label,
        )
        return {
            "sentence": sentence,
            "sentence_label": sentence_label,
            "sentence_type": parsed.sentence_type,
            "structure": tuple(parsed.structure),
            "tokens": [token.text if hasattr(token, "text") else str(token) for token in parsed.tokens],
            "action_count": len(parsed.action_tuples),
            "relation_count": len(parsed.relation_tuples),
            "reward_count": len(parsed.reward_tuples),
            "surprise_count": len(parsed.surprise_tuples),
            "event_entries_added": len(self.short_memory.short_memory_event) - before_event_count,
            "relation_entries_added": len(self.short_memory.short_memory_relation) - before_relation_count,
            "reward_entries_added": len(self.short_memory.short_memory_reward) - before_reward_count,
            "surprise_entries_added": len(self.short_memory.short_memory_surprise) - before_surprise_count,
            "states": states,
        }

    def observe_many(
        self,
        sentences: Sequence[str],
        *,
        start_time_position: Optional[int] = None,
        base_score: float = 1.0,
        adjective_relation_types: Optional[Sequence[Any]] = None,
    ):
        """Parse and store multiple sentences in sequence.

        Input:
            sentences: ordered sentence list.
            start_time_position: optional first time step.
            base_score: initial attention score.
            adjective_relation_types: per-sentence overrides.
        Output:
            dict: sentence count, start time, memory sizes, and collected states.
        """
        grammar = self._grammar()
        required_actions = []
        for sentence, overrides in zip(sentences, adjective_relation_types or [None] * len(sentences)):
            parsed = grammar.parse_sentence(
                sentence,
                adjective_relation_types=overrides,
                short_memory=self.short_memory,
            )
            required_actions.extend(action_tuple.action for action_tuple in parsed.action_tuples)
        self._sync_world_model_with_actions(required_actions)

        all_states = grammar.sentences_to_short_memory(
            sentences=sentences,
            short_memory=self.short_memory,
            world_model=self.world_model,
            start_time_position=start_time_position,
            base_score=base_score,
            adjective_relation_types=adjective_relation_types,
        )
        return {
            "sentence_count": len(sentences),
            "start_time_position": None if start_time_position is None else int(start_time_position),
            "event_count": len(self.short_memory.short_memory_event),
            "relation_count": len(self.short_memory.short_memory_relation),
            "states": all_states,
        }

    def inspect_memory(self, kind: str = "all", order_by: str = "time"):
        """Inspect current short-memory content.

        Input:
            kind: all/event/relation.
            order_by: time/attention.
        Output:
            dict or list: requested memory view.
        """
        if kind == "all":
            return self.short_memory.get_content_view(order_by=order_by)
        if kind == "event":
            return self.short_memory.get_event_content_view(order_by=order_by)
        if kind == "relation":
            return self.short_memory.get_relation_content_view(order_by=order_by)
        if kind == "reward":
            return self.short_memory.get_reward_content_view(order_by=order_by)
        if kind == "surprise":
            return self.short_memory.get_surprise_content_view(order_by=order_by)
        raise ValueError("kind must be one of: all/event/relation/reward/surprise")

    def inspect_focus(self, steps=None):
        """Return the current focus event used by the world model.

        Input:
            steps: optional event window size.
            epochs: number of optimizer steps executed inside this training call.
        Output:
            dict | None: focused event summary.
        """
        focus = self.short_memory.get_focus_entry(steps=steps)
        if focus is None:
            return None
        return self._event_entry_to_dict(focus)

    def inspect_instance(self, instance_id: str):
        """Inspect one noun instance inside short memory.

        Input:
            instance_id: noun instance identifier.
        Output:
            dict: existence flag, embedding norm, and related entries.
        """
        noun_embedding = self.short_memory.get_noun_embedding(instance_id)
        metadata = self.short_memory.get_noun_instance_metadata(instance_id)
        related_events = [
            self._event_entry_to_dict(entry)
            for entry in self.short_memory.short_memory_event
            if entry.noun_instance_id == instance_id
        ]
        related_relations = [
            self._relation_entry_to_dict(entry)
            for entry in self.short_memory.short_memory_relation
            if entry.source_instance_id == instance_id or entry.target_instance_id == instance_id
        ]
        return {
            "instance_id": instance_id,
            "exists": noun_embedding is not None,
            "embedding_norm": None if noun_embedding is None else float(noun_embedding.norm().item()),
            "metadata": metadata,
            "event_count": len(related_events),
            "relation_count": len(related_relations),
            "events": related_events,
            "relations": related_relations,
        }

    def inspect_relation_clone(self, relation_kind: str, relation_name: str):
        """Inspect one short-memory relation clone.

        Input:
            relation_kind: relation family.
            relation_name: relation label.
        Output:
            dict: clone existence, shape/norm, and related entries.
        """
        clone = self.short_memory.get_relation_clone(relation_kind, relation_name)
        related_entries = [
            self._relation_entry_to_dict(entry)
            for entry in self.short_memory.short_memory_relation
            if entry.relation_kind == relation_kind and entry.relation_name == relation_name
        ]
        return {
            "relation_kind": relation_kind,
            "relation_name": relation_name,
            "exists": clone is not None,
            "shape": None if clone is None else tuple(int(dim) for dim in clone.shape),
            "norm": None if clone is None else float(clone.norm().item()),
            "entry_count": len(related_entries),
            "entries": related_entries,
        }

    def rebuild_instance(
        self,
        instance_id: str,
        step_scale: Optional[float] = None,
        *,
        reward_model=None,
        reward_encoder=None,
    ):
        """Rebuild one noun instance embedding from short-memory relation/reward evidence.

        Input:
            instance_id: noun instance identifier.
            step_scale: optional scaling factor.
            reward_model: optional SubjectEventRewardNet for including reward losses.
            reward_encoder: optional RewardEncoder for including reward losses.
        Output:
            dict: whether an update happened and the resulting norm.
        """
        if reward_model is not None and reward_encoder is not None:
            embedding = self.short_memory.rebuild_instance_embedding_from_relation_and_reward(
                instance_id,
                reward_model=reward_model,
                reward_encoder=reward_encoder,
                step_scale=step_scale,
            )
        else:
            embedding = self.short_memory.rebuild_instance_embedding(instance_id, step_scale=step_scale)
        return {
            "instance_id": instance_id,
            "updated": embedding is not None,
            "embedding_norm": None if embedding is None else float(embedding.norm().item()),
        }

    def set_instance_profile(
        self,
        instance_id: str,
        *,
        entity_kind: Optional[str] = None,
        gender: Optional[str] = None,
    ):
        """Update lightweight metadata for one noun instance.

        Input:
            instance_id: noun instance identifier.
            entity_kind: one of person, object, unknown.
            gender: one of male, female, unknown.
        Output:
            dict: updated instance metadata.
        """
        metadata = self.short_memory.update_noun_instance_metadata(
            instance_id,
            entity_kind=entity_kind,
            gender=gender,
        )
        return {
            "instance_id": instance_id,
            "metadata": metadata,
        }

    def focus_instance(self, instance_id: str, *, target_score: float = 100.0):
        """Raise the attention score of all memory entries involving one instance.

        Input:
            instance_id: noun instance identifier.
            target_score: score floor applied to matching event and relation entries.
        Output:
            dict: updated counts plus refreshed focus and instance summary.
        """
        updated = self.short_memory.focus_instance(instance_id, target_score=target_score)
        return {
            **updated,
            "focus": self.inspect_focus(),
            "instance": self.inspect_instance(instance_id),
        }

    def update_relation_clone(
        self,
        relation_kind: str,
        relation_name: str,
        *,
        step_scale: Optional[float] = None,
    ):
        """Explicitly update one short-memory relation clone.

        Input:
            relation_kind: relation family.
            relation_name: relation label.
            step_scale: optional scaling factor.
        Output:
            dict: whether the clone was updated and its norm.
        """
        relation_clone = self.short_memory.update_relation_clone(
            relation_kind,
            relation_name,
            step_scale=step_scale,
        )
        return {
            "relation_kind": relation_kind,
            "relation_name": relation_name,
            "updated": relation_clone is not None,
            "norm": None if relation_clone is None else float(relation_clone.norm().item()),
        }

    def update_all_relation_clones(
        self,
        relation_kind: Optional[str] = None,
        *,
        step_scale: Optional[float] = None,
    ):
        """Explicitly update every short-memory relation clone.

        Input:
            relation_kind: optional relation family filter.
            step_scale: optional scaling factor.
        Output:
            list[dict]: one update summary per relation clone.
        """
        return self.short_memory.update_all_relation_clones(
            relation_kind=relation_kind,
            step_scale=step_scale,
        )

    def sync_relation_clone_to_knowledge(
        self,
        relation_kind: str,
        relation_name: str,
        *,
        save: bool = True,
    ):
        """Copy one learned short-memory relation clone back into long-term knowledge.

        Input:
            relation_kind: relation family, such as noun_noun_relation or adj_noun_relation.
            relation_name: relation label to sync.
            save: whether to persist updated long-term knowledge to disk.
        Output:
            dict: sync status, target family, and resulting clone norm.
        """
        clone = self.short_memory.get_relation_clone(relation_kind, relation_name)
        if clone is None:
            return {
                "relation_kind": relation_kind,
                "relation_name": relation_name,
                "synced": False,
                "reason": "clone_not_found",
            }

        rm, arm, kt = self.question._ctx()
        with torch.no_grad():
            if relation_kind == "noun_noun_relation":
                if relation_name not in rm.relation_list:
                    return {
                        "relation_kind": relation_kind,
                        "relation_name": relation_name,
                        "synced": False,
                        "reason": "relation_not_found",
                    }
                relation_index = rm.relation_list.index(relation_name)
                kt.knowledge_map_one.relations[relation_index].weight.copy_(clone)
                target_model = "knowledge_map"
            elif relation_kind == "adj_noun_relation":
                if relation_name not in arm.adj_relation_list:
                    return {
                        "relation_kind": relation_kind,
                        "relation_name": relation_name,
                        "synced": False,
                        "reason": "relation_not_found",
                    }
                relation_index = arm.adj_relation_list.index(relation_name)
                kt.adj_map_one.relations[relation_index].weight.copy_(clone)
                target_model = "adj_map"
            else:
                raise ValueError("relation_kind must be 'noun_noun_relation' or 'adj_noun_relation'")

        if save:
            rm.save_relation_data()
            arm.save_adj_relation_data()
            torch.save(kt.knowledge_map_one.state_dict(), kt.MODEL_PATH)
            torch.save(kt.adj_map_one.state_dict(), kt.ADJ_MODEL_PATH)

        return {
            "relation_kind": relation_kind,
            "relation_name": relation_name,
            "synced": True,
            "saved": bool(save),
            "target_model": target_model,
            "norm": float(clone.norm().item()),
        }

    def clear_short_memory(self):
        """Clear all short-memory content and instance/relation caches.

        Input:
            None.
        Output:
            dict: memory sizes before and after clearing.
        """
        before = {
            "event_count": len(self.short_memory.short_memory_event),
            "relation_count": len(self.short_memory.short_memory_relation),
            "noun_instance_count": len(self.short_memory.noun_instance_memory),
            "action_instance_count": len(self.short_memory.action_instance_memory),
            "noun_relation_clone_count": len(self.short_memory.noun_relation_memory),
            "adj_relation_clone_count": len(self.short_memory.adj_relation_memory),
        }
        self.short_memory.clear()
        after = {
            "event_count": len(self.short_memory.short_memory_event),
            "relation_count": len(self.short_memory.short_memory_relation),
            "noun_instance_count": len(self.short_memory.noun_instance_memory),
            "action_instance_count": len(self.short_memory.action_instance_memory),
            "noun_relation_clone_count": len(self.short_memory.noun_relation_memory),
            "adj_relation_clone_count": len(self.short_memory.adj_relation_memory),
        }
        return {
            "before": before,
            "after": after,
        }

    def train_joint_knowledge(self, *, save: bool = True):
        """Run one joint-average training step for knowledge_map and adj_map.

        Input:
            save: whether to persist the updated long-term models and learning-rate maps.
        Output:
            dict: training summary.
        """
        import importlib

        training = importlib.import_module("knowledge.training")
        training._load_training_state()
        relation_map, _, _, lr_per_embedding, lr_relation = training.rm.load_relation_data()
        adj_relation_map, _, _, lr_per_adjective, lr_adj_relation = training.arm.load_adj_relation_data()
        if adj_relation_map is False:
            raise FileNotFoundError(
                "adj_relation_data.npz not found. Generate adjective relation data before joint training."
            )

        training.train_joint_average(
            knowledge_map_one=training.knowledge_map_one,
            adj_map_one=training.adj_map_one,
            relation_map=relation_map,
            adj_relation_map=adj_relation_map,
            lr_per_embedding=lr_per_embedding,
            lr_relation=lr_relation,
            lr_per_adjective=lr_per_adjective,
            lr_adj_relation=lr_adj_relation,
        )
        if save:
            training._save_training_state()

        return {
            "trained": True,
            "saved": bool(save),
            "knowledge_embedding_norm": float(training.knowledge_map_one.embedding.weight.norm().item()),
            "adj_embedding_norm": float(training.adj_map_one.adjective_embedding.weight.norm().item()),
        }

    def build_world_input(self, steps=None):
        """Build the current world-model input tensor from short-memory events.

        Input:
            steps: optional event window size.
        Output:
            dict: tensor shape and the actual input tensor.
        """
        world_input = self.short_memory.build_world_model_event_input(steps=steps)
        return {
            "shape": tuple(int(dim) for dim in world_input.shape),
            "tensor": world_input,
        }

    def _build_training_memory_before_event(self, target_entry):
        training_memory = ShortMemory(
            maxlen=self.short_memory.maxlen,
            device=self.short_memory.device,
            state_dim=self.short_memory.state_dim,
            relation_update_mode=self.short_memory.relation_update_mode,
            relation_update_frequency=self.short_memory.relation_update_frequency,
            relation_step_scale=self.short_memory.relation_step_scale,
            relation_clone_update_mode=self.short_memory.relation_clone_update_mode,
            relation_clone_update_frequency=self.short_memory.relation_clone_update_frequency,
            relation_clone_step_scale=self.short_memory.relation_clone_step_scale,
        )
        selected_entries = [
            entry
            for entry in self.short_memory.short_memory_event
            if entry.time_position < target_entry.time_position
        ]
        for entry in selected_entries:
            noun_embedding = self.short_memory.get_noun_embedding(entry.noun_instance_id)
            action_embedding = self.short_memory.get_action_embedding(entry.action_instance_id)
            if noun_embedding is None or action_embedding is None:
                continue
            training_memory.append_event(
                noun_embedding=noun_embedding,
                action_embedding=action_embedding,
                score=entry.score,
                noun_type=entry.noun_type,
                action_type=entry.action_type,
                time_position=entry.time_position,
                pair_index=entry.pair_index,
                noun_text=entry.noun_text,
                action_text=entry.action_text,
                noun_instance_id=entry.noun_instance_id,
                action_instance_id=entry.action_instance_id,
                role=entry.role,
                adjectives=list(entry.adjectives),
                pair_kind=entry.pair_kind,
                polarity=entry.polarity,
                accept_label=entry.accept_label,
                question_label=entry.question_label,
                sentence_label=getattr(entry, "sentence_label", "none"),
                info_pair=dict(entry.info_pair),
            )
        return training_memory

    def train_event_from_instance(
        self,
        instance_id: str,
        *,
        target_score: float = 100.0,
        epochs: int = 10,
    ):
        """Train the world model to predict the latest event of one focused instance.

        Input:
            instance_id: noun instance identifier.
            target_score: score floor used to focus the instance before selecting the target event.
            epochs: number of optimizer steps executed inside this training call.
        Output:
            dict: focus summary, selected target event, training input shape, and training result.
        """
        focus_result = self.focus_instance(instance_id, target_score=target_score)
        target_candidates = [
            entry
            for entry in self.short_memory.short_memory_event
            if entry.noun_instance_id == instance_id
        ]
        if not target_candidates:
            raise ValueError(f"instance '{instance_id}' has no event entries")

        target_time = max(entry.time_position for entry in target_candidates)
        target_entry = [
            entry for entry in self.short_memory.short_memory_event
            if entry.noun_instance_id == instance_id and entry.time_position == target_time
        ][-1]

        training_memory = self._build_training_memory_before_event(target_entry)
        if len(training_memory.short_memory_event) == 0:
            raise ValueError("cannot train event prediction without earlier time-step events")

        focus_entry = training_memory.get_focus_entry()
        if focus_entry is None or focus_entry.action_type is None:
            raise ValueError("training input does not contain a valid focus action")
        if target_entry.action_type is None:
            raise ValueError("target event has no action_type")

        train_result = self.world_model.training_step_next_event(
            short_memory=training_memory,
            action_type=int(focus_entry.action_type),
            target_action_type=int(target_entry.action_type),
            optimizer=self.world_optimizer,
            steps=None,
            epochs=epochs,
        )
        world_input = training_memory.build_world_model_event_input()
        return {
            "instance_id": instance_id,
            "focus": focus_result,
            "input_shape": tuple(int(dim) for dim in world_input.shape),
            "input_focus": self._event_entry_to_dict(focus_entry),
            "target_event": self._event_entry_to_dict(target_entry),
            "train_result": train_result,
        }

    def predict_next_event(self, action_type: int, *, steps=None, score: float = 0.5):
        """Predict the next event from current short-memory events.

        Input:
            action_type: action selector used by the world model.
            steps: optional event window size.
            score: score assigned to the predicted event.
        Output:
            dict: predicted action, focus info, and a structured predicted event.
        """
        result = self.world_model.predict_next_event(
            self.short_memory,
            action_type=action_type,
            steps=steps,
            score=score,
        )
        result["predicted_event"] = result["predicted_event_dict"]
        return result

    def train_next_event(
        self,
        action_type: int,
        *,
        target_action_type: Optional[int] = None,
        target_action_embedding=None,
        steps=None,
        epochs: int = 10,
    ):
        """Run one world-model training step from current short memory.

        Input:
            action_type: action selector used by the world model.
            target_action_type: target discrete action type.
            target_action_embedding: optional direct target embedding.
            steps: optional event window size.
        Output:
            dict: training loss, predicted type, and focus metadata.
        """
        return self.world_model.training_step_next_event(
            short_memory=self.short_memory,
            action_type=action_type,
            target_action_type=target_action_type,
            target_action_embedding=target_action_embedding,
            optimizer=self.world_optimizer,
            steps=steps,
            epochs=epochs,
        )

    def evaluate_next_event(
        self,
        action_type: int,
        *,
        target_action_type: int,
        steps=None,
        score: float = 0.5,
    ):
        """Evaluate next-event prediction against a target action without training.

        Input:
            action_type: action selector used by the world model.
            target_action_type: target discrete action type.
            steps: optional event window size.
            score: score assigned to the predicted event.
        Output:
            dict: loss, predicted action type, ranking scores, and predicted event summary.
        """
        with torch.no_grad():
            prediction = self.world_model.predict_next_event(
                self.short_memory,
                action_type=action_type,
                steps=steps,
                score=score,
            )
            pred_action = prediction["pred_action"]
            pred_action_type = prediction["pred_action_type"]
            target_embedding = self.world_model.get_action_embedding(target_action_type).detach()
            loss = torch.nn.functional.mse_loss(pred_action, target_embedding).item()
            top_indices, top_scores = self.world_model.infer_action_type(
                pred_action,
                top_k=self.world_model.model_count,
            )

        return {
            "loss": float(loss),
            "pred_action_type": int(pred_action_type),
            "top_indices": top_indices.tolist(),
            "top_scores": [round(float(score.item()), 4) for score in top_scores],
            "predicted_event": prediction["predicted_event_dict"],
        }

    def append_predicted_event(self, predicted_event, **kwargs):
        """Write a predicted event back into short memory.

        Input:
            predicted_event: dict or PredictedEvent object.
            **kwargs: optional writeback fields.
        Output:
            dict: stored event data after writeback.
        """
        stored = self.world_model.append_predicted_event(
            self.short_memory,
            predicted_event,
            **kwargs,
        )
        return stored.as_dict()

    def rollout(self, action_type: int, *, steps: int = 1, score: float = 0.5):
        """Run multi-step autoregressive next-event prediction.

        Input:
            action_type: initial action selector.
            steps: rollout step count.
            score: score assigned to each predicted event.
        Output:
            list[dict]: predicted event summaries in rollout order.
        """
        outputs = []
        current_action_type = int(action_type)
        for _ in range(int(steps)):
            result = self.world_model.autoregressive_step(
                short_memory=self.short_memory,
                action_type=current_action_type,
                score=score,
            )
            outputs.append(result["predicted_event_dict"])
            current_action_type = result["pred_action_type"]
        return outputs

    def learn_from_sentence(
        self,
        sentence: str,
        *,
        noun_relation_type: Optional[int] = None,
        adjective_relation_types=None,
        infer_missing: bool = False,
        save: bool = True,
    ):
        """Run the long-term language-learning path on one sentence.

        Input:
            sentence: raw sentence text.
            noun_relation_type: optional noun-noun relation override.
            adjective_relation_types: optional adjective relation mapping.
            infer_missing: whether grammar may infer missing labels.
            save: whether to persist learned knowledge.
        Output:
            dict: original sentence, extracted samples, and training results.
        """
        import importlib

        training = importlib.import_module("knowledge.training")
        samples, results = training.train_sentence_online(
            sentence,
            noun_relation_type=noun_relation_type,
            adjective_relation_types=adjective_relation_types,
            infer_missing=infer_missing,
            save=save,
        )
        return {
            "sentence": sentence,
            "samples": samples,
            "results": results,
        }

    def inspect_vocab(self):
        """Inspect the currently defined long-term vocabulary.

        Input:
            None.
        Output:
            dict: noun list, adjective list, and relation list.
        """
        rm, arm, _ = self.question._ctx()
        return {
            "noun_list": [noun for noun in rm.noun_list if not noun.startswith("noun_")],
            "adj_list": [adj for adj in arm.adjective_list if not adj.startswith("adj_")],
            "relation_list": list(rm.relation_list),
        }

    def resolve_relation_type(self, relation_kind: str, relation_name: str):
        """Resolve a relation name into its discrete relation type.

        Input:
            relation_kind: relation family, such as noun_noun_relation or adj_noun_relation.
            relation_name: relation label.
        Output:
            int | None: 1-based relation type if found, otherwise None.
        """
        rm, arm, _ = self.question._ctx()
        relation_name = str(relation_name).lower()
        if relation_kind == "noun_noun_relation":
            if relation_name not in rm.relation_list:
                return None
            return int(rm.relation_list.index(relation_name) + 1)
        if relation_kind == "adj_noun_relation":
            if relation_name not in arm.adj_relation_list:
                return None
            return int(arm.adj_relation_list.index(relation_name) + 1)
        raise ValueError("relation_kind must be one of: noun_noun_relation, adj_noun_relation")

    def recall_noun_action(
        self,
        noun: Optional[str] = None,
        action: Optional[str] = None,
    ):
        """Recall stored noun-action permissions from long-term noun_action memory.

        Input:
            noun: optional noun filter.
            action: optional action filter.
        Output:
            list[dict]: matching noun-action memory entries.
        """
        import importlib

        noun_action_map = importlib.import_module("knowledge.noun_action_map")
        noun_key = None if noun is None else str(noun).lower()
        action_key = None if action is None else str(action).lower()

        if noun_key is not None and action_key is not None:
            allowed = noun_action_map.can_noun_do_action(noun_key, action_key)
            return [
                {
                    "noun": noun_key,
                    "action": action_key,
                    "value": 1 if allowed else 0,
                }
            ]

        if noun_key is not None:
            results = noun_action_map.get_actions_for_noun(noun_key, only_allowed=True)
        elif action_key is not None:
            results = noun_action_map.get_nouns_for_action(action_key, only_allowed=True)
        else:
            results = []
            for noun_name in self.inspect_vocab()["noun_list"]:
                results.extend(noun_action_map.get_actions_for_noun(noun_name, only_allowed=True))

        normalized = []
        for item in results:
            current_noun = str(item.get("noun", "")).lower()
            current_action = str(item.get("action", "")).lower()
            if noun_key is not None and current_noun != noun_key:
                continue
            if action_key is not None and current_action != action_key:
                continue
            normalized.append(
                {
                    "noun": current_noun,
                    "action": current_action,
                    "value": int(item.get("value", 0)),
                }
            )
        return normalized

    def recall_relation_value(self, relation_kind: str, source_noun: str, target_value: str):
        """Look up the exact stored value for one long-term relation slot.

        Input:
            relation_kind: noun_noun_relation or adj_noun_relation.
            source_noun: source noun text.
            target_value: target noun or adjective text.
        Output:
            int | None: stored relation type, or 0 when the slot is empty, or None if unknown.
        """
        rm, arm, _ = self.question._ctx()
        source_key = str(source_noun).lower()
        target_key = str(target_value).lower()

        if source_key not in rm.noun_list:
            return None
        source_idx = rm.noun_list.index(source_key)

        if relation_kind == "noun_noun_relation":
            if target_key not in rm.noun_list:
                return None
            target_idx = rm.noun_list.index(target_key)
            return int(rm.relation_map[source_idx, target_idx])

        if relation_kind == "adj_noun_relation":
            if target_key not in arm.adjective_list:
                return None
            target_idx = arm.adjective_list.index(target_key)
            return int(arm.adj_relation_map[source_idx, target_idx])

        raise ValueError("relation_kind must be one of: noun_noun_relation, adj_noun_relation")

    def remember_noun_relation(
        self,
        source_noun: str,
        target_noun: str,
        relation_type: int,
        *,
        save: bool = True,
    ):
        """Write one supervised noun-noun relation into long-term relation_map.

        Input:
            source_noun: source noun text.
            target_noun: target noun text.
            relation_type: 1-based relation type.
            save: whether to persist the map update.
        Output:
            dict: write summary including created flag and stored relation type.
        """
        import importlib

        rm = importlib.import_module("knowledge.relation_map")
        rm.load_relation_data()
        created, source_idx, target_idx, stored_relation_type = rm.add_relation_by_type(
            source_noun,
            target_noun,
            relation_type,
        )
        if save:
            rm.save_relation_data()
        return {
            "kind": "noun_noun_relation",
            "source_noun": str(source_noun).lower(),
            "target_noun": str(target_noun).lower(),
            "relation_type": int(stored_relation_type),
            "created": bool(created),
            "source_idx": int(source_idx),
            "target_idx": int(target_idx),
            "saved": bool(save),
        }

    def remember_adj_relation(
        self,
        noun: str,
        adjective: str,
        relation_type: int,
        *,
        save: bool = True,
    ):
        """Write one supervised adj-noun relation into long-term adj_relation_map.

        Input:
            noun: noun text.
            adjective: adjective text.
            relation_type: 1-based adjective relation type.
            save: whether to persist the map update.
        Output:
            dict: write summary including created flag and stored relation type.
        """
        import importlib

        arm = importlib.import_module("knowledge.adj_relation_map")
        arm.load_adj_relation_data()
        created, noun_idx, adjective_idx, stored_relation_type = arm.add_adj_relation_by_type(
            noun,
            adjective,
            relation_type,
        )
        if save:
            arm.save_adj_relation_data()
        return {
            "kind": "adj_noun_relation",
            "noun": str(noun).lower(),
            "adjective": str(adjective).lower(),
            "relation_type": int(stored_relation_type),
            "created": bool(created),
            "noun_idx": int(noun_idx),
            "adjective_idx": int(adjective_idx),
            "saved": bool(save),
        }

    def remember_noun_action(
        self,
        noun: str,
        action: str,
        *,
        save: bool = True,
    ):
        """Write one supervised noun-action relation into long-term noun_action_map.

        Input:
            noun: noun text.
            action: action text.
            save: whether to persist the map update.
        Output:
            dict: write summary including indices and stored value.
        """
        import importlib

        noun_action_map = importlib.import_module("knowledge.noun_action_map")
        noun_action_map.load_noun_action_data()
        noun_idx, action_idx, value = noun_action_map.add_noun_action(noun, action)
        if save:
            noun_action_map.save_noun_action_data()
        return {
            "kind": "noun_action_relation",
            "noun": str(noun).lower(),
            "action": str(action).lower(),
            "noun_idx": int(noun_idx),
            "action_idx": int(action_idx),
            "value": int(value),
            "saved": bool(save),
        }

    def re_predict_question(self, question: ProposedQuestion):
        """Re-run the appropriate predictor for an existing proposed question.

        Input:
            question: a previously generated ProposedQuestion.
        Output:
            ProposedQuestion: refreshed prediction after learning or state changes.
        """
        if question.kind == "adj_noun":
            return self.question.predict_adjective(question.source_noun, question.relation_type)
        if question.question_target == "noun":
            return self.question.predict_noun_from_relation(question.source_noun, question.relation_type)
        return self.question.predict_relation_between_nouns(question.source_noun, question.target_noun)

    def recall(
        self,
        noun: Optional[str] = None,
        adjective: Optional[str] = None,
        relation_type: Optional[int] = None,
    ):
        """Recall stored long-term noun-noun and adj-noun relations.

        Input:
            noun: optional noun filter.
            adjective: optional adjective filter.
            relation_type: optional relation-type filter.
        Output:
            dict: lists of matching noun_noun and adj_noun relations.
        """
        rm, arm, _ = self.question._ctx()
        noun_key = None if noun is None else noun.lower()
        adjective_key = None if adjective is None else adjective.lower()
        relation_type = None if relation_type is None else int(relation_type)

        noun_relations = []
        if noun_key is not None and noun_key in rm.noun_list:
            noun_idx = rm.noun_list.index(noun_key)
            for target_idx, raw_relation_type in enumerate(rm.relation_map[noun_idx]):
                rt = int(raw_relation_type)
                if rt == 0 or (relation_type is not None and rt != relation_type):
                    continue
                relation_name = rm.relation_list[rt - 1] if rt - 1 < len(rm.relation_list) else f"relation_{rt}"
                target_noun = rm.noun_list[target_idx] if target_idx < len(rm.noun_list) else f"noun_{target_idx}"
                noun_relations.append(
                    {
                        "source_noun": noun_key,
                        "target_noun": target_noun,
                        "relation_type": rt,
                        "relation_name": relation_name,
                    }
                )
            for source_idx, row in enumerate(rm.relation_map):
                rt = int(row[noun_idx])
                if rt == 0 or (relation_type is not None and rt != relation_type):
                    continue
                relation_name = rm.relation_list[rt - 1] if rt - 1 < len(rm.relation_list) else f"relation_{rt}"
                source_noun = rm.noun_list[source_idx] if source_idx < len(rm.noun_list) else f"noun_{source_idx}"
                noun_relations.append(
                    {
                        "source_noun": source_noun,
                        "target_noun": noun_key,
                        "relation_type": rt,
                        "relation_name": relation_name,
                    }
                )

        adj_relations = []
        noun_idx = rm.noun_list.index(noun_key) if noun_key is not None and noun_key in rm.noun_list else None
        adjective_idx = (
            arm.adjective_list.index(adjective_key)
            if adjective_key is not None and adjective_key in arm.adjective_list
            else None
        )

        if noun_idx is not None:
            for adj_idx, raw_relation_type in enumerate(arm.adj_relation_map[noun_idx]):
                rt = int(raw_relation_type)
                if rt == 0 or (relation_type is not None and rt != relation_type):
                    continue
                adjective_name = arm.adjective_list[adj_idx] if adj_idx < len(arm.adjective_list) else f"adj_{adj_idx}"
                if adjective_key is not None and adjective_name != adjective_key:
                    continue
                relation_name = arm.adj_relation_list[rt - 1] if rt - 1 < len(arm.adj_relation_list) else f"relation_{rt}"
                adj_relations.append(
                    {
                        "noun": noun_key,
                        "adjective": adjective_name,
                        "relation_type": rt,
                        "relation_name": relation_name,
                    }
                )
        elif adjective_idx is not None:
            for source_idx in range(arm.adj_relation_map.shape[0]):
                rt = int(arm.adj_relation_map[source_idx, adjective_idx])
                if rt == 0 or (relation_type is not None and rt != relation_type):
                    continue
                source_noun = rm.noun_list[source_idx] if source_idx < len(rm.noun_list) else f"noun_{source_idx}"
                relation_name = arm.adj_relation_list[rt - 1] if rt - 1 < len(arm.adj_relation_list) else f"relation_{rt}"
                adj_relations.append(
                    {
                        "noun": source_noun,
                        "adjective": adjective_key,
                        "relation_type": rt,
                        "relation_name": relation_name,
                    }
                )

        return {
            "noun_noun": noun_relations,
            "adj_noun": adj_relations,
        }

    def predict(
        self,
        kind: str = "sample",
        *,
        noun: Optional[str] = None,
        relation_type: Optional[int] = None,
        source_noun: Optional[str] = None,
        target_noun: Optional[str] = None,
        answer_text: Optional[str] = None,
        corrected_target: Optional[str] = None,
        corrected_relation_type: Optional[int] = None,
        rng: Optional[random.Random] = None,
        save: bool = True,
    ):
        """Run high-level knowledge prediction and optional feedback learning.

        Input:
            kind: sample, adj_noun, noun_noun, or relation.
            noun/source_noun/target_noun/relation_type: prediction arguments by kind.
            answer_text/corrected_target/corrected_relation_type: optional feedback payload.
            rng: optional random source.
            save: whether to persist feedback learning.
        Output:
            dict: status plus prediction/question/result payload.
        """
        if kind == "sample":
            question = self.think(rng=rng)
            if question is None:
                return {"status": "no_question", "question": None}
            if answer_text is None:
                return {"status": "question", "question": question}
            return {
                "status": "learned",
                "question": question,
                "result": self.answer_question(
                    question,
                    answer_text,
                    corrected_target=corrected_target,
                    corrected_relation_type=corrected_relation_type,
                    save=save,
                ),
            }

        if kind == "adj_noun":
            if noun is None or relation_type is None:
                raise ValueError("adj_noun prediction requires noun and relation_type")
            prediction = self.question.predict_adjective(noun, relation_type)
        elif kind == "noun_noun":
            if noun is None or relation_type is None:
                raise ValueError("noun_noun prediction requires noun and relation_type")
            prediction = self.question.predict_noun_from_relation(noun, relation_type)
        elif kind == "relation":
            if source_noun is None or target_noun is None:
                raise ValueError("relation prediction requires source_noun and target_noun")
            prediction = self.question.predict_relation_between_nouns(source_noun, target_noun)
        else:
            raise ValueError("kind must be one of: sample, adj_noun, noun_noun, relation")

        status = self._prediction_status(prediction.confidence)
        if status != "question":
            return {
                "status": status,
                "question": None,
                "prediction": prediction,
            }
        if answer_text is None:
            return {
                "status": "question",
                "question": prediction,
            }
        return {
            "status": "learned",
            "question": prediction,
            "result": self.answer_question(
                prediction,
                answer_text,
                corrected_target=corrected_target,
                corrected_relation_type=corrected_relation_type,
                save=save,
            ),
        }

    def think(self, rng: Optional[random.Random] = None) -> Optional[ProposedQuestion]:
        """Sample one uncertainty-driven question from the knowledge layer.

        Input:
            rng: optional random generator.
        Output:
            ProposedQuestion | None: one question to ask, or None if nothing suitable was found.
        """
        return self.question.sample_question(rng or random.Random())

    def what(
        self,
        token: str,
        *,
        position: Optional[int] = None,
        tokens: Optional[list[str]] = None,
    ) -> TokenWhatResult:
        """Inspect whether a token is known and what part-of-speech it likely is.

        Input:
            token: token text.
            position: optional token position in the sentence.
            tokens: optional full token sequence for context.
        Output:
            TokenWhatResult: known/unknown status, guessed POS, prompt, and candidates.
        """
        return self.question.what_is_token(token, position=position, tokens=tokens)

    def register_token_pos(self, token: str, pos: str, *, save: bool = True):
        """Register one token as a noun, adjective, action, or relation token.

        Input:
            token: token text.
            pos: noun, adj, action, or relation.
            save: whether to persist the vocabulary update.
        Output:
            dict: token registration summary.
        """
        return self.question.register_token_pos(token, pos, save=save)

    def register_relation_name(
        self,
        relation_name: str,
        *,
        relation_family: str = "noun_noun_relation",
        save: bool = True,
    ):
        """Register one relation label into the requested long-term relation family.

        Input:
            relation_name: relation label text.
            relation_family: noun_noun_relation or adj_noun_relation.
            save: whether to persist the vocabulary update.
        Output:
            dict: relation registration summary.
        """
        import importlib

        relation_name = str(relation_name).lower()
        if relation_family == "noun_noun_relation":
            result = self.question.register_token_pos(relation_name, "relation", save=save)
            result["relation_family"] = relation_family
            return result
        if relation_family == "adj_noun_relation":
            arm = importlib.import_module("knowledge.adj_relation_map")
            arm.load_adj_relation_data()
            created = False
            if relation_name not in arm.adj_relation_list:
                if len(arm.adj_relation_list) >= arm.adj_relation_num:
                    raise ValueError(
                        f"adj_relation_list is full; cannot register new adjective relation '{relation_name}'"
                    )
                arm.adj_relation_list.append(relation_name)
                created = True
            if save:
                arm.save_adj_relation_data()
            return {
                "token": relation_name,
                "pos": "relation",
                "relation_family": relation_family,
                "created": bool(created),
                "saved": bool(save),
            }
        raise ValueError("relation_family must be one of: noun_noun_relation, adj_noun_relation")

    def register_adjective_relation_hint(
        self,
        adjective: str,
        relation_name: str,
        *,
        save: bool = True,
    ):
        """Register the default relation hint used when an adjective appears in grammar parsing.

        Input:
            adjective: adjective token.
            relation_name: default adjective relation label.
            save: whether to persist the updated adjective relation vocabulary.
        Output:
            dict: hint registration summary.
        """
        import importlib

        grammar = importlib.import_module("prototype.grammar")
        adjective = str(adjective).lower()
        relation_name = str(relation_name).lower()
        self.question.register_token_pos(adjective, "adj", save=save)
        relation_result = self.register_relation_name(
            relation_name,
            relation_family="adj_noun_relation",
            save=save,
        )
        previous = grammar.ADJECTIVE_RELATION_HINTS.get(adjective)
        grammar.ADJECTIVE_RELATION_HINTS[adjective] = relation_name
        return {
            "adjective": adjective,
            "relation_name": relation_name,
            "previous_relation_name": previous,
            "updated": previous != relation_name,
            "saved": bool(save),
            "relation_registration": relation_result,
        }
    def learn_noun_relation(self, source_noun: str, target_noun: str, relation_type: int, save: bool = True):
        """Directly learn one noun-noun relation in long-term knowledge.

        Input:
            source_noun: source noun.
            target_noun: target noun.
            relation_type: discrete relation type.
            save: whether to persist the update.
        Output:
            dict: learning summary with relation type and loss.
        """
        return self.question.direct_learn_noun_relation(
            source_noun,
            target_noun,
            relation_type,
            save=save,
        )

    def learn_adj_relation(self, noun: str, adjective: str, relation_type: int, save: bool = True):
        """Directly learn one adj-noun relation in long-term knowledge.

        Input:
            noun: noun token.
            adjective: adjective token.
            relation_type: discrete adjective relation type.
            save: whether to persist the update.
        Output:
            dict: learning summary with relation type and loss.
        """
        return self.question.direct_learn_adj_relation(
            noun,
            adjective,
            relation_type,
            save=save,
        )

    def answer_question(
        self,
        question: ProposedQuestion,
        answer_text: str,
        corrected_target: Optional[str] = None,
        corrected_relation_type: Optional[int] = None,
        save: bool = True,
    ) -> AnswerResult:
        """Apply a yes/no answer to a proposed question and learn from the feedback.

        Input:
            question: proposed question object.
            answer_text: 'yes' or 'no'.
            corrected_target: optional corrected noun/adjective target for a 'no' answer.
            corrected_relation_type: optional corrected relation type for relation questions.
            save: whether to persist the learned update.
        Output:
            AnswerResult: accepted/learned flags, message, learned target, and optional loss.
        """
        if question.kind == "adj_noun":
            return self.question.apply_adj_answer(
                question,
                answer_text,
                corrected_adjective=corrected_target,
                save=save,
            )
        if question.question_target == "noun":
            return self.question.apply_noun_answer(
                question,
                answer_text,
                corrected_target_noun=corrected_target,
                save=save,
            )
        return self.question.apply_relation_answer(
            question,
            answer_text,
            corrected_relation_type=corrected_relation_type,
            save=save,
        )


def available_reasoning_modes():
    return {
        "what": "inspect whether a token is known and ask for its part-of-speech when needed",
        "where": "type or relation lookup",
        "when": "type or relation lookup",
        "recall": "retrieve stored noun_noun and adj_noun relations from memory maps",
        "predict": "predict a relation target and ask only when confidence is in the uncertainty band",
        "thinking": "predict a relation target first, then ask when confidence is in the uncertainty band",
        "learning": "learn directly from provided noun_noun or adj_noun supervision",
        "observe": "parse sentence input and write relation/event information into short memory",
        "inspect": "inspect current short memory state and relation clones",
        "update": "rebuild noun instances or explicitly update short-memory relation clones",
        "imagine": "predict and optionally append next events from short memory",
        "train": "train the world model from current short memory",
        "learn_from_sentence": "learn long-term language relations from sentence input",
    }


def available_consciousness_functions():
    return Consciousness().available_functions()


def has_relation(i: int, j: int, relation_type: int) -> bool:
    return int(relation_map[i][j]) == int(relation_type)


__all__ = [
    "AnswerResult",
    "Consciousness",
    "ProposedQuestion",
    "QuestionEngine",
    "available_consciousness_functions",
    "available_reasoning_modes",
    "has_relation",
]





