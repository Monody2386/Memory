"""High-level command wrappers built on top of the Consciousness API."""

from dataclasses import dataclass, field
from typing import Any, Optional

from .consciousness import Consciousness
from grammar_layer import parse_sentence


@dataclass
class HighLevelCommands:
    """Task-level commands composed from low-level consciousness interfaces."""

    consciousness: Consciousness = field(default_factory=Consciousness)
    _emotion_reward_encoder: Any = field(default=None, init=False, repr=False)
    _subject_emotion_reward_model: Any = field(default=None, init=False, repr=False)
    _subject_emotion_reward_engine: Any = field(default=None, init=False, repr=False)
    _subject_emotion_reward_trainer: Any = field(default=None, init=False, repr=False)
    _object_emotion_reward_model: Any = field(default=None, init=False, repr=False)
    _object_emotion_reward_engine: Any = field(default=None, init=False, repr=False)
    _object_emotion_reward_trainer: Any = field(default=None, init=False, repr=False)
    _surprise_encoder: Any = field(default=None, init=False, repr=False)
    _surprise_model: Any = field(default=None, init=False, repr=False)
    _surprise_engine: Any = field(default=None, init=False, repr=False)
    _surprise_trainer: Any = field(default=None, init=False, repr=False)

    def _surprise_stack(self):
        if self._surprise_engine is None:
            from surprise import SubjectEventSurpriseEngine, SubjectEventSurpriseNet, SubjectEventSurpriseTrainer, SurpriseEncoder
            from world.world_model import action_dim, noun_dim

            self._surprise_encoder = SurpriseEncoder(self.consciousness)
            self._surprise_model = SubjectEventSurpriseNet(
                noun_dim=noun_dim,
                action_dim=action_dim,
            )
            self._surprise_engine = SubjectEventSurpriseEngine(
                self._surprise_model,
                self._surprise_encoder,
            )
            self._surprise_trainer = SubjectEventSurpriseTrainer(
                self._surprise_model,
                self._surprise_encoder,
            )
        return (
            self._surprise_encoder,
            self._surprise_model,
            self._surprise_engine,
            self._surprise_trainer,
        )

    def _emotion_reward_stack(self, role: str = "subject"):
        """Create one reward model per event role: subject actor reward or object receiver reward."""
        role = str(role).strip().lower()
        if role not in {"subject", "object"}:
            raise ValueError("role must be one of: subject, object")

        if self._emotion_reward_encoder is None:
            from reward import RewardEncoder
            self._emotion_reward_encoder = RewardEncoder(self.consciousness)

        from reward import SubjectEventRewardEngine, SubjectEventRewardNet, SubjectEventRewardTrainer
        from world.world_model import action_dim, noun_dim

        if role == "subject":
            if self._subject_emotion_reward_engine is None:
                self._subject_emotion_reward_model = SubjectEventRewardNet(
                    noun_dim=noun_dim,
                    action_dim=action_dim,
                )
                self._subject_emotion_reward_engine = SubjectEventRewardEngine(
                    self._subject_emotion_reward_model,
                    self._emotion_reward_encoder,
                )
                self._subject_emotion_reward_trainer = SubjectEventRewardTrainer(
                    self._subject_emotion_reward_model,
                    self._emotion_reward_encoder,
                )
            return (
                self._emotion_reward_encoder,
                self._subject_emotion_reward_model,
                self._subject_emotion_reward_engine,
                self._subject_emotion_reward_trainer,
            )

        if self._object_emotion_reward_engine is None:
            self._object_emotion_reward_model = SubjectEventRewardNet(
                noun_dim=noun_dim,
                action_dim=action_dim,
            )
            self._object_emotion_reward_engine = SubjectEventRewardEngine(
                self._object_emotion_reward_model,
                self._emotion_reward_encoder,
            )
            self._object_emotion_reward_trainer = SubjectEventRewardTrainer(
                self._object_emotion_reward_model,
                self._emotion_reward_encoder,
            )
        return (
            self._emotion_reward_encoder,
            self._object_emotion_reward_model,
            self._object_emotion_reward_engine,
            self._object_emotion_reward_trainer,
        )

    def _noun_instance_ids(self, noun: str):
        noun = str(noun).strip().lower()
        matches = []
        for instance_id, metadata in self.consciousness.short_memory.noun_instance_metadata.items():
            noun_text = str(metadata.get("noun_text") or "").lower()
            if noun_text == noun:
                matches.append(instance_id)
        return matches

    def train_emotion_reward(self, *, epochs: int = 20):
        """Train the subject emotion predictor from reward sentences in short memory."""
        from reward import reward_samples_from_short_memory

        _, _, _, trainer = self._emotion_reward_stack(role="subject")
        samples = reward_samples_from_short_memory(self.consciousness.short_memory)
        history = trainer.train_epochs(samples, epochs=epochs)
        return {
            "command": "train_emotion_reward",
            "role": "subject",
            "sample_count": len(samples),
            "epochs": int(epochs),
            "last_result": None if not history else history[-1],
            "history": history,
        }

    def train_object_emotion_reward(self, samples, *, epochs: int = 20):
        """Train the object emotion predictor from explicit object-reward samples."""
        _, _, _, trainer = self._emotion_reward_stack(role="object")
        samples = list(samples)
        history = trainer.train_epochs(samples, epochs=epochs)
        return {
            "command": "train_object_emotion_reward",
            "role": "object",
            "sample_count": len(samples),
            "epochs": int(epochs),
            "last_result": None if not history else history[-1],
            "history": history,
        }

    def predict_emotion(self, noun: str):
        """Predict reward for noun's latest event, routing by whether noun is subject or object."""
        from reward import reward_input_from_subject_event, subject_events_from_short_memory

        noun = str(noun).strip().lower()
        instance_ids = self._noun_instance_ids(noun)
        events = subject_events_from_short_memory(self.consciousness.short_memory)

        candidate_events = []
        for event in events:
            is_subject = (
                event.subject_instance_id in instance_ids
                or str(event.subject_text or "").lower() == noun
            )
            is_object = (
                event.object_instance_id in instance_ids
                or str(event.object_text or "").lower() == noun
            )
            if is_subject:
                candidate_events.append((event, "subject"))
            if is_object:
                candidate_events.append((event, "object"))

        candidate_events.sort(
            key=lambda item: (
                -1 if item[0].time_position is None else int(item[0].time_position),
                -1 if item[0].subject_pair_index is None else int(item[0].subject_pair_index),
            ),
            reverse=True,
        )

        if not candidate_events:
            return {
                "command": "predict_emotion",
                "status": "no_event",
                "noun": noun,
                "instance_ids": instance_ids,
                "event_role": None,
                "reward_score": None,
                "reward_label": None,
                "event": None,
            }

        latest_event, event_role = candidate_events[0]
        _, _, engine, _ = self._emotion_reward_stack(role=event_role)
        prediction = engine.predict(reward_input_from_subject_event(latest_event))

        selected_instance_id = (
            latest_event.subject_instance_id
            if event_role == "subject"
            else latest_event.object_instance_id
        )
        event_view = {
            "subject_text": latest_event.subject_text,
            "subject_instance_id": latest_event.subject_instance_id,
            "action_text": latest_event.action_text,
            "object_text": latest_event.object_text,
            "object_instance_id": latest_event.object_instance_id,
            "time_position": latest_event.time_position,
            "subject_pair_index": latest_event.subject_pair_index,
        }
        return {
            "command": "predict_emotion",
            "status": "predicted",
            "noun": noun,
            "instance_ids": instance_ids,
            "selected_instance_id": selected_instance_id,
            "event_role": event_role,
            "event": event_view,
            "reward_score": prediction.score,
            "reward_label": prediction.label,
            "prediction": {
                "score": prediction.score,
                "label": prediction.label,
                "model_role": event_role,
                "subject_text": prediction.subject_text,
                "action_text": prediction.action_text,
                "object_text": prediction.object_text,
                "subject_instance_id": prediction.subject_instance_id,
                "object_instance_id": prediction.object_instance_id,
            },
        }

    def _event_entry_matches(self, entry, *, noun: str, action: Optional[str]) -> bool:
        if getattr(entry, "question_label", "none") == "question":
            return False
        noun_key = str(noun).strip().lower()
        action_key = None if action is None else str(action).strip().lower()
        if str(getattr(entry, "noun_text", "") or "").lower() != noun_key:
            return False
        if action_key is not None and str(getattr(entry, "action_text", "") or "").lower() != action_key:
            return False
        return True

    def _event_group_for_entry(self, focus_entry):
        focus_key = (getattr(focus_entry, "time_position", None), getattr(focus_entry, "event_index", None))
        if focus_key[1] is not None:
            return [
                entry for entry in self.consciousness.short_memory.short_memory_event
                if (entry.time_position, entry.event_index) == focus_key
            ]
        return [
            entry for entry in self.consciousness.short_memory.short_memory_event
            if entry.time_position == focus_key[0]
        ]

    def _subject_event_from_group(self, group):
        from reward import SubjectEvent

        subject_entries = [entry for entry in group if entry.role == "subject"]
        object_entries = [entry for entry in group if entry.role == "object"]
        if not subject_entries:
            return None
        subject_entry = sorted(subject_entries, key=lambda entry: entry.pair_index)[-1]
        object_entry = None
        if object_entries:
            object_entry = sorted(object_entries, key=lambda entry: entry.pair_index)[0]
        return SubjectEvent(
            subject_instance_id=subject_entry.noun_instance_id,
            subject_text=subject_entry.noun_text,
            action_text=subject_entry.action_text,
            object_instance_id=None if object_entry is None else object_entry.noun_instance_id,
            object_text=None if object_entry is None else object_entry.noun_text,
            time_position=subject_entry.time_position,
            subject_pair_index=subject_entry.pair_index,
            object_pair_index=None if object_entry is None else object_entry.pair_index,
        )

    def surprise_event(self, noun: str, action: Optional[str] = None):
        """Predict surprise for the latest stored event matching noun/action."""
        from surprise import surprise_input_from_subject_event

        noun_key = str(noun).strip().lower()
        action_key = None if action is None else str(action).strip().lower()
        matches = [
            entry for entry in self.consciousness.short_memory.short_memory_event
            if self._event_entry_matches(entry, noun=noun_key, action=action_key)
        ]
        matches.sort(
            key=lambda entry: (
                -1 if entry.time_position is None else int(entry.time_position),
                -1 if entry.event_index is None else int(entry.event_index),
                -1 if entry.pair_index is None else int(entry.pair_index),
            ),
            reverse=True,
        )
        if not matches:
            return {
                "command": "surprise_event",
                "status": "no_event",
                "noun": noun_key,
                "action": action_key,
                "event": None,
                "surprise_score": None,
                "surprise_label": None,
            }

        focus_entry = matches[0]
        group = self._event_group_for_entry(focus_entry)
        event = self._subject_event_from_group(group)
        if event is None:
            return {
                "command": "surprise_event",
                "status": "no_subject_event",
                "noun": noun_key,
                "action": action_key,
                "event_index": focus_entry.event_index,
                "time_position": focus_entry.time_position,
                "event": None,
                "surprise_score": None,
                "surprise_label": None,
            }

        _, _, engine, _ = self._surprise_stack()
        prediction = engine.predict(surprise_input_from_subject_event(event))
        event_view = {
            "subject_text": event.subject_text,
            "subject_instance_id": event.subject_instance_id,
            "action_text": event.action_text,
            "object_text": event.object_text,
            "object_instance_id": event.object_instance_id,
            "time_position": event.time_position,
            "event_index": focus_entry.event_index,
            "subject_pair_index": event.subject_pair_index,
            "object_pair_index": event.object_pair_index,
        }
        return {
            "command": "surprise_event",
            "status": "predicted",
            "noun": noun_key,
            "action": action_key,
            "matched_role": focus_entry.role,
            "event_index": focus_entry.event_index,
            "time_position": focus_entry.time_position,
            "event": event_view,
            "surprise_score": prediction.score,
            "surprise_label": prediction.label,
            "prediction": {
                "score": prediction.score,
                "label": prediction.label,
                "subject_text": prediction.subject_text,
                "action_text": prediction.action_text,
                "object_text": prediction.object_text,
                "subject_instance_id": prediction.subject_instance_id,
                "object_instance_id": prediction.object_instance_id,
            },
        }

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
            "sentence_label": observation.get("sentence_label"),
            "sentence_type": observation["sentence_type"],
            "structure": observation["structure"],
            "tokens": observation["tokens"],
            "action_count": observation["action_count"],
            "relation_count": observation["relation_count"],
            "reward_count": observation.get("reward_count", 0),
            "surprise_count": observation.get("surprise_count", 0),
            "event_entries_added": observation["event_entries_added"],
            "relation_entries_added": observation["relation_entries_added"],
            "reward_entries_added": observation.get("reward_entries_added", 0),
            "surprise_entries_added": observation.get("surprise_entries_added", 0),
            "states": observation["states"],
            "updated_instances": rebuilt_instances,
            "focus": focus,
        }

    def encode(
        self,
        sentence: str,
        *,
        time_position: Optional[int] = None,
        base_score: float = 1.0,
        adjective_relation_types: Optional[Any] = None,
        auto_accept: bool = True,
        confirm_threshold: float = 50.0,
        yes_threshold: float = 30.0,
        no_threshold: float = 70.0,
        instance_epochs: int = 5,
        value_model_epochs: int = 1,
        step_scale: Optional[float] = None,
        event_surprise_target: float = -50.0,
        train_instance_embeddings: bool = True,
        train_value_models: bool = True,
    ):
        """Encode one sentence with the fully automatic non-interactive update flow.

        The pipeline is fixed as:
        question_what -> understand -> accept -> question(interact=False) ->
        train_confirmed_info_pairs(confirmed_yes)
        """
        what_result = self.question_what(
            sentence,
            time_position=time_position,
            base_score=base_score,
            adjective_relation_types=adjective_relation_types,
        )
        if what_result.get("status") == "question":
            return {
                "command": "encode",
                "status": "question_what",
                "sentence": sentence,
                "question_what_result": what_result,
                "understand_result": None,
                "accept_result": None,
                "question_result": None,
                "confirmed_yes_count": 0,
                "confirmed_yes": [],
                "training_result": None,
            }

        understand_result = what_result.get("understand_result")
        sentence_label = None if understand_result is None else understand_result.get("sentence_label")

        accept_result = self.accept(sentence_label=sentence_label)
        question_result = self.question(
            sentence_label=sentence_label,
            confirm_threshold=confirm_threshold,
            yes_threshold=yes_threshold,
            no_threshold=no_threshold,
            auto_accept=auto_accept,
            interact=False,
        )

        confirmed_yes = list(question_result.get("confirmed_yes", []))
        training_result = self.train_confirmed_info_pairs(
            confirmed_yes,
            instance_epochs=instance_epochs,
            value_model_epochs=value_model_epochs,
            step_scale=step_scale,
            event_surprise_target=event_surprise_target,
            train_instance_embeddings=train_instance_embeddings,
            train_value_models=train_value_models,
        )

        return {
            "command": "encode",
            "status": "encoded",
            "sentence": sentence,
            "sentence_label": sentence_label,
            "question_what_result": what_result,
            "understand_result": understand_result,
            "accept_result": accept_result,
            "question_result": question_result,
            "confirmed_yes_count": len(confirmed_yes),
            "confirmed_yes": confirmed_yes,
            "training_result": training_result,
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


    def inf_question(self, info_pair: dict, *, reason: str = "large_diff") -> dict:
        """Build a placeholder confirmation question for one non-question info pair."""
        return {
            "type": "inf_question",
            "reason": reason,
            "kind": info_pair.get("pair_kind"),
            "sentence_label": info_pair.get("sentence_label"),
            "question_label": info_pair.get("question_label", "none"),
            "diff_value": info_pair.get("diff_value", "none"),
            "accept_label": info_pair.get("accept_label", "none"),
            "memory_entry": info_pair,
        }

    def question(
        self,
        *,
        sentence_index: Optional[int] = None,
        sentence_label: Optional[str] = None,
        confirm_threshold: float = 50.0,
        yes_threshold: float = 30.0,
        no_threshold: float = 70.0,
        order_by: str = "time",
        auto_accept: bool = True,
        interact: bool = True,
    ):
        """Inspect one sentence's info pairs and select which pairs are confirmed yes.

        question() is the filtering phase: low-diff non-question pairs become pending yes
        confirmations, high-diff non-question pairs become inf_question items in interactive
        mode, and question pairs are answered from |diff_value| without training.
        """
        if sentence_label is None and sentence_index is not None:
            sentence_label = f"sentence:{int(sentence_index)}"

        memory_view = self.consciousness.inspect_memory(kind="all", order_by=order_by)
        all_entries = []
        for kind, entries in memory_view.items():
            for entry in entries:
                item = dict(entry)
                item["memory_kind"] = kind
                all_entries.append(item)

        def _sentence_sort_key(entry: dict):
            label = str(entry.get("sentence_label") or "none")
            if label.startswith("sentence:"):
                try:
                    return (1, int(label.split(":", 1)[1]))
                except ValueError:
                    pass
            return (0, int(entry.get("time_position", -1)), int(entry.get("pair_index", -1)))

        if sentence_label is None:
            labeled_entries = [entry for entry in all_entries if entry.get("sentence_label") not in {None, "none"}]
            if labeled_entries:
                sentence_label = str(max(labeled_entries, key=_sentence_sort_key).get("sentence_label"))

        if sentence_label is None:
            return {
                "command": "question",
                "status": "empty",
                "sentence_label": None,
                "sentence_index": sentence_index,
                "auto_accept_result": None,
                "inspected_pair_count": 0,
                "question_count": 0,
                "questions": [],
                "answer_count": 0,
                "answers": [],
                "training": {"reward": None, "surprise": None},
            }

        sentence_entries = [
            entry for entry in all_entries
            if str(entry.get("sentence_label") or "none") == str(sentence_label)
        ]

        def _has_diff(entry: dict) -> bool:
            diff = entry.get("diff_value", "none")
            return diff not in {None, "none"}

        auto_accept_result = None
        if auto_accept and sentence_entries and any(not _has_diff(entry) for entry in sentence_entries):
            auto_accept_result = self.accept(sentence_label=str(sentence_label), order_by=order_by)
            memory_view = self.consciousness.inspect_memory(kind="all", order_by=order_by)
            all_entries = []
            for kind, entries in memory_view.items():
                for entry in entries:
                    item = dict(entry)
                    item["memory_kind"] = kind
                    all_entries.append(item)
            sentence_entries = [
                entry for entry in all_entries
                if str(entry.get("sentence_label") or "none") == str(sentence_label)
            ]

        questions = []
        answers = []
        auto_updates = []
        confirmed_yes = []
        skipped_updates = []
        training = {"instance_embedding": None}
        threshold = float(confirm_threshold)

        def _auto_yes(entry: dict, reason: str):
            question_item = self.inf_question(entry, reason=reason)
            confirmed_yes.append(question_item)
            result = {
                "command": "question",
                "accepted": True,
                "action": "pending_yes_confirmation",
                "kind": entry.get("memory_kind"),
                "reason": reason,
                "question": question_item,
            }
            auto_updates.append(result)
            return result

        if not bool(interact):
            for entry in sentence_entries:
                if not _has_diff(entry):
                    continue
                if str(entry.get("question_label") or "none") == "question":
                    skipped_updates.append({
                        "reason": "question_pair_skipped_in_non_interactive_mode",
                        "memory_kind": entry.get("memory_kind"),
                        "memory_entry": entry,
                    })
                    continue
                diff_value = float(entry.get("diff_value"))
                abs_diff = abs(diff_value)
                if abs_diff <= threshold:
                    _auto_yes(entry, reason="small_diff_auto_accept")
                else:
                    skipped_updates.append({
                        "reason": "large_diff_non_interactive_skip",
                        "diff_value": diff_value,
                        "abs_diff_value": abs_diff,
                        "threshold": threshold,
                        "memory_kind": entry.get("memory_kind"),
                        "memory_entry": entry,
                    })

            return {
                "command": "question",
                "status": "updated",
                "interact": False,
                "sentence_label": str(sentence_label),
                "sentence_index": sentence_index,
                "thresholds": {
                    "confirm_threshold": float(confirm_threshold),
                    "yes_threshold": float(yes_threshold),
                    "no_threshold": float(no_threshold),
                },
                "auto_accept_result": auto_accept_result,
                "inspected_pair_count": len(sentence_entries),
                "question_count": 0,
                "questions": [],
                "answer_count": 0,
                "answers": [],
                "training": training,
                "confirmed_yes_count": len(confirmed_yes),
                "confirmed_yes": confirmed_yes,
                "auto_update_count": len(auto_updates),
                "auto_updates": auto_updates,
                "skipped_update_count": len(skipped_updates),
                "skipped_updates": skipped_updates,
                "trained_pair_count": 0,
                "trained_reward_count": 0,
                "trained_surprise_count": 0,
            }

        for entry in sentence_entries:
            if not _has_diff(entry):
                continue
            diff_value = float(entry.get("diff_value"))
            abs_diff = abs(diff_value)
            if str(entry.get("question_label") or "none") == "question":
                if abs_diff >= float(no_threshold):
                    answer = "NO"
                elif abs_diff <= float(yes_threshold):
                    answer = "YES"
                else:
                    answer = "UNKNOWN"
                answers.append(
                    {
                        "type": "answer_question_pair",
                        "answer": answer,
                        "diff_value": diff_value,
                        "abs_diff_value": abs_diff,
                        "memory_kind": entry.get("memory_kind"),
                        "memory_entry": entry,
                    }
                )
                continue

            if abs_diff <= threshold:
                _auto_yes(entry, reason="small_diff_auto_accept")
            else:
                questions.append(self.inf_question(entry, reason="large_diff"))

        return {
            "command": "question",
            "status": "ok",
            "interact": True,
            "sentence_label": str(sentence_label),
            "sentence_index": sentence_index,
            "thresholds": {
                "confirm_threshold": float(confirm_threshold),
                "yes_threshold": float(yes_threshold),
                "no_threshold": float(no_threshold),
            },
            "auto_accept_result": auto_accept_result,
            "inspected_pair_count": len(sentence_entries),
            "question_count": len(questions),
            "questions": questions,
            "answer_count": len(answers),
            "answers": answers,
            "confirmed_yes_count": len(confirmed_yes),
            "confirmed_yes": confirmed_yes,
            "auto_update_count": len(auto_updates),
            "auto_updates": auto_updates,
            "skipped_update_count": len(skipped_updates),
            "skipped_updates": skipped_updates,
            "training": training,
        }

    def accept(
        self,
        *,
        sentence_label: Optional[str] = None,
        time_position: Optional[int] = None,
        event_index: Optional[int] = None,
        order_by: str = "time",
        include_rewards: bool = True,
        save: bool = True,
    ):
        """Evaluate all info pairs from one sentence.

        The default scope is the latest sentence_label produced by understand(sentence).
        Relation pairs are checked against long-term memory; reward pairs use the reward model;
        event and surprise pairs use the surprise model. accept_label is numeric in [-100, 100]:
        100 means same/expected, -100 means conflict, and 0 means no stored evidence or neutral.
        """
        del save
        from reward import SubjectEventRewardInput
        from surprise import SubjectEventSurpriseInput

        event_entries = self.consciousness.inspect_memory(kind="event", order_by=order_by)
        relation_entries = self.consciousness.inspect_memory(kind="relation", order_by=order_by)
        reward_entries = self.consciousness.inspect_memory(kind="reward", order_by=order_by) if include_rewards else []
        surprise_entries = self.consciousness.inspect_memory(kind="surprise", order_by=order_by)
        all_entries = list(event_entries) + list(relation_entries) + list(reward_entries) + list(surprise_entries)

        def _sentence_sort_key(entry: dict):
            label = str(entry.get("sentence_label") or "none")
            if label.startswith("sentence:"):
                try:
                    return (1, int(label.split(":", 1)[1]))
                except ValueError:
                    pass
            return (0, int(entry.get("time_position", -1)), int(entry.get("pair_index", -1)))

        if sentence_label is None:
            labeled_entries = [entry for entry in all_entries if entry.get("sentence_label") not in {None, "none"}]
            if labeled_entries:
                sentence_label = str(max(labeled_entries, key=_sentence_sort_key).get("sentence_label"))

        if time_position is None and event_index is None and sentence_label is None:
            focus = self.consciousness.inspect_focus()
            if focus is not None:
                time_position = int(focus["time_position"])
                event_index = focus.get("event_index")
            elif all_entries:
                time_position = max((int(entry["time_position"]) for entry in all_entries), default=None)

        if sentence_label is None and time_position is None:
            return {
                "command": "accept",
                "status": "empty",
                "sentence_label": None,
                "time_position": None,
                "event_index": None,
                "evidence": {"event": [], "relation": [], "reward": [], "surprise": []},
                "labeled_pair_count": 0,
                "labeled_pairs": [],
                "accepted_write_count": 0,
                "accepted_writes": [],
                "issue_count": 0,
                "issues": [],
            }

        def _entry_in_accept_scope(entry):
            if sentence_label is not None:
                return str(entry.get("sentence_label") or "none") == str(sentence_label)
            if int(entry.get("time_position", -1)) != int(time_position):
                return False
            if event_index is None:
                return True
            return entry.get("event_index") == event_index

        current_events = [entry for entry in event_entries if _entry_in_accept_scope(entry)]
        current_relations = [entry for entry in relation_entries if _entry_in_accept_scope(entry)]
        current_rewards = [entry for entry in reward_entries if _entry_in_accept_scope(entry)]
        current_surprises = [entry for entry in surprise_entries if _entry_in_accept_scope(entry)]

        labeled_pairs = []
        issues = []
        accepted_writes = []

        def _relation_accept_score(stored_value, expected_value, polarity: int = 1) -> float:
            if stored_value in {None, 0}:
                return 0.0
            same = stored_value == expected_value
            if int(polarity) == -1:
                same = not same
            return -50.0 if same else 50.0

        def _score_issue_type(score: float) -> str:
            if score >= 0.0:
                return "same"
            return "conflict"

        def _mark_accept_label(kind: str, entry: dict, label, diff_value="none") -> None:
            if kind == "event":
                entries = self.consciousness.short_memory.short_memory_event
            elif kind == "relation":
                entries = self.consciousness.short_memory.short_memory_relation
            elif kind == "reward":
                entries = self.consciousness.short_memory.short_memory_reward
            elif kind == "surprise":
                entries = self.consciousness.short_memory.short_memory_surprise
            else:
                return

            label_value = float(label)
            for memory_entry in entries:
                if str(getattr(memory_entry, "sentence_label", "none")) != str(entry.get("sentence_label", "none")):
                    continue
                if int(getattr(memory_entry, "time_position", -1)) != int(entry.get("time_position", -1)):
                    continue
                if int(getattr(memory_entry, "pair_index", -1)) != int(entry.get("pair_index", -1)):
                    continue
                if kind == "event" and getattr(memory_entry, "event_index", None) != entry.get("event_index"):
                    continue
                setattr(memory_entry, "accept_label", label_value)
                setattr(memory_entry, "diff_value", diff_value)
                memory_entry.info_pair["accept_label"] = label_value
                memory_entry.info_pair["diff_value"] = diff_value
                entry["accept_label"] = label_value
                entry["diff_value"] = diff_value
                return

        def _predict_instance_relation(entry: dict):
            source_instance_id = entry.get("source_instance_id")
            if source_instance_id is None:
                return None

            relation_kind = entry.get("pair_kind")
            relation_name = entry.get("relation_name")
            source_embedding = self.consciousness.short_memory.get_noun_embedding(source_instance_id)
            relation_weight = self.consciousness.short_memory.ensure_relation_clone(relation_kind, relation_name)
            if source_embedding is None or relation_weight is None:
                return None

            rm, arm, kt = self.consciousness.short_memory._load_language_context()
            predicted_target = relation_weight @ source_embedding.to(relation_weight.device).view(-1)
            if relation_kind == "noun_noun_relation":
                top_indices, top_scores = kt.knowledge_map_one.query_similarity(predicted_target.detach().cpu(), top_k=1)
                predicted_index = int(top_indices.view(-1)[0].item())
                predicted_text = rm.noun_list[predicted_index] if predicted_index < len(rm.noun_list) else None
            elif relation_kind == "adj_noun_relation":
                top_indices, top_scores = kt.adj_map_one.query_adjective_similarity(predicted_target.detach().cpu(), top_k=1)
                predicted_index = int(top_indices.view(-1)[0].item())
                predicted_text = arm.adjective_list[predicted_index] if predicted_index < len(arm.adjective_list) else None
            else:
                return None

            score = float(top_scores.view(-1)[0].item())
            target_text = str(entry.get("target_text") or "").lower()
            same = predicted_text is not None and str(predicted_text).lower() == target_text
            if int(entry.get("polarity", 1)) == -1:
                same = not same
            return {
                "stored_value": predicted_text,
                "prediction_score": score,
                "accept_label": -50.0 if same else 50.0,
                "method": "instance_relation_prediction",
            }

        for entry in current_relations:
            relation_kind = entry["pair_kind"]
            relation_name = entry["relation_name"]
            source_text = entry.get("source_text")
            target_text = entry.get("target_text")
            if not source_text or not target_text:
                continue

            relation_type = self.consciousness.resolve_relation_type(relation_kind, relation_name)
            relation_polarity = int(entry.get("polarity", 1))
            instance_prediction = _predict_instance_relation(entry)
            if instance_prediction is not None:
                stored_value = instance_prediction["stored_value"]
                label = instance_prediction["accept_label"]
                accept_method = instance_prediction["method"]
                prediction_score = instance_prediction["prediction_score"]
            else:
                stored_value = self.consciousness.recall_relation_value(
                    relation_kind,
                    source_text,
                    target_text,
                )
                label = _relation_accept_score(stored_value, relation_type, relation_polarity)
                accept_method = "relation_map_lookup"
                prediction_score = None

            diff_value = float(label) + 50.0
            _mark_accept_label("relation", entry, label, diff_value)
            pair_label = {
                "kind": relation_kind,
                "accept_label": label,
                "diff_value": diff_value,
                "source_noun": source_text,
                "target": target_text,
                "relation_name": relation_name,
                "relation_type": relation_type,
                "stored_value": stored_value,
                "prediction_score": prediction_score,
                "accept_method": accept_method,
                "polarity": relation_polarity,
                "memory_entry": entry,
            }
            labeled_pairs.append(pair_label)
            if label > 0.0:
                issues.append({**pair_label, "issue_type": "conflict"})

        def _object_for_subject(subject_entry: dict):
            candidates = [
                entry for entry in current_events
                if entry.get("role") == "object"
                and entry.get("sentence_label") == subject_entry.get("sentence_label")
                and entry.get("time_position") == subject_entry.get("time_position")
                and entry.get("event_index") == subject_entry.get("event_index")
                and int(entry.get("pair_index", -1)) < int(subject_entry.get("pair_index", -1))
            ]
            if not candidates:
                return None
            return max(candidates, key=lambda item: int(item.get("pair_index", -1)))

        _, _, surprise_engine, _ = self._surprise_stack()
        event_predictions = {}
        for subject_entry in [entry for entry in current_events if entry.get("role") == "subject"]:
            object_entry = _object_for_subject(subject_entry)
            prediction = surprise_engine.predict(
                SubjectEventSurpriseInput(
                    subject_text=subject_entry.get("noun_text"),
                    subject_instance_id=subject_entry.get("noun_instance_id"),
                    action_text=subject_entry.get("action_text"),
                    object_text=None if object_entry is None else object_entry.get("noun_text"),
                    object_instance_id=None if object_entry is None else object_entry.get("noun_instance_id"),
                )
            )
            label = float(prediction.score) * int(subject_entry.get("polarity", 1))
            event_key = (
                subject_entry.get("sentence_label"),
                subject_entry.get("time_position"),
                subject_entry.get("event_index"),
            )
            event_predictions[event_key] = {
                "label": max(-100.0, min(100.0, label)),
                "prediction": prediction,
            }

        for entry in current_events:
            event_key = (entry.get("sentence_label"), entry.get("time_position"), entry.get("event_index"))
            prediction_info = event_predictions.get(event_key)
            if prediction_info is None:
                continue
            label = prediction_info["label"]
            diff_value = float(label)
            _mark_accept_label("event", entry, label, diff_value)
            pair_label = {
                "kind": "event_surprise_estimate",
                "accept_label": label,
                "diff_value": diff_value,
                "source_noun": entry.get("noun_text"),
                "target": entry.get("action_text"),
                "relation_name": "surprise_model",
                "relation_type": None,
                "stored_value": label,
                "polarity": int(entry.get("polarity", 1)),
                "prediction_label": prediction_info["prediction"].label,
                "memory_entry": entry,
            }
            labeled_pairs.append(pair_label)
            if label < 0.0:
                issues.append({**pair_label, "issue_type": _score_issue_type(label)})

        _, _, reward_engine, _ = self._emotion_reward_stack(role="subject")
        for entry in current_rewards:
            prediction = reward_engine.predict(
                SubjectEventRewardInput(
                    subject_text=entry.get("subject_text"),
                    subject_instance_id=entry.get("subject_instance_id"),
                    action_text=entry.get("action_text"),
                    object_text=entry.get("object_text"),
                    object_instance_id=entry.get("object_instance_id"),
                )
            )
            label = max(-100.0, min(100.0, float(prediction.score) * int(entry.get("polarity", 1))))
            diff_value = float(entry.get("reward_value", 0.0)) - float(label)
            _mark_accept_label("reward", entry, label, diff_value)
            pair_label = {
                "kind": "subject_event_reward",
                "accept_label": label,
                "diff_value": diff_value,
                "source_noun": entry.get("subject_text"),
                "target": entry.get("object_text") or entry.get("action_text"),
                "relation_name": "reward_model",
                "relation_type": None,
                "stored_value": label,
                "polarity": int(entry.get("polarity", 1)),
                "prediction_label": prediction.label,
                "memory_entry": entry,
            }
            labeled_pairs.append(pair_label)
            if label < 0.0:
                issues.append({**pair_label, "issue_type": _score_issue_type(label)})

        for entry in current_surprises:
            prediction = surprise_engine.predict(
                SubjectEventSurpriseInput(
                    subject_text=entry.get("subject_text"),
                    subject_instance_id=entry.get("subject_instance_id"),
                    action_text=entry.get("action_text"),
                    object_text=entry.get("object_text"),
                    object_instance_id=entry.get("object_instance_id"),
                )
            )
            label = max(-100.0, min(100.0, float(prediction.score) * int(entry.get("polarity", 1))))
            diff_value = float(entry.get("surprise_value", 0.0)) - float(label)
            _mark_accept_label("surprise", entry, label, diff_value)
            pair_label = {
                "kind": "subject_event_surprise",
                "accept_label": label,
                "diff_value": diff_value,
                "source_noun": entry.get("subject_text"),
                "target": entry.get("object_text") or entry.get("action_text"),
                "relation_name": "surprise_model",
                "relation_type": None,
                "stored_value": label,
                "polarity": int(entry.get("polarity", 1)),
                "prediction_label": prediction.label,
                "memory_entry": entry,
            }
            labeled_pairs.append(pair_label)
            if label < 0.0:
                issues.append({**pair_label, "issue_type": _score_issue_type(label)})

        status = "accepted" if not issues else "needs_review"
        return {
            "command": "accept",
            "status": status,
            "sentence_label": sentence_label,
            "time_position": None if time_position is None else int(time_position),
            "event_index": event_index,
            "evidence": {
                "event": current_events,
                "relation": current_relations,
                "reward": current_rewards,
                "surprise": current_surprises,
            },
            "labeled_pair_count": len(labeled_pairs),
            "labeled_pairs": labeled_pairs,
            "accepted_write_count": len(accepted_writes),
            "accepted_writes": accepted_writes,
            "issue_count": len(issues),
            "issues": issues,
        }


    def _memory_entry_matches(self, memory_entry, entry: dict, kind: str) -> bool:
        if str(getattr(memory_entry, "sentence_label", "none")) != str(entry.get("sentence_label", "none")):
            return False
        if int(getattr(memory_entry, "time_position", -1)) != int(entry.get("time_position", -1)):
            return False
        if int(getattr(memory_entry, "pair_index", -1)) != int(entry.get("pair_index", -1)):
            return False
        if kind == "event" and getattr(memory_entry, "event_index", None) != entry.get("event_index"):
            return False
        return True

    def _remove_short_memory_entry(self, entry: dict) -> dict:
        kind = str(entry.get("memory_kind") or "")
        memory = self.consciousness.short_memory
        stores = {
            "event": memory.short_memory_event,
            "relation": memory.short_memory_relation,
            "reward": memory.short_memory_reward,
            "surprise": memory.short_memory_surprise,
        }
        store = stores.get(kind)
        if store is None:
            return {"removed": False, "reason": f"unsupported_memory_kind:{kind}"}

        removed = []
        for index in range(len(store) - 1, -1, -1):
            if self._memory_entry_matches(store[index], entry, kind):
                removed.append(dict(getattr(store[index], "info_pair", {}) or {}))
                del store[index]

        prune = getattr(memory, "_prune_instance_stores", None)
        if callable(prune):
            prune()
        return {
            "removed": bool(removed),
            "removed_count": len(removed),
            "removed_entries": list(reversed(removed)),
        }

    def _event_surprise_sample_from_entry(self, entry: dict, *, surprise_value: float, weight: float):
        from surprise import SubjectEventSurpriseSample

        memory = self.consciousness.short_memory
        event_key = (
            str(entry.get("sentence_label") or "none"),
            int(entry.get("time_position", -1)),
            entry.get("event_index"),
        )
        group = [
            item for item in memory.short_memory_event
            if (
                str(getattr(item, "sentence_label", "none")),
                int(getattr(item, "time_position", -1)),
                getattr(item, "event_index", None),
            ) == event_key
        ]
        subject_entry = next((item for item in group if getattr(item, "role", None) == "subject"), None)
        object_entry = next((item for item in group if getattr(item, "role", None) == "object"), None)
        if subject_entry is None:
            return None
        return SubjectEventSurpriseSample(
            subject_text=subject_entry.noun_text,
            action_text=subject_entry.action_text,
            object_text=None if object_entry is None else object_entry.noun_text,
            subject_instance_id=subject_entry.noun_instance_id,
            object_instance_id=None if object_entry is None else object_entry.noun_instance_id,
            surprise_value=float(surprise_value),
            weight=float(weight),
            source="answer_inf_question_event_confirmed",
        )

    def _confirmed_entry_from_item(self, item: dict) -> dict:
        if "memory_entry" in item:
            return dict(item["memory_entry"])
        if "question" in item and isinstance(item["question"], dict):
            question = item["question"]
            if "memory_entry" in question:
                return dict(question["memory_entry"])
        return dict(item)

    def _confirmed_kind(self, entry: dict) -> str:
        kind = str(entry.get("memory_kind") or "")
        if kind:
            return kind
        pair_kind = str(entry.get("pair_kind") or "")
        if pair_kind in {"noun_noun_relation", "adj_noun_relation"}:
            return "relation"
        if pair_kind == "subject_event_reward":
            return "reward"
        if pair_kind == "subject_event_surprise":
            return "surprise"
        return "event"

    def _confirmed_instance_ids(self, entry: dict) -> list[str]:
        kind = self._confirmed_kind(entry)
        instance_ids = []
        if kind == "relation":
            if entry.get("source_instance_id") is not None:
                instance_ids.append(entry.get("source_instance_id"))
        elif kind == "reward":
            if entry.get("subject_instance_id") is not None:
                instance_ids.append(entry.get("subject_instance_id"))
        elif kind == "surprise":
            for key in ("subject_instance_id", "object_instance_id"):
                if entry.get(key) is not None:
                    instance_ids.append(entry.get(key))
        elif kind == "event":
            if entry.get("noun_instance_id") is not None:
                instance_ids.append(entry.get("noun_instance_id"))
        return list(dict.fromkeys(str(item) for item in instance_ids if item is not None))

    def _entry_weight(self, entry: dict) -> float:
        diff = entry.get("diff_value", "none")
        try:
            return max(1e-6, abs(float(diff)))
        except (TypeError, ValueError):
            try:
                return max(1e-6, abs(float(entry.get("score", 1.0))))
            except (TypeError, ValueError):
                return 1.0

    def _weighted_mean_torch_losses(self, losses, weights):
        import torch

        stacked = torch.stack(losses)
        weight_tensor = torch.tensor(
            [max(0.0, float(weight)) for weight in weights],
            dtype=stacked.dtype,
            device=stacked.device,
        )
        if float(weight_tensor.sum().item()) <= 1e-6:
            return stacked.mean()
        return (stacked * weight_tensor).sum() / weight_tensor.sum().clamp_min(1e-6)

    def _loss_for_confirmed_relation_entry(self, entry: dict, instance_id: str, trainable_embedding, rm, arm, kt):
        import torch.nn.functional as F

        if str(entry.get("source_instance_id")) != str(instance_id):
            return None
        relation_kind = str(entry.get("pair_kind") or "")
        relation_name = entry.get("relation_name")
        relation_weight = self.consciousness.short_memory.ensure_relation_clone(relation_kind, relation_name)
        if relation_weight is None:
            return None
        relation_weight = relation_weight.to(trainable_embedding.device)

        if relation_kind == "adj_noun_relation":
            adjective_key = str(entry.get("target_text") or "").lower()
            if not adjective_key:
                return None
            if adjective_key not in arm.adjective_list:
                arm.adjective_list.append(adjective_key)
            adjective_idx = arm.adjective_list.index(adjective_key)
            target_embedding = kt.adj_map_one.adjective_embedding.weight.data[adjective_idx].to(trainable_embedding.device)
        elif relation_kind == "noun_noun_relation":
            target_text = entry.get("target_text")
            if not target_text:
                return None
            target_instance_id = entry.get("target_instance_id")
            target_embedding = self.consciousness.short_memory.get_noun_embedding(target_instance_id)
            if target_embedding is None:
                _, target_embedding, _ = self.consciousness.short_memory.ensure_noun_instance(
                    str(target_text),
                    target_instance_id,
                    noun_type=entry.get("target_type"),
                )
            target_embedding = target_embedding.to(trainable_embedding.device).detach().clone()
        else:
            return None

        loss = F.mse_loss(relation_weight @ trainable_embedding, target_embedding)
        if int(entry.get("polarity", 1)) == -1:
            loss = -loss
        return loss

    def _loss_for_confirmed_reward_entry(self, entry: dict, instance_id: str, trainable_embedding, reward_model, reward_encoder):
        import torch
        import torch.nn.functional as F

        if str(entry.get("subject_instance_id")) != str(instance_id):
            return None
        subject_embedding = trainable_embedding
        action_embedding = reward_encoder.encode_action(action_text=entry.get("action_text"))
        object_embedding = reward_encoder.encode_noun(
            noun_text=entry.get("object_text"),
            noun_instance_id=entry.get("object_instance_id"),
        )
        prediction = reward_model(
            subject_embedding=subject_embedding,
            action_embedding=action_embedding,
            object_embedding=object_embedding,
        ).view(-1)[0]
        target = torch.tensor(float(entry.get("reward_value", 0.0)), dtype=prediction.dtype, device=prediction.device)
        return F.smooth_l1_loss(prediction, target)

    def _loss_for_confirmed_surprise_entry(self, entry: dict, instance_id: str, trainable_embedding, surprise_model, surprise_encoder):
        import torch
        import torch.nn.functional as F

        subject_embedding = surprise_encoder.encode_noun(
            noun_text=entry.get("subject_text"),
            noun_instance_id=entry.get("subject_instance_id"),
        )
        object_embedding = surprise_encoder.encode_noun(
            noun_text=entry.get("object_text"),
            noun_instance_id=entry.get("object_instance_id"),
        )
        if str(entry.get("subject_instance_id")) == str(instance_id):
            subject_embedding = trainable_embedding
        elif str(entry.get("object_instance_id")) == str(instance_id):
            object_embedding = trainable_embedding
        else:
            return None
        action_embedding = surprise_encoder.encode_action(action_text=entry.get("action_text"))
        prediction = surprise_model(
            subject_embedding=subject_embedding,
            action_embedding=action_embedding,
            object_embedding=object_embedding,
        ).view(-1)[0]
        target = torch.tensor(float(entry.get("surprise_value", 0.0)), dtype=prediction.dtype, device=prediction.device)
        return F.smooth_l1_loss(prediction, target)

    def _value_model_samples_from_confirmed_entries(self, entries, *, event_surprise_target: float = -50.0):
        from reward import SubjectEventRewardSample
        from surprise import SubjectEventSurpriseSample

        reward_samples = []
        surprise_samples = []
        for entry in entries:
            kind = self._confirmed_kind(entry)
            weight = self._entry_weight(entry)
            if kind == "reward":
                reward_samples.append(
                    SubjectEventRewardSample(
                        subject_text=entry.get("subject_text"),
                        action_text=entry.get("action_text"),
                        object_text=entry.get("object_text"),
                        subject_instance_id=entry.get("subject_instance_id"),
                        object_instance_id=entry.get("object_instance_id"),
                        reward_value=float(entry.get("reward_value", 0.0)),
                        weight=weight,
                        source="confirmed_info_pair",
                    )
                )
            elif kind == "surprise":
                surprise_samples.append(
                    SubjectEventSurpriseSample(
                        subject_text=entry.get("subject_text"),
                        action_text=entry.get("action_text"),
                        object_text=entry.get("object_text"),
                        subject_instance_id=entry.get("subject_instance_id"),
                        object_instance_id=entry.get("object_instance_id"),
                        surprise_value=float(entry.get("surprise_value", 0.0)),
                        weight=weight,
                        source="confirmed_info_pair",
                    )
                )
            elif kind == "event":
                sample = self._event_surprise_sample_from_entry(
                    entry,
                    surprise_value=float(event_surprise_target),
                    weight=weight,
                )
                if sample is not None:
                    surprise_samples.append(sample)
        return reward_samples, surprise_samples

    def _train_confirmed_value_models(
        self,
        entries,
        *,
        value_model_epochs: int = 1,
        event_surprise_target: float = -50.0,
    ):
        reward_samples, surprise_samples = self._value_model_samples_from_confirmed_entries(
            entries,
            event_surprise_target=event_surprise_target,
        )
        reward_history = []
        surprise_history = []
        if reward_samples:
            _, _, _, reward_trainer = self._emotion_reward_stack(role="subject")
            reward_history = reward_trainer.train_epochs(reward_samples, epochs=int(value_model_epochs))
        if surprise_samples:
            _, _, _, surprise_trainer = self._surprise_stack()
            surprise_history = surprise_trainer.train_epochs(surprise_samples, epochs=int(value_model_epochs))
        return {
            "enabled": True,
            "epoch_count": int(value_model_epochs),
            "reward": {
                "sample_count": len(reward_samples),
                "history": reward_history,
                "last_result": None if not reward_history else reward_history[-1],
            },
            "surprise": {
                "sample_count": len(surprise_samples),
                "history": surprise_history,
                "last_result": None if not surprise_history else surprise_history[-1],
            },
        }

    def train_confirmed_info_pairs(
        self,
        confirmed_items,
        *,
        epochs: Optional[int] = None,
        instance_epochs: Optional[int] = None,
        value_model_epochs: int = 1,
        step_scale: Optional[float] = None,
        event_surprise_target: float = -50.0,
        train_instance_embeddings: bool = True,
        train_value_models: bool = True,
    ):
        """Train from confirmed yes info pairs as a second phase after question().

        The update is split into two targets. Instance embeddings are grouped by
        short-memory instance_id and updated with averaged gradients. Reward/surprise
        model parameters are trained separately as batched value-model updates.
        """
        import torch

        if instance_epochs is None:
            instance_epochs = 5 if epochs is None else int(epochs)

        entries = [self._confirmed_entry_from_item(item) for item in list(confirmed_items or [])]
        for entry in entries:
            entry["memory_kind"] = self._confirmed_kind(entry)

        grouped = {}
        for entry in entries:
            for instance_id in self._confirmed_instance_ids(entry):
                grouped.setdefault(instance_id, []).append(entry)

        empty_result = {
            "command": "train_confirmed_info_pairs",
            "status": "empty",
            "confirmed_pair_count": len(entries),
            "instance_training": {
                "enabled": bool(train_instance_embeddings),
                "epoch_count": int(instance_epochs),
                "instance_count": 0,
                "updates": [],
            },
            "value_model_training": {
                "enabled": bool(train_value_models),
                "epoch_count": int(value_model_epochs),
                "reward": {"sample_count": 0, "history": [], "last_result": None},
                "surprise": {"sample_count": 0, "history": [], "last_result": None},
            },
        }
        if not entries:
            return empty_result

        memory = self.consciousness.short_memory
        reward_encoder, reward_model, _, _ = self._emotion_reward_stack(role="subject")
        surprise_encoder, surprise_model, _, _ = self._surprise_stack()

        instance_result = {
            "enabled": bool(train_instance_embeddings),
            "epoch_count": int(instance_epochs),
            "instance_count": len(grouped),
            "updates": [],
        }

        if train_instance_embeddings and grouped:
            rm, arm, kt = memory._load_language_context()
            frozen_params = list(reward_model.parameters()) + list(surprise_model.parameters())
            previous_requires_grad = [param.requires_grad for param in frozen_params]
            for param in frozen_params:
                param.requires_grad_(False)

            updates = {instance_id: {"instance_id": instance_id, "entry_count": len(items), "loss_history": []} for instance_id, items in grouped.items()}
            scale = memory.relation_step_scale if step_scale is None else float(step_scale)

            try:
                for _ in range(int(instance_epochs)):
                    for instance_id, instance_entries in grouped.items():
                        metadata = memory.get_noun_instance_metadata(instance_id) or {}
                        noun_text = metadata.get("noun_text")
                        if noun_text is None:
                            for entry in instance_entries:
                                noun_text = entry.get("source_text") or entry.get("subject_text") or entry.get("noun_text")
                                if noun_text:
                                    break
                        if noun_text is None:
                            updates[instance_id].setdefault("skipped_reasons", []).append("missing_noun_text")
                            continue

                        noun_type, base_embedding, resolved_instance_id = memory.ensure_noun_instance(str(noun_text), instance_id)
                        trainable_embedding = base_embedding.detach().clone().requires_grad_(True)
                        losses = []
                        weights = []

                        for entry in instance_entries:
                            kind = self._confirmed_kind(entry)
                            loss = None
                            if kind == "relation":
                                loss = self._loss_for_confirmed_relation_entry(
                                    entry,
                                    instance_id,
                                    trainable_embedding,
                                    rm,
                                    arm,
                                    kt,
                                )
                            elif kind == "reward":
                                loss = self._loss_for_confirmed_reward_entry(
                                    entry,
                                    instance_id,
                                    trainable_embedding,
                                    reward_model,
                                    reward_encoder,
                                )
                            elif kind == "surprise":
                                loss = self._loss_for_confirmed_surprise_entry(
                                    entry,
                                    instance_id,
                                    trainable_embedding,
                                    surprise_model,
                                    surprise_encoder,
                                )
                            elif kind == "event":
                                sample = self._event_surprise_sample_from_entry(
                                    entry,
                                    surprise_value=float(event_surprise_target),
                                    weight=self._entry_weight(entry),
                                )
                                if sample is not None:
                                    event_entry = {
                                        "subject_text": sample.subject_text,
                                        "subject_instance_id": sample.subject_instance_id,
                                        "action_text": sample.action_text,
                                        "object_text": sample.object_text,
                                        "object_instance_id": sample.object_instance_id,
                                        "surprise_value": sample.surprise_value,
                                    }
                                    loss = self._loss_for_confirmed_surprise_entry(
                                        event_entry,
                                        instance_id,
                                        trainable_embedding,
                                        surprise_model,
                                        surprise_encoder,
                                    )
                            if loss is not None:
                                losses.append(loss)
                                weights.append(self._entry_weight(entry))

                        if not losses:
                            updates[instance_id].setdefault("skipped_reasons", []).append("no_losses")
                            continue

                        total_loss = self._weighted_mean_torch_losses(losses, weights)
                        total_loss.backward()
                        if trainable_embedding.grad is None:
                            updates[instance_id].setdefault("skipped_reasons", []).append("no_gradient")
                            continue

                        noun_idx = noun_type if noun_type is not None else rm._ensure_noun(str(noun_text).lower())
                        lr = float(rm.lr_per_embedding[int(noun_idx)]) * scale
                        with torch.no_grad():
                            updated_embedding = trainable_embedding - lr * trainable_embedding.grad
                        trainable_embedding.grad.zero_()
                        memory.store_noun_instance(resolved_instance_id, updated_embedding.detach(), noun_text=str(noun_text))
                        updates[instance_id]["loss_history"].append(float(total_loss.item()))
                        updates[instance_id]["final_embedding_norm"] = float(updated_embedding.norm().item())
            finally:
                for param, requires_grad in zip(frozen_params, previous_requires_grad):
                    param.requires_grad_(requires_grad)

            instance_result["updates"] = list(updates.values())
        elif not train_instance_embeddings:
            instance_result["updates"] = []

        if train_value_models:
            value_model_result = self._train_confirmed_value_models(
                entries,
                value_model_epochs=int(value_model_epochs),
                event_surprise_target=float(event_surprise_target),
            )
        else:
            value_model_result = {
                "enabled": False,
                "epoch_count": int(value_model_epochs),
                "reward": {"sample_count": 0, "history": [], "last_result": None},
                "surprise": {"sample_count": 0, "history": [], "last_result": None},
            }

        status = "trained" if (train_instance_embeddings or train_value_models) else "skipped"
        return {
            "command": "train_confirmed_info_pairs",
            "status": status,
            "confirmed_pair_count": len(entries),
            "instance_training": instance_result,
            "value_model_training": value_model_result,
            # Backward-compatible summary fields.
            "epoch_count": int(instance_epochs),
            "instance_count": int(instance_result.get("instance_count", 0)),
            "updates": list(instance_result.get("updates", [])),
        }

    def answer_inf_question(
        self,
        question: dict,
        answer_text: str,
        *,
        save: bool = True,
        event_surprise_target: float = -50.0,
        step_scale: Optional[float] = None,
    ):
        """Apply a yes/no answer to one inf_question item returned by question().

        no deletes the original short-memory info pair. yes only marks the pair as confirmed;
        call train_confirmed_info_pairs() as the second phase to update instance embeddings.
        """
        normalized_answer = str(answer_text).strip().lower()
        if normalized_answer not in {"y", "yes", "n", "no"}:
            raise ValueError("answer_text must be one of: y, yes, n, no")

        entry = dict(question.get("memory_entry", question))
        kind = str(entry.get("memory_kind") or "")
        if not kind:
            pair_kind = str(entry.get("pair_kind") or question.get("kind") or "")
            if pair_kind in {"noun_noun_relation", "adj_noun_relation"}:
                kind = "relation"
            elif pair_kind == "subject_event_reward":
                kind = "reward"
            elif pair_kind == "subject_event_surprise":
                kind = "surprise"
            else:
                kind = "event"
            entry["memory_kind"] = kind

        if normalized_answer in {"n", "no"}:
            return {
                "command": "answer_inf_question",
                "accepted": False,
                "action": "delete_info_pair",
                "kind": kind,
                "question": question,
                "result": self._remove_short_memory_entry(entry),
            }

        return {
            "command": "answer_inf_question",
            "accepted": True,
            "action": "pending_yes_confirmation",
            "kind": kind,
            "question": question,
            "confirmed_yes": [self.inf_question(entry, reason="user_confirmed_yes")],
            "result": {
                "status": "confirmed",
                "training_required": True,
            },
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
        entry = question.get("memory_entry", {})
        source_noun = question.get("source_noun") or entry.get("source_text") or entry.get("subject_text") or entry.get("noun_text")
        target = question.get("target") or entry.get("target_text") or entry.get("object_text") or entry.get("action_text")
        relation_type = question.get("relation_type") or entry.get("relation_type")
        relation_name = question.get("relation_name") or entry.get("relation_name")
        if relation_type is None and kind in {"noun_noun_relation", "adj_noun_relation"} and relation_name is not None:
            relation_type = self.consciousness.resolve_relation_type(kind, relation_name)
        if kind == "noun_noun_relation":
            result = self.consciousness.remember_noun_relation(
                source_noun,
                target,
                int(relation_type),
                save=save,
            )
        elif kind == "adj_noun_relation":
            result = self.consciousness.remember_adj_relation(
                source_noun,
                target,
                int(relation_type),
                save=save,
            )
        elif kind == "noun_action_relation":
            result = self.consciousness.remember_noun_action(
                source_noun,
                target,
                save=save,
            )
        elif kind == "subject_event_reward":
            result = self.train_emotion_reward(epochs=1)
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












