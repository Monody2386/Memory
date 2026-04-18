from __future__ import annotations

from dataclasses import dataclass
import importlib
import os
import random
from typing import List, Literal, Optional, Sequence

import torch
import torch.nn.functional as F

QuestionKind = Literal["adj_noun", "noun_noun"]
NounQuestionTarget = Literal["noun", "relation"]
TokenPos = Literal["noun", "pronoun", "possessive", "possessive_noun", "article", "adj", "be", "helper", "negative", "action", "actioned", "reward", "relation", "unknown"]


@dataclass
class ProposedQuestion:
    kind: QuestionKind
    prompt: str
    confidence: float
    source_noun: str
    predicted_target: str
    relation_type: int
    relation_name: str
    target: str
    target_noun: Optional[str] = None
    adjective: Optional[str] = None
    question_target: Optional[NounQuestionTarget] = None
    top_candidates: Optional[List[str]] = None
    top_scores: Optional[List[float]] = None


@dataclass
class AnswerResult:
    accepted: bool
    learned: bool
    kind: QuestionKind
    message: str
    relation_type: int
    source_noun: str
    target_value: Optional[str] = None
    loss: Optional[float] = None


@dataclass
class TokenWhatResult:
    token: str
    predicted_pos: TokenPos
    status: str
    prompt: str
    candidates: List[str]
    source: str


class QuestionEngine:
    def __init__(
        self,
        ask_confidence_threshold: float = 0.92,
        min_ask_confidence_threshold: float = 0.15,
        bind_slot_confidence_threshold: float = 0.35,
    ):
        self.ask_confidence_threshold = float(ask_confidence_threshold)
        self.min_ask_confidence_threshold = float(min_ask_confidence_threshold)
        self.bind_slot_confidence_threshold = float(bind_slot_confidence_threshold)
        if self.min_ask_confidence_threshold >= self.ask_confidence_threshold:
            raise ValueError(
                "min_ask_confidence_threshold must be smaller than ask_confidence_threshold"
            )

    def _ctx(self):
        rm = importlib.import_module("knowledge.relation_map")
        arm = importlib.import_module("knowledge.adj_relation_map")
        kt = importlib.import_module("knowledge.training")

        rm.load_relation_data()
        arm.load_adj_relation_data()
        if os.path.exists(kt.MODEL_PATH):
            kt.knowledge_map_one.load_state_dict(torch.load(kt.MODEL_PATH, map_location="cpu"))
        if os.path.exists(kt.ADJ_MODEL_PATH):
            kt.adj_map_one.load_state_dict(
                torch.load(kt.ADJ_MODEL_PATH, map_location="cpu"),
                strict=False,
            )
        return rm, arm, kt

    def _extract_predicted_index(self, token: str, prefix: str) -> Optional[int]:
        if token.startswith(prefix):
            suffix = token[len(prefix):]
            if suffix.isdigit():
                return int(suffix)
        return None

    def _ensure_noun_registered(self, rm, noun: str) -> int:
        return rm._ensure_noun(noun.lower())

    def _ensure_adjective_registered(self, arm, adjective: str) -> int:
        return arm._ensure_adjective(adjective.lower())

    def _bind_or_register_noun(self, rm, noun: str, question: Optional[ProposedQuestion] = None) -> int:
        noun = noun.lower()
        if noun in rm.noun_list:
            return rm.noun_list.index(noun)
        if question is not None and question.question_target == "noun":
            predicted_idx = self._extract_predicted_index(question.predicted_target, "noun_")
            if (
                predicted_idx is not None
                and float(question.confidence) >= self.bind_slot_confidence_threshold
                and not rm.is_defined_noun_index(predicted_idx)
            ):
                return rm.bind_noun_to_index(noun, predicted_idx)
        return rm._ensure_noun(noun)

    def _bind_or_register_adjective(self, arm, adjective: str, question: Optional[ProposedQuestion] = None) -> int:
        adjective = adjective.lower()
        if adjective in arm.adjective_list:
            return arm.adjective_list.index(adjective)
        if question is not None and question.kind == "adj_noun":
            predicted_idx = self._extract_predicted_index(question.predicted_target, "adj_")
            if (
                predicted_idx is not None
                and float(question.confidence) >= self.bind_slot_confidence_threshold
                and not arm.is_defined_adjective_index(predicted_idx)
            ):
                return arm.bind_adjective_to_index(adjective, predicted_idx)
        return arm._ensure_adjective(adjective)

    def _token_lexicons(self):
        rm, arm, _ = self._ctx()
        action_vocab = importlib.import_module("world.action_vocab")
        relation_tokens = set()
        for relation in rm.relation_list:
            relation_tokens.update(relation.lower().split())
        grammar = importlib.import_module("grammar_layer")
        adjective_hints = set(grammar.ADJECTIVE_RELATION_HINTS.keys())
        return {
            "nouns": {noun.lower() for noun in rm.noun_list},
            "pronouns": set(grammar.PRONOUN_LIST),
            "possessives": set(grammar.POSSESSIVE_LIST),
            "possessive_nouns": set(grammar.POSSESSIVE_NOUN_LIST),
            "articles": set(grammar.ARTICLE_LIST),
            "helpers": set(grammar.HELPER_WORD_SET),
            "negative_words": set(grammar.NEGATIVE_WORD_SET),
            "adjectives": set(arm.adjective_list) | adjective_hints,
            "actions": {action.lower() for action in action_vocab.action_list},
            "actioned_words": {grammar.object_action_form(action.lower()) for action in action_vocab.action_list}
            | set(grammar.ACTIONED_TO_ACTIVE.keys()),
            "be_verbs": {"am", "is", "are", "was", "were", "be", "been", "being"},
            "relations": {relation.lower() for relation in rm.relation_list},
            "relation_tokens": relation_tokens,
            "rm": rm,
            "arm": arm,
        }

    def what_is_token(
        self,
        token: str,
        *,
        position: Optional[int] = None,
        tokens: Optional[Sequence[str]] = None,
    ) -> TokenWhatResult:
        token = token.lower()
        lexicons = self._token_lexicons()

        if token in lexicons["nouns"]:
            return TokenWhatResult(
                token=token,
                predicted_pos="noun",
                status="known",
                prompt=f"'{token}' is already known as a noun.",
                candidates=["noun"],
                source="noun_list",
            )
        if token in lexicons["possessive_nouns"]:
            return TokenWhatResult(
                token=token,
                predicted_pos="possessive_noun",
                status="known",
                prompt=f"'{token}' is already known as a standalone possessive noun.",
                candidates=["possessive_noun"],
                source="possessive_noun_list",
            )
        if token in lexicons["possessives"]:
            return TokenWhatResult(
                token=token,
                predicted_pos="possessive",
                status="known",
                prompt=f"'{token}' is already known as a possessive marker.",
                candidates=["possessive"],
                source="possessive_list",
            )
        if token in lexicons["articles"]:
            return TokenWhatResult(
                token=token,
                predicted_pos="article",
                status="known",
                prompt=f"'{token}' is already known as an article.",
                candidates=["article"],
                source="article_list",
            )
        if token in lexicons["pronouns"]:
            return TokenWhatResult(
                token=token,
                predicted_pos="pronoun",
                status="known",
                prompt=f"'{token}' is already known as a pronoun.",
                candidates=["pronoun"],
                source="pronoun_list",
            )
        if token in lexicons["helpers"]:
            return TokenWhatResult(
                token=token,
                predicted_pos="helper",
                status="known",
                prompt=f"'{token}' is already known as a grammar helper.",
                candidates=["helper"],
                source="helper_word_list",
            )
        if token in lexicons["negative_words"]:
            return TokenWhatResult(
                token=token,
                predicted_pos="negative",
                status="known",
                prompt=f"'{token}' is already known as a negation word.",
                candidates=["negative"],
                source="negative_word_list",
            )
        if token in lexicons["adjectives"]:
            return TokenWhatResult(
                token=token,
                predicted_pos="adj",
                status="known",
                prompt=f"'{token}' is already known as an adjective.",
                candidates=["adj"],
                source="adjective_list",
            )
        if token in lexicons["be_verbs"]:
            return TokenWhatResult(
                token=token,
                predicted_pos="be",
                status="known",
                prompt=f"'{token}' is already known as a be-verb.",
                candidates=["be"],
                source="be_verb_list",
            )
        if token in lexicons["actioned_words"]:
            return TokenWhatResult(
                token=token,
                predicted_pos="actioned",
                status="known",
                prompt=f"'{token}' is already known as an actioned/passive action form.",
                candidates=["actioned"],
                source="actioned_form_list",
            )
        if token in lexicons["actions"]:
            return TokenWhatResult(
                token=token,
                predicted_pos="action",
                status="known",
                prompt=f"'{token}' is already known as an action.",
                candidates=["action"],
                source="action_list",
            )
        if token in lexicons["relations"] or token in lexicons["relation_tokens"]:
            return TokenWhatResult(
                token=token,
                predicted_pos="relation",
                status="known",
                prompt=f"'{token}' is already known as a relation token.",
                candidates=["relation"],
                source="relation_list",
            )

        guessed_pos: TokenPos = "noun"
        source = "default_guess"
        if position == 1:
            guessed_pos = "action"
            source = "position_heuristic"
        if tokens is not None:
            lowered_tokens = [item.lower() for item in tokens]
            joined = " ".join(lowered_tokens)
            for relation in lexicons["relations"]:
                if relation in joined and token in relation.split():
                    guessed_pos = "relation"
                    source = "relation_phrase_heuristic"
                    break

        prompt = (
            f"I do not know the token '{token}'. What is it in this sentence: "
            f"noun, pronoun, possessive, possessive_noun, article, adj, be, helper, negative, action, actioned, reward, or relation? Current best guess: {guessed_pos}."
        )
        return TokenWhatResult(
            token=token,
            predicted_pos=guessed_pos,
            status="question",
            prompt=prompt,
            candidates=["noun", "pronoun", "possessive", "possessive_noun", "article", "adj", "be", "helper", "negative", "action", "actioned", "reward", "relation"],
            source=source,
        )

    def register_token_pos(self, token: str, pos: TokenPos, save: bool = True) -> dict:
        token = token.lower()
        if pos == "unknown":
            raise ValueError("Cannot register token with pos='unknown'")

        rm, arm, kt = self._ctx()
        created = False
        if pos == "noun":
            if token not in rm.noun_list:
                rm._ensure_noun(token)
                created = True
        elif pos == "pronoun":
            created = False
        elif pos == "possessive":
            created = False
        elif pos == "possessive_noun":
            created = False
        elif pos == "article":
            created = False
        elif pos == "adj":
            if token not in arm.adjective_list:
                arm._ensure_adjective(token)
                created = True
        elif pos == "be":
            created = False
        elif pos == "helper":
            created = False
        elif pos == "negative":
            created = False
        elif pos == "action":
            action_vocab = importlib.import_module("world.action_vocab")
            if token not in action_vocab.action_list:
                action_vocab.ensure_action(token)
                created = True
        elif pos == "actioned":
            created = False
        elif pos == "reward":
            created = False
        elif pos == "relation":
            if token not in rm.relation_list:
                if len(rm.relation_list) >= rm.relation_num:
                    raise ValueError(
                        f"relation_list is full; cannot register new relation '{token}'"
                    )
                rm.relation_list.append(token)
                created = True
        else:
            raise ValueError("pos must be one of: noun, pronoun, possessive, possessive_noun, article, adj, be, helper, negative, action, actioned, reward, relation")

        if save:
            rm.save_relation_data()
            arm.save_adj_relation_data()
            torch.save(kt.knowledge_map_one.state_dict(), kt.MODEL_PATH)
            torch.save(kt.adj_map_one.state_dict(), kt.ADJ_MODEL_PATH)

        return {
            "token": token,
            "pos": pos,
            "created": created,
        }

    def predict_adjective(self, noun: str, relation_type: int, top_k: int = 3) -> ProposedQuestion:
        rm, arm, kt = self._ctx()
        noun_idx = self._ensure_noun_registered(rm, noun)
        noun_tensor = torch.tensor(noun_idx, dtype=torch.long)
        rel_idx = int(relation_type) - 1
        if rel_idx < 0 or rel_idx >= len(kt.adj_map_one.relations):
            raise ValueError(f"relation_type must be in [1, {len(kt.adj_map_one.relations)}]")

        with torch.no_grad():
            noun_embedding = kt.knowledge_map_one.embedding(noun_tensor)
            predicted_adj = kt.adj_map_one.relations[rel_idx](noun_embedding)
            top_indices, top_scores = kt.adj_map_one.query_adjective_similarity(predicted_adj, top_k=top_k)

        best_idx = int(top_indices[0].item())
        best_adj = arm.adjective_list[best_idx] if best_idx < len(arm.adjective_list) else f"adj_{best_idx}"
        relation_name = arm.adj_relation_list[rel_idx]
        candidates = [
            arm.adjective_list[int(idx)] if int(idx) < len(arm.adjective_list) else f"adj_{int(idx)}"
            for idx in top_indices.tolist()
        ]
        scores = [float(score.item()) for score in top_scores]
        prompt = f"When focusing on {relation_name}, is the {relation_name} adjective of {noun.lower()} '{best_adj}'?"
        return ProposedQuestion(
            kind="adj_noun",
            prompt=prompt,
            confidence=scores[0],
            source_noun=noun.lower(),
            predicted_target=best_adj,
            relation_type=int(relation_type),
            relation_name=relation_name,
            target="adjective",
            adjective=best_adj,
            top_candidates=candidates,
            top_scores=scores,
        )

    def predict_noun_from_relation(self, noun: str, relation_type: int, top_k: int = 3) -> ProposedQuestion:
        rm, _, kt = self._ctx()
        noun_idx = self._ensure_noun_registered(rm, noun)
        noun_tensor = torch.tensor(noun_idx, dtype=torch.long)
        rel_idx = int(relation_type) - 1
        if rel_idx < 0 or rel_idx >= len(kt.knowledge_map_one.relations):
            raise ValueError(f"relation_type must be in [1, {len(kt.knowledge_map_one.relations)}]")

        with torch.no_grad():
            noun_embedding = kt.knowledge_map_one.embedding(noun_tensor)
            predicted_target = kt.knowledge_map_one.relations[rel_idx](noun_embedding)
            top_indices, top_scores = kt.knowledge_map_one.query_similarity(predicted_target, top_k=top_k)

        best_idx = int(top_indices[0].item())
        best_noun = rm.noun_list[best_idx] if best_idx < len(rm.noun_list) else f"noun_{best_idx}"
        relation_name = rm.relation_list[rel_idx] if rel_idx < len(rm.relation_list) else f"relation_{relation_type}"
        candidates = [
            rm.noun_list[int(idx)] if int(idx) < len(rm.noun_list) else f"noun_{int(idx)}"
            for idx in top_indices.tolist()
        ]
        scores = [float(score.item()) for score in top_scores]
        prompt = f"Does {noun.lower()} {relation_name} {best_noun}?"
        return ProposedQuestion(
            kind="noun_noun",
            prompt=prompt,
            confidence=scores[0],
            source_noun=noun.lower(),
            predicted_target=best_noun,
            relation_type=int(relation_type),
            relation_name=relation_name,
            target="noun",
            target_noun=best_noun,
            question_target="noun",
            top_candidates=candidates,
            top_scores=scores,
        )

    def predict_relation_between_nouns(self, source_noun: str, target_noun: str) -> ProposedQuestion:
        rm, _, kt = self._ctx()
        source_idx = self._ensure_noun_registered(rm, source_noun)
        target_idx = self._ensure_noun_registered(rm, target_noun)
        source_tensor = torch.tensor(source_idx, dtype=torch.long)
        target_tensor = torch.tensor(target_idx, dtype=torch.long)

        with torch.no_grad():
            source_embedding = kt.knowledge_map_one.embedding(source_tensor)
            target_embedding = kt.knowledge_map_one.embedding(target_tensor)
            relation_scores = []
            for relation_type, relation_layer in enumerate(kt.knowledge_map_one.relations, start=1):
                pred_target = relation_layer(source_embedding)
                score = F.cosine_similarity(
                    pred_target.unsqueeze(0),
                    target_embedding.unsqueeze(0),
                    dim=1,
                ).item()
                relation_scores.append((relation_type, float(score)))

        relation_scores.sort(key=lambda item: item[1], reverse=True)
        best_relation_type, best_score = relation_scores[0]
        rel_idx = best_relation_type - 1
        relation_name = rm.relation_list[rel_idx] if rel_idx < len(rm.relation_list) else f"relation_{best_relation_type}"
        prompt = f"Is the relation between {source_noun.lower()} and {target_noun.lower()} '{relation_name}'?"
        return ProposedQuestion(
            kind="noun_noun",
            prompt=prompt,
            confidence=best_score,
            source_noun=source_noun.lower(),
            predicted_target=relation_name,
            relation_type=int(best_relation_type),
            relation_name=relation_name,
            target="relation",
            target_noun=target_noun.lower(),
            question_target="relation",
            top_candidates=[
                rm.relation_list[item[0] - 1] if item[0] - 1 < len(rm.relation_list) else f"relation_{item[0]}"
                for item in relation_scores[:3]
            ],
            top_scores=[item[1] for item in relation_scores[:3]],
        )

    def should_ask(self, question: ProposedQuestion) -> bool:
        confidence = float(question.confidence)
        return self.min_ask_confidence_threshold < confidence < self.ask_confidence_threshold

    def propose_adj_question(self, noun: str, relation_type: int, top_k: int = 3) -> Optional[ProposedQuestion]:
        question = self.predict_adjective(noun, relation_type, top_k=top_k)
        return question if self.should_ask(question) else None

    def propose_noun_question(self, noun: str, relation_type: int, top_k: int = 3) -> Optional[ProposedQuestion]:
        question = self.predict_noun_from_relation(noun, relation_type, top_k=top_k)
        return question if self.should_ask(question) else None

    def propose_relation_question(self, source_noun: str, target_noun: str) -> Optional[ProposedQuestion]:
        question = self.predict_relation_between_nouns(source_noun, target_noun)
        return question if self.should_ask(question) else None

    def sample_question(self, rng: random.Random) -> Optional[ProposedQuestion]:
        rm, arm, _ = self._ctx()
        noun_indices = list(range(len(rm.noun_list)))
        rng.shuffle(noun_indices)

        adj_relation_types = list(range(1, len(arm.adj_relation_list) + 1))
        rng.shuffle(adj_relation_types)
        noun_relation_types = list(range(1, len(rm.relation_list) + 1))
        rng.shuffle(noun_relation_types)

        for noun_idx in noun_indices:
            noun = rm.noun_list[noun_idx]
            if noun.startswith("noun_"):
                continue
            for relation_type in adj_relation_types:
                question = self.propose_adj_question(noun, relation_type)
                if question is not None:
                    return question
            for relation_type in noun_relation_types:
                question = self.propose_noun_question(noun, relation_type)
                if question is not None:
                    return question

        source_indices = noun_indices[:]
        target_indices = noun_indices[:]
        rng.shuffle(source_indices)
        rng.shuffle(target_indices)
        for source_idx in source_indices:
            source_noun = rm.noun_list[source_idx]
            if source_noun.startswith("noun_"):
                continue
            for target_idx in target_indices:
                if target_idx == source_idx:
                    continue
                target_noun = rm.noun_list[target_idx]
                if target_noun.startswith("noun_"):
                    continue
                question = self.propose_relation_question(source_noun, target_noun)
                if question is not None:
                    return question
        return None

    def _save(self, rm, arm, kt):
        rm.save_relation_data()
        arm.save_adj_relation_data()
        torch.save(kt.knowledge_map_one.state_dict(), kt.MODEL_PATH)
        torch.save(kt.adj_map_one.state_dict(), kt.ADJ_MODEL_PATH)

    def direct_learn_noun_relation(self, source_noun: str, target_noun: str, relation_type: int, save: bool = True):
        rm, arm, kt = self._ctx()
        i_idx = self._ensure_noun_registered(rm, source_noun)
        j_idx = self._bind_or_register_noun(rm, target_noun)
        _, i_idx, j_idx, relation_type = rm.add_relation_by_type(source_noun, target_noun, relation_type)
        loss = kt.train_random(
            kt.knowledge_map_one,
            i_idx,
            j_idx,
            relation_type,
            rm.lr_per_embedding,
            rm.lr_relation,
        )
        if save:
            self._save(rm, arm, kt)
        return {
            "mode": "learning",
            "kind": "noun_noun",
            "source_noun": source_noun.lower(),
            "target_noun": target_noun.lower(),
            "relation_type": int(relation_type),
            "loss": float(loss),
        }

    def direct_learn_adj_relation(self, noun: str, adjective: str, relation_type: int, save: bool = True):
        rm, arm, kt = self._ctx()
        noun_idx = self._ensure_noun_registered(rm, noun)
        adjective_idx = self._bind_or_register_adjective(arm, adjective)
        _, noun_idx, adjective_idx, relation_type = arm.add_adj_relation_by_type(noun, adjective, relation_type)
        loss = kt.train_adj_random(
            kt.adj_map_one,
            noun_idx,
            adjective_idx,
            relation_type,
            rm.lr_per_embedding,
            arm.lr_per_adjective,
            arm.lr_adj_relation,
        )
        if save:
            self._save(rm, arm, kt)
        return {
            "mode": "learning",
            "kind": "adj_noun",
            "noun": noun.lower(),
            "adjective": adjective.lower(),
            "relation_type": int(relation_type),
            "loss": float(loss),
        }

    def apply_adj_answer(
        self,
        question: ProposedQuestion,
        answer_text: str,
        corrected_adjective: Optional[str] = None,
        save: bool = True,
    ) -> AnswerResult:
        rm, arm, kt = self._ctx()
        answer = answer_text.strip().lower()
        if answer not in {"yes", "no"}:
            raise ValueError("answer_text must be 'yes' or 'no'")

        adjective = (question.adjective or question.predicted_target) if answer == "yes" else corrected_adjective
        if not adjective:
            return AnswerResult(
                accepted=False,
                learned=False,
                kind="adj_noun",
                message="Prediction rejected; provide corrected_adjective to learn from a 'no' answer.",
                relation_type=question.relation_type,
                source_noun=question.source_noun,
                target_value=question.predicted_target,
            )
        adjective = adjective.lower()

        noun_idx = self._ensure_noun_registered(rm, question.source_noun)
        adjective_idx = self._bind_or_register_adjective(arm, adjective, question=question)
        _, noun_idx, adjective_idx, relation_type = arm.add_adj_relation_by_type(
            question.source_noun,
            adjective,
            question.relation_type,
        )
        loss = kt.train_adj_random(
            kt.adj_map_one,
            noun_idx,
            adjective_idx,
            relation_type,
            rm.lr_per_embedding,
            arm.lr_per_adjective,
            arm.lr_adj_relation,
        )
        if save:
            self._save(rm, arm, kt)
        return AnswerResult(
            accepted=answer == "yes",
            learned=True,
            kind="adj_noun",
            message=f"Learned adjective '{adjective}' for {question.source_noun} under relation {question.relation_name}.",
            relation_type=question.relation_type,
            source_noun=question.source_noun,
            target_value=adjective,
            loss=float(loss),
        )

    def apply_noun_answer(
        self,
        question: ProposedQuestion,
        answer_text: str,
        corrected_target_noun: Optional[str] = None,
        save: bool = True,
    ) -> AnswerResult:
        rm, arm, kt = self._ctx()
        answer = answer_text.strip().lower()
        if answer not in {"yes", "no"}:
            raise ValueError("answer_text must be 'yes' or 'no'")
        if question.question_target != "noun":
            raise ValueError("apply_noun_answer expects a noun-target question")

        target_noun = (question.target_noun or question.predicted_target) if answer == "yes" else corrected_target_noun
        if not target_noun:
            return AnswerResult(
                accepted=False,
                learned=False,
                kind="noun_noun",
                message="Prediction rejected; provide corrected_target_noun to learn from a 'no' answer.",
                relation_type=question.relation_type,
                source_noun=question.source_noun,
                target_value=question.predicted_target,
            )
        target_noun = target_noun.lower()

        i_idx = self._ensure_noun_registered(rm, question.source_noun)
        j_idx = self._bind_or_register_noun(rm, target_noun, question=question)
        _, i_idx, j_idx, relation_type = rm.add_relation_by_type(
            question.source_noun,
            target_noun,
            question.relation_type,
        )
        loss = kt.train_random(
            kt.knowledge_map_one,
            i_idx,
            j_idx,
            relation_type,
            rm.lr_per_embedding,
            rm.lr_relation,
        )
        if save:
            self._save(rm, arm, kt)
        return AnswerResult(
            accepted=answer == "yes",
            learned=True,
            kind="noun_noun",
            message=f"Learned that {question.source_noun} {question.relation_name} {target_noun}.",
            relation_type=question.relation_type,
            source_noun=question.source_noun,
            target_value=target_noun,
            loss=float(loss),
        )

    def apply_relation_answer(
        self,
        question: ProposedQuestion,
        answer_text: str,
        corrected_relation_type: Optional[int] = None,
        save: bool = True,
    ) -> AnswerResult:
        rm, arm, kt = self._ctx()
        answer = answer_text.strip().lower()
        if answer not in {"yes", "no"}:
            raise ValueError("answer_text must be 'yes' or 'no'")
        if question.question_target != "relation":
            raise ValueError("apply_relation_answer expects a relation-target question")

        relation_type = question.relation_type if answer == "yes" else corrected_relation_type
        if relation_type is None:
            return AnswerResult(
                accepted=False,
                learned=False,
                kind="noun_noun",
                message="Prediction rejected; provide corrected_relation_type to learn from a 'no' answer.",
                relation_type=question.relation_type,
                source_noun=question.source_noun,
                target_value=question.target_noun,
            )

        i_idx = self._ensure_noun_registered(rm, question.source_noun)
        j_idx = self._ensure_noun_registered(rm, question.target_noun)
        _, i_idx, j_idx, learned_relation_type = rm.add_relation_by_type(
            question.source_noun,
            question.target_noun,
            int(relation_type),
        )
        loss = kt.train_random(
            kt.knowledge_map_one,
            i_idx,
            j_idx,
            learned_relation_type,
            rm.lr_per_embedding,
            rm.lr_relation,
        )
        if save:
            self._save(rm, arm, kt)
        relation_name = rm.relation_list[learned_relation_type - 1] if learned_relation_type - 1 < len(rm.relation_list) else f"relation_{learned_relation_type}"
        return AnswerResult(
            accepted=answer == "yes",
            learned=True,
            kind="noun_noun",
            message=f"Learned relation '{relation_name}' between {question.source_noun} and {question.target_noun}.",
            relation_type=int(learned_relation_type),
            source_noun=question.source_noun,
            target_value=question.target_noun,
            loss=float(loss),
        )


__all__ = [
    "AnswerResult",
    "ProposedQuestion",
    "QuestionEngine",
    "TokenWhatResult",
]
