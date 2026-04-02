"""High-level orchestration for retrieval and awareness functions.

Question-driven prediction and relation learning are exposed here as awareness
interfaces, while detailed implementation stays in ``prototype.question``.
"""

from dataclasses import dataclass, field
import random
from typing import Optional

from knowledge.relation_map import relation_map
from .question import AnswerResult, ProposedQuestion, QuestionEngine, TokenWhatResult


@dataclass
class Consciousness:
    ask_confidence_threshold: float = 0.92
    min_ask_confidence_threshold: float = 0.15
    bind_slot_confidence_threshold: float = 0.35
    question: QuestionEngine = field(init=False)

    def __post_init__(self):
        self.question = QuestionEngine(
            ask_confidence_threshold=self.ask_confidence_threshold,
            min_ask_confidence_threshold=self.min_ask_confidence_threshold,
            bind_slot_confidence_threshold=self.bind_slot_confidence_threshold,
        )

    def available_functions(self):
        return {
            "recall": "retrieve stored noun_noun and adj_noun relations from memory maps",
            "predict": "run prediction, decide whether confidence warrants asking, and optionally learn from feedback",
            "thinking": "predict relation targets, evaluate confidence, and surface one question to ask",
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

    def recall(
        self,
        noun: Optional[str] = None,
        adjective: Optional[str] = None,
        relation_type: Optional[int] = None,
    ):
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
        return self.question.sample_question(rng or random.Random())

    def what(
        self,
        token: str,
        *,
        position: Optional[int] = None,
        tokens: Optional[list[str]] = None,
    ) -> TokenWhatResult:
        return self.question.what_is_token(token, position=position, tokens=tokens)

    def learn_noun_relation(self, source_noun: str, target_noun: str, relation_type: int, save: bool = True):
        return self.question.direct_learn_noun_relation(
            source_noun,
            target_noun,
            relation_type,
            save=save,
        )

    def learn_adj_relation(self, noun: str, adjective: str, relation_type: int, save: bool = True):
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
