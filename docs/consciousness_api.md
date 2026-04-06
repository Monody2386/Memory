# Consciousness API

This document summarizes the current high-level interfaces exposed by `prototype/consciousness.py`.

## Overview

`Consciousness` is the high-level control layer that connects:
- long-term knowledge recall and question-driven learning
- short-memory observation and inspection
- short-memory update operations
- world-model prediction, training, and rollout

The interfaces below are intended to be the main command-level operations for interacting with memory and reasoning.

## Construction

### `Consciousness(...)`
Purpose:
Create a high-level controller with its own `QuestionEngine`, `ShortMemory`, `WorldModel`, and optimizer.

Input:
- `ask_confidence_threshold: float`
- `min_ask_confidence_threshold: float`
- `bind_slot_confidence_threshold: float`
- `short_memory_maxlen: int`
- `short_memory_device: str`

Output:
- A `Consciousness` instance.

## Interface Summary

### `available_functions()`
Purpose:
Return a summary of the currently exposed high-level command interfaces.

Input:
- None.

Output:
- `dict[str, str]`: interface name to short description.

## Observation Interfaces

### `observe(sentence, *, time_position=None, base_score=1.0, adjective_relation_types=None)`
Purpose:
Parse one sentence and write its relation/event groups into short memory.

Input:
- `sentence: str`: raw sentence text.
- `time_position: Optional[int]`: explicit time step. If omitted, append after current memory.
- `base_score: float`: initial attention score for created memory entries.
- `adjective_relation_types`: optional adjective-to-relation mapping used by grammar.

Output:
- `dict` with:
  - `sentence`
  - `sentence_type`
  - `structure`
  - `tokens`
  - `action_count`
  - `relation_count`
  - `event_entries_added`
  - `relation_entries_added`
  - `states`

### `observe_many(sentences, *, start_time_position=None, base_score=1.0, adjective_relation_types=None)`
Purpose:
Parse and store multiple sentences in sequence.

Input:
- `sentences: Sequence[str]`: ordered sentence list.
- `start_time_position: Optional[int]`: first time step. If omitted, use the next free step.
- `base_score: float`: initial attention score for created entries.
- `adjective_relation_types: Optional[Sequence[Any]]`: per-sentence overrides.

Output:
- `dict` with:
  - `sentence_count`
  - `start_time_position`
  - `event_count`
  - `relation_count`
  - `states`

## Memory Inspection Interfaces

### `inspect_memory(kind='all', order_by='time')`
Purpose:
Inspect current short-memory content.

Input:
- `kind: str`: `all`, `event`, or `relation`.
- `order_by: str`: `time` or `attention`.

Output:
- `dict` or `list`: the requested memory view.

### `inspect_focus(steps=None)`
Purpose:
Return the current focus event used by the world model.

Input:
- `steps: Optional[int]`: optional event window size.

Output:
- `dict | None`: focused event summary, or `None` if memory is empty.

### `inspect_instance(instance_id)`
Purpose:
Inspect one noun instance inside short memory.

Input:
- `instance_id: str`: noun instance identifier.

Output:
- `dict` with:
  - `instance_id`
  - `exists`
  - `embedding_norm`
  - `event_count`
  - `relation_count`
  - `events`
  - `relations`

### `inspect_relation_clone(relation_kind, relation_name)`
Purpose:
Inspect one short-memory relation clone.

Input:
- `relation_kind: str`: relation family.
- `relation_name: str`: relation label.

Output:
- `dict` with:
  - `relation_kind`
  - `relation_name`
  - `exists`
  - `shape`
  - `norm`
  - `entry_count`
  - `entries`

### `build_world_input(steps=None)`
Purpose:
Build the current world-model input tensor from short-memory events.

Input:
- `steps: Optional[int]`: optional event window size.

Output:
- `dict` with:
  - `shape`
  - `tensor`

### `inspect_vocab()`
Purpose:
Inspect the currently defined long-term vocabulary.

Input:
- None.

Output:
- `dict` with:
  - `noun_list`
  - `adj_list`
  - `relation_list`

## Short-Memory Update Interfaces

### `rebuild_instance(instance_id, step_scale=None)`
Purpose:
Rebuild one noun instance embedding from current short-memory relations.

Input:
- `instance_id: str`: noun instance identifier.
- `step_scale: Optional[float]`: optional scaling factor.

Output:
- `dict` with:
  - `instance_id`
  - `updated`
  - `embedding_norm`

### `update_relation_clone(relation_kind, relation_name, *, step_scale=None)`
Purpose:
Explicitly update one short-memory relation clone.

Input:
- `relation_kind: str`
- `relation_name: str`
- `step_scale: Optional[float]`

Output:
- `dict` with:
  - `relation_kind`
  - `relation_name`
  - `updated`
  - `norm`

### `update_all_relation_clones(relation_kind=None, *, step_scale=None)`
Purpose:
Explicitly update every short-memory relation clone, optionally filtered by relation kind.

Input:
- `relation_kind: Optional[str]`
- `step_scale: Optional[float]`

Output:
- `list[dict]`: one update summary per relation clone.

## World-Model Interfaces

### `predict_next_event(action_type, *, steps=None, score=0.5)`
Purpose:
Predict the next event from current short-memory events.

Input:
- `action_type: int`: action selector used by the world model.
- `steps: Optional[int]`: optional event window size.
- `score: float`: score assigned to the predicted event.

Output:
- `dict` containing world-model prediction data, including:
  - `pred_action`
  - `pred_action_type`
  - focus information
  - `predicted_event`
  - `predicted_event_dict`

### `train_next_event(action_type, *, target_action_type=None, target_action_embedding=None, steps=None)`
Purpose:
Run one world-model training step from current short memory.

Input:
- `action_type: int`
- `target_action_type: Optional[int]`
- `target_action_embedding: Optional[Tensor]`
- `steps: Optional[int]`

Output:
- `dict` with:
  - `loss`
  - `pred_action`
  - `pred_action_type`
  - focus information

### `evaluate_next_event(action_type, *, target_action_type, steps=None, score=0.5)`
Purpose:
Evaluate next-event prediction against a target action without training.

Input:
- `action_type: int`
- `target_action_type: int`
- `steps: Optional[int]`
- `score: float`

Output:
- `dict` with:
  - `loss`
  - `pred_action_type`
  - `top_indices`
  - `top_scores`
  - `predicted_event`

### `append_predicted_event(predicted_event, **kwargs)`
Purpose:
Write a predicted event back into short memory.

Input:
- `predicted_event`: predicted event dict or `PredictedEvent` object.
- `**kwargs`: optional writeback fields passed through to world-model writeback.

Output:
- `dict`: stored event data after writeback.

### `rollout(action_type, *, steps=1, score=0.5)`
Purpose:
Run multi-step autoregressive next-event prediction.

Input:
- `action_type: int`: initial action selector.
- `steps: int`: rollout step count.
- `score: float`: score assigned to each predicted event.

Output:
- `list[dict]`: predicted event summaries in rollout order.

## Long-Term Language Learning Interfaces

### `learn_from_sentence(sentence, *, noun_relation_type=None, adjective_relation_types=None, infer_missing=False, save=True)`
Purpose:
Run the long-term language-learning path on one sentence.

Input:
- `sentence: str`
- `noun_relation_type: Optional[int]`
- `adjective_relation_types`
- `infer_missing: bool`
- `save: bool`

Output:
- `dict` with:
  - `sentence`
  - `samples`
  - `results`

### `learn_noun_relation(source_noun, target_noun, relation_type, save=True)`
Purpose:
Directly learn one noun-noun relation in long-term knowledge.

Input:
- `source_noun: str`
- `target_noun: str`
- `relation_type: int`
- `save: bool`

Output:
- `dict`: learning summary with relation type and loss.

### `learn_adj_relation(noun, adjective, relation_type, save=True)`
Purpose:
Directly learn one adj-noun relation in long-term knowledge.

Input:
- `noun: str`
- `adjective: str`
- `relation_type: int`
- `save: bool`

Output:
- `dict`: learning summary with relation type and loss.

## Knowledge Recall and Question Interfaces

### `recall(noun=None, adjective=None, relation_type=None)`
Purpose:
Recall stored long-term noun-noun and adj-noun relations.

Input:
- `noun: Optional[str]`
- `adjective: Optional[str]`
- `relation_type: Optional[int]`

Output:
- `dict` with:
  - `noun_noun`
  - `adj_noun`

### `predict(kind='sample', *, noun=None, relation_type=None, source_noun=None, target_noun=None, answer_text=None, corrected_target=None, corrected_relation_type=None, rng=None, save=True)`
Purpose:
Run high-level knowledge prediction and optional feedback learning.

Input:
- `kind: str`: `sample`, `adj_noun`, `noun_noun`, or `relation`.
- Additional arguments depend on `kind`.
- `answer_text`, `corrected_target`, `corrected_relation_type`: optional feedback payload.
- `rng: Optional[random.Random]`
- `save: bool`

Output:
- `dict`: status plus prediction/question/result payload.

### `think(rng=None)`
Purpose:
Sample one uncertainty-driven question from the knowledge layer.

Input:
- `rng: Optional[random.Random]`

Output:
- `ProposedQuestion | None`

### `what(token, *, position=None, tokens=None)`
Purpose:
Inspect whether a token is known and what part-of-speech it likely is.

Input:
- `token: str`
- `position: Optional[int]`
- `tokens: Optional[list[str]]`

Output:
- `TokenWhatResult`

### `re_predict_question(question)`
Purpose:
Re-run the appropriate predictor for an existing proposed question.

Input:
- `question: ProposedQuestion`

Output:
- `ProposedQuestion`

### `answer_question(question, answer_text, corrected_target=None, corrected_relation_type=None, save=True)`
Purpose:
Apply a yes/no answer to a proposed question and learn from the feedback.

Input:
- `question: ProposedQuestion`
- `answer_text: str`
- `corrected_target: Optional[str]`
- `corrected_relation_type: Optional[int]`
- `save: bool`

Output:
- `AnswerResult`

## Notes

- `observe(...)` and `observe_many(...)` operate on short memory and may synchronize the world model action vocabulary.
- `update_relation_clone(...)` and `update_all_relation_clones(...)` only affect short-memory relation clones, not long-term knowledge parameters.
- `rebuild_instance(...)` updates noun instance embeddings in short memory, not concept embeddings in long-term knowledge.
- `predict_next_event(...)`, `train_next_event(...)`, and `rollout(...)` operate on short-memory event content through the world model.
