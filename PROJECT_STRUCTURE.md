# Project Structure

This project is organized into Python packages plus a small set of
root-level compatibility wrappers.

## Packages

### `knowledge/`

- `knowledge/relation_map.py`
  Stores the global noun list, relation list, relation matrix, and learning rates.
- `knowledge/knowledge_map.py`
  Defines the `knowledge_map` model and the training functions `train_random` and `train_average`.
- `knowledge/training.py`
  Wraps the training lifecycle: load state, feed relations, save state.
- `knowledge/train_knowledge.py`
  Entry-style script for relation feeding and next-word prediction.

### `world/`

- `world/world_model.py`
  Defines the attention-based action models and the `WorldModel` wrapper.
- `world/train_world_model.py`
  Provides training loops for world-model samples, with optional validation.

### `short_memory/`

- `short_memory/shortmemory.py`
  Defines `ShortMemory`, `ScoredTensorQueue`, and the event/relation/reward/surprise memory entries.
- `world/shortmemory.py`
  Compatibility wrapper that re-exports the standalone short-memory package for older imports.

### `grammar_layer/`

- `grammar_layer/grammar.py`
  Defines the rule-based tokenizer, part-of-speech tagging, sentence extraction, instance resolution, and short-memory conversion helpers.
- `grammar_layer/grammar_routes.py`
  Defines the sentence pattern routing table used by the grammar extractor.
- `prototype/grammar.py` and `prototype/grammar_routes.py`
  Compatibility wrappers that re-export the standalone grammar layer for older imports.

### `prototype/`

- `prototype/consciousness.py`
  High-level controller for question handling, memory inspection, learning, and world-model interaction.
- `prototype/episode_memory.py`
  Prototype episodic memory container.
- `prototype/event_model.py`
  Empty placeholder.
- `prototype/train_data_generator.py`
  Empty placeholder.
- `prototype/adj.py`
  Small embedding-adjustment experiment.

## Saved Data

- `relation_data.npz`
  Persisted relation graph state.
- `knowledge_map_one.pt`
  Persisted knowledge model weights.

## Top-Level Compatibility Wrappers

- Legacy file names such as `Knowledge_map.py`, `World_model.py`, and `Consciousness.py` are still present at the repo root.
- They now re-export the packaged implementations so existing scripts and IDE tabs keep working during the transition.
- `train_relation_map.py` and `Feed_relations.py` remain compatibility shims for older imports.

## Recommended Entry Points

- Knowledge workflow: `knowledge/train_knowledge.py`
- World-model workflow: `world/train_world_model.py`

## Current Cleanup Notes

- `relation_num` is aligned with the knowledge model capacity.
- Prototype modules now expose small, explicit helpers instead of ambiguous placeholders.
