import os

import numpy as np

import knowledge.relation_map as noun_rm
from world.action_vocab import action_list, ensure_action, set_action_list

noun_number = noun_rm.noun_number
action_number = 200

noun_action_map = np.zeros((noun_number, action_number), dtype=np.int64)


def _placeholder_name(index: int) -> str:
    return f"action_{int(index)}"


def _normalize_action_name(action: str) -> str:
    return str(action).lower()


def is_defined_action_index(index: int) -> bool:
    index = int(index)
    return 0 <= index < len(action_list) and action_list[index] != _placeholder_name(index)


def sync_action_list_with_world_model(action_names=None):
    return set_action_list(action_names or action_list)


def bind_action_to_index(action: str, index: int) -> int:
    action = _normalize_action_name(action)
    index = int(index)
    if index < 0 or index >= action_number:
        raise ValueError(f"action index must be in [0, {action_number - 1}]")

    if action in action_list:
        return action_list.index(action)

    while len(action_list) <= index:
        action_list.append(_placeholder_name(len(action_list)))

    current_value = action_list[index]
    if current_value == action or current_value == _placeholder_name(index):
        action_list[index] = action
        return index

    raise ValueError(f"action slot {index} is already bound to '{current_value}'")


def _ensure_action(action: str) -> int:
    if len(action_list) >= action_number and _normalize_action_name(action) not in action_list:
        raise ValueError("action_list is full; cannot register a new action")
    return ensure_action(action)


def _normalize_value(value: int) -> int:
    value = int(value)
    if value not in {0, 1}:
        raise ValueError("noun_action_map only accepts 0 or 1")
    return value


def set_noun_action(noun: str, action: str, value: int = 1):
    noun_idx = noun_rm._ensure_noun(noun)
    action_idx = _ensure_action(action)
    noun_action_map[noun_idx, action_idx] = _normalize_value(value)
    return noun_idx, action_idx, int(noun_action_map[noun_idx, action_idx])


def add_noun_action(noun: str, action: str):
    return set_noun_action(noun, action, value=1)


def remove_noun_action(noun: str, action: str):
    return set_noun_action(noun, action, value=0)


def can_noun_do_action(noun, action) -> bool:
    noun_idx = noun if isinstance(noun, int) else noun_rm._ensure_noun(noun)
    action_idx = action if isinstance(action, int) else _ensure_action(action)
    return int(noun_action_map[noun_idx, action_idx]) == 1


def get_actions_for_noun(noun, only_allowed: bool = True):
    noun_idx = noun if isinstance(noun, int) else noun_rm._ensure_noun(noun)
    values = noun_action_map[noun_idx]
    results = []
    for action_idx, raw_value in enumerate(values):
        value = int(raw_value)
        if only_allowed and value != 1:
            continue
        action_name = action_list[action_idx] if action_idx < len(action_list) else _placeholder_name(action_idx)
        results.append(
            {
                "noun_idx": int(noun_idx),
                "noun": noun_rm.noun_list[int(noun_idx)] if int(noun_idx) < len(noun_rm.noun_list) else f"noun_{int(noun_idx)}",
                "action_idx": int(action_idx),
                "action": action_name,
                "value": value,
            }
        )
    return results


def get_nouns_for_action(action, only_allowed: bool = True):
    action_idx = action if isinstance(action, int) else _ensure_action(action)
    values = noun_action_map[:, action_idx]
    results = []
    for noun_idx, raw_value in enumerate(values):
        value = int(raw_value)
        if only_allowed and value != 1:
            continue
        noun_name = noun_rm.noun_list[noun_idx] if noun_idx < len(noun_rm.noun_list) else f"noun_{noun_idx}"
        results.append(
            {
                "noun_idx": int(noun_idx),
                "noun": noun_name,
                "action_idx": int(action_idx),
                "action": action_list[int(action_idx)] if int(action_idx) < len(action_list) else _placeholder_name(action_idx),
                "value": value,
            }
        )
    return results


def save_noun_action_data(file_path="noun_action_data.npz"):
    np.savez(
        file_path,
        noun_action_map=noun_action_map,
        noun_list=np.array(noun_rm.noun_list, dtype=object),
        action_list=np.array(action_list, dtype=object),
    )


def load_noun_action_data(file_path="noun_action_data.npz"):
    global noun_action_map

    if not os.path.exists(file_path):
        return False

    data = np.load(file_path, allow_pickle=True)
    noun_action_map = data["noun_action_map"].astype(np.int64, copy=False)
    noun_rm.noun_list = data["noun_list"].tolist()
    set_action_list(data["action_list"].tolist())
    return noun_action_map, noun_rm.noun_list, action_list


if __name__ == "__main__":
    save_noun_action_data()
