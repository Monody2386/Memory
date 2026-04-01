from typing import Iterable, List

NO_ACTION_NAME = "no_action"
DEFAULT_ACTION_LIST = ["action1", "action2", "action3"]
action_list = list(DEFAULT_ACTION_LIST)


def _normalize_action_name(action: str) -> str:
    return str(action).lower()


def set_action_list(action_names: Iterable[str]) -> List[str]:
    normalized = []
    for action in action_names:
        action_name = _normalize_action_name(action)
        if action_name == NO_ACTION_NAME:
            continue
        if action_name not in normalized:
            normalized.append(action_name)
    action_list[:] = normalized
    return action_list


def get_action_list() -> List[str]:
    return action_list


def get_full_action_type_list() -> List[str]:
    return [NO_ACTION_NAME] + list(action_list)


def ensure_action(action: str) -> int:
    action_name = _normalize_action_name(action)
    if action_name == NO_ACTION_NAME:
        raise ValueError("no_action is reserved and not part of the trainable action vocabulary")
    if action_name not in action_list:
        action_list.append(action_name)
    return action_list.index(action_name)


def action_type_to_name(action_type: int) -> str:
    action_type = int(action_type)
    if action_type == 0:
        return NO_ACTION_NAME
    return action_list[action_type - 1]
