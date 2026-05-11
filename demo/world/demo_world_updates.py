import pathlib
import sys

import torch

if __package__ is None or __package__ == "":
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

from short_memory import ShortMemory
from world.world_model import (
    WorldModel,
    action_dim,
    attention_dim,
    hidden_dim,
    noun_dim,
    value_dim,
)


def _vector_head(tensor: torch.Tensor, n: int = 6):
    values = tensor.detach().cpu().view(-1)[:n].tolist()
    return [round(float(value), 4) for value in values]


def _embedding_snapshot(memory: ShortMemory, instance_id: str):
    embedding = memory.get_noun_embedding(instance_id)
    if embedding is None:
        return None
    return {
        "instance_id": instance_id,
        "head": _vector_head(embedding),
        "norm": round(float(torch.norm(embedding).item()), 4),
    }


def _print_section(title: str):
    print()
    print("=" * 16, title, "=" * 16)


def build_demo_world():
    torch.manual_seed(7)

    model = WorldModel(
        noun_dim=noun_dim,
        action_dim=action_dim,
        attention_dim=attention_dim,
        value_dim=value_dim,
        hidden_dim=hidden_dim,
        action_names=["eat", "move", "wash"],
        action_space_names=["put_in", "put_on", "move_near"],
        action_space_relations={
            "put_in": "in",
            "put_on": "on",
            "move_near": "near",
        },
    )

    memory = ShortMemory(maxlen=20, device="cpu")

    apple_embedding = torch.linspace(-1.0, 1.0, noun_dim)
    box_embedding = torch.linspace(0.8, -0.8, noun_dim)
    table_embedding = torch.sin(torch.linspace(0.0, 3.14159, noun_dim))

    seed_action = model.get_action_embedding(1).detach().clone()

    memory.append_event(
        noun_embedding=apple_embedding,
        action_embedding=seed_action,
        score=1.0,
        noun_type=1,
        action_type=1,
        noun_text="apple",
        action_text="eat",
        noun_instance_id="apple#1",
        time_position=0,
        event_index=0,
    )
    memory.append_event(
        noun_embedding=box_embedding,
        action_embedding=seed_action,
        score=1.0,
        noun_type=2,
        action_type=1,
        noun_text="box",
        action_text="eat",
        noun_instance_id="box#1",
        time_position=0,
        event_index=1,
    )
    memory.append_event(
        noun_embedding=table_embedding,
        action_embedding=seed_action,
        score=1.0,
        noun_type=3,
        action_type=1,
        noun_text="table",
        action_text="eat",
        noun_instance_id="table#1",
        time_position=0,
        event_index=2,
    )

    return model, memory


def run_demo():
    model, memory = build_demo_world()

    _print_section("Initial State")
    print("apple embedding:", _embedding_snapshot(memory, "apple#1"))
    print("box embedding:", _embedding_snapshot(memory, "box#1"))
    print("table embedding:", _embedding_snapshot(memory, "table#1"))
    print("space_state:", memory.get_space_content_view())
    print("last_event:", memory.get_focus_entry().info_pair)

    _print_section("Step 1: Instance Update")
    before = memory.get_noun_embedding("apple#1")
    update_result = model.update_instance_embedding(
        memory,
        noun_instance_id="apple#1",
        action_type=2,
        noun_type=1,
        noun_text="apple",
        action_text="move",
        score=0.7,
    )
    after = memory.get_noun_embedding("apple#1")
    print("update_result:", {
        "noun_instance_id": update_result.noun_instance_id,
        "action_type": update_result.action_type,
        "time_position": update_result.time_position,
        "event_index": update_result.event_index,
        "old_head": _vector_head(update_result.old_embedding),
        "new_head": _vector_head(update_result.new_embedding),
        "delta_norm": round(float(torch.norm(after - before).item()), 4),
    })
    print("apple embedding now:", _embedding_snapshot(memory, "apple#1"))
    print("last_event:", memory.get_focus_entry().info_pair)

    _print_section("Step 2: Space Update apple -> box")
    put_in_result = model.update_space_state(
        memory,
        source_instance_id="apple#1",
        target_instance_id="box#1",
        action_space_type=1,
        noun_type=1,
        noun_text="apple",
        action_text="put_in",
        score=0.8,
    )
    print("space_update_result:", put_in_result.as_dict())
    print("apple summary:", memory.get_spatial_summary("apple#1"))
    print("box outgoing:", [fact.to_dict() for fact in memory.space_state.get_outgoing("box#1")])
    print("last_event:", memory.get_focus_entry().info_pair)

    _print_section("Step 3: Space Update box -> table")
    put_on_result = model.update_space_state(
        memory,
        source_instance_id="box#1",
        target_instance_id="table#1",
        action_space_type=2,
        noun_type=2,
        noun_text="box",
        action_text="put_on",
        score=0.85,
    )
    print("space_update_result:", put_on_result.as_dict())
    print("box summary:", memory.get_spatial_summary("box#1"))
    print("table outgoing:", [fact.to_dict() for fact in memory.space_state.get_outgoing("table#1")])
    print("space_state:", memory.get_space_content_view())

    _print_section("Step 4: Space Update apple near table")
    near_result = model.update_space_state(
        memory,
        source_instance_id="apple#1",
        target_instance_id="table#1",
        action_space_type=3,
        noun_type=1,
        noun_text="apple",
        action_text="move_near",
        score=0.9,
        replace_family=False,
    )
    print("space_update_result:", near_result.as_dict())
    print("apple summary:", memory.get_spatial_summary("apple#1"))
    print("table outgoing:", [fact.to_dict() for fact in memory.space_state.get_outgoing("table#1")])

    _print_section("Final Short Memory View")
    content = memory.get_content_view(order_by="time")
    print("event_count:", len(content["event"]))
    print("recent_events:")
    for item in content["event"][-4:]:
        print(item)
    print("space:")
    for item in content["space"]:
        print(item)


if __name__ == "__main__":
    run_demo()
