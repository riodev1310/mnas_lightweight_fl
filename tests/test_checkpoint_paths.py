from pathlib import Path

from checkpointing import CheckpointManager


def test_client_checkpoint_path_matches_paper_faithful_layout(tmp_path: Path):
    manager = CheckpointManager(tmp_path)

    client_path = manager.client_checkpoint_path("heterogeneous", 10, 1, 0)

    assert client_path == (
        tmp_path
        / "checkpoints"
        / "heterogeneous"
        / "clients_10"
        / "round_001"
        / "client_000.pt"
    )


def test_server_checkpoint_api_is_not_exposed():
    manager = CheckpointManager("/tmp/example")

    assert not hasattr(manager, "server_checkpoint_path")
    assert not hasattr(manager, "save_server_checkpoint")
