from pathlib import Path

from checkpointing import CheckpointManager


def test_client_checkpoint_path_matches_paper_faithful_layout(tmp_path: Path):
    manager = CheckpointManager(tmp_path)

    client_path = manager.client_checkpoint_path("heterogeneous", 10, 1, 0)
    round_dir = manager.client_round_dir("heterogeneous", 10, 1)

    assert client_path == (
        tmp_path
        / "checkpoints"
        / "heterogeneous"
        / "clients_10"
        / "round_001"
        / "client_000.pt"
    )
    assert round_dir == tmp_path / "checkpoints" / "heterogeneous" / "clients_10" / "round_001"


def test_missing_resume_round_checkpoints_raise_clear_error(tmp_path: Path):
    manager = CheckpointManager(tmp_path)

    try:
        manager.load_client_round_checkpoints("heterogeneous", 10, 55)
    except FileNotFoundError as exc:
        assert "Missing 10 client checkpoints" in str(exc)
        assert "round 55" in str(exc)
        return
    raise AssertionError("Expected missing resume checkpoints to raise FileNotFoundError")


def test_server_checkpoint_api_is_not_exposed():
    manager = CheckpointManager("/tmp/example")

    assert not hasattr(manager, "server_checkpoint_path")
    assert not hasattr(manager, "save_server_checkpoint")
