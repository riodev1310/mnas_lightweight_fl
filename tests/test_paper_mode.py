from config.default_config import MNASConfig
from runner.run_experiment import build_parser


def test_mnas_config_rejects_homogeneous_mode():
    cfg = MNASConfig()
    cfg.experiment.mode = "homogeneous"

    try:
        cfg.validate()
    except ValueError as exc:
        assert "heterogeneous personalized architectures" in str(exc)
        return
    raise AssertionError("Expected homogeneous mode to be rejected")


def test_cli_does_not_expose_mode_switch():
    help_text = build_parser().format_help()

    assert "--mode" not in help_text


def test_cli_exposes_resume_from_round():
    help_text = build_parser().format_help()

    assert "--resume-from-round" in help_text
