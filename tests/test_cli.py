import json

from colonist_rl.cli import main


def test_board_command_reports_both_supported_topologies(capsys):
    assert main(["board", "--players", "6"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["tiles"] == 30
    assert result["simulation_available"] is False


def test_help_describes_training_entrypoint(capsys):
    try:
        main(["--help"])
    except SystemExit as exc:
        assert exc.code == 0
    assert "train" in capsys.readouterr().out


def test_train_and_resume_restore_checkpoint_progress(tmp_path, capsys):
    checkpoint = tmp_path / "resume.pt"
    assert (
        main(
            [
                "train",
                "--updates",
                "1",
                "--rollout",
                "4",
                "--epochs",
                "1",
                "--hidden-size",
                "16",
                "--warmstart-decisions",
                "0",
                "--checkpoint",
                str(checkpoint),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "train",
                "--resume",
                str(checkpoint),
                "--updates",
                "1",
                "--rollout",
                "4",
                "--epochs",
                "1",
                "--checkpoint",
                str(checkpoint),
            ]
        )
        == 0
    )
    from colonist_rl.training import load_checkpoint

    assert load_checkpoint(checkpoint)[1]["steps"] == 8


def test_export_command_produces_a_valid_player_observation(tmp_path):
    from colonist_rl.perception import view_from_dict

    output = tmp_path / "observation.json"
    assert main(["export", "--output", str(output)]) == 0
    view = view_from_dict(json.loads(output.read_text()))
    assert len(view.board.tiles) == 19
    assert view.legal_actions
