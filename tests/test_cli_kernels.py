import json

from sr_adapter.cli import main


def test_cli_kernels_status_json(capsys) -> None:
    exit_code = main(["kernels", "status", "--json"])
    assert exit_code == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert set(payload.keys()) == {"text", "layout"}


def test_cli_kernels_warm_summary(capsys) -> None:
    exit_code = main(["kernels", "warm"])
    assert exit_code == 0
    captured = capsys.readouterr()
    lines = [line for line in captured.out.splitlines() if line.strip()]
    assert any(line.startswith("Text kernel:") for line in lines)
