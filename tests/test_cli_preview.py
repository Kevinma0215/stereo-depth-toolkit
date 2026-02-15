from typer.testing import CliRunner
from stereo_depth.cli.app import app

runner = CliRunner()

def test_preview_parsing(monkeypatch):
    calls = {}

    def fake_open_source(**kwargs):
        calls["open_source"] = kwargs
        return object()  # fake cap

    def fake_preview_sbs(cap, splitter, recorder):
        calls["preview_sbs"] = {
            "cap": cap,
            "swap_lr": getattr(splitter, "swap_lr", None),
            "recorder": recorder,
        }

    # monkeypatch import paths: 要 patch 「preview_cmd 內使用到的名字」
    import stereo_depth.cli.preview_cmd as preview_cmd
    monkeypatch.setattr(preview_cmd, "open_source", fake_open_source)
    monkeypatch.setattr(preview_cmd, "preview_sbs", fake_preview_sbs)

    result = runner.invoke(app, ["preview", "--device", "2", "--width", "2560", "--height", "720", "--swap-lr"])
    assert result.exit_code == 0

    assert calls["open_source"]["device"] == 2
    assert calls["open_source"]["width"] == 2560
    assert calls["open_source"]["height"] == 720
    assert calls["open_source"]["video"] is None
    assert calls["open_source"]["path"] is None
    assert calls["preview_sbs"]["swap_lr"] is True
