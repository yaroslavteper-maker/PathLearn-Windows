"""Driving the converter from inside the app.

Nothing here downloads torch or converts a real model: what needs pinning is
the plumbing — which interpreter is chosen, what the command line says, and
that a half-built environment is reported rather than used.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from pathlearn.extractors import conversion
from pathlearn.extractors.conversion import (ConversionError, EnvironmentStatus,
                                             convert_command, find_environment,
                                             inspect_environment,
                                             interpreter_for, requirements_path,
                                             script_path, setup_command)


class TestPaths:
    def test_script_is_found_in_the_checkout(self):
        assert script_path().is_file()
        assert script_path().name == "convert_extractor.py"

    def test_requirements_sit_beside_the_script(self):
        assert requirements_path().is_file()
        assert "torch" in requirements_path().read_text(encoding="utf-8")

    def test_interpreter_layout_matches_the_platform(self, tmp_path):
        got = interpreter_for(tmp_path / "venv")
        assert got.parent.name == ("Scripts" if sys.platform == "win32" else "bin")

    def test_a_missing_checkout_is_explained(self, monkeypatch, tmp_path):
        monkeypatch.setattr(conversion, "__file__",
                            str(tmp_path / "a" / "b" / "conversion.py"))
        with pytest.raises(ConversionError, match="source checkout"):
            script_path()


class TestInspectEnvironment:
    def test_none_is_not_ready(self):
        assert not inspect_environment(None).ready

    def test_a_missing_file_is_not_ready(self, tmp_path):
        assert not inspect_environment(tmp_path / "nope.exe").ready

    def test_this_interpreter_lacks_torch(self):
        """The app venv deliberately has no torch — that is the whole design."""
        status = inspect_environment(Path(sys.executable))
        assert status.interpreter is not None
        assert "torch" in status.missing
        assert not status.ready

    def test_the_summary_names_what_is_missing(self):
        status = inspect_environment(Path(sys.executable))
        assert "torch" in status.summary()

    def test_a_ready_environment_reports_ready(self, monkeypatch):
        monkeypatch.setattr(conversion.subprocess, "run",
                            lambda *a, **k: subprocess.CompletedProcess(
                                a[0], 0, stdout="", stderr=""))
        status = inspect_environment(Path(sys.executable))
        assert status.ready
        assert "Ready" in status.summary()

    def test_a_crashing_probe_is_an_error_not_a_crash(self, monkeypatch):
        monkeypatch.setattr(conversion.subprocess, "run",
                            lambda *a, **k: subprocess.CompletedProcess(
                                a[0], 1, stdout="", stderr="DLL load failed"))
        status = inspect_environment(Path(sys.executable))
        assert not status.ready
        assert "DLL load failed" in status.summary()

    def test_a_launch_failure_is_reported(self, monkeypatch):
        def boom(*a, **k):
            raise OSError("not executable")
        monkeypatch.setattr(conversion.subprocess, "run", boom)
        status = inspect_environment(Path(sys.executable))
        assert "not executable" in status.summary()


class TestFindEnvironment:
    def test_a_ready_preference_wins(self, monkeypatch, tmp_path):
        chosen = tmp_path / "python.exe"
        chosen.write_text("", encoding="utf-8")
        monkeypatch.setattr(conversion, "inspect_environment",
                            lambda p: EnvironmentStatus(Path(p))
                            if p == chosen else EnvironmentStatus(Path(p),
                                                                 missing=("torch",)))
        assert find_environment(chosen).interpreter == chosen

    def test_the_default_venv_is_found_when_it_exists(self):
        """Someone who built the venv from the docs must not be asked again."""
        status = find_environment(None)
        default = interpreter_for(conversion.DEFAULT_VENV)
        if default.is_file():
            assert status.ready, status.summary()
            assert status.interpreter == default
        else:
            assert not status.ready
        assert status.summary()

    def test_nothing_available_still_reports(self, monkeypatch):
        monkeypatch.setattr(conversion, "inspect_environment",
                            lambda p: EnvironmentStatus(None))
        status = find_environment(None)
        assert not status.ready
        assert "No conversion environment" in status.summary()

    def test_an_unusable_preference_does_not_win(self, tmp_path):
        status = find_environment(tmp_path / "ghost.exe")
        assert status.interpreter != tmp_path / "ghost.exe"


class TestCommands:
    def test_setup_creates_then_installs(self, tmp_path):
        venv, reqs = tmp_path / "v", tmp_path / "r.txt"
        create, install = setup_command(venv, reqs)
        assert create[1:3] == ["-m", "venv"]
        assert install[1:4] == ["-m", "pip", "install"]
        assert str(reqs) in install
        assert install[0] == str(interpreter_for(venv))

    def test_a_built_in_is_passed_by_name(self, tmp_path):
        command = convert_command(Path("py"), tmp_path, known="uni2-h")
        assert "uni2-h" in command
        assert "--repo" not in command

    def test_a_repo_carries_its_settings(self, tmp_path):
        command = convert_command(Path("py"), tmp_path, repo="owner/model",
                                  name="thing", revision=3, input_size=384)
        assert command[command.index("--repo") + 1] == "owner/model"
        assert command[command.index("--name") + 1] == "thing"
        assert command[command.index("--revision") + 1] == "3"
        assert command[command.index("--input-size") + 1] == "384"

    def test_the_output_directory_is_always_given(self, tmp_path):
        command = convert_command(Path("py"), tmp_path, known="phikon-v1")
        assert command[command.index("--out") + 1] == str(tmp_path)

    def test_offline_is_opt_in(self, tmp_path):
        assert "--offline" not in convert_command(Path("py"), tmp_path,
                                                  known="phikon-v1")
        assert "--offline" in convert_command(Path("py"), tmp_path,
                                              known="phikon-v1", offline=True)

    def test_naming_nothing_is_refused(self, tmp_path):
        with pytest.raises(ConversionError, match="Pick a built-in"):
            convert_command(Path("py"), tmp_path)


class TestRunStreaming:
    def test_output_arrives_line_by_line(self):
        lines = []
        code = conversion.run_streaming(
            [sys.executable, "-c", "print('one');print('two')"],
            progress=lines.append)
        assert code == 0
        assert lines == ["one", "two"]

    def test_stderr_is_interleaved_not_lost(self):
        """Conversion failures arrive on stderr; losing them loses the reason."""
        lines = []
        conversion.run_streaming(
            [sys.executable, "-c", "import sys; print('bad', file=sys.stderr)"],
            progress=lines.append)
        assert "bad" in lines

    def test_the_exit_code_comes_back(self):
        assert conversion.run_streaming([sys.executable, "-c", "raise SystemExit(3)"]) == 3

    def test_a_missing_executable_is_explained(self, tmp_path):
        with pytest.raises(ConversionError, match="Could not start"):
            conversion.run_streaming([str(tmp_path / "nope.exe")])

    def test_cancelling_stops_it(self):
        with pytest.raises(ConversionError, match="Cancelled"):
            conversion.run_streaming(
                [sys.executable, "-u", "-c",
                 "import time\nfor i in range(500):\n print(i)\n time.sleep(0.01)"],
                should_cancel=lambda: True)


class TestConvertSheet:
    @pytest.fixture
    def sheet(self, qtbot, tmp_path):
        from pathlearn.ui.sheets.convert_model import ConvertModelSheet

        widget = ConvertModelSheet(tmp_path / "Extractors")
        qtbot.addWidget(widget)
        yield widget
        widget.reject()

    def test_it_reports_the_environment(self, sheet):
        assert sheet.env_label.text()

    def test_built_in_is_the_default_choice(self, sheet):
        assert sheet.builtin_radio.isChecked()
        assert sheet.builtin_combo.isEnabled()
        assert not sheet.repo_edit.isEnabled()

    def test_choosing_other_enables_the_repo_fields(self, sheet):
        sheet.repo_radio.setChecked(True)
        assert sheet.repo_edit.isEnabled()
        assert sheet.revision.isEnabled()
        assert not sheet.builtin_combo.isEnabled()

    def test_convert_is_blocked_until_a_repo_is_named(self, sheet):
        sheet.repo_radio.setChecked(True)
        sheet.repo_edit.setText("")
        assert not sheet.convert_button.isEnabled()
        sheet.repo_edit.setText("owner/model")
        sheet._update_ready()
        assert sheet.convert_button.isEnabled() == sheet.status.ready

    def test_the_built_in_command_names_the_model(self, sheet):
        if not sheet.status.ready:
            pytest.skip("no conversion environment on this machine")
        index = sheet.builtin_combo.findData("uni2-h")
        sheet.builtin_combo.setCurrentIndex(index)
        assert "uni2-h" in sheet._command()

    def test_the_custom_command_carries_the_fields(self, sheet):
        if not sheet.status.ready:
            pytest.skip("no conversion environment on this machine")
        sheet.repo_radio.setChecked(True)
        sheet.repo_edit.setText("owner/model")
        sheet.name_edit.setText("thing")
        sheet.revision.setValue(2)
        command = sheet._command()
        assert command[command.index("--repo") + 1] == "owner/model"
        assert command[command.index("--name") + 1] == "thing"
        assert command[command.index("--revision") + 1] == "2"

    def test_it_installs_into_the_extractors_folder(self, sheet, tmp_path):
        assert str(tmp_path / "Extractors") in sheet.destination.text()

    def test_a_bad_interpreter_is_reported_not_adopted(self, sheet, tmp_path):
        ghost = tmp_path / "ghost.exe"
        sheet.set_interpreter(ghost)
        assert not sheet.status.ready
        assert "ghost" in sheet.log.toPlainText()

    def test_the_app_interpreter_is_refused_for_lacking_torch(self, sheet):
        sheet.set_interpreter(Path(sys.executable))
        assert not sheet.status.ready
        assert "torch" in sheet.log.toPlainText()

    def test_it_emits_converted_on_success(self, qtbot, sheet):
        with qtbot.waitSignal(sheet.converted, timeout=1000):
            sheet._on_convert_done(0)
        assert sheet.wrote_something

    def test_a_failure_does_not_claim_success(self, sheet):
        sheet._on_convert_done(2)
        assert not sheet.wrote_something
        assert "failed" in sheet.log.toPlainText()
        assert "gated" in sheet.log.toPlainText()
