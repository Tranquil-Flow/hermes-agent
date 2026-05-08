"""Regression tests for install.sh Python environment sanitization.

When install.sh is launched from another Python-driven tool session, inherited
PYTHONPATH/PYTHONHOME can shadow the freshly installed checkout. The installer
must sanitize those vars both during installation and at runtime launch.
"""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"


def test_install_script_unsets_pythonpath_and_pythonhome_early() -> None:
    text = INSTALL_SH.read_text()

    # During install, inherited Python env must be sanitized before pip/venv use.
    assert 'unset PYTHONPATH' in text
    assert 'unset PYTHONHOME' in text


def test_hermes_launcher_wrapper_clears_python_env_before_exec() -> None:
    text = INSTALL_SH.read_text()

    # Wrapper should clear env and forward args untouched to the venv entrypoint.
    assert 'cat > "$command_link_dir/hermes" <<EOF' in text
    assert 'unset PYTHONPATH' in text
    assert 'unset PYTHONHOME' in text
    assert 'exec "$HERMES_BIN" "\\$@"' in text


def test_installer_does_not_overwrite_venv_console_script_with_self_wrapper() -> None:
    text = INSTALL_SH.read_text()

    assert 'if [ "$command_link_dir" = "$(dirname "$HERMES_BIN")" ]; then' in text
    guard = text.index('if [ "$command_link_dir" = "$(dirname "$HERMES_BIN")" ]; then')
    wrapper = text.index('cat > "$command_link_dir/hermes" <<EOF')
    assert guard < wrapper
    assert 'log_success "hermes command ready at $HERMES_BIN"' in text
    assert 'return 0' in text[guard:wrapper]
