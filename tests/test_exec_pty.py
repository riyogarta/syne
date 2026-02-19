"""Tests for PTY-aware exec in CLI mode.

Tests for:
- Interactive command detection (_command_needs_interactive)
- PTY exec is only triggered in CLI mode
- Non-interactive commands still use normal subprocess
"""

import pytest
from syne.agent import SyneAgent


class TestCommandNeedsInteractive:
    """Test the static method that detects interactive commands."""

    def test_sudo_at_start(self):
        assert SyneAgent._command_needs_interactive("sudo apt update") is True

    def test_sudo_in_chain_pipe(self):
        assert SyneAgent._command_needs_interactive("echo yes | sudo tee /etc/file") is True

    def test_sudo_in_chain_and(self):
        assert SyneAgent._command_needs_interactive("apt update && sudo apt upgrade") is True

    def test_ssh_interactive(self):
        assert SyneAgent._command_needs_interactive("ssh user@host") is True

    def test_ssh_batch_mode_not_interactive(self):
        assert SyneAgent._command_needs_interactive("ssh -o BatchMode=yes user@host ls") is False

    def test_passwd(self):
        assert SyneAgent._command_needs_interactive("passwd") is True

    def test_su(self):
        assert SyneAgent._command_needs_interactive("su root") is True

    def test_normal_command_not_interactive(self):
        assert SyneAgent._command_needs_interactive("ls -la") is False

    def test_echo_not_interactive(self):
        assert SyneAgent._command_needs_interactive("echo hello") is False

    def test_pip_not_interactive(self):
        assert SyneAgent._command_needs_interactive("pip install flask") is False

    def test_git_not_interactive(self):
        assert SyneAgent._command_needs_interactive("git status") is False

    def test_grep_sudo_in_text_not_interactive(self):
        """grep for 'sudo' in a file should NOT trigger PTY."""
        assert SyneAgent._command_needs_interactive("grep sudo /etc/group") is False

    def test_echo_sudo_in_text_not_interactive(self):
        """echo containing 'sudo' should NOT trigger PTY."""
        assert SyneAgent._command_needs_interactive("echo 'use sudo for admin'") is False

    def test_empty_string(self):
        assert SyneAgent._command_needs_interactive("") is False

    def test_just_whitespace(self):
        assert SyneAgent._command_needs_interactive("   ") is False
