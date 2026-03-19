# tests/unit/test_config_opencode_server.py
"""Tests for OpenCode server config getters."""

import os
from unittest.mock import patch

from nexus_mcp.config import (
    get_opencode_server_password,
    get_opencode_server_url,
    get_opencode_server_username,
)


class TestOpenCodeServerConfig:
    """Tests for opencode_server config functions."""

    def test_url_default(self):
        with patch.dict(os.environ, {}, clear=True):
            assert get_opencode_server_url() == "http://127.0.0.1:4096"

    def test_url_from_env(self):
        with patch.dict(os.environ, {"NEXUS_OPENCODE_SERVER_URL": "http://myhost:9999"}):
            assert get_opencode_server_url() == "http://myhost:9999"

    def test_username_default(self):
        with patch.dict(os.environ, {}, clear=True):
            assert get_opencode_server_username() == "opencode"

    def test_username_from_env(self):
        with patch.dict(os.environ, {"NEXUS_OPENCODE_SERVER_USERNAME": "admin"}):
            assert get_opencode_server_username() == "admin"

    def test_password_default_none(self):
        with patch.dict(os.environ, {}, clear=True):
            assert get_opencode_server_password() is None

    def test_password_from_env(self):
        with patch.dict(os.environ, {"NEXUS_OPENCODE_SERVER_PASSWORD": "secret"}):
            assert get_opencode_server_password() == "secret"
