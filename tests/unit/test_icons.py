"""Tests for nexus_mcp.icons — validates icon constants and base64 integrity."""

import base64

from mcp.types import Icon

from nexus_mcp.icons import SERVER_ICONS, TOOL_CONFIG_ICONS, TOOL_EXEC_ICONS

ALL_ICON_LISTS = [
    ("SERVER_ICONS", SERVER_ICONS),
    ("TOOL_EXEC_ICONS", TOOL_EXEC_ICONS),
    ("TOOL_CONFIG_ICONS", TOOL_CONFIG_ICONS),
]


class TestIconConstants:
    """Each exported constant is a non-empty list[Icon] with valid structure."""

    def test_all_constants_are_non_empty_lists(self) -> None:
        for name, icons in ALL_ICON_LISTS:
            assert isinstance(icons, list), f"{name} is not a list"
            assert len(icons) > 0, f"{name} is empty"

    def test_all_items_are_icon_instances(self) -> None:
        for name, icons in ALL_ICON_LISTS:
            for i, icon in enumerate(icons):
                assert isinstance(icon, Icon), f"{name}[{i}] is not an Icon"

    def test_src_is_svg_data_uri(self) -> None:
        prefix = "data:image/svg+xml;base64,"
        for name, icons in ALL_ICON_LISTS:
            for i, icon in enumerate(icons):
                assert icon.src.startswith(prefix), f"{name}[{i}].src does not start with {prefix}"

    def test_mime_type_is_svg(self) -> None:
        for name, icons in ALL_ICON_LISTS:
            for i, icon in enumerate(icons):
                assert icon.mimeType == "image/svg+xml", (
                    f"{name}[{i}].mimeType is {icon.mimeType!r}, expected 'image/svg+xml'"
                )

    def test_base64_payload_decodes(self) -> None:
        prefix = "data:image/svg+xml;base64,"
        for name, icons in ALL_ICON_LISTS:
            for i, icon in enumerate(icons):
                payload = icon.src[len(prefix) :]
                try:
                    decoded = base64.b64decode(payload)
                except Exception as e:
                    raise AssertionError(f"{name}[{i}] base64 decode failed: {e}") from e
                assert b"<svg" in decoded, f"{name}[{i}] decoded payload does not contain '<svg'"
