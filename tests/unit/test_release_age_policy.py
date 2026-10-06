import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEPENDABOT = ROOT / ".github" / "dependabot.yml"
DEPENDENCY_SAFETY = ROOT / ".github" / "workflows" / "dependency-safety.yml"
PYPROJECT = ROOT / "pyproject.toml"


def test_release_age_settings_agree() -> None:
    dependabot = DEPENDABOT.read_text(encoding="utf-8")
    cooldowns = re.findall(r"^\s+default-days: (\d+)\s*$", dependabot, re.M)
    updaters = re.findall(r"^\s+- package-ecosystem:", dependabot, re.M)
    assert cooldowns, "dependabot.yml has no cooldown.default-days"
    assert len(cooldowns) == len(updaters), "every Dependabot updater needs a cooldown"

    safety = DEPENDENCY_SAFETY.read_text(encoding="utf-8")
    gate = re.findall(r"^\s+minimum_release_age_days: (\d+)\s*$", safety, re.M)
    assert len(gate) == 1, "dependency-safety.yml needs one minimum_release_age_days"

    exclude_newer = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["tool"]["uv"][
        "exclude-newer"
    ]
    uv_days = re.fullmatch(r"(\d+) days", exclude_newer)
    assert uv_days, f"tool.uv.exclude-newer must be 'N days', got {exclude_newer!r}"

    settings = {
        **{f"dependabot.yml cooldown #{i}": int(d) for i, d in enumerate(cooldowns, 1)},
        "dependency-safety.yml minimum_release_age_days": int(gate[0]),
        "pyproject.toml tool.uv.exclude-newer": int(uv_days.group(1)),
    }
    assert len(set(settings.values())) == 1, f"release-age settings disagree: {settings}"
