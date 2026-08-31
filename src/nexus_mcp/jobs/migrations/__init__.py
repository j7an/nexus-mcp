"""Forward-only, checksummed SQLite schema migrations."""

import hashlib
from dataclasses import dataclass, field

from nexus_mcp.jobs.migrations.v0001_initial import INITIAL_STATEMENTS

__all__ = ["MIGRATIONS", "Migration"]


@dataclass(frozen=True, slots=True)
class Migration:
    """One immutable ordered collection of SQLite schema statements."""

    migration_id: str
    statements: tuple[str, ...]
    checksum: str = field(init=False)

    def __post_init__(self) -> None:
        """Bind the recorded checksum to the exact ordered SQL payload."""
        if not self.migration_id or not self.statements:
            raise ValueError("a migration requires an identity and at least one statement")
        payload = "\n-- statement boundary --\n".join(self.statements).encode()
        object.__setattr__(self, "checksum", hashlib.sha256(payload).hexdigest())


MIGRATIONS = (Migration("v0001_initial", INITIAL_STATEMENTS),)
