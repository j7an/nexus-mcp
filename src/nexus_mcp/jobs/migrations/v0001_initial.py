"""Initial durable job database schema."""

__all__ = ["INITIAL_STATEMENTS"]

INITIAL_STATEMENTS = (
    """
    CREATE TABLE schema_migrations (
      migration_id TEXT PRIMARY KEY,
      checksum TEXT NOT NULL,
      applied_at_ms INTEGER NOT NULL
    )
    """,
    """
    CREATE TABLE workspaces (
      workspace_id TEXT PRIMARY KEY,
      canonical_path TEXT NOT NULL UNIQUE,
      display_name TEXT,
      config_ref TEXT,
      created_at_ms INTEGER NOT NULL,
      updated_at_ms INTEGER NOT NULL
    )
    """,
    """
    CREATE TABLE sessions (
      session_id TEXT PRIMARY KEY,
      workspace_id TEXT NOT NULL REFERENCES workspaces(workspace_id),
      backend_id TEXT NOT NULL,
      owner_id TEXT NOT NULL,
      access_policy TEXT NOT NULL CHECK (access_policy IN ('private', 'workspace')),
      parent_session_id TEXT REFERENCES sessions(session_id),
      created_at_ms INTEGER NOT NULL,
      updated_at_ms INTEGER NOT NULL
    )
    """,
    """
    CREATE TABLE jobs (
      job_id TEXT PRIMARY KEY,
      session_id TEXT REFERENCES sessions(session_id),
      workspace_id TEXT NOT NULL REFERENCES workspaces(workspace_id),
      backend_id TEXT NOT NULL,
      owner_id TEXT NOT NULL,
      access_policy TEXT NOT NULL CHECK (access_policy IN ('private', 'workspace')),
      operation_kind TEXT NOT NULL CHECK (
        operation_kind IN ('turn','fork','review','diagnostics')
      ),
      operation_json TEXT NOT NULL,
      operation_schema_version INTEGER NOT NULL,
      request_hash TEXT NOT NULL,
      requested_config_json TEXT NOT NULL,
      requested_config_schema_version INTEGER NOT NULL,
      resolved_config_json TEXT,
      resolved_config_schema_version INTEGER,
      state TEXT NOT NULL CHECK (
        state IN ('queued','running','input_required','completed','failed','cancelled')
      ),
      phase TEXT,
      cancel_requested_at_ms INTEGER,
      retry_at_ms INTEGER,
      lease_owner TEXT,
      lease_generation INTEGER NOT NULL DEFAULT 0,
      lease_expires_at_ms INTEGER,
      created_at_ms INTEGER NOT NULL,
      updated_at_ms INTEGER NOT NULL,
      terminal_at_ms INTEGER
    )
    """,
    """
    CREATE TABLE job_attempts (
      job_id TEXT NOT NULL REFERENCES jobs(job_id),
      attempt_number INTEGER NOT NULL,
      phase TEXT NOT NULL,
      owner_id TEXT NOT NULL,
      lease_generation INTEGER NOT NULL,
      started_at_ms INTEGER NOT NULL,
      ended_at_ms INTEGER,
      error_json TEXT,
      error_schema_version INTEGER,
      PRIMARY KEY (job_id, attempt_number)
    )
    """,
    """
    CREATE TABLE provider_references (
      provider_reference_id TEXT PRIMARY KEY,
      backend_id TEXT NOT NULL,
      kind TEXT NOT NULL,
      value TEXT NOT NULL,
      session_id TEXT REFERENCES sessions(session_id),
      job_id TEXT REFERENCES jobs(job_id),
      attempt_number INTEGER,
      created_at_ms INTEGER NOT NULL,
      CHECK (session_id IS NOT NULL OR job_id IS NOT NULL)
    )
    """,
    """
    CREATE TABLE pending_inputs (
      input_id TEXT PRIMARY KEY,
      job_id TEXT NOT NULL REFERENCES jobs(job_id),
      kind TEXT NOT NULL CHECK (kind IN ('approval','permission','question','form')),
      request_json TEXT NOT NULL,
      request_schema_version INTEGER NOT NULL,
      response_json TEXT,
      response_schema_version INTEGER,
      status TEXT NOT NULL CHECK (status IN ('pending','resolved','expired')),
      provider_reference_id TEXT REFERENCES provider_references(provider_reference_id),
      created_at_ms INTEGER NOT NULL,
      resolved_at_ms INTEGER
    )
    """,
    """
    CREATE TABLE job_events (
      job_id TEXT NOT NULL REFERENCES jobs(job_id),
      sequence INTEGER NOT NULL,
      event_type TEXT NOT NULL,
      payload_json TEXT NOT NULL,
      payload_schema_version INTEGER NOT NULL,
      attempt_number INTEGER,
      created_at_ms INTEGER NOT NULL,
      provider_event_type TEXT,
      provider_event_id TEXT,
      provider_reference_id TEXT REFERENCES provider_references(provider_reference_id),
      PRIMARY KEY (job_id, sequence)
    )
    """,
    """
    CREATE TABLE job_results (
      job_id TEXT PRIMARY KEY REFERENCES jobs(job_id),
      outcome_kind TEXT NOT NULL CHECK (outcome_kind IN ('succeeded','failed','cancelled')),
      payload_json TEXT,
      payload_schema_version INTEGER,
      error_json TEXT,
      error_schema_version INTEGER,
      created_at_ms INTEGER NOT NULL
    )
    """,
    """
    CREATE TABLE idempotency_keys (
      idempotency_id TEXT PRIMARY KEY,
      principal_id TEXT NOT NULL,
      workspace_id TEXT NOT NULL REFERENCES workspaces(workspace_id),
      command_family TEXT NOT NULL,
      idempotency_key TEXT NOT NULL,
      source_session_id TEXT REFERENCES sessions(session_id),
      request_hash TEXT NOT NULL,
      job_id TEXT NOT NULL REFERENCES jobs(job_id),
      created_at_ms INTEGER NOT NULL
    )
    """,
    """
    CREATE TABLE runtime_leases (
      runtime_key TEXT PRIMARY KEY,
      owner_id TEXT NOT NULL,
      lease_generation INTEGER NOT NULL,
      endpoint TEXT,
      lease_expires_at_ms INTEGER NOT NULL,
      heartbeat_at_ms INTEGER NOT NULL
    )
    """,
    """
    CREATE UNIQUE INDEX one_nonterminal_job_per_session
    ON jobs(session_id)
    WHERE session_id IS NOT NULL
      AND state IN ('queued', 'running', 'input_required')
    """,
    """
    CREATE UNIQUE INDEX scoped_idempotency_key
    ON idempotency_keys(
      principal_id, workspace_id, command_family, idempotency_key,
      ifnull(source_session_id, '')
    )
    """,
    """
    CREATE UNIQUE INDEX job_event_sequence
    ON job_events(job_id, sequence)
    """,
    """
    CREATE UNIQUE INDEX provider_reference_identity
    ON provider_references(
      backend_id, kind, value,
      ifnull(session_id, ''), ifnull(job_id, ''), ifnull(attempt_number, -1)
    )
    """,
    """
    CREATE INDEX claimable_jobs
    ON jobs(state, retry_at_ms, lease_expires_at_ms, created_at_ms)
    """,
    """
    CREATE INDEX jobs_by_workspace_created
    ON jobs(workspace_id, created_at_ms DESC, job_id DESC)
    """,
)
