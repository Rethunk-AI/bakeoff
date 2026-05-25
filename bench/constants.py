"""Project-wide constants for the bakeoff harness."""

from uuid import UUID

# Namespace UUIDs for deterministic UUID5 generation.
# These are fixed project constants — never change after first model/creator ingestion.
# Generated once (250942ZMAY26) and committed per bakeoff#13/#15 design agreement.
#
# Usage:
#   model_id  = uuid.uuid5(BAKEOFF_MODEL_NAMESPACE, model_hash)           # primary
#   model_id  = uuid.uuid5(BAKEOFF_MODEL_NAMESPACE, f"{url}|{params}|{size}")  # provisional
#   creator_id = uuid.uuid5(BAKEOFF_CREATOR_NAMESPACE, homepage)           # primary
#   creator_id = uuid.uuid5(BAKEOFF_CREATOR_NAMESPACE, display_name)       # provisional
BAKEOFF_MODEL_NAMESPACE: UUID = UUID("dbc87c22-6b6d-4f89-9757-d186f0f62719")
BAKEOFF_CREATOR_NAMESPACE: UUID = UUID("efaf4f4e-cbd1-43f7-a535-f72079513d00")
