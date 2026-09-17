"""W&B helpers: start_wandb_run (resume='allow') and log_wandb_artifact."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.sae_config import to_jsonable


def start_wandb_run(
    *,
    project: str,
    entity: str | None,
    run_name: str | None,
    run_id: str | None,
    group: str | None,
    config: dict[str, Any],
    tags: list[str] | None = None,
) -> Any:
    import wandb

    run = wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        id=run_id,
        group=group,
        tags=tags,
        resume="allow",
        config=config,
    )
    return run


def finish_wandb_run() -> None:
    import wandb

    wandb.finish()


def log_wandb_artifact(
    *,
    name: str,
    artifact_type: str,
    files: list[Path],
    metadata: dict[str, Any] | None = None,
    aliases: list[str] | None = None,
) -> None:
    import wandb

    artifact = wandb.Artifact(name, type=artifact_type, metadata=metadata or {})
    for file_path in files:
        artifact.add_file(str(file_path))
    wandb.log_artifact(artifact, aliases=aliases or None)


def save_wandb_metadata(path: Path, run: Any) -> None:
    payload = {
        "wandb_id": run.id if run else None,
        "wandb_project": run.project if run else None,
        "wandb_run_name": run.name if run else None,
        "wandb_run_url": run.url if run else None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2), encoding="utf-8")
