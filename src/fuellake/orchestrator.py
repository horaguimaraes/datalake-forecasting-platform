"""A small DAG runner: dependency ordering, retries, failure propagation and a durable run log.

It deliberately does what an Airflow/Prefect DAG does at this scale (each task declares its
upstream tasks; failures skip everything downstream) without needing a scheduler or a database,
so the whole pipeline runs with one command, in CI, and inside unit tests.
"""

from __future__ import annotations

import json
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SUCCESS, FAILED, SKIPPED = "success", "failed", "skipped"


class CycleError(ValueError):
    """The task graph has a dependency cycle."""


@dataclass
class Task:
    name: str
    fn: Callable[[], Any]
    deps: tuple[str, ...] = ()
    retries: int = 0
    retry_delay: float = 2.0
    description: str = ""


@dataclass
class TaskResult:
    name: str
    status: str
    attempts: int = 0
    duration_s: float = 0.0
    error: str | None = None
    output: Any = None


@dataclass
class RunResult:
    run_id: str
    started_at: str
    finished_at: str
    results: dict[str, TaskResult] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return all(r.status == SUCCESS for r in self.results.values())

    @property
    def failed(self) -> list[str]:
        return [n for n, r in self.results.items() if r.status == FAILED]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "ok": self.ok,
            "tasks": {
                name: {
                    "status": r.status,
                    "attempts": r.attempts,
                    "duration_s": round(r.duration_s, 3),
                    "error": r.error,
                    "output": _jsonable(r.output),
                }
                for name, r in self.results.items()
            },
        }


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return repr(value)


class Pipeline:
    def __init__(self, tasks: list[Task]):
        self.tasks = {t.name: t for t in tasks}
        if len(self.tasks) != len(tasks):
            raise ValueError("Duplicate task names")
        for task in tasks:
            for dep in task.deps:
                if dep not in self.tasks:
                    raise ValueError(f"Task '{task.name}' depends on unknown task '{dep}'")
        self.order = self._toposort()

    def _toposort(self) -> list[str]:
        """Kahn's algorithm; ties broken by declaration order so runs are deterministic."""
        indegree = {name: len(t.deps) for name, t in self.tasks.items()}
        children: dict[str, list[str]] = {name: [] for name in self.tasks}
        for name, task in self.tasks.items():
            for dep in task.deps:
                children[dep].append(name)
        ready = [n for n in self.tasks if indegree[n] == 0]
        order: list[str] = []
        while ready:
            node = ready.pop(0)
            order.append(node)
            for child in children[node]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    ready.append(child)
        if len(order) != len(self.tasks):
            stuck = sorted(set(self.tasks) - set(order))
            raise CycleError(f"Dependency cycle involving: {', '.join(stuck)}")
        return order

    def select(self, only: list[str] | None = None, start_from: str | None = None) -> list[str]:
        """Tasks to run. `only` runs those plus nothing else; `start_from` runs it and everything downstream."""
        if only:
            unknown = set(only) - set(self.tasks)
            if unknown:
                raise ValueError(f"Unknown task(s): {', '.join(sorted(unknown))}")
            return [n for n in self.order if n in only]
        if start_from:
            if start_from not in self.tasks:
                raise ValueError(f"Unknown task: {start_from}")
            downstream = {start_from}
            for name in self.order:
                if any(dep in downstream for dep in self.tasks[name].deps):
                    downstream.add(name)
            return [n for n in self.order if n in downstream]
        return list(self.order)

    def run(
        self,
        only: list[str] | None = None,
        start_from: str | None = None,
        run_dir: Path | None = None,
        sleep: Callable[[float], None] = time.sleep,
        log: Callable[[str], None] = lambda message: None,
    ) -> RunResult:
        selected = self.select(only, start_from)
        started = datetime.now(UTC)
        run = RunResult(run_id=started.strftime("%Y%m%dT%H%M%S%fZ"), started_at=started.isoformat(), finished_at="")

        for name in selected:
            task = self.tasks[name]
            blocked = [d for d in task.deps if d in run.results and run.results[d].status != SUCCESS]
            if blocked:
                run.results[name] = TaskResult(name, SKIPPED, error=f"upstream not successful: {', '.join(blocked)}")
                log(f"- {name}: skipped ({', '.join(blocked)} did not succeed)")
                continue

            result = TaskResult(name, FAILED)
            began = time.perf_counter()
            for attempt in range(1, task.retries + 2):
                result.attempts = attempt
                try:
                    result.output = task.fn()
                    result.status, result.error = SUCCESS, None
                    break
                except Exception as error:  # noqa: BLE001 - a task may fail in any way; we record it
                    result.error = f"{type(error).__name__}: {error}"
                    detail = traceback.format_exc(limit=3)
                    log(f"- {name}: attempt {attempt} failed: {result.error}\n{detail}")
                    if attempt <= task.retries:
                        sleep(task.retry_delay * attempt)
            result.duration_s = time.perf_counter() - began
            run.results[name] = result
            log(f"- {name}: {result.status} in {result.duration_s:.1f}s (attempts: {result.attempts})")

        run.finished_at = datetime.now(UTC).isoformat()
        if run_dir is not None:
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / f"{run.run_id}.json").write_text(json.dumps(run.to_dict(), indent=2), encoding="utf-8")
        return run

    def to_mermaid(self) -> str:
        lines = ["flowchart LR"]
        for name in self.order:
            task = self.tasks[name]
            if not task.deps:
                lines.append(f"    {name}")
            for dep in task.deps:
                lines.append(f"    {dep} --> {name}")
        return "\n".join(lines)
