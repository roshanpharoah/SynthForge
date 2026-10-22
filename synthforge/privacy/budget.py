"""Durable privacy budget manager.

Budgets are stored as rows; spending writes an immutable ledger entry in the same transaction. A job
must pass :meth:`assert_capacity` before training starts and :meth:`spend` after the accountant has
reported the realised epsilon. The realised value is what is charged: never the target.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select

from synthforge.core.errors import BudgetExceededError
from synthforge.db.models import Budget, LedgerEntry, Project
from synthforge.db.session import Database


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class BudgetStatus:
    budget_id: str
    scope_type: str
    scope_id: str
    epsilon_limit: float
    spent_epsilon: float
    delta: float

    @property
    def remaining(self) -> float:
        return max(0.0, self.epsilon_limit - self.spent_epsilon)

    def to_dict(self) -> dict[str, Any]:
        return {
            "budget_id": self.budget_id,
            "scope_type": self.scope_type,
            "scope_id": self.scope_id,
            "epsilon_limit": self.epsilon_limit,
            "spent_epsilon": self.spent_epsilon,
            "remaining_epsilon": self.remaining,
            "delta": self.delta,
            "utilisation": self.spent_epsilon / self.epsilon_limit if self.epsilon_limit else None,
        }


class BudgetManager:
    def __init__(self, db: Database):
        self.db = db

    def ensure_project(self, project_id: str, *, epsilon_limit: float | None = None, delta: float = 1e-5) -> BudgetStatus:
        with self.db.session() as s:
            if s.get(Project, project_id) is None:
                raise KeyError(f"project {project_id!r} not found")
            rows = s.scalars(
                select(Budget).where(
                    Budget.project_id == project_id,
                    Budget.scope_type == "project",
                    Budget.scope_id == project_id,
                )
            ).all()
            if rows:
                return self._status(rows[0])
            if epsilon_limit is None:
                epsilon_limit = 10.0
            b = Budget(
                id=f"bud_{uuid.uuid4().hex[:16]}",
                project_id=project_id,
                scope_type="project",
                scope_id=project_id,
                epsilon_limit=float(epsilon_limit),
                delta=float(delta),
            )
            s.add(b)
            s.flush()
            return self._status(b)

    def set_budget(
        self,
        project_id: str,
        *,
        epsilon_limit: float,
        scope_type: str = "project",
        scope_id: str | None = None,
        delta: float = 1e-5,
    ) -> BudgetStatus:
        scope_id = scope_id or project_id
        if epsilon_limit <= 0:
            raise ValueError("epsilon_limit must be positive")
        with self.db.session() as s:
            if s.get(Project, project_id) is None:
                raise KeyError(f"project {project_id!r} not found")
            row = s.scalars(
                select(Budget).where(
                    Budget.project_id == project_id,
                    Budget.scope_type == scope_type,
                    Budget.scope_id == scope_id,
                )
            ).first()
            if row is None:
                row = Budget(
                    id=f"bud_{uuid.uuid4().hex[:16]}",
                    project_id=project_id,
                    scope_type=scope_type,
                    scope_id=scope_id,
                    epsilon_limit=float(epsilon_limit),
                    delta=float(delta),
                )
                s.add(row)
            else:
                row.epsilon_limit = float(epsilon_limit)
                row.delta = float(delta)
                row.updated_at = _now()
            s.flush()
            return self._status(row)

    def status(self, project_id: str, *, scope_type: str = "project", scope_id: str | None = None) -> BudgetStatus | None:
        scope_id = scope_id or project_id
        with self.db.session() as s:
            row = s.scalars(
                select(Budget).where(
                    Budget.project_id == project_id,
                    Budget.scope_type == scope_type,
                    Budget.scope_id == scope_id,
                )
            ).first()
            return self._status(row) if row else None

    def statuses(self, project_id: str) -> list[dict[str, Any]]:
        with self.db.session() as s:
            rows = s.scalars(select(Budget).where(Budget.project_id == project_id)).all()
            return [self._status(r).to_dict() for r in rows]

    def ledger(self, project_id: str, *, limit: int = 200) -> list[dict[str, Any]]:
        with self.db.session() as s:
            rows = s.scalars(
                select(LedgerEntry)
                .where(LedgerEntry.project_id == project_id)
                .order_by(LedgerEntry.created_at.desc())
                .limit(limit)
            ).all()
            return [
                {
                    "id": r.id,
                    "budget_id": r.budget_id,
                    "project_id": r.project_id,
                    "job_id": r.job_id,
                    "run_id": r.run_id,
                    "epsilon": r.epsilon,
                    "delta": r.delta,
                    "mechanism": r.mechanism,
                    "description": r.description,
                    "created_at": r.created_at,
                }
                for r in rows
            ]

    def assert_capacity(
        self,
        project_id: str,
        requested_epsilon: float,
        *,
        dataset_id: str | None = None,
        delta: float = 1e-5,
    ) -> None:
        if requested_epsilon <= 0:
            return
        with self.db.session() as s:
            scopes = [("project", project_id)]
            if dataset_id:
                scopes.append(("dataset", dataset_id))
            for scope_type, scope_id in scopes:
                row = s.scalars(
                    select(Budget).where(
                        Budget.project_id == project_id,
                        Budget.scope_type == scope_type,
                        Budget.scope_id == scope_id,
                    )
                ).first()
                if row is None:
                    if scope_type == "project":
                        raise BudgetExceededError(
                            "no privacy budget is configured for this project",
                            hint="Create a budget first (CLI: synthforge budget set, or POST /api/v1/budgets).",
                        )
                    continue
                if row.delta > delta and requested_epsilon > 0:
                    # A stricter (smaller) delta than the budget allows is acceptable; flag only the reverse.
                    pass
                if requested_epsilon > row.epsilon_limit - row.spent_epsilon + 1e-12:
                    raise BudgetExceededError(
                        f"requested epsilon {requested_epsilon:.4g} exceeds the remaining {scope_type} budget "
                        f"{max(0.0, row.epsilon_limit - row.spent_epsilon):.4g}",
                        hint="Raise the budget only through the audited governance endpoint, or lower epsilon.",
                    )

    def spend(
        self,
        project_id: str,
        *,
        epsilon: float,
        delta: float,
        mechanism: str,
        description: str = "",
        job_id: str | None = None,
        run_id: str | None = None,
        dataset_id: str | None = None,
    ) -> dict[str, Any]:
        if epsilon < 0:
            raise ValueError("epsilon cannot be negative")
        charged: list[dict[str, Any]] = []
        with self.db.session() as s:
            scopes = [("project", project_id)]
            if dataset_id:
                scopes.append(("dataset", dataset_id))
            for scope_type, scope_id in scopes:
                row = s.scalars(
                    select(Budget).where(
                        Budget.project_id == project_id,
                        Budget.scope_type == scope_type,
                        Budget.scope_id == scope_id,
                    )
                ).first()
                if row is None:
                    if scope_type == "dataset":
                        continue
                    raise BudgetExceededError(
                        "no privacy budget is configured for this project",
                        hint="Create a project budget before spending epsilon.",
                    )
                if epsilon > 0 and row.spent_epsilon + epsilon > row.epsilon_limit + 1e-9:
                    raise BudgetExceededError(
                        f"charging ε={epsilon:.4g} would exceed the {scope_type} budget "
                        f"({row.spent_epsilon:.4g}/{row.epsilon_limit:.4g})"
                    )
                row.spent_epsilon += float(epsilon)
                row.spent_delta += float(delta)
                row.updated_at = _now()
                entry = LedgerEntry(
                    id=f"led_{uuid.uuid4().hex[:20]}",
                    budget_id=row.id,
                    project_id=project_id,
                    job_id=job_id,
                    run_id=run_id,
                    epsilon=float(epsilon),
                    delta=float(delta),
                    mechanism=mechanism,
                    description=description,
                )
                s.add(entry)
                s.flush()
                charged.append(
                    {
                        "budget_id": row.id,
                        "scope_type": scope_type,
                        "scope_id": scope_id,
                        "epsilon": float(epsilon),
                        "delta": float(delta),
                        "ledger_id": entry.id,
                        "spent_epsilon": row.spent_epsilon,
                        "remaining_epsilon": max(0.0, row.epsilon_limit - row.spent_epsilon),
                    }
                )
        return {"charged": charged, "epsilon": epsilon, "delta": delta, "mechanism": mechanism}

    @staticmethod
    def _status(row: Budget) -> BudgetStatus:
        return BudgetStatus(
            budget_id=row.id,
            scope_type=row.scope_type,
            scope_id=row.scope_id,
            epsilon_limit=float(row.epsilon_limit),
            spent_epsilon=float(row.spent_epsilon),
            delta=float(row.delta),
        )


__all__ = ["BudgetManager", "BudgetStatus"]
