# api/orchestrator.py — stub (full implementation in Task 6)
import asyncio
from collections import defaultdict

_subscribers: dict[str, list[asyncio.Queue]] = defaultdict(list)


def subscribe_job(job_id: str, queue: asyncio.Queue) -> None:
    _subscribers[job_id].append(queue)


def unsubscribe_job(job_id: str, queue: asyncio.Queue) -> None:
    if queue in _subscribers.get(job_id, []):
        _subscribers[job_id].remove(queue)


def _push(job_id: str, event_type: str, data: dict) -> None:
    for q in list(_subscribers.get(job_id, [])):
        q.put_nowait({"type": event_type, "data": data})


async def run_pipeline(job_id: str, backend) -> None:
    pass  # replaced in Task 6
