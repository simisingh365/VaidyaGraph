"""
VaidyaGraph - In-memory session store.

A "session" holds the frozen AgentState from a single /diagnose run plus
the running chat_history produced by follow-up /ask calls.

Scope
-----
This is DEMO-grade storage: a process-local dict with a bounded LRU cap.
It is NOT suitable for:
    * multi-worker deployments (each uvicorn worker has its own dict)
    * horizontal scaling (no shared backend)
    * persistence across restarts

Upgrade path (documented for reviewers)
---------------------------------------
    * Single-box production     -> SQLite with JSON column
    * Multi-worker / multi-box  -> Redis with TTL-bound keys
    * Long-term memory          -> Postgres + pgvector for semantic recall

Keeping the abstraction tight (get / put / append_turn) means any of those
swaps is a drop-in replacement.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import OrderedDict
from typing import Optional

from core.state import AgentState, ChatTurn

logger = logging.getLogger(__name__)

# Cap the store so a long-running demo can't OOM. LRU eviction - oldest
# session by last-access goes first.
_MAX_SESSIONS = 128

# Soft TTL (seconds). Sessions older than this are evicted on the next
# touch. 2 hours is plenty for a walkthrough; adjust as needed.
_SESSION_TTL_SEC = 2 * 60 * 60


class _SessionStore:
    def __init__(self) -> None:
        self._data: "OrderedDict[str, dict]" = OrderedDict()
        self._lock = threading.Lock()

    def create(self, state: AgentState) -> str:
        """Register a new session and return its id."""
        session_id = uuid.uuid4().hex[:16]
        # Ensure chat_history exists so later appends don't have to guard.
        state.setdefault("chat_history", [])
        with self._lock:
            self._data[session_id] = {
                "state": state,
                "created_at": time.time(),
                "last_access": time.time(),
            }
            self._evict_if_needed_locked()
        logger.info("Session created: %s (total=%d)", session_id, len(self._data))
        return session_id

    def get(self, session_id: str) -> Optional[AgentState]:
        with self._lock:
            entry = self._data.get(session_id)
            if entry is None:
                return None
            if time.time() - entry["created_at"] > _SESSION_TTL_SEC:
                # Expired.
                self._data.pop(session_id, None)
                return None
            # Mark as recently used for LRU.
            entry["last_access"] = time.time()
            self._data.move_to_end(session_id)
            return entry["state"]

    def append_turn(self, session_id: str, turn: ChatTurn) -> bool:
        """Append a turn to the session's chat_history. Returns success."""
        with self._lock:
            entry = self._data.get(session_id)
            if entry is None:
                return False
            state = entry["state"]
            history = state.setdefault("chat_history", [])
            history.append(turn)
            entry["last_access"] = time.time()
            self._data.move_to_end(session_id)
            return True

    def clear_history(self, session_id: str) -> bool:
        """Reset chat_history for a session but keep the panel output."""
        with self._lock:
            entry = self._data.get(session_id)
            if entry is None:
                return False
            entry["state"]["chat_history"] = []
            return True

    def _evict_if_needed_locked(self) -> None:
        """Drop the oldest entries until under cap. Caller holds the lock."""
        while len(self._data) > _MAX_SESSIONS:
            oldest_id, _ = self._data.popitem(last=False)
            logger.info("Evicting session (cap hit): %s", oldest_id)


# Module-level singleton. Import as:
#     from core.sessions import session_store
session_store = _SessionStore()


__all__ = ["session_store"]
