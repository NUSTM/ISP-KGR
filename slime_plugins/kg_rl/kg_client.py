"""Async HTTP client for the KG query server (Flask service at port 5501).

Wraps the sync `requests.post` calls in ISP-KGR/infer_reasoning_k_beam.py and
ISP-KGR/k_beam_score.py with aiohttp so they can be `await`ed from the slime
rollout function.

The KG server itself (kg_query_server.py) is unchanged — copied verbatim into
examples/kg_rl/.
"""

from __future__ import annotations

import logging
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)


class KGClient:
    """Single-process aiohttp client. Hold one per rollout function call."""

    def __init__(
        self,
        kg_query_url: str = "http://localhost:5501/kg_query",
        distance_api_url: str = "http://localhost:5501/entity_distance",
        max_concurrency: int = 64,
        kg_query_timeout: float = 90.0,
        distance_timeout: float = 300.0,
    ):
        self.kg_query_url = kg_query_url
        self.distance_api_url = distance_api_url
        self.max_concurrency = max_concurrency
        self.kg_query_timeout = kg_query_timeout
        self.distance_timeout = distance_timeout
        self._session: aiohttp.ClientSession | None = None

    async def __aenter__(self):
        connector = aiohttp.TCPConnector(limit=self.max_concurrency, enable_cleanup_closed=True)
        timeout = aiohttp.ClientTimeout(total=max(self.kg_query_timeout, self.distance_timeout))
        self._session = aiohttp.ClientSession(connector=connector, timeout=timeout)
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def _kg_query(self, query_data: dict, timeout: float) -> dict:
        """Mirror infer_reasoning_k_beam.py:62-79 (kg_query_request), async."""
        try:
            assert self._session is not None, "Use KGClient as async context manager"
            async with self._session.post(
                self.kg_query_url,
                json={"query": query_data},
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as resp:
                if resp.status != 200:
                    return {}
                payload = await resp.json()
                if payload.get("status") == "success":
                    return payload.get("results", {})
                return {}
        except Exception as e:
            logger.warning("kg_query timed out / failed: %s", e)
            return {}

    async def query_full_sparql(self, sparql: str) -> list:
        """Run a free-form SPARQL string. Mirror infer_reasoning_k_beam.py:81-89."""
        query_data = {
            "type": "sparql",
            "content": sparql,
            "parameters": {"idx": 0},
        }
        result = await self._kg_query(query_data, timeout=30.0)
        return result.get("results", [])

    async def query_node_relation(
        self,
        mid: str | None,
        sparql: str = "",
        question: str = "",
    ) -> list:
        """Query relations adjacent to a node. Mirror infer_reasoning_k_beam.py:91-112."""
        if mid is None:
            return []
        query_data = {
            "type": "node",
            "content": mid,
            "entity_query": True,
            "parameters": {
                "idx": 0,
                "previous_sparql": sparql,
                "question": question,
                "filter_k": 10,
                "filter_threshold": 0.0,
            },
        }
        result = await self._kg_query(query_data, timeout=self.kg_query_timeout)
        return result.get("results", []) or []

    async def entity_distance(
        self,
        set_a: list[str],
        set_b: list[str],
        max_distance: int = 3,
        early_stop_global_min: int = 1,
    ) -> float | None:
        """POST to /entity_distance. Returns the distance scalar or None on failure.
        Mirror k_beam_score.py:108-126.
        """
        if not set_a or not set_b:
            return None
        payload: dict[str, Any] = {
            "set_a": set_a,
            "set_b": set_b,
            "max_distance": max_distance,
            "early_stop_global_min": early_stop_global_min,
        }
        try:
            assert self._session is not None, "Use KGClient as async context manager"
            async with self._session.post(
                self.distance_api_url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.distance_timeout),
            ) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
                d = data.get("distance", max_distance)
                return max(0.0, min(float(max_distance), float(d)))
        except Exception as e:
            logger.warning("entity_distance failed: %s", e)
            return None
