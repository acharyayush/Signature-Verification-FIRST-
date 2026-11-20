"""Regression tests for the per-dataset breakdown's handling of unscorable slices.

``training.evaluate_per_dataset`` runs the shared verification protocol on each dataset
slice and turns the result into a FAR / FRR / EER block. A slice is only comparable when
it holds all three query kinds: the signer's remaining genuine signatures, forgeries of
that signer, and genuine signatures of other signers. A grouping finer than a dataset --
for example ``{dataset}/{Genuine|Forgery}`` with ``--group-depth 2`` -- yields a
Genuine-only slice with no skilled forgeries and a Forgery-only slice with no genuine
images.

Before the fix the Genuine-only slice reached ``training.evaluate._metrics`` with an empty
``deep_skilled`` array and crashed the whole run inside ``rates_at_threshold``. These tests
pin down that ``_slice_scores`` reports it as a ``ValueError`` instead, so ``_collect`` can
skip the slice and the run finishes.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from training.evaluate_per_dataset import _missing_query_kinds, _slice_scores
from training.stage3_metric_learning import SignatureRecord


def _record(signer: str, is_forgery: bool) -> SignatureRecord:
    kind = "F" if is_forgery else "G"
    return SignatureRecord(Path(f"/tmp/{signer}_{kind}.png"), signer, is_forgery)


def _slice(genuine_per_signer: int, forgery_per_signer: int) -> tuple[list, np.ndarray]:
    """Two signers with the requested counts, plus unit-length toy embeddings."""
    records: list[SignatureRecord] = []
    for signer in ("signer_a", "signer_b"):
        for _ in range(genuine_per_signer):
            records.append(_record(signer, False))
        for _ in range(forgery_per_signer):
            records.append(_record(signer, True))
    rng = np.random.default_rng(0)
    embeddings = rng.normal(size=(len(records), 8))
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    return records, embeddings


class MissingQueryKindTests(unittest.TestCase):
    def test_reports_only_the_empty_kinds(self) -> None:
        scores = {
            "deep_genuine": np.array([0.5]),
            "deep_skilled": np.array([]),
            "deep_random": np.array([0.1]),
        }
        self.assertEqual(_missing_query_kinds(scores), ["skilled"])

    def test_reports_nothing_for_a_complete_slice(self) -> None:
        scores = {
            key: np.array([0.5])
            for key in ("deep_genuine", "deep_skilled", "deep_random")
        }
        self.assertEqual(_missing_query_kinds(scores), [])


class SliceScoresTests(unittest.TestCase):
    def _score(self, records, embeddings) -> dict[str, np.ndarray]:
        keys = ["toy"] * len(records)
        return _slice_scores(
            records,
            embeddings,
            [],
            keys,
            np.arange(len(records)),
            2,  # references
            None,  # scorer (deep-only)
            None,  # max_random_queries
            None,  # hand_normalizer
        )

    def test_genuine_only_slice_raises_instead_of_reaching_the_metrics(self) -> None:
        # The ``{dataset}/Genuine`` slice: no forgeries, so an empty skilled array would
        # otherwise reach ``rates_at_threshold`` and crash the whole evaluation.
        records, embeddings = _slice(genuine_per_signer=4, forgery_per_signer=0)
        with self.assertRaisesRegex(ValueError, "skilled"):
            self._score(records, embeddings)

    def test_forgery_only_slice_raises_during_enrolment(self) -> None:
        # The ``{dataset}/Forgery`` slice: no genuine images, so a signer cannot supply
        # the references. This is the message ``_collect`` already printed as "skipped".
        records, embeddings = _slice(genuine_per_signer=0, forgery_per_signer=3)
        with self.assertRaisesRegex(ValueError, "too few genuine"):
            self._score(records, embeddings)

    def test_complete_slice_returns_all_three_kinds(self) -> None:
        records, embeddings = _slice(genuine_per_signer=4, forgery_per_signer=2)
        scores = self._score(records, embeddings)
        self.assertEqual(scores["deep_genuine"].size, 4)  # 2 signers * (4 - 2 refs)
        self.assertEqual(scores["deep_skilled"].size, 4)  # 2 signers * 2 forgeries
        self.assertEqual(scores["deep_random"].size, 4)  # each signer vs the other's 2


if __name__ == "__main__":
    unittest.main()
