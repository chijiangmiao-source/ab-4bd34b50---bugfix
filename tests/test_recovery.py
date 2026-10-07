"""Rule tests for the ARIES recovery engine and the audit store.

Run: python -m unittest -v tests.test_recovery
(no third-party dependencies)
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app"))

from recovery import RecoveryError, recover  # noqa: E402
from storage import AuditStore  # noqa: E402

ZEROS = "00" * 4096


def zeros_with(offset, hexbytes):
    """A 4096-byte page hex string with *hexbytes* written at *offset*."""
    raw = bytearray(4096)
    patch = bytes.fromhex(hexbytes)
    raw[offset : offset + len(patch)] = patch
    return bytes(raw).hex()


def page(pno, data=ZEROS, page_lsn=None):
    p = {"page": pno, "data": data}
    if page_lsn is not None:
        p["pageLSN"] = page_lsn
    return p


def begin(lsn, xid):
    return {"lsn": lsn, "type": "begin", "xid": xid}


def upd(lsn, xid, prev, pno, offset, before, after):
    return {
        "lsn": lsn, "type": "update", "xid": xid, "prevLSN": prev,
        "page": pno, "offset": offset, "before": before, "after": after,
    }


def commit(lsn, xid, prev):
    return {"lsn": lsn, "type": "commit", "xid": xid, "prevLSN": prev}


def abort(lsn, xid, prev):
    return {"lsn": lsn, "type": "abort", "xid": xid, "prevLSN": prev}


def end(lsn, xid, prev):
    return {"lsn": lsn, "type": "end", "xid": xid, "prevLSN": prev}


def run(audit_id="A", pages=None, wal=None):
    return recover({"auditId": audit_id, "pages": pages or [], "wal": wal or []})


class CommittedRedoTests(unittest.TestCase):
    def test_committed_missing_update_is_redone_and_kept(self):
        wal = [
            begin(10, "T1"),
            upd(20, "T1", 10, 1, 0, "00000000", "aaaaaaaa"),
            commit(30, "T1", 20),
            end(40, "T1", 30),
        ]
        v = run("commit-redo", [page(1, ZEROS, page_lsn=10)], wal)

        self.assertEqual(v["committedTransactions"], ["T1"])
        self.assertEqual(v["loserTransactions"], [])
        self.assertEqual(v["redone"], 1)
        self.assertEqual(v["undone"], 0)
        summary = v["pages"][0]
        self.assertTrue(summary["changed"])
        self.assertEqual(summary["pageLSNBefore"], 10)
        self.assertEqual(summary["pageLSNAfter"], 20)
        self.assertTrue(summary["data"].startswith("aaaaaaaa"))
        self.assertEqual(len(summary["sha256After"]), 64)

    def test_already_flushed_update_is_skipped(self):
        wal = [
            begin(10, "T1"),
            upd(20, "T1", 10, 1, 0, "00000000", "aaaaaaaa"),
            commit(30, "T1", 20),
            end(40, "T1", 30),
        ]
        # Crashed image already carries the update (pageLSN 20 + after bytes).
        v = run("already-disk", [page(1, zeros_with(0, "aaaaaaaa"), page_lsn=20)], wal)
        self.assertEqual(v["redone"], 0)
        self.assertFalse(v["pages"][0]["changed"])
        redo_rows = [t for t in v["trace"] if t["phase"] == "redo" and t["type"] == "update"]
        self.assertIn("跳过", redo_rows[0]["decision"])


class LoserUndoTests(unittest.TestCase):
    def sample_loser_payload(self, page7_lsn=30):
        # Mirrors /api/sample: T1 commits; T2 (loser) update LSN60 is
        # missing from the crashed page image and must be redone then undone.
        pages = [
            page(7, zeros_with(0, "11" * 16), page_lsn=page7_lsn),
            page(9, zeros_with(0, "ab" * 8), page_lsn=20),
        ]
        wal = [
            begin(10, "T1"),
            upd(20, "T1", 10, 9, 0, "00" * 8, "ab" * 8),
            upd(30, "T1", 20, 7, 0, "00" * 16, "11" * 16),
            begin(35, "T2"),
            {"lsn": 40, "type": "checkpoint",
             "transactions": {"T1": 30, "T2": 35},
             "dirtyPages": {"7": 30}},
            upd(60, "T2", 35, 7, 16, "00000000", "22222222"),
            commit(70, "T1", 30),
            end(80, "T1", 70),
        ]
        return pages, wal

    def test_loser_update_redone_then_undone(self):
        pages, wal = self.sample_loser_payload()
        v = run("loser-rollback", pages, wal)

        self.assertEqual(v["committedTransactions"], ["T1"])
        self.assertEqual(v["loserTransactions"], ["T2"])
        self.assertEqual(v["redone"], 1)   # only LSN60; LSN30 already on page
        self.assertEqual(v["undone"], 1)

        p7 = next(p for p in v["pages"] if p["page"] == 7)
        # T1 committed bytes remain; T2 bytes are rolled back to before-image.
        self.assertTrue(p7["data"].startswith("11" * 16))
        self.assertTrue(p7["data"][32:40] == "00" * 4)
        # Undo rolled pageLSN from 60 back along T2's prevLSN chain (35).
        self.assertEqual(p7["pageLSNBefore"], 30)
        self.assertEqual(p7["pageLSNAfter"], 35)

        p9 = next(p for p in v["pages"] if p["page"] == 9)
        self.assertFalse(p9["changed"])  # LSN20 was already on the image

    def test_undo_uses_interval_bytes_not_scalar_pagelsn(self):
        # Crashed image *already contains* T2's after bytes but with an
        # older pageLSN: redo is idempotently skipped, yet at undo time the
        # interval still carries exactly the loser update's after-image and
        # no committed transaction touched those positions.  Rollback is
        # therefore unambiguous and must proceed (the old scalar-pageLSN
        # guard wrongly rejected this interleaving-safe case).
        pages, wal = self.sample_loser_payload(page7_lsn=30)
        raw = bytearray.fromhex(pages[0]["data"])
        raw[16:20] = bytes.fromhex("22222222")
        pages[0]["data"] = bytes(raw).hex()
        v = run("undo-interval", pages, wal)
        p7 = next(p for p in v["pages"] if p["page"] == 7)
        self.assertEqual(v["undone"], 1)
        self.assertTrue(p7["data"][32:40] == "00" * 4)
        # Committed T1 bytes remain untouched.
        self.assertTrue(p7["data"].startswith("11" * 16))

    def test_aborted_without_end_is_also_rolled_back(self):
        wal = [
            begin(10, "T9"),
            upd(20, "T9", 10, 3, 0, "0000", "cccc"),
            abort(30, "T9", 20),
            # no end record before crash
        ]
        v = run("aborted-no-end", [page(3, ZEROS, page_lsn=10)], wal)
        self.assertEqual(v["abortedTransactions"], ["T9"])
        self.assertEqual(v["loserTransactions"], ["T9"])
        self.assertEqual(v["undone"], 1)
        self.assertTrue(v["pages"][0]["data"].startswith("0000"))


class InterleavedSamePageTests(unittest.TestCase):
    """Multiple transactions successively modify the SAME page.

    The crashed image may omit some/all updates.  After recovery the visible
    bytes, page summary (pageLSN/SHA-256) and the per-record undo evidence
    must agree: committed bytes survive, loser bytes never linger.
    """

    def _wal_loser_first_winner_later(self):
        # Loser T1 writes [0,4) and never commits; winner T2 writes the
        # non-overlapping [8,12) afterwards and commits.
        return [
            begin(10, "T1"),
            upd(20, "T1", 10, 1, 0, "00000000", "aaaaaaaa"),
            begin(30, "T2"),
            upd(40, "T2", 30, 1, 8, "00000000", "bbbbbbbb"),
            commit(50, "T2", 40),
            end(60, "T2", 50),
        ]

    def test_loser_first_committed_later_non_overlapping(self):
        # The headline defect: crashed image contains NEITHER update.
        wal = self._wal_loser_first_winner_later()
        v = run("interleave-missing", [page(1, ZEROS, page_lsn=10)], wal)

        p1 = v["pages"][0]
        raw = bytes.fromhex(p1["data"])
        aa = bytes.fromhex("aaaaaaaa")
        bb = bytes.fromhex("bbbbbbbb")
        # loser bytes gone, committed bytes present
        self.assertEqual(raw[0:4], b"\x00\x00\x00\x00")
        self.assertEqual(raw[8:12], bb)
        # nothing else changed
        self.assertEqual(raw[4:8], b"\x00\x00\x00\x00")
        self.assertEqual(raw[12:], b"\x00" * (4096 - 12))
        self.assertNotIn(aa, raw)
        # counters/evidence
        self.assertEqual(v["redone"], 2)
        self.assertEqual(v["undone"], 1)
        self.assertEqual(v["loserTransactions"], ["T1"])
        self.assertEqual(v["committedTransactions"], ["T2"])
        undo_rows = [t for t in v["trace"]
                     if t["phase"] == "undo" and t["type"] == "update"]
        self.assertEqual(len(undo_rows), 1)
        self.assertTrue(undo_rows[0]["changed"])
        self.assertEqual(undo_rows[0]["restoredBytes"], 4)
        self.assertEqual(undo_rows[0]["committedBytesKept"], 0)
        # The surviving committed update LSN 40 fixes the frozen pageLSN;
        # evidence and summary must agree.
        self.assertEqual(p1["pageLSNAfter"], 40)
        self.assertEqual(undo_rows[0]["pageLSNAfter"], 40)

    def test_frozen_summary_sha_matches_data(self):
        import hashlib
        wal = self._wal_loser_first_winner_later()
        v = run("interleave-sha", [page(1, ZEROS, page_lsn=10)], wal)
        p1 = v["pages"][0]
        self.assertEqual(
            hashlib.sha256(bytes.fromhex(p1["data"])).hexdigest(),
            p1["sha256After"],
        )
        self.assertNotEqual(p1["sha256After"], p1["sha256Before"])

    def test_winner_bytes_already_flushed_loser_never_flushed(self):
        # Crashed image already carries the committed [8,12) bytes
        # (pageLSN 40); the loser's [0,4) write never reached disk either
        # (zeros there).  Redo skips both; undo is an idempotent no-op.
        wal = self._wal_loser_first_winner_later()
        img = bytearray(4096)
        img[8:12] = bytes.fromhex("bbbbbbbb")
        v = run("interleave-winner-disk",
                [page(1, bytes(img).hex(), page_lsn=40)], wal)
        raw = bytes.fromhex(v["pages"][0]["data"])
        self.assertEqual(raw[0:4], b"\x00\x00\x00\x00")
        self.assertEqual(raw[8:12], bytes.fromhex("bbbbbbbb"))
        self.assertEqual(v["redone"], 0)
        self.assertEqual(v["undone"], 0)

    def test_loser_bytes_flushed_under_later_winner_pagelsn_still_undone(self):
        # pageLSN already stands at the winner's LSN 40 and the image
        # physically carries BOTH non-overlapping writes (redo skips both).
        # Undo must still remove the loser bytes despite the higher
        # scalar pageLSN — the old guard leaked them.
        wal = self._wal_loser_first_winner_later()
        img = bytearray(4096)
        img[0:4] = bytes.fromhex("aaaaaaaa")
        img[8:12] = bytes.fromhex("bbbbbbbb")
        v = run("interleave-both-disk",
                [page(1, bytes(img).hex(), page_lsn=40)], wal)
        raw = bytes.fromhex(v["pages"][0]["data"])
        self.assertEqual(raw[0:4], b"\x00\x00\x00\x00")
        self.assertEqual(raw[8:12], bytes.fromhex("bbbbbbbb"))
        self.assertEqual(v["redone"], 0)
        self.assertEqual(v["undone"], 1)

    def test_multiple_interleaved_modifications_same_page(self):
        # T1 and T2 alternate on the same page; T2 commits, T1 loses.
        wal = [
            begin(10, "T1"),
            upd(20, "T1", 10, 1, 0, "00000000", "aaaaaaaa"),
            begin(30, "T2"),
            upd(40, "T2", 30, 1, 4, "00000000", "bbbbbbbb"),
            upd(50, "T1", 20, 1, 8, "00000000", "cccccccc"),
            upd(60, "T2", 40, 1, 12, "00000000", "dddddddd"),
            commit(70, "T2", 60),
            end(80, "T2", 70),
        ]
        v = run("interleave-multi", [page(1, ZEROS, page_lsn=None)], wal)
        raw = bytes.fromhex(v["pages"][0]["data"])
        # Both T1 ranges rolled back; both T2 ranges visible.
        self.assertEqual(raw[0:4], b"\x00\x00\x00\x00")
        self.assertEqual(raw[4:8], bytes.fromhex("bbbbbbbb"))
        self.assertEqual(raw[8:12], b"\x00\x00\x00\x00")
        self.assertEqual(raw[12:16], bytes.fromhex("dddddddd"))
        self.assertEqual(v["undone"], 2)
        undo_lsns = [t["lsn"] for t in v["trace"]
                     if t["phase"] == "undo" and t["type"] == "update"]
        # Reverse-LSN order along T1's chain.
        self.assertEqual(undo_lsns, [50, 20])

    def test_multiple_failing_transactions_interleaved(self):
        wal = [
            begin(10, "A"),
            upd(20, "A", 10, 1, 0, "00000000", "aaaaaaaa"),
            begin(30, "B"),
            upd(40, "B", 30, 1, 4, "00000000", "bbbbbbbb"),
            abort(50, "A", 20),
            # B crashes without even an abort record; neither writes end.
        ]
        v = run("interleave-two-losers", [page(1, ZEROS, page_lsn=None)], wal)
        raw = bytes.fromhex(v["pages"][0]["data"])
        self.assertEqual(raw[0:8], b"\x00" * 8)
        self.assertEqual(sorted(v["loserTransactions"]), ["A", "B"])
        self.assertEqual(v["undone"], 2)

    def test_later_committed_write_touches_previous_loser_position(self):
        # Loser writes [0,4); a later committed txn overwrites the SAME
        # positions (its before-image is exactly the loser's after-image).
        wal = [
            begin(10, "L"),
            upd(20, "L", 10, 1, 0, "00000000", "aaaaaaaa"),
            begin(30, "W"),
            upd(40, "W", 30, 1, 0, "aaaaaaaa", "cccccccc"),
            commit(50, "W", 40),
            end(60, "W", 50),
        ]
        v = run("overlap-same-bytes", [page(1, ZEROS, page_lsn=10)], wal)
        raw = bytes.fromhex(v["pages"][0]["data"])
        # The committed value wins; the loser bytes never reappear.
        self.assertEqual(raw[0:4], bytes.fromhex("cccccccc"))
        undo_rows = [t for t in v["trace"]
                     if t["phase"] == "undo" and t["type"] == "update"]
        self.assertEqual(len(undo_rows), 1)
        self.assertFalse(undo_rows[0]["changed"])
        self.assertEqual(undo_rows[0]["committedBytesKept"], 4)
        self.assertEqual(undo_rows[0]["restoredBytes"], 0)

    def test_committed_write_partially_overlaps_loser_interval(self):
        # Loser writes 8 bytes [0,8); a later committed txn overwrites the
        # middle 2 bytes [2,4).  Undo restores before only on [0,2)∪[4,8),
        # the committed middle stays.
        wal = [
            begin(10, "L"),
            upd(20, "L", 10, 1, 0, "0000000000000000", "aaaaaaaaaaaaaaaa"),
            begin(30, "W"),
            upd(40, "W", 30, 1, 2, "aaaa", "bbbb"),
            commit(50, "W", 40),
            end(60, "W", 50),
        ]
        v = run("overlap-partial", [page(1, ZEROS, page_lsn=10)], wal)
        raw = bytes.fromhex(v["pages"][0]["data"])
        self.assertEqual(raw[0:2], b"\x00\x00")
        self.assertEqual(raw[2:4], bytes.fromhex("bbbb"))
        self.assertEqual(raw[4:8], b"\x00\x00\x00\x00")
        undo_rows = [t for t in v["trace"]
                     if t["phase"] == "undo" and t["type"] == "update"]
        self.assertEqual(undo_rows[0]["restoredBytes"], 6)
        self.assertEqual(undo_rows[0]["committedBytesKept"], 2)

    def test_loser_then_winner_then_another_loser_same_positions(self):
        # L1 writes [0,4), W commits over [0,4), L2 then writes [0,4) and
        # crashes.  Undoing L2 must RE-MATERIALIZE W's committed bytes
        # (L2's before-image), and undoing L1 must keep them.
        wal = [
            begin(10, "L1"),
            upd(20, "L1", 10, 1, 0, "00000000", "aaaaaaaa"),
            begin(30, "W"),
            upd(40, "W", 30, 1, 0, "aaaaaaaa", "cccccccc"),
            commit(50, "W", 40),
            end(60, "W", 50),
            begin(70, "L2"),
            upd(80, "L2", 70, 1, 0, "cccccccc", "dddddddd"),
        ]
        v = run("loser-committed-loser", [page(1, ZEROS, page_lsn=10)], wal)
        raw = bytes.fromhex(v["pages"][0]["data"])
        self.assertEqual(raw[0:4], bytes.fromhex("cccccccc"))
        # L2's rewrite physically re-materializes the committed value; L1's
        # undo then finds its interval fully committed-covered (a recorded
        # no-op), so exactly one page rewrite plus two undo evidence rows.
        undo_rows = [t for t in v["trace"]
                     if t["phase"] == "undo" and t["type"] == "update"]
        self.assertEqual([t["lsn"] for t in undo_rows], [80, 20])
        self.assertEqual(v["undone"], 1)
        self.assertFalse(undo_rows[1]["changed"])
        self.assertEqual(undo_rows[1]["committedBytesKept"], 4)

    def test_loser_chain_multiple_writes_same_range_fully_undone(self):
        wal = [
            begin(10, "L"),
            upd(20, "L", 10, 1, 0, "00000000", "aaaaaaaa"),
            upd(30, "L", 20, 1, 0, "aaaaaaaa", "cccccccc"),
        ]
        v = run("loser-twice", [page(1, ZEROS, page_lsn=None)], wal)
        raw = bytes.fromhex(v["pages"][0]["data"])
        self.assertEqual(raw[0:4], b"\x00\x00\x00\x00")
        self.assertEqual(v["undone"], 2)

    def test_undo_rejects_corrupt_unprotected_bytes(self):
        # pageLSN already covers every update so redo does not touch the
        # page, but the loser's interval holds garbage (neither its after
        # nor its before) and no committed txn covers those positions:
        # the before-image chain is untrustworthy -> stable rejection.
        wal = self._wal_loser_first_winner_later()
        img = bytearray(4096)
        img[0:4] = bytes.fromhex("ffffffff")   # corrupt loser interval
        img[8:12] = bytes.fromhex("bbbbbbbb")  # winner bytes intact
        with self.assertRaises(RecoveryError) as ctx:
            run("interleave-corrupt",
                [page(1, bytes(img).hex(), page_lsn=60)], wal)
        self.assertIn("前像链异常", str(ctx.exception))

    def test_undo_accepts_committed_value_when_loser_after_expected(self):
        # Post-redo bytes at the loser interval are the COMMITTED winner's
        # value (later winner overwrote the position): keeping them is
        # correct and must not be mistaken for a corrupt before-image.
        wal = [
            begin(10, "L"),
            upd(20, "L", 10, 1, 0, "00000000", "aaaaaaaa"),
            begin(30, "W"),
            upd(40, "W", 30, 1, 0, "aaaaaaaa", "cccccccc"),
            commit(50, "W", 40),
            end(60, "W", 50),
        ]
        # Crashed image already holds the final committed value cccc.
        img = bytearray(4096)
        img[0:4] = bytes.fromhex("cccccccc")
        v = run("committed-covers-loser",
                [page(1, bytes(img).hex(), page_lsn=50)], wal)
        raw = bytes.fromhex(v["pages"][0]["data"])
        self.assertEqual(raw[0:4], bytes.fromhex("cccccccc"))


class AnalysisTests(unittest.TestCase):
    def test_checkpoint_seeds_tables_and_redo_starts_at_min_reclsn(self):
        wal = [
            begin(10, "T0"),
            upd(15, "T0", 10, 2, 0, "0000", "dddd"),
            begin(20, "T1"),
            upd(25, "T1", 20, 1, 0, "0000", "eeee"),
            {"lsn": 30, "type": "checkpoint",
             "transactions": {"T0": 15, "T1": 25},
             "dirtyPages": {"1": 25}},   # page 2 flushed before checkpoint
            upd(35, "T1", 25, 1, 2, "0000", "ffff"),
        ]
        # Page 2 was flushed before the checkpoint: its crashed image
        # already carries LSN15 (and must not be redone/undone).
        v = run("cp-analysis",
                [page(1, ZEROS, page_lsn=None),
                 page(2, zeros_with(0, "dddd"), page_lsn=15)], wal)
        # Redo starts at 25 (min DPT recLSN); LSN15 on flushed page 2 is
        # not revisited.
        self.assertEqual(v["redoStartLSN"], 25)
        redo_lsns = [t["lsn"] for t in v["trace"]
                     if t["phase"] == "redo" and t["type"] == "update"]
        self.assertEqual(redo_lsns, [25, 35])
        # Both T0 and T1 are losers; undo walks both chains in reverse LSN,
        # including flushed page 2 (a committed-disk update of a loser is
        # still rolled back).
        self.assertEqual(sorted(v["loserTransactions"]), ["T0", "T1"])
        undo_lsns = [t["lsn"] for t in v["trace"]
                     if t["phase"] == "undo" and t["type"] == "update"]
        self.assertEqual(undo_lsns, [35, 25, 15])

    def test_no_checkpoint_analyzes_whole_wal(self):
        wal = [
            begin(10, "T1"),
            upd(20, "T1", 10, 1, 0, "0000", "aaaa"),
        ]
        v = run("no-cp", [page(1, ZEROS)], wal)
        # Analysis of the full WAL still builds the DPT; redo starts at the
        # only recLSN, loser T1 is then rolled back to the zero image.
        self.assertEqual(v["redoStartLSN"], 20)
        self.assertEqual(v["loserTransactions"], ["T1"])
        self.assertEqual(v["redone"], 1)
        self.assertEqual(v["undone"], 1)
        self.assertFalse(v["pages"][0]["changed"])


class StableRejectionTests(unittest.TestCase):
    def _base(self):
        return [
            begin(10, "T1"),
            upd(20, "T1", 10, 1, 0, "0000", "aaaa"),
            commit(30, "T1", 20),
            end(40, "T1", 30),
        ], [page(1, ZEROS, page_lsn=10)]

    def test_broken_prevlsn_chain(self):
        wal, pages = self._base()
        wal[1]["prevLSN"] = 999  # does not exist
        with self.assertRaises(RecoveryError) as ctx:
            run("broken-chain", pages, wal)
        self.assertIn("断链", str(ctx.exception))

    def test_prevlsn_points_into_other_transaction(self):
        wal, pages = self._base()
        wal.append(begin(50, "T2"))
        wal.append(upd(60, "T2", 20, 1, 4, "0000", "bbbb"))  # T1's LSN!
        with self.assertRaises(RecoveryError) as ctx:
            run("cross-txn-chain", pages, wal)
        self.assertIn("属于事务", str(ctx.exception))

    def _wal_with_second_committed_txn(self):
        return [
            begin(10, "T1"),
            upd(20, "T1", 10, 1, 0, "0000", "aaaa"),
            begin(50, "T2"),
            upd(60, "T2", 50, 1, 8, "0000", "bbbb"),
        ]

    def test_chain_out_of_order_predecessor(self):
        # T2 update points at begin 10 (T1) -> cross txn; instead build a
        # genuine out-of-order chain: two T2 updates whose middle link is
        # skipped.
        wal = [
            begin(10, "T2"),
            upd(20, "T2", 10, 1, 0, "0000", "aaaa"),
            upd(30, "T2", 20, 1, 4, "0000", "bbbb"),
            upd(40, "T2", 20, 1, 8, "0000", "cccc"),  # skips LSN30
        ]
        with self.assertRaises(RecoveryError) as ctx:
            run("chain-order", [page(1, ZEROS)], wal)
        self.assertIn("前驱链失序", str(ctx.exception))

    def test_duplicate_lsn(self):
        wal, pages = self._base()
        wal.append({"lsn": 20, "type": "begin", "xid": "T2"})
        with self.assertRaises(RecoveryError) as ctx:
            run("dup-lsn", pages, wal)
        self.assertIn("重复 LSN", str(ctx.exception))

    def test_wal_must_be_sorted(self):
        wal, pages = self._base()
        wal[1]["lsn"] = 50
        wal[2]["lsn"] = 25
        # rebuild ordering-independent fields consistently
        with self.assertRaises(RecoveryError) as ctx:
            run("unsorted", pages, wal)
        self.assertIn("升序", str(ctx.exception))

    def test_wrong_before_image_during_redo(self):
        wal, _ = self._base()
        # Page carries garbage where the before-image is expected.
        bad = [page(1, zeros_with(0, "ffff"), page_lsn=10)]
        with self.assertRaises(RecoveryError) as ctx:
            run("bad-before", bad, wal)
        self.assertIn("错误前像", str(ctx.exception))

    def test_out_of_range_interval(self):
        wal = [
            begin(10, "T1"),
            upd(20, "T1", 10, 1, 4095, "0000", "aaaa"),  # [4095,4097) > 4096
        ]
        with self.assertRaises(RecoveryError) as ctx:
            run("oob", [page(1)], wal)
        self.assertIn("越界", str(ctx.exception))

    def test_interval_ending_exactly_at_page_boundary_is_valid(self):
        wal = [
            begin(10, "T1"),
            upd(20, "T1", 10, 1, 4094, "0000", "aaaa"),  # [4094,4096) half-open
            commit(30, "T1", 20),
            end(40, "T1", 30),
        ]
        v = run("edge-ok", [page(1, ZEROS, page_lsn=10)], wal)
        self.assertEqual(v["redone"], 1)

    def test_unequal_before_after_length(self):
        wal = [
            begin(10, "T1"),
            upd(20, "T1", 10, 1, 0, "0000", "aa"),
        ]
        with self.assertRaises(RecoveryError) as ctx:
            run("unequal", [page(1)], wal)
        self.assertIn("等长", str(ctx.exception))

    def test_reference_to_ended_transaction(self):
        wal, pages = self._base()
        wal.append(upd(50, "T1", 40, 1, 4, "0000", "bbbb"))  # T1 already ended
        with self.assertRaises(RecoveryError) as ctx:
            run("ended-ref", pages, wal)
        self.assertIn("已结束事务", str(ctx.exception))

    def test_end_must_follow_terminator(self):
        wal = [begin(10, "T1"), upd(20, "T1", 10, 1, 0, "0000", "aaaa"),
               end(30, "T1", 20)]
        with self.assertRaises(RecoveryError) as ctx:
            run("bad-end", [page(1, ZEROS)], wal)
        self.assertIn("end 必须紧跟", str(ctx.exception))

    def test_record_after_commit_rejected(self):
        wal = [
            begin(10, "T1"),
            upd(20, "T1", 10, 1, 0, "0000", "aaaa"),
            commit(30, "T1", 20),
            upd(40, "T1", 30, 1, 4, "0000", "bbbb"),
        ]
        with self.assertRaises(RecoveryError):
            run("after-commit", [page(1, ZEROS)], wal)

    def test_checkpoint_dangling_dirty_reclsn_rejected(self):
        wal = [
            begin(10, "T1"),
            {"lsn": 20, "type": "checkpoint",
             "transactions": {}, "dirtyPages": {"1": 999}},
        ]
        with self.assertRaises(RecoveryError) as ctx:
            run("cp-dangle", [page(1, ZEROS)], wal)
        self.assertIn("断链", str(ctx.exception))

    def test_checkpoint_null_lastlsn_with_unverifiable_chain_rejected(self):
        # T3 is active at checkpoint with a null chain tail while its
        # predecessor (LSN 5) is absent from the WAL window: the chain
        # cannot be verified and recovery is stably rejected.
        wal = [
            begin(10, "T1"),
            upd(15, "T3", 5, 1, 0, "0000", "aaaa"),   # begin(5) outside window
            {"lsn": 20, "type": "checkpoint",
             "transactions": {"T3": None}, "dirtyPages": {}},
            upd(30, "T3", 15, 1, 4, "0000", "bbbb"),
        ]
        with self.assertRaises(RecoveryError):
            run("cp-null-lastlsn", [page(1, ZEROS)], wal)

    def test_checkpoint_reference_after_checkpoint_lsn_rejected(self):
        wal = [
            begin(10, "T1"),
            {"lsn": 20, "type": "checkpoint",
             "transactions": {"T1": 30}, "dirtyPages": {}},
            upd(30, "T1", 10, 1, 0, "0000", "aaaa"),
        ]
        with self.assertRaises(RecoveryError) as ctx:
            run("cp-future-lastlsn", [page(1, ZEROS)], wal)
        self.assertIn("不能晚于检查点", str(ctx.exception))

    def test_limits_enforced(self):
        pages = [page(i) for i in range(49)]
        with self.assertRaises(RecoveryError) as ctx:
            run("too-many-pages", pages, [])
        self.assertIn("48", str(ctx.exception))

        wal = [begin(i + 1, f"T{i}") for i in range(129)]
        with self.assertRaises(RecoveryError) as ctx:
            run("too-many-wal", [page(0)], wal)
        self.assertIn("128", str(ctx.exception))

    def test_missing_audit_id_rejected(self):
        with self.assertRaises(RecoveryError):
            recover({"pages": [page(0)], "wal": []})


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = os.path.join(self.tmp.name, "audit.db")
        self.store = AuditStore(self.db)

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def test_rejection_clears_prior_success_evidence(self):
        good_verdict = run("stable-id", [page(1, ZEROS, 10)],
                           [begin(10, "T1"), upd(20, "T1", 10, 1, 0, "0000", "aaaa"),
                            commit(30, "T1", 20), end(40, "T1", 30)])
        self.store.save_accepted("stable-id", {"auditId": "stable-id"}, good_verdict)
        frozen = self.store.get("stable-id")
        self.assertEqual(frozen["status"], "accepted")
        self.assertIn("verdict", frozen)

        # Same audit id, now a corrupted predecessor chain -> stable reject.
        bad_request = {"auditId": "stable-id", "pages": [page(1)],
                       "wal": [begin(10, "T1"),
                               upd(20, "T1", 999, 1, 0, "0000", "aaaa")]}
        self.store.save_rejected("stable-id", bad_request, "LSN 20 的 prevLSN 999 不存在（断链）")

        after = self.store.get("stable-id")
        self.assertEqual(after["status"], "rejected")
        self.assertNotIn("verdict", after)
        self.assertIn("断链", after["error"])

    def test_unrelated_audit_survives_other_rejection(self):
        self.store.save_accepted("keep", {}, {"auditId": "keep"})
        self.store.save_rejected("drop", {}, "bad")
        self.assertEqual(self.store.get("keep")["status"], "accepted")
        self.assertEqual(self.store.get("drop")["status"], "rejected")


if __name__ == "__main__":
    unittest.main(verbosity=2)
