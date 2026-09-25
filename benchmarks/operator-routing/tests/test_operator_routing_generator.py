"""The generator's PRNG, split invariants and digest stability."""

from __future__ import annotations

import hashlib
import unittest

from _support import DATA_DIR, data_on_disk, gen, lock, prefix, score

# Examples per split regenerated in memory by the invariant tests. The full
# splits are checked too when they are on disk.
N_MEMORY = 400


def tsv_rows(tsv: bytes):
    for line in tsv.decode("utf-8").splitlines():
        ident, label, regime, budget, tokens, facts = line.split("\t")
        parsed = []
        for chunk in facts.split(";"):
            r, e, c, v, ent = chunk.split(",")
            parsed.append((int(r), int(e), c, int(v), int(ent)))
        yield ident, int(label), int(regime), int(budget), [int(t) for t in tokens.split(",")], parsed


def split_rows(split: str):
    """Every row of the split on disk, or the in-memory prefix when it is absent."""
    if data_on_disk():
        return list(tsv_rows((DATA_DIR / f"{split}.tsv").read_bytes()))
    return list(tsv_rows(prefix(split, N_MEMORY)[1]))


class Prng(unittest.TestCase):
    def test_splitmix64_reference_vectors(self):
        # The published splitmix64 sequence for state 0.
        stream = gen.Stream(0)
        self.assertEqual(
            [stream.next() for _ in range(3)],
            [0xE220A8397B1DCDAF, 0x6E789E6AA1B965F4, 0x06C45D188009454F],
        )

    def test_single_value_splitmix64_is_one_output_of_a_stream(self):
        for x in (0, 1, 20260925, (1 << 64) - 1):
            self.assertEqual(gen.splitmix64(x), gen.Stream(x).next())

    def test_fnv1a64_reference_vectors(self):
        self.assertEqual(gen.fnv1a64(b""), 0xCBF29CE484222325)
        self.assertEqual(gen.fnv1a64(b"a"), 0xAF63DC4C8601EC8C)
        self.assertEqual(gen.fnv1a64(b"foobar"), 0x85944171F73967E8)
        self.assertEqual(score.fnv1a64(b"foobar"), 0x85944171F73967E8)

    def test_pinned_example_seeds(self):
        # Cross-language vectors (README.md, PRNG): the Rust loader can check these.
        self.assertEqual(gen.fnv1a64(b"train"), 0xDEE795A6C5087209)
        self.assertEqual(gen.example_seed("train", 0), 0x0E0645255755A741)
        stream = gen.Stream(gen.example_seed("train", 0))
        self.assertEqual(
            [stream.next() for _ in range(3)],
            [0x607D08C136D17281, 0x0A5AD4C5A0F9E481, 0xC3DDCFB2E72A2CAA],
        )
        self.assertEqual(gen.example_seed("test_iid", 1), 0x81D60976A260ED72)
        self.assertEqual(gen.example_seed("ood_payload", 2999), 0x6439903D016197D9)

    def test_randbelow_stays_below_n(self):
        top = ((1 << 53) - 1) / 2.0**53
        for n in range(1, 1001):
            self.assertLess(int(top * n), n)

    def test_confidence_bucket_is_never_ambiguous(self):
        for k in range(1000):
            c = gen.confidence(k)
            expected = (2 * k + 1) // 400  # floor((k + 0.5) / 200), exactly
            self.assertEqual(gen.bucket(c), expected)
            self.assertEqual(gen.bucket(float(f"{c:.4f}")), expected)
            self.assertEqual(score.bucket_of(f"{c:.4f}"), expected)


class SplitInvariants(unittest.TestCase):
    def test_held_out_cells_only_in_compose_epi(self):
        for split in gen.SPLITS:
            rows = split_rows(split)
            held = sum((f[0], f[1]) in gen.HELD_OUT for row in rows for f in row[5])
            with self.subTest(split=split):
                if split == "ood_compose_epi":
                    self.assertGreater(held, 0)
                else:
                    self.assertEqual(held, 0)

    def test_evidence_under_interventional_only_in_compose_regime(self):
        for split in gen.SPLITS:
            rows = split_rows(split)
            cells = sum(
                row[2] == gen.INTERVENTIONAL and f[0] == gen.EVIDENCE for row in rows for f in row[5]
            )
            with self.subTest(split=split):
                if split == "ood_compose_regime":
                    self.assertGreater(cells, 0)
                    self.assertTrue(all(row[2] == gen.INTERVENTIONAL for row in rows))
                else:
                    self.assertEqual(cells, 0)

    def test_every_example_has_a_live_fact(self):
        for split in gen.SPLITS:
            with self.subTest(split=split):
                self.assertTrue(all(any(f[3] == gen.LIVE for f in row[5]) for row in split_rows(split)))

    def test_split_acceptance_rules(self):
        epi = split_rows("ood_compose_epi")
        self.assertTrue(all(
            any(f[3] == gen.LIVE and score.bucket_of(f[2]) > 0 and (f[0], f[1]) in gen.HELD_OUT for f in row[5])
            for row in epi
        ))
        regime = split_rows("ood_compose_regime")
        self.assertTrue(all(
            any(f[3] == gen.LIVE and score.bucket_of(f[2]) > 0 and f[0] == gen.EVIDENCE for f in row[5])
            for row in regime
        ))

    def test_entities_tokens_and_lengths(self):
        for split in gen.SPLITS:
            base = 256 if split == "ood_payload" else 0
            length = 72 if split == "ood_distractors" else 36
            for ident, _label, regime, budget, tokens, facts in split_rows(split):
                with self.subTest(split=split, id=ident):
                    self.assertEqual(len(facts), 6)
                    self.assertTrue(all(base <= f[4] < base + 256 for f in facts))
                    self.assertEqual(len(tokens), length)
                    expected = []
                    for i, (r, e, c, v, _ent) in enumerate(facts):
                        t0 = 1 + 24 * i
                        expected += [t0 + r, t0 + 9 + e, t0 + 15 + score.bucket_of(c), t0 + 20 + v]
                    expected += [145 + regime, 149 + budget]
                    fillers = sorted(t for t in tokens if t >= 152)
                    self.assertEqual(sorted(t for t in tokens if t < 152), sorted(expected))
                    self.assertEqual(len(fillers), length - 26)
                    self.assertTrue(all(152 <= t <= 215 for t in fillers))

    def test_token_range_and_ids_everywhere(self):
        for split in gen.SPLITS:
            rows = split_rows(split)
            with self.subTest(split=split):
                self.assertTrue(all(0 < t < gen.VOCAB for row in rows for t in row[4]))
                ids = [row[0] for row in rows]
                self.assertEqual(ids, [f"{split}-{j:05d}" for j in range(len(rows))])
                if data_on_disk():
                    self.assertEqual(len(rows), gen.SPLIT_SIZES[split])


class Digests(unittest.TestCase):
    def test_lock_matches_the_design(self):
        document = lock()
        self.assertEqual(document["generator_version"], 1)
        self.assertEqual(document["benchmark_seed"], 20260925)
        self.assertTrue(document["type_codebook_fingerprint"].startswith("2b6f8175"))
        self.assertEqual({s: v["n"] for s, v in document["splits"].items()}, {
            "train": 16000, "val": 2000, "test_iid": 3000, "ood_compose_epi": 3000,
            "ood_compose_regime": 3000, "ood_distractors": 3000, "ood_validity": 3000,
            "ood_payload": 3000,
        })

    def test_regenerated_prefixes_match_the_lock(self):
        document = lock()
        n = document["prefix"]["n"]
        for split in gen.SPLITS:
            jsonl, tsv, _labels = prefix(split, n)
            pinned = document["prefix"]["splits"][split]
            with self.subTest(split=split):
                self.assertEqual(hashlib.sha256(jsonl).hexdigest(), pinned["jsonl_sha256"])
                self.assertEqual(hashlib.sha256(tsv).hexdigest(), pinned["tsv_sha256"])

    def test_regeneration_is_byte_identical(self):
        # Same bytes on a second render in the same process (no hidden state).
        first = gen.render_split("ood_distractors", 20)
        second = gen.render_split("ood_distractors", 20)
        self.assertEqual(first, second)

    @unittest.skipUnless(data_on_disk(), "generated splits are not on disk")
    def test_files_on_disk_match_the_lock(self):
        document = lock()
        state = gen.FNV_OFFSET
        for split in gen.SPLITS:
            for kind in ("jsonl", "tsv"):
                data = (DATA_DIR / f"{split}.{kind}").read_bytes()
                with self.subTest(split=split, kind=kind):
                    self.assertEqual(hashlib.sha256(data).hexdigest(), document["splits"][split][kind]["sha256"])
                if kind == "tsv":
                    state = gen.fnv1a64(data, state)
                    labels = "".join(format(int(line.split(b"\t")[1]), "x") for line in data.splitlines())
                    self.assertEqual(gen.hex64(gen.fnv1a64(labels.encode("ascii"))),
                                     document["splits"][split]["label_fnv1a64"])
        self.assertEqual(gen.hex64(state), document["data_fnv1a64"])


if __name__ == "__main__":
    unittest.main()
