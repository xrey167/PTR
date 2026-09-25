"""Gate G0's data bands.

references.json (written by references.py on the full splits) must record every
band as passing, for the data the lock pins. The label-only bands are also
recomputed here on a regenerated prefix, so they are checked even where the
splits and references.json are absent.
"""

from __future__ import annotations

import json
import unittest
from collections import Counter

from _support import REFERENCES_PATH, gen, lock, prefix, score_items_from_tsv

BAND_IDS = {
    "test_iid_majority", "train_min_operator_share",
    "iid_decisive_validity", "iid_decisive_regime", "iid_decisive_budget",
    "iid_decisive_confidence", "iid_decisive_epistemic",
    "compose_epi_heldout_decisive", "compose_regime_regime_decisive", "iid_exact_ties",
    "hand_weighted", "bound_additive", "raw_blind_ceiling", "nuisance_only",
}
REFERENCE_NAMES = {
    "train_majority", "count_router", "unbound_bag_of_attributes", "bound_additive",
    "hand_primary", "hand_weighted", "validity_blind_rule", "raw_blind_exact_bayes", "nuisance_only",
}
N_SUBSET = 1000


@unittest.skipUnless(REFERENCES_PATH.is_file(), "references.json has not been written")
class RecordedBands(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.doc = json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))

    def test_references_describe_the_pinned_data(self):
        self.assertEqual(self.doc["data"]["data_fnv1a64"], lock()["data_fnv1a64"])
        self.assertTrue(self.doc["data"]["all_files_match_lock"])
        self.assertEqual(self.doc["type_codebook_fingerprint"], gen.CODEBOOK_FINGERPRINT)

    def test_every_band_is_recorded_with_its_value_and_passes(self):
        bands = {b["id"]: b for b in self.doc["bands"]}
        self.assertEqual(set(bands), BAND_IDS)
        for ident, entry in bands.items():
            with self.subTest(band=ident):
                self.assertIsInstance(entry["measured"], float)
                lower_ok = entry["lower"] is None or entry["measured"] >= entry["lower"]
                upper_ok = entry["upper"] is None or entry["measured"] <= entry["upper"]
                self.assertEqual(entry["pass"], lower_ok and upper_ok)
                self.assertTrue(entry["pass"], f"{ident}: {entry['measured']}")
        self.assertTrue(self.doc["all_bands_pass"])

    def test_every_reference_line_is_present(self):
        references = self.doc["references"]
        self.assertEqual(set(references), REFERENCE_NAMES)
        for name, values in references.items():
            self.assertEqual(set(values), {"val", "test_iid", "ood_compose_epi", "ood_compose_regime",
                                           "ood_distractors", "ood_validity", "ood_payload"}, name)
        self.assertEqual(self.doc["no_transfer"]["ood_compose_regime"]["max_no_transfer_on_transfer_subset"], 0.0)
        self.assertEqual(self.doc["no_transfer"]["ood_compose_epi"]["ignore_heldout_on_heldout_subset"], 0.0)


class LabelBandsOnAPrefix(unittest.TestCase):
    """The label-only bands, recomputed on the first 1000 examples of each split."""

    @classmethod
    def setUpClass(cls):
        cls.items = {s: score_items_from_tsv(s, prefix(s, N_SUBSET)[1])
                     for s in ("test_iid", "ood_compose_epi", "ood_compose_regime")}

    def test_iid_majority_and_ties(self):
        items = self.items["test_iid"]
        counts = Counter(i.label for i in items)
        self.assertLessEqual(max(counts.values()) / len(items), 0.25)
        ties = sum(sum(1 for v in i.z if v >= max(i.z) - 1e-9) > 1 for i in items)
        self.assertLessEqual(ties / len(items), 0.02)

    def test_iid_decisive_rates(self):
        items = self.items["test_iid"]
        for tag in ("validity", "regime", "budget", "confidence", "epistemic"):
            rate = sum(tag in i.tags for i in items) / len(items)
            with self.subTest(tag=tag):
                self.assertGreaterEqual(rate, 0.12)
                self.assertLessEqual(rate, 0.40)

    def test_compose_rates(self):
        epi = self.items["ood_compose_epi"]
        regime = self.items["ood_compose_regime"]
        self.assertGreaterEqual(sum("heldout" in i.tags for i in epi) / len(epi), 0.25)
        self.assertGreaterEqual(sum("regime" in i.tags for i in regime) / len(regime), 0.40)


if __name__ == "__main__":
    unittest.main()
