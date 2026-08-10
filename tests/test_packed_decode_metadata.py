import unittest
from unittest.mock import patch

import numpy as np

from nanovllm.engine.gpu_block_table import BlockTableDelta
from nanovllm.engine.packed_decode_metadata import (
    DELTA_WORDS,
    HEADER_WORDS,
    MAGIC,
    ROW_WORDS,
    PackedDecodeMetadata,
)
from nanovllm.engine.sequence import Sequence


class PackedDecodeMetadataTest(unittest.TestCase):
    def setUp(self):
        self.original_block_size = Sequence.block_size
        Sequence.block_size = 4

    def tearDown(self):
        Sequence.block_size = self.original_block_size

    def make_sequence(self, token_ids, blocks, slot, epoch=1, temperature=0.6):
        seq = Sequence(token_ids)
        seq.block_table = list(blocks)
        seq.block_table_version = 1
        seq.runtime_slot = slot
        seq.runtime_slot_epoch = epoch
        seq.temperature = temperature
        return seq

    def test_layout_padding_and_delta_records(self):
        packed = PackedDecodeMetadata(8, 4, 4, pin_memory=False, device="cpu")
        seqs = [
            self.make_sequence([1, 2, 3, 4, 5], [10, 11], 2, temperature=0.5),
            self.make_sequence([6, 7, 8], [20], 5, temperature=0.8),
        ]
        deltas = [
            BlockTableDelta(2, 0, 10, 1, True),
            BlockTableDelta(2, 1, 11, 1, True),
            BlockTableDelta(5, 0, 20, 3, True),
        ]
        view = packed.prepare(seqs, 4, deltas, include_temperatures=True)

        self.assertEqual(view.active_block_width, 2)
        self.assertEqual(view.padding_rows, 2)
        self.assertEqual(view.delta_offset_words, HEADER_WORDS + 4 * ROW_WORDS)
        self.assertEqual(
            view.active_bytes,
            (HEADER_WORDS + 4 * ROW_WORDS + 3 * DELTA_WORDS) * 4,
        )
        self.assertEqual(packed.cpu_words[:7].tolist(), [MAGIC, 2, 4, 2, 3, 40, 52])
        self.assertEqual(packed.rows["input_id"][:4].tolist(), [5, 8, 0, 0])
        self.assertEqual(packed.rows["position"][:4].tolist(), [4, 2, 0, 0])
        self.assertEqual(packed.rows["slot_mapping"][:4].tolist(), [44, 82, -1, -1])
        self.assertEqual(packed.rows["context_len"][:4].tolist(), [5, 3, 0, 0])
        self.assertEqual(packed.rows["runtime_slot"][:4].tolist(), [2, 5, -1, -1])
        self.assertTrue(
            np.allclose(packed.rows["temperature"][:4], [0.5, 0.8, 1.0, 1.0])
        )
        delta_words = packed.cpu_words[40:52].reshape(-1, DELTA_WORDS)
        self.assertEqual(
            delta_words.tolist(),
            [[2, 0, 10, 1], [2, 1, 11, 1], [5, 0, 20, 3]],
        )

    def test_single_copy_uses_active_prefix_and_shared_storage(self):
        packed = PackedDecodeMetadata(4, 4, 4, pin_memory=False, device="cpu")
        seq = self.make_sequence([1, 2, 3, 4], [7], 0)
        view = packed.prepare([seq], 1, [], include_temperatures=False)
        self.assertEqual(
            packed.cpu_blob.data_ptr(), packed.cpu_bytes.__array_interface__["data"][0]
        )
        packed.gpu_blob.fill_(255)
        packed.copy_to_gpu(view.active_bytes)
        self.assertEqual(
            packed.gpu_blob[: view.active_bytes].tolist(),
            packed.cpu_blob[: view.active_bytes].tolist(),
        )
        self.assertTrue(
            all(value == 255 for value in packed.gpu_blob[view.active_bytes :].tolist())
        )

    def test_prepare_has_no_torch_tensor_allocation(self):
        packed = PackedDecodeMetadata(4, 4, 4, pin_memory=False, device="cpu")
        seq = self.make_sequence([1, 2, 3, 4], [7], 0)
        with (
            patch("torch.tensor", side_effect=AssertionError("torch.tensor called")),
            patch("torch.empty", side_effect=AssertionError("torch.empty called")),
        ):
            packed.prepare([seq], 1, [], include_temperatures=True)

    def test_batch_shrink_reinitializes_padding_rows(self):
        packed = PackedDecodeMetadata(8, 4, 4, pin_memory=False, device="cpu")
        large = [self.make_sequence([1, 2, 3, 4], [row + 1], row) for row in range(8)]
        packed.prepare(large, 8, [], include_temperatures=True)
        small = [self.make_sequence([9, 10, 11, 12], [31], 0)]
        packed.prepare(small, 1, [], include_temperatures=True)
        medium = [self.make_sequence([13, 14, 15, 16], [40 + row], row) for row in range(3)]
        packed.prepare(medium, 4, [], include_temperatures=True)
        self.assertEqual(packed.rows["context_len"][:4].tolist(), [4, 4, 4, 0])
        self.assertEqual(packed.rows["runtime_slot"][:4].tolist(), [0, 1, 2, -1])


if __name__ == "__main__":
    unittest.main()
