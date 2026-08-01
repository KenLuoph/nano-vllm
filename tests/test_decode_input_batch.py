import unittest
from unittest.mock import patch

import numpy as np
import torch

from nanovllm.engine.decode_input_batch import DecodeInputBatch


class FakeSequence:
    def __init__(self, length, last_token, block_table, block_size=4, temperature=1.0):
        self.num_tokens = length
        self.last_token = last_token
        self.block_table = list(block_table)
        self.block_size = block_size
        self.temperature = temperature

    def __len__(self):
        return self.num_tokens

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - ((self.num_tokens - 1) // self.block_size) * self.block_size


class DecodeInputBatchTest(unittest.TestCase):
    def make_batch(self, max_batch_size=16, max_num_blocks=8):
        return DecodeInputBatch(
            max_batch_size,
            max_num_blocks,
            block_size=4,
            pin_memory=False,
            allocate_temperature_gpu=False,
        )

    def test_vector_metadata_and_slot_mapping(self):
        batch = self.make_batch()
        seqs = [
            FakeSequence(5, 101, [3, 7], temperature=0.5),
            FakeSequence(8, 102, [4, 9], temperature=0.7),
            FakeSequence(3, 103, [11], temperature=0.9),
        ]
        view = batch.prepare_decode(seqs, graph_bucket=4, include_temperatures=True)

        self.assertEqual(view.active_block_width, 2)
        self.assertEqual(view.padding_rows, 1)
        self.assertEqual(batch.input_ids_cpu[:4].tolist(), [101, 102, 103, 0])
        self.assertEqual(batch.positions_cpu[:4].tolist(), [4, 7, 2, 0])
        self.assertEqual(batch.context_lens_cpu[:4].tolist(), [5, 8, 3, 0])
        self.assertEqual(batch.slot_mapping_cpu[:4].tolist(), [28, 39, 46, -1])
        self.assertEqual(
            batch.block_tables_cpu[:4, :2].tolist(),
            [[3, 7], [4, 9], [11, -1], [-1, -1]],
        )
        self.assertEqual(
            batch.temperatures_cpu[:3].tolist(),
            [np.float32(0.5), np.float32(0.7), np.float32(0.9)],
        )

    def test_bucket_padding_for_batch_five(self):
        batch = self.make_batch()
        seqs = [FakeSequence(4, 10 + i, [i]) for i in range(5)]
        batch.prepare_decode(seqs, graph_bucket=8, include_temperatures=False)

        self.assertEqual(batch.input_ids_cpu[5:8].tolist(), [0, 0, 0])
        self.assertEqual(batch.positions_cpu[5:8].tolist(), [0, 0, 0])
        self.assertEqual(batch.slot_mapping_cpu[5:8].tolist(), [-1, -1, -1])
        self.assertEqual(batch.context_lens_cpu[5:8].tolist(), [0, 0, 0])
        self.assertEqual(batch.block_tables_cpu[5:8, :1].tolist(), [[-1], [-1], [-1]])

    def test_batch_shrink_clears_every_accessible_row(self):
        batch = self.make_batch()
        large = [FakeSequence(8, 100 + i, [20 + i, 40 + i]) for i in range(16)]
        batch.prepare_decode(large, graph_bucket=16, include_temperatures=False)
        small = [FakeSequence(3, 200 + i, [60 + i]) for i in range(4)]
        view = batch.prepare_decode(small, graph_bucket=4, include_temperatures=False)

        self.assertEqual(view.active_block_width, 1)
        self.assertEqual(batch.input_ids_cpu[:4].tolist(), [200, 201, 202, 203])
        self.assertEqual(batch.block_tables_cpu[:4, :1].tolist(), [[60], [61], [62], [63]])
        self.assertTrue(all(length == 3 for length in batch.context_lens_cpu[:4].tolist()))

    def test_block_width_large_small_large(self):
        batch = self.make_batch(max_batch_size=8, max_num_blocks=8)
        batch.prepare_decode(
            [FakeSequence(16, 1, [1, 2, 3, 4])], 1, include_temperatures=False
        )
        batch.prepare_decode([FakeSequence(4, 2, [9])], 1, include_temperatures=False)
        view = batch.prepare_decode(
            [FakeSequence(12, 3, [10, 11, 12])], 1, include_temperatures=False
        )

        self.assertEqual(view.active_block_width, 3)
        self.assertEqual(batch.block_tables_cpu[0, :3].tolist(), [10, 11, 12])

    def test_short_rows_are_padded_with_minus_one(self):
        batch = self.make_batch()
        seqs = [
            FakeSequence(12, 1, [1, 2, 3]),
            FakeSequence(4, 2, [8]),
        ]
        batch.prepare_decode(seqs, graph_bucket=2, include_temperatures=False)
        self.assertEqual(batch.block_tables_cpu[:2, :3].tolist(), [[1, 2, 3], [8, -1, -1]])

    def test_temperature_is_only_written_when_requested(self):
        batch = self.make_batch()
        batch.temperatures_cpu.fill_(-123.0)
        seq = FakeSequence(4, 1, [2], temperature=0.25)

        view = batch.prepare_decode([seq], 1, include_temperatures=False)
        self.assertIsNone(view.temperatures)
        self.assertEqual(batch.temperatures_cpu[0].item(), -123.0)
        view = batch.prepare_decode([seq], 1, include_temperatures=True)
        self.assertIsNotNone(view.temperatures)
        self.assertAlmostEqual(batch.temperatures_cpu[0].item(), 0.25)

    def test_numpy_views_share_tensor_storage(self):
        batch = self.make_batch()
        pairs = [
            (batch.input_ids_cpu, batch.input_ids_np),
            (batch.positions_cpu, batch.positions_np),
            (batch.slot_mapping_cpu, batch.slot_mapping_np),
            (batch.context_lens_cpu, batch.context_lens_np),
            (batch.block_tables_cpu, batch.block_tables_np),
            (batch.temperatures_cpu, batch.temperatures_np),
        ]
        for tensor, array in pairs:
            self.assertEqual(tensor.data_ptr(), array.__array_interface__["data"][0])
        batch.input_ids_np[0] = 77
        self.assertEqual(batch.input_ids_cpu[0].item(), 77)

    def test_rejects_capacity_overflow(self):
        batch = self.make_batch(max_batch_size=4, max_num_blocks=2)
        with self.assertRaises(ValueError):
            batch.prepare_decode([FakeSequence(12, 1, [1, 2, 3])], 1, False)

    def test_steady_state_does_not_call_tensor_allocators(self):
        batch = self.make_batch()
        seqs = [FakeSequence(4, 1, [2]), FakeSequence(8, 2, [3, 4])]
        with (
            patch("torch.tensor", side_effect=AssertionError("torch.tensor called")),
            patch("torch.empty", side_effect=AssertionError("torch.empty called")),
        ):
            batch.prepare_decode(seqs, graph_bucket=2, include_temperatures=True)


if __name__ == "__main__":
    unittest.main()
