import pickle
import unittest
from types import SimpleNamespace

import torch

from nanovllm.engine.gpu_block_table import GpuBlockTableMirror
from nanovllm.engine.runtime_slots import RuntimeSlotManager
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams


class RuntimeSlotTest(unittest.TestCase):
    def setUp(self):
        self.original_block_size = Sequence.block_size
        Sequence.block_size = 4

    def tearDown(self):
        Sequence.block_size = self.original_block_size

    def test_slot_reuse_increments_epoch_and_rejects_stale_owner(self):
        manager = RuntimeSlotManager(1)
        first = Sequence([1, 2, 3, 4])
        second = Sequence([5, 6, 7, 8])
        first_lease = manager.acquire(first)
        manager.release(first)
        second_lease = manager.acquire(second)

        self.assertEqual(first_lease.slot, second_lease.slot)
        self.assertGreater(second_lease.epoch, first_lease.epoch)
        first.runtime_slot = first_lease.slot
        first.runtime_slot_epoch = first_lease.epoch
        with self.assertRaises(RuntimeError):
            manager.validate(first)

    def test_scheduler_holds_slot_only_while_request_is_running(self):
        config = SimpleNamespace(
            max_num_seqs=2,
            max_num_batched_tokens=16,
            eos=-1,
            num_kvcache_blocks=16,
            kvcache_block_size=4,
        )
        scheduler = Scheduler(config)
        seq = Sequence(
            [1, 2, 3, 4],
            SamplingParams(max_tokens=2, ignore_eos=True),
        )
        scheduler.add(seq)

        seqs, is_prefill = scheduler.schedule()
        self.assertTrue(is_prefill)
        self.assertNotEqual(seq.runtime_slot, -1)
        scheduler.postprocess(seqs, [9], is_prefill=True)
        active_slot = seq.runtime_slot

        seqs, is_prefill = scheduler.schedule()
        self.assertFalse(is_prefill)
        self.assertEqual(seq.runtime_slot, active_slot)
        scheduler.postprocess(seqs, [10], is_prefill=False)
        self.assertEqual(seq.runtime_slot, -1)
        self.assertTrue(scheduler.is_finished())

    def test_runtime_metadata_is_preserved_by_tp_serialization(self):
        seq = Sequence([1, 2, 3, 4])
        seq.runtime_slot = 3
        seq.runtime_slot_epoch = 7
        seq.block_table = [11, 12]
        seq.block_table_version = 4

        restored = pickle.loads(pickle.dumps(seq))
        self.assertEqual(restored.seq_id, seq.seq_id)
        self.assertEqual(restored.runtime_slot, 3)
        self.assertEqual(restored.runtime_slot_epoch, 7)
        self.assertEqual(restored.block_table_version, 4)
        self.assertEqual(restored.block_table, [11, 12])

    def test_waiting_prefill_does_not_overcommit_runtime_slots(self):
        config = SimpleNamespace(
            max_num_seqs=1,
            max_num_batched_tokens=16,
            eos=-1,
            num_kvcache_blocks=16,
            kvcache_block_size=4,
        )
        scheduler = Scheduler(config)
        first = Sequence(
            [1, 2, 3, 4],
            SamplingParams(max_tokens=3, ignore_eos=True),
        )
        second = Sequence(
            [5, 6, 7, 8],
            SamplingParams(max_tokens=2, ignore_eos=True),
        )
        scheduler.add(first)
        seqs, is_prefill = scheduler.schedule()
        scheduler.postprocess(seqs, [9], is_prefill)
        scheduler.add(second)

        seqs, is_prefill = scheduler.schedule()
        self.assertFalse(is_prefill)
        self.assertEqual(seqs, [first])
        self.assertEqual(list(scheduler.waiting), [second])
        self.assertEqual(sum(owner is not None for owner in scheduler.runtime_slots.owners), 1)


class GpuBlockTableMirrorTest(unittest.TestCase):
    def setUp(self):
        self.original_block_size = Sequence.block_size
        Sequence.block_size = 4

    def tearDown(self):
        Sequence.block_size = self.original_block_size

    def test_reference_gather_delta_and_epoch_reuse(self):
        slots = RuntimeSlotManager(2)
        first = Sequence([1, 2, 3, 4])
        second = Sequence([5, 6, 7, 8])
        slots.acquire(first)
        slots.acquire(second)
        first.block_table = [5, 19]
        first.block_table_version = 1
        second.block_table = [7]
        second.block_table_version = 1

        mirror = GpuBlockTableMirror(2, 4, device="cpu")
        initial = mirror.synchronize([first, second])
        self.assertEqual(len(initial), 8)
        gathered = mirror.gather_reference([second, first], 4, 2)
        expected = torch.tensor(
            [[7, -1], [5, 19], [-1, -1], [-1, -1]], dtype=torch.int32
        )
        self.assertTrue(torch.equal(gathered, expected))
        self.assertEqual(mirror.synchronize([first, second]), [])

        first.block_table.append(23)
        first.block_table_version += 1
        incremental = mirror.synchronize([first])
        self.assertEqual(len(incremental), 1)
        self.assertEqual(incremental[0].column, 2)
        self.assertEqual(incremental[0].block_id, 23)

        old_slot = first.runtime_slot
        old_epoch = first.runtime_slot_epoch
        slots.release(first)
        replacement = Sequence([9, 10, 11, 12])
        lease = slots.acquire(replacement)
        self.assertEqual(lease.slot, old_slot)
        self.assertGreater(lease.epoch, old_epoch)
        replacement.block_table = [31]
        replacement.block_table_version = 1
        replacement_deltas = mirror.synchronize([replacement])
        self.assertEqual(len(replacement_deltas), 4)
        self.assertTrue(all(delta.full_row for delta in replacement_deltas))
        self.assertEqual(mirror.master_block_tables[old_slot].tolist(), [31, -1, -1, -1])


if __name__ == "__main__":
    unittest.main()
