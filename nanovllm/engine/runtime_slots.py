from collections import deque
from dataclasses import dataclass

from nanovllm.engine.sequence import Sequence


@dataclass(frozen=True, slots=True)
class RuntimeSlotLease:
    slot: int
    epoch: int


class RuntimeSlotManager:
    """Owns stable runtime rows for sequences participating in decode."""

    def __init__(self, max_num_seqs: int):
        if max_num_seqs < 1:
            raise ValueError("max_num_seqs must be positive")
        self.max_num_seqs = max_num_seqs
        self.free_slots = deque(range(max_num_seqs))
        self.epochs = [0] * max_num_seqs
        self.owners: list[int | None] = [None] * max_num_seqs

    def acquire(self, seq: Sequence) -> RuntimeSlotLease:
        if seq.runtime_slot != -1:
            self.validate(seq)
            return RuntimeSlotLease(seq.runtime_slot, seq.runtime_slot_epoch)
        if not self.free_slots:
            raise RuntimeError("no free runtime slots")
        slot = self.free_slots.popleft()
        self.epochs[slot] += 1
        epoch = self.epochs[slot]
        self.owners[slot] = seq.seq_id
        seq.runtime_slot = slot
        seq.runtime_slot_epoch = epoch
        return RuntimeSlotLease(slot, epoch)

    def release(self, seq: Sequence) -> None:
        if seq.runtime_slot == -1:
            return
        self.validate(seq)
        slot = seq.runtime_slot
        self.owners[slot] = None
        self.free_slots.append(slot)
        seq.runtime_slot = -1
        seq.runtime_slot_epoch = 0

    def validate(self, seq: Sequence) -> None:
        slot = seq.runtime_slot
        if not 0 <= slot < self.max_num_seqs:
            raise RuntimeError(f"invalid runtime slot {slot} for sequence {seq.seq_id}")
        if self.owners[slot] != seq.seq_id:
            raise RuntimeError(
                f"runtime slot {slot} belongs to {self.owners[slot]}, not {seq.seq_id}"
            )
        if self.epochs[slot] != seq.runtime_slot_epoch:
            raise RuntimeError(
                f"runtime slot {slot} epoch is {self.epochs[slot]}, "
                f"not {seq.runtime_slot_epoch}"
            )
