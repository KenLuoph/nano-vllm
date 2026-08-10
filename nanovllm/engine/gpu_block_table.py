from dataclasses import dataclass

import numpy as np
import torch

from nanovllm.engine.sequence import Sequence


@dataclass(frozen=True, slots=True)
class BlockTableDelta:
    runtime_slot: int
    column: int
    block_id: int
    epoch: int
    full_row: bool = False


class GpuBlockTableMirror:
    """Reference implementation of a versioned GPU-resident block-table mirror."""

    def __init__(
        self,
        max_num_seqs: int,
        max_num_blocks: int,
        *,
        device: torch.device | str = "cuda",
    ):
        if max_num_seqs < 1 or max_num_blocks < 1:
            raise ValueError("mirror dimensions must be positive")
        self.max_num_seqs = max_num_seqs
        self.max_num_blocks = max_num_blocks
        self.device = torch.device(device)
        self.master_block_tables = torch.full(
            (max_num_seqs, max_num_blocks),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        self.host_rows = np.full((max_num_seqs, max_num_blocks), -1, dtype=np.int32)
        self.seen_epochs = np.zeros(max_num_seqs, dtype=np.int64)
        self.seen_versions = np.full(max_num_seqs, -1, dtype=np.int64)

    def synchronize(self, seqs: list[Sequence]) -> list[BlockTableDelta]:
        deltas = []
        for seq in seqs:
            slot = seq.runtime_slot
            if not 0 <= slot < self.max_num_seqs:
                raise RuntimeError(f"sequence {seq.seq_id} has no active runtime slot")
            if len(seq.block_table) > self.max_num_blocks:
                raise ValueError(
                    f"sequence {seq.seq_id} block table exceeds mirror capacity"
                )

            epoch_changed = self.seen_epochs[slot] != seq.runtime_slot_epoch
            version_changed = self.seen_versions[slot] != seq.block_table_version
            if not epoch_changed and not version_changed:
                continue

            new_row = np.full(self.max_num_blocks, -1, dtype=np.int32)
            new_row[: len(seq.block_table)] = seq.block_table
            if epoch_changed:
                changed_columns = range(self.max_num_blocks)
            else:
                changed_columns = np.flatnonzero(self.host_rows[slot] != new_row).tolist()

            for column in changed_columns:
                deltas.append(
                    BlockTableDelta(
                        runtime_slot=slot,
                        column=int(column),
                        block_id=int(new_row[column]),
                        epoch=seq.runtime_slot_epoch,
                        full_row=epoch_changed,
                    )
                )

            self.host_rows[slot] = new_row
            self.seen_epochs[slot] = seq.runtime_slot_epoch
            self.seen_versions[slot] = seq.block_table_version
            # This is intentionally a clear PyTorch reference. Day 7 replaces it
            # with packed deltas consumed by a captured Triton kernel.
            self.master_block_tables[slot].copy_(
                torch.from_numpy(new_row).to(self.device)
            )
        return deltas

    def gather_reference(
        self,
        seqs: list[Sequence],
        graph_bucket: int,
        active_block_width: int,
    ) -> torch.Tensor:
        if len(seqs) > graph_bucket:
            raise ValueError("batch does not fit graph bucket")
        if not 0 < active_block_width <= self.max_num_blocks:
            raise ValueError("invalid active block width")
        gathered = torch.full(
            (graph_bucket, active_block_width),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        if seqs:
            slots = torch.tensor(
                [seq.runtime_slot for seq in seqs],
                dtype=torch.int64,
                device=self.device,
            )
            gathered[: len(seqs)].copy_(
                self.master_block_tables.index_select(0, slots)[:, :active_block_width]
            )
        return gathered
