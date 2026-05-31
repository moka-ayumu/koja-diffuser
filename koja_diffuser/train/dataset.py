import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import lancedb
from typing import Literal
from collections import defaultdict
import numpy as np
import math


class NameDataset(Dataset):
    def __init__(self, t: Literal["ko", "ja"], max_len: int):
        db = lancedb.connect("./koja_diffuser/data/generated")
        table_name = "data_korea" if t == "ko" else "data"
        col_name = "name" if t == "ko" else "hiragana"

        raw_df = db.open_table(table_name).to_pandas()

        filtered_df = raw_df[raw_df[col_name].str.len() < max_len].reset_index(
            drop=True
        )

        self.names = filtered_df[col_name].values
        self.ages = filtered_df["age"].values // 10
        self.count = len(filtered_df)

    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        return {"name": self.names[idx], "age": self.ages[idx]}


def get_dataloader(
    t: Literal["ko", "ja"], max_len: int, batch_size: int = 5000, shuffle: bool = True
):
    dataset = NameDataset(t, max_len)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True
    )
    return dataloader


class Stage2NameDataset(IterableDataset):
    def __init__(
        self,
        max_len: int,
        batch_size: int = 5000,
        shuffle: bool = True,
        seed: int | None = None,
    ):
        self.ko_dataset = NameDataset("ko", max_len)
        self.ja_dataset = NameDataset("ja", max_len)

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed

        self.ko_by_age = self._group_by_age(self.ko_dataset)
        self.ja_by_age = self._group_by_age(self.ja_dataset)

        ko_ages = set(self.ko_by_age.keys())
        ja_ages = set(self.ja_by_age.keys())

        missing_in_ko = ja_ages - ko_ages
        missing_in_ja = ko_ages - ja_ages

        if missing_in_ko or missing_in_ja:
            raise ValueError(
                f"ko/ja 양쪽에 모두 존재하지 않는 age_group이 있습니다. "
                f"missing_in_ko={missing_in_ko}, missing_in_ja={missing_in_ja}"
            )

        self.age_groups = sorted(ko_ages)

    @staticmethod
    def _normalize_age_group(age: int) -> int:
        return min(max(int(age), 0), 9)

    def _group_by_age(self, dataset: NameDataset):
        grouped = defaultdict(list)

        for idx, age in enumerate(dataset.ages):
            age_group = self._normalize_age_group(age)
            grouped[age_group].append(idx)

        return dict(grouped)

    def _take_with_recycle(
        self,
        indices: list[int],
        pos: int,
        take_size: int,
    ):
        n = len(indices)
        result = []

        primary_take = min(take_size, n - pos)
        if primary_take > 0:
            result.extend(indices[pos : pos + primary_take])
            pos += primary_take

        remain = take_size - primary_take

        if remain > 0:
            repeat_count = remain // n
            tail_count = remain % n

            for _ in range(repeat_count):
                result.extend(indices)

            if tail_count > 0:
                result.extend(indices[:tail_count])

        return result, pos

    def __iter__(self):
        rng = np.random.default_rng(self.seed)

        ko_by_age = {}
        ja_by_age = {}

        for age in self.age_groups:
            ko_indices = list(self.ko_by_age[age])
            ja_indices = list(self.ja_by_age[age])

            if self.shuffle:
                rng.shuffle(ko_indices)
                rng.shuffle(ja_indices)

            ko_by_age[age] = ko_indices
            ja_by_age[age] = ja_indices

        ko_pos = {age: 0 for age in self.age_groups}
        ja_pos = {age: 0 for age in self.age_groups}

        age_idx = 0

        while age_idx < len(self.age_groups):
            batch_ko_indices = []
            batch_ja_indices = []
            batch_ages = []

            remaining_batch = self.batch_size

            while remaining_batch > 0 and age_idx < len(self.age_groups):
                age = self.age_groups[age_idx]

                ko_indices = ko_by_age[age]
                ja_indices = ja_by_age[age]

                ko_remaining = len(ko_indices) - ko_pos[age]
                ja_remaining = len(ja_indices) - ja_pos[age]

                if ko_remaining == 0 and ja_remaining == 0:
                    age_idx += 1
                    continue

                take_size = min(
                    remaining_batch,
                    max(ko_remaining, ja_remaining),
                )

                ko_taken, ko_pos[age] = self._take_with_recycle(
                    ko_indices,
                    ko_pos[age],
                    take_size,
                )

                ja_taken, ja_pos[age] = self._take_with_recycle(
                    ja_indices,
                    ja_pos[age],
                    take_size,
                )

                batch_ko_indices.extend(ko_taken)
                batch_ja_indices.extend(ja_taken)
                batch_ages.extend([age] * take_size)

                remaining_batch -= take_size

                if ko_pos[age] == len(ko_indices) and ja_pos[age] == len(ja_indices):
                    age_idx += 1

            if len(batch_ko_indices) == 0:
                break

            batch_age_tensor = torch.tensor(
                batch_ages,
                dtype=torch.long,
            )

            yield {
                "ko": {
                    "name": self.ko_dataset.names[batch_ko_indices].tolist(),
                    "age": batch_age_tensor.clone(),
                },
                "ja": {
                    "name": self.ja_dataset.names[batch_ja_indices].tolist(),
                    "age": batch_age_tensor.clone(),
                },
                "age_group": batch_age_tensor,
            }

    def __len__(self):
        total_aligned_samples = sum(
            max(
                len(self.ko_by_age[age]),
                len(self.ja_by_age[age]),
            )
            for age in self.age_groups
        )

        return math.ceil(total_aligned_samples / self.batch_size)


def get_stage2_dataloader(
    batch_size: int = 5000,
    max_len: int = 10000,
    shuffle: bool = True,
    seed: int | None = None,
):
    dataset = Stage2NameDataset(
        max_len=max_len,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
    )

    return DataLoader(
        dataset,
        batch_size=None,  # Dataset이 이미 batch 단위로 yield함
        num_workers=0,
        pin_memory=True,
    )
