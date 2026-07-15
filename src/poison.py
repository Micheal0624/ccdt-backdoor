import random
from dataclasses import dataclass
from typing import Dict, List

from torch.utils.data import Dataset

from .adaptive_loss import GROUP_TO_ID
from .triggers import apply_trigger, apply_dual_trigger
from .utils import get_dataset_label


@dataclass
class PoisonRecord:
    base_index: int
    mode: str
    config_id: int
    target: int
    p1: str
    p2: str
    p1_alt: str = ""
    p2_alt: str = ""


def make_poison_index_splits(
    dataset: Dataset,
    poison_rate: float,
    configs: List[Dict],
    seed: int,
) -> Dict[int, List[int]]:
    rng = random.Random(seed)
    n_total = len(dataset)
    n_selected = int(round(poison_rate * n_total))
    n_per_config = max(1, n_selected // len(configs))

    all_indices = list(range(n_total))
    rng.shuffle(all_indices)

    used = set()
    splits = {}

    for cfg in configs:
        target = int(cfg["target"])
        chosen = []

        for idx in all_indices:
            if idx in used:
                continue
            y = get_dataset_label(dataset, idx)
            if int(y) == target:
                continue

            chosen.append(idx)
            used.add(idx)

            if len(chosen) >= n_per_config:
                break

        if len(chosen) < n_per_config:
            raise RuntimeError(
                f"Not enough candidates for config {cfg['name']} target={target}"
            )

        splits[int(cfg["config_id"])] = chosen

    return splits


class CCDTPoisonedDataset(Dataset):
    def __init__(
        self,
        base_dataset: Dataset,
        configs: List[Dict],
        poison_splits: Dict[int, List[int]],
        method: str,
        image_size: int = 32,
        trigger_size: int = 4,
        trigger_margin: int = 2,
        return_group: bool = False,
    ):
        self.base_dataset = base_dataset
        self.configs = configs
        self.poison_splits = poison_splits
        self.method = method
        self.image_size = image_size
        self.trigger_size = trigger_size
        self.trigger_margin = trigger_margin
        self.return_group = bool(return_group)

        self.records: List[PoisonRecord] = []
        self._build_records()

    def _build_records(self) -> None:
        valid_methods = {"single", "naive_dual", "wo_invalid", "full"}
        if self.method not in valid_methods:
            raise ValueError(f"Unknown method={self.method}, valid={valid_methods}")

        for cfg in self.configs:
            cid = int(cfg["config_id"])
            indices = self.poison_splits[cid]

            for idx in indices:
                if self.method == "single":
                    self.records.append(PoisonRecord(
                        base_index=idx,
                        mode="single_t1_positive",
                        config_id=cid,
                        target=int(cfg["target"]),
                        p1=cfg["p1"],
                        p2=cfg["p2"],
                    ))
                    self.records.append(PoisonRecord(
                        base_index=idx,
                        mode="single_t2_positive",
                        config_id=cid,
                        target=int(cfg["target"]),
                        p1=cfg["p1"],
                        p2=cfg["p2"],
                    ))

                elif self.method == "naive_dual":
                    self.records.append(PoisonRecord(
                        base_index=idx,
                        mode="valid_positive",
                        config_id=cid,
                        target=int(cfg["target"]),
                        p1=cfg["p1"],
                        p2=cfg["p2"],
                    ))

                elif self.method == "wo_invalid":
                    self.records.append(PoisonRecord(
                        base_index=idx,
                        mode="single_t1_negative",
                        config_id=cid,
                        target=int(cfg["target"]),
                        p1=cfg["p1"],
                        p2=cfg["p2"],
                    ))
                    self.records.append(PoisonRecord(
                        base_index=idx,
                        mode="single_t2_negative",
                        config_id=cid,
                        target=int(cfg["target"]),
                        p1=cfg["p1"],
                        p2=cfg["p2"],
                    ))
                    self.records.append(PoisonRecord(
                        base_index=idx,
                        mode="valid_positive",
                        config_id=cid,
                        target=int(cfg["target"]),
                        p1=cfg["p1"],
                        p2=cfg["p2"],
                    ))

                elif self.method == "full":
                    self.records.append(PoisonRecord(
                        base_index=idx,
                        mode="single_t1_negative",
                        config_id=cid,
                        target=int(cfg["target"]),
                        p1=cfg["p1"],
                        p2=cfg["p2"],
                    ))
                    self.records.append(PoisonRecord(
                        base_index=idx,
                        mode="single_t2_negative",
                        config_id=cid,
                        target=int(cfg["target"]),
                        p1=cfg["p1"],
                        p2=cfg["p2"],
                    ))
                    self.records.append(PoisonRecord(
                        base_index=idx,
                        mode="valid_positive",
                        config_id=cid,
                        target=int(cfg["target"]),
                        p1=cfg["p1"],
                        p2=cfg["p2"],
                    ))

                    for other_cfg in self.configs:
                        other_cid = int(other_cfg["config_id"])
                        if other_cid == cid:
                            continue

                        self.records.append(PoisonRecord(
                            base_index=idx,
                            mode="invalid_negative",
                            config_id=cid,
                            target=int(cfg["target"]),
                            p1=cfg["p1"],
                            p2=cfg["p2"],
                            p1_alt=cfg["p1"],
                            p2_alt=other_cfg["p2"],
                        ))

    def __len__(self):
        return len(self.base_dataset) + len(self.records)

    @staticmethod
    def _group_for_mode(mode: str) -> int:
        if mode == "valid_positive" or mode in {"single_t1_positive", "single_t2_positive"}:
            return GROUP_TO_ID["valid"]
        if mode in {"single_t1_negative", "single_t2_negative"}:
            return GROUP_TO_ID["single"]
        if mode == "invalid_negative":
            return GROUP_TO_ID["invalid"]
        raise ValueError(f"Unknown poison mode for grouping: {mode}")

    def __getitem__(self, index):
        if index < len(self.base_dataset):
            x, y = self.base_dataset[index]
            if self.return_group:
                return x, y, GROUP_TO_ID["clean"]
            return x, y

        record = self.records[index - len(self.base_dataset)]
        x, y = self.base_dataset[record.base_index]
        y = int(y)

        if record.mode == "single_t1_positive":
            x = apply_trigger(
                x, 1, record.p1,
                self.image_size, self.trigger_size, self.trigger_margin
            )
            y = int(record.target)

        elif record.mode == "single_t2_positive":
            x = apply_trigger(
                x, 2, record.p2,
                self.image_size, self.trigger_size, self.trigger_margin
            )
            y = int(record.target)

        elif record.mode == "single_t1_negative":
            x = apply_trigger(
                x, 1, record.p1,
                self.image_size, self.trigger_size, self.trigger_margin
            )

        elif record.mode == "single_t2_negative":
            x = apply_trigger(
                x, 2, record.p2,
                self.image_size, self.trigger_size, self.trigger_margin
            )

        elif record.mode == "valid_positive":
            x = apply_dual_trigger(
                x, record.p1, record.p2,
                self.image_size, self.trigger_size, self.trigger_margin
            )
            y = int(record.target)

        elif record.mode == "invalid_negative":
            x = apply_dual_trigger(
                x, record.p1_alt, record.p2_alt,
                self.image_size, self.trigger_size, self.trigger_margin
            )

        else:
            raise ValueError(f"Unknown poison mode: {record.mode}")

        if self.return_group:
            return x, y, self._group_for_mode(record.mode)
        return x, y
