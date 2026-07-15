#!/usr/bin/env python3

import argparse
import copy
import importlib.util
import json
import re
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
BASE_SCRIPT = PROJECT / "tools/evaluate_out_of_library_invalid.py"
DEFAULT_MANIFEST = PROJECT / "common_pair_manifest_12.json"

RUN_PATTERN = re.compile(
    r"^(?P<dataset>cifar10|cifar100|gtsrb)_"
    r"(?P<model>resnet18|vgg11)_"
    r"full_dyn_pr0\.05_"
    r"kc(?P<kc>4|8)_"
    r"seed(?P<seed>0|1|2)$"
)


def load_base_module():
    spec = importlib.util.spec_from_file_location(
        "out_of_library_base",
        BASE_SCRIPT,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"无法加载基础脚本：{BASE_SCRIPT}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    wrapper_parser = argparse.ArgumentParser(
        add_help=False,
    )
    wrapper_parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
    )

    wrapper_args, remaining_args = (
        wrapper_parser.parse_known_args()
    )

    manifest_path = wrapper_args.manifest.resolve()

    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"统一组合清单不存在：{manifest_path}"
        )

    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    groups = manifest.get("groups", {})

    if len(groups) != 12:
        raise RuntimeError(
            f"Manifest 分组数量异常：{len(groups)}，应为12"
        )

    base = load_base_module()

    def build_common_pairs(
        run_name,
        configs,
        grid_positions,
        pairs_per_category,
    ):
        match = RUN_PATTERN.match(run_name)

        if not match:
            raise ValueError(
                f"无法解析运行名称：{run_name}"
            )

        dataset = match.group("dataset")
        model = match.group("model")
        kc = int(match.group("kc"))

        group_key = f"{dataset}|{model}|{kc}"

        if group_key not in groups:
            raise KeyError(
                f"Manifest 缺少分组：{group_key}"
            )

        group = groups[group_key]
        manifest_pairs = group["pairs"]

        p1_used = {
            base.position_xy(config["p1"])
            for config in configs
        }

        p2_used = {
            base.position_xy(config["p2"])
            for config in configs
        }

        all_used = p1_used | p2_used

        grid_coordinates = {
            base.position_xy(position)
            for position in grid_positions
        }

        current_unused = (
            grid_coordinates - all_used
        )

        selected = {}

        for category in base.CATEGORIES:
            available = manifest_pairs.get(
                category,
                [],
            )

            if pairs_per_category > len(available):
                raise RuntimeError(
                    f"{group_key}/{category} "
                    f"请求{pairs_per_category}组，"
                    f"但清单只有{len(available)}组"
                )

            chosen = copy.deepcopy(
                available[:pairs_per_category]
            )

            for pair in chosen:
                p1 = base.position_xy(pair["p1"])
                p2 = base.position_xy(pair["p2"])

                if category == "seen_p1_unseen_p2":
                    valid = (
                        p1 in p1_used
                        and p2 in current_unused
                    )

                elif category == "unseen_p1_seen_p2":
                    valid = (
                        p1 in current_unused
                        and p2 in p2_used
                    )

                elif category == "unseen_p1_unseen_p2":
                    valid = (
                        p1 in current_unused
                        and p2 in current_unused
                        and p1 != p2
                    )

                else:
                    raise ValueError(
                        f"未知类别：{category}"
                    )

                if not valid:
                    raise RuntimeError(
                        f"{run_name} 组合验证失败："
                        f"{category} "
                        f"{base.position_name(pair['p1'])}+"
                        f"{base.position_name(pair['p2'])}"
                    )

            selected[category] = chosen

        print(
            f"[COMMON MANIFEST] {group_key} | "
            f"seed={match.group('seed')} | "
            f"pairs="
            f"{len(selected['seen_p1_unseen_p2'])}/"
            f"{len(selected['unseen_p1_seen_p2'])}/"
            f"{len(selected['unseen_p1_unseen_p2'])}",
            flush=True,
        )

        return (
            selected,
            sorted(all_used),
            sorted(current_unused),
        )

    base.build_out_of_library_pairs = (
        build_common_pairs
    )

    # 移除 wrapper 自己的 --manifest 参数，
    # 其余参数交给原始评估脚本解析。
    sys.argv = [sys.argv[0]] + remaining_args

    base.main()


if __name__ == "__main__":
    main()
