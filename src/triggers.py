from typing import Dict, List, Union, Tuple

import torch

Position = Union[str, Dict, List, Tuple]


def _position_to_slice(
    position: Position,
    image_size: int,
    trigger_size: int,
    margin: int,
):
    if isinstance(position, dict):
        if "y" not in position or "x" not in position:
            raise ValueError(f"Position dict must contain y and x: {position}")
        y0 = int(position["y"])
        x0 = int(position["x"])

    elif isinstance(position, (list, tuple)) and len(position) == 2:
        y0 = int(position[0])
        x0 = int(position[1])

    elif isinstance(position, str):
        if position == "top_left":
            y0 = margin
            x0 = margin
        elif position == "top_right":
            y0 = margin
            x0 = image_size - margin - trigger_size
        elif position == "bottom_left":
            y0 = image_size - margin - trigger_size
            x0 = margin
        elif position == "bottom_right":
            y0 = image_size - margin - trigger_size
            x0 = image_size - margin - trigger_size
        else:
            raise ValueError(f"Unknown position: {position}")
    else:
        raise ValueError(f"Unsupported position type: {position}")

    y0 = max(0, min(y0, image_size - trigger_size))
    x0 = max(0, min(x0, image_size - trigger_size))

    y1 = y0 + trigger_size
    x1 = x0 + trigger_size
    return y0, y1, x0, x1


def position_to_name(position: Position) -> str:
    if isinstance(position, dict):
        if "name" in position:
            return str(position["name"])
        return f"y{int(position['y'])}_x{int(position['x'])}"
    if isinstance(position, (list, tuple)) and len(position) == 2:
        return f"y{int(position[0])}_x{int(position[1])}"
    return str(position)


def apply_trigger(
    x: torch.Tensor,
    trigger_id: int,
    position: Position,
    image_size: int = 32,
    trigger_size: int = 4,
    margin: int = 2,
) -> torch.Tensor:
    out = x.clone()
    y0, y1, x0, x1 = _position_to_slice(
        position, image_size, trigger_size, margin
    )

    if out.dim() == 3:
        patch = out[:, y0:y1, x0:x1]
    elif out.dim() == 4:
        patch = out[:, :, y0:y1, x0:x1]
    else:
        raise ValueError(f"Expected CHW or BCHW tensor, got shape {tuple(out.shape)}")

    if trigger_id == 1:
        patch.zero_()
        if out.dim() == 3:
            patch[0, :, :] = 1.0
        else:
            patch[:, 0, :, :] = 1.0

    elif trigger_id == 2:
        patch.zero_()
        if out.dim() == 3:
            patch[2, :, :] = 1.0
        else:
            patch[:, 2, :, :] = 1.0

    else:
        raise ValueError(f"Unknown trigger_id: {trigger_id}")

    return out


def apply_dual_trigger(
    x: torch.Tensor,
    p1: Position,
    p2: Position,
    image_size: int = 32,
    trigger_size: int = 4,
    margin: int = 2,
) -> torch.Tensor:
    out = apply_trigger(
        x,
        trigger_id=1,
        position=p1,
        image_size=image_size,
        trigger_size=trigger_size,
        margin=margin,
    )
    out = apply_trigger(
        out,
        trigger_id=2,
        position=p2,
        image_size=image_size,
        trigger_size=trigger_size,
        margin=margin,
    )
    return out


def get_default_configs(
    kc: int,
    target_labels: List[int],
) -> List[Dict]:
    # Eight deterministic fixed relations.
    # The original first four configurations are preserved exactly.
    # The additional four form a rotationally symmetric clockwise cycle.
    pairs = [
        ("top_left", "bottom_right"),
        ("top_right", "bottom_left"),
        ("bottom_left", "top_right"),
        ("bottom_right", "top_left"),
        ("top_left", "top_right"),
        ("top_right", "bottom_right"),
        ("bottom_right", "bottom_left"),
        ("bottom_left", "top_left"),
    ]

    if kc > len(pairs):
        raise ValueError(f"Current default implementation supports kc <= {len(pairs)}.")

    if len(target_labels) < kc:
        raise ValueError("target_labels length must be >= kc.")

    configs = []
    for k in range(kc):
        p1, p2 = pairs[k]
        configs.append({
            "config_id": k,
            "name": f"c{k + 1}",
            "p1": p1,
            "p2": p2,
            "target": int(target_labels[k]),
        })
    return configs



def get_fixed_diverse_configs(
    kc: int,
    target_labels: List[int],
    image_size: int = 32,
    trigger_size: int = 4,
    margin: int = 2,
) -> List[Dict]:
    """
    Human-designed fixed-diverse baseline.

    Uses a deterministic 6x6 spatial grid. For Kc=8, all 16 physical
    trigger positions are unique. No dataset-dependent search is used.
    """
    if kc < 1 or kc > 8:
        raise ValueError("fixed_diverse supports 1 <= kc <= 8.")

    if len(target_labels) < kc:
        raise ValueError("target_labels length must be >= kc.")

    lo = int(margin)
    hi = int(image_size - margin - trigger_size)

    if hi <= lo:
        raise ValueError(
            f"Invalid spatial range: lo={lo}, hi={hi}, "
            f"image_size={image_size}, trigger_size={trigger_size}"
        )

    # Same coordinate scale as a six-point search grid.
    axis = [
        int(round(lo + i * (hi - lo) / 5.0))
        for i in range(6)
    ]

    # Sixteen unique points arranged symmetrically.
    pair_indices = [
        ((0, 0), (5, 5)),
        ((0, 5), (5, 0)),
        ((0, 2), (5, 3)),
        ((0, 3), (5, 2)),
        ((2, 0), (3, 5)),
        ((3, 0), (2, 5)),
        ((1, 1), (4, 4)),
        ((1, 4), (4, 1)),
    ]

    def make_position(index_pair):
        yi, xi = index_pair
        y = axis[yi]
        x = axis[xi]
        return {
            "name": f"fixed_diverse_y{y}_x{x}",
            "y": y,
            "x": x,
        }

    configs = []

    for config_id in range(kc):
        p1_index, p2_index = pair_indices[config_id]
        p1 = make_position(p1_index)
        p2 = make_position(p2_index)

        configs.append({
            "config_id": config_id,
            "name": f"fd{config_id + 1}",
            "p1": p1,
            "p2": p2,
            "target": int(target_labels[config_id]),
        })

    # Integrity check: Kc configurations must use 2*Kc unique locations.
    used_positions = []

    for config in configs:
        for key in ("p1", "p2"):
            position = config[key]
            used_positions.append(
                (int(position["y"]), int(position["x"]))
            )

    if len(set(used_positions)) != 2 * kc:
        raise RuntimeError(
            "fixed_diverse contains repeated physical positions."
        )

    return configs
