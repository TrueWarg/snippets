import torch

# todo add numpy analog also
def xcycwh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    return torch.cat((
        boxes[:, :2] - boxes[:, 2:] / 2,
        boxes[:, :2] + boxes[:, 2:] / 2
    ), boxes.dim() - 1)


def xyxy_to_xcycwh(boxes: torch.Tensor) -> torch.Tensor:
    return torch.cat((
        (boxes[:, :2] + boxes[:, 2:]) / 2,
        boxes[:, 2:] - boxes[:, :2]
    ), boxes.dim() - 1)
