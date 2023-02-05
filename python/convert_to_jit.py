import importlib
import sys
from pathlib import Path
from typing import NamedTuple

import torch

sys.path.append(str(Path(__file__).resolve().parent / "FairMOT" / "src" / "lib"))
sys.path.append(str(Path(__file__).resolve().parent / "pytorch-dcnv2"))

sys.modules["dcn_v2"] = importlib.import_module("dcn")
sys.modules["dcn_v2"].__dict__["DCN"] = sys.modules["dcn_v2"].__dict__["DCNv2"]

from models.networks.pose_dla_dcn import DLASeg


class DLASegOutput(NamedTuple):
    hm: torch.Tensor
    wh: torch.Tensor
    id: torch.Tensor
    reg: torch.Tensor


class DLASegCustom(DLASeg):
    def forward(self, x):
        return DLASegOutput(**super().forward(x)[-1])


def main():
    weights_dir = Path(__file__).resolve().parents[1] / "weights"
    weights_path = weights_dir / "fairmot_dla34.pth"
    converted_weights_path = weights_dir / "fairmot_dla34_jit.pth"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = DLASegCustom(
        "dla34",
        {"hm": 1, "wh": 4, "id": 128, "reg": 2},
        pretrained=False,
        down_ratio=4,
        final_kernel=1,
        last_level=5,
        head_conv=256,
    )
    checkpoint = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"], strict=False)

    model = model.to(device)
    model.eval()

    example_input = torch.rand(1, 3, 480, 864).to(device)
    script_module = torch.jit.trace(model, example_input)
    script_module.save(converted_weights_path)


if __name__ == "__main__":
    main()
