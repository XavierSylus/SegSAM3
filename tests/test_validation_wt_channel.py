import torch
from torch.utils.data import DataLoader, Dataset

from src.client import ImageOnlyTrainer


class _SingleSampleDataset(Dataset):
    def __init__(self):
        self.image = torch.zeros(3, 2, 2, dtype=torch.float32)

        # Loader convention:
        # ch0 = background, ch1 = WT, ch2 = ET
        wt = torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.float32)
        bg = 1.0 - wt
        et = torch.zeros_like(wt)
        self.mask = torch.stack([bg, wt, et], dim=0)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.image, self.mask


class _DummyModel(torch.nn.Module):
    def forward(self, images):
        b, _, h, w = images.shape
        logits = torch.full((b, 3, h, w), -10.0, dtype=images.dtype, device=images.device)

        # WT channel predicts the mask perfectly.
        logits[:, 1, 0, 0] = 10.0

        # BG channel predicts the opposite region.
        logits[:, 0, 0, 0] = -10.0
        logits[:, 0, 0, 1] = 10.0
        logits[:, 0, 1, 0] = 10.0
        logits[:, 0, 1, 1] = 10.0
        return {"logits": logits}


def test_validate_uses_wt_channel_1_for_multichannel_outputs():
    trainer = ImageOnlyTrainer(
        private_loader=None,
        public_loader=None,
        device="cpu",
        local_epochs=1,
    )
    loader = DataLoader(_SingleSampleDataset(), batch_size=1, shuffle=False)
    model = _DummyModel()

    metrics = trainer.validate(model, loader, compute_hd95=False)
    assert metrics["dice"] > 0.99, f"Expected Dice near 1.0 when WT channel is correct, got {metrics['dice']:.6f}"
