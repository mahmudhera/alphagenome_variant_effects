from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch


def organism_to_index(organism: str) -> int:
    value = organism.lower()
    if value in {"human", "homo_sapiens", "homo-sapiens", "0"}:
        return 0
    if value in {"mouse", "mus_musculus", "mus-musculus", "1"}:
        return 1
    raise ValueError(f"Unknown organism {organism!r}; expected human or mouse")


@dataclass
class BackendConfig:
    backend: str
    weights: Optional[str]
    device: str
    organism: str
    named_outputs: bool = True
    include_padding: bool = False
    compile_model: bool = False


class BaseBackend:
    def predict(self, onehot_batch: torch.Tensor) -> Any:
        raise NotImplementedError


class MockBackend(BaseBackend):
    """Small deterministic backend for testing repository I/O without AlphaGenome."""

    def __init__(self, device: str = "cpu", organism_index: int = 0):
        self.device = torch.device(device)
        self.organism_index = organism_index

    def predict(self, onehot_batch: torch.Tensor) -> dict[str, Any]:
        x = onehot_batch.to(self.device)
        # x is B,L,4 with A,C,G,T. Create simple deterministic signals.
        gc = x[..., 1] + x[..., 2]
        at = x[..., 0] + x[..., 3]
        purine = x[..., 0] + x[..., 2]
        atac_1 = torch.stack([gc, at, purine], dim=-1)

        pooled = _avg_pool_1d_channel_last(atac_1, window=8)
        rna_128 = torch.stack(
            [pooled[..., 0] * 0.7 + pooled[..., 1] * 0.1, pooled[..., 2] * 0.5],
            dim=-1,
        )
        atac_128 = pooled

        contact_base = _avg_pool_1d_channel_last(gc.unsqueeze(-1), window=max(1, x.shape[1] // 8)).squeeze(-1)
        contact = contact_base.unsqueeze(2) * contact_base.unsqueeze(1)
        contact = contact.unsqueeze(-1)

        return {
            "atac": {"1": atac_1, "128": atac_128},
            "rna_seq": {"128": rna_128},
            "contact_maps": {"native": contact},
        }


class AlphaGenomePyTorchBackend(BaseBackend):
    """Adapter for an installed or vendored alphagenome_pytorch module.

    This class intentionally does not import Hugging Face libraries. It only calls
    AlphaGenome.from_pretrained() on a local artifact path.
    """

    def __init__(
        self,
        weights: str,
        device: str = "cuda",
        organism_index: int = 0,
        named_outputs: bool = True,
        include_padding: bool = False,
        compile_model: bool = False,
    ):
        weights_path = Path(weights)
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights file not found: {weights_path}")

        try:
            from alphagenome_pytorch import AlphaGenome  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "Could not import alphagenome_pytorch. Install it from PyPI/GitHub or vendor it "
                "on PYTHONPATH. This wrapper does not include the AlphaGenome architecture."
            ) from exc

        self.device = torch.device(device)
        self.organism_index = organism_index
        self.named_outputs = named_outputs
        self.include_padding = include_padding

        self.model = AlphaGenome.from_pretrained(str(weights_path), device=str(self.device))
        self.model.eval()

        if named_outputs:
            self._try_attach_track_metadata(organism_index)

        if compile_model:
            self.model = torch.compile(self.model)  # type: ignore[attr-defined]

    def _try_attach_track_metadata(self, organism_index: int) -> None:
        try:
            from alphagenome_pytorch.named_outputs import TrackMetadataCatalog  # type: ignore
        except Exception:
            return
        try:
            try:
                catalog = TrackMetadataCatalog.load_builtin(organism=organism_index)
            except TypeError:
                catalog = TrackMetadataCatalog.load_builtin("human" if organism_index == 0 else "mouse")
            self.model.set_track_metadata_catalog(catalog)
        except Exception:
            # Predictions still work without named metadata; output will use track indices.
            return

    def predict(self, onehot_batch: torch.Tensor) -> Any:
        x = onehot_batch.to(self.device, non_blocking=True)
        with torch.no_grad():
            # Accommodate small API differences between alphagenome_pytorch versions.
            if self.named_outputs:
                try:
                    return self.model.predict(
                        x,
                        organism_index=self.organism_index,
                        named_outputs=True,
                        include_padding=self.include_padding,
                    )
                except TypeError:
                    try:
                        return self.model.predict(x, self.organism_index, named_outputs=True)
                    except TypeError:
                        pass
            try:
                return self.model.predict(x, organism_index=self.organism_index)
            except TypeError:
                return self.model.predict(x, self.organism_index)


def create_backend(config: BackendConfig) -> BaseBackend:
    org_idx = organism_to_index(config.organism)
    if config.backend == "mock":
        return MockBackend(device=config.device, organism_index=org_idx)
    if config.backend == "alphagenome-pytorch":
        if not config.weights:
            raise ValueError("--weights is required for --backend alphagenome-pytorch")
        return AlphaGenomePyTorchBackend(
            weights=config.weights,
            device=config.device,
            organism_index=org_idx,
            named_outputs=config.named_outputs,
            include_padding=config.include_padding,
            compile_model=config.compile_model,
        )
    raise ValueError(f"Unsupported backend: {config.backend}")


def _avg_pool_1d_channel_last(x: torch.Tensor, window: int) -> torch.Tensor:
    if window <= 1:
        return x
    batch, length, channels = x.shape
    trim = (length // window) * window
    if trim == 0:
        return x.mean(dim=1, keepdim=True)
    y = x[:, :trim, :].reshape(batch, trim // window, window, channels)
    return y.mean(dim=2)
