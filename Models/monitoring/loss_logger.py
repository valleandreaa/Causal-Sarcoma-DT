import csv
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt

__all__ = ["LossLogger", "load_loss_log", "plot_loss_curves"]

class LossLogger:
    """Log training and validation losses to a CSV-style log file."""

    def __init__(self, filepath):
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        # Create file with header if it does not exist
        if not self.filepath.exists():
            with open(self.filepath, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "epoch",
                    "gen_loss",
                    "disc_loss",
                    "cycle_loss",
                    "identity_loss",
                    "val_gen_loss",
                    "val_disc_loss",
                    "val_cycle_loss",
                    "val_identity_loss",
                    "gx_loss",
                    "gy_loss",
                    "dx_loss",
                    "dy_loss",
                    "val_gx_loss",
                    "val_gy_loss",
                    "val_dx_loss",
                    "val_dy_loss",
                ])

    def log(
        self,
        epoch: int,
        gen_loss: float,
        disc_loss: float,
        cycle_loss: Optional[float] = None,
        identity_loss: Optional[float] = None,
        val_gen_loss: Optional[float] = None,
        val_disc_loss: Optional[float] = None,
        val_cycle_loss: Optional[float] = None,
        val_identity_loss: Optional[float] = None,

        gx_loss: Optional[float] = None,
        gy_loss: Optional[float] = None,
        dx_loss: Optional[float] = None,
        dy_loss: Optional[float] = None,
        val_gx_loss: Optional[float] = None,
        val_gy_loss: Optional[float] = None,
        val_dx_loss: Optional[float] = None,
        val_dy_loss: Optional[float] = None,
    ) -> None:
        """Append a row of losses to the CSV file."""
        with open(self.filepath, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                gen_loss,
                disc_loss,
                cycle_loss if cycle_loss is not None else "",
                identity_loss if identity_loss is not None else "",
                val_gen_loss if val_gen_loss is not None else "",
                val_disc_loss if val_disc_loss is not None else "",
                val_cycle_loss if val_cycle_loss is not None else "",
                val_identity_loss if val_identity_loss is not None else "",
                gx_loss if gx_loss is not None else "",
                gy_loss if gy_loss is not None else "",
                dx_loss if dx_loss is not None else "",
                dy_loss if dy_loss is not None else "",
                val_gx_loss if val_gx_loss is not None else "",
                val_gy_loss if val_gy_loss is not None else "",
                val_dx_loss if val_dx_loss is not None else "",
                val_dy_loss if val_dy_loss is not None else "",
            ])


def load_loss_log(filepath) -> Dict[str, List[float]]:
    """Load the logged losses from ``filepath`` and return them as lists."""
    filepath = Path(filepath)
    data: Dict[str, List[float]] = {}
    if not filepath.exists():
        return data
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, value in row.items():
                data.setdefault(key, [])
                if value == "" or value is None:
                    data[key].append(float("nan"))
                else:
                    data[key].append(float(value))
    return data


def plot_loss_curves(filepath, save_to=None) -> None:
    """Plot training and validation losses stored in ``filepath``."""
    data = load_loss_log(filepath)
    if not data:
        raise FileNotFoundError(f"No loss log found at {filepath}")

    epochs = data.get("epoch", [])
    fig, ax = plt.subplots(5, 1, figsize=(10, 25))

    ax[0].plot(epochs, data.get("gen_loss", []), label="Train Generator")
    ax[0].plot(epochs, data.get("val_gen_loss", []), label="Val Generator")
    ax[0].plot(epochs, data.get("disc_loss", []), label="Train Discriminator")
    ax[0].plot(epochs, data.get("val_disc_loss", []), label="Val Discriminator")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].legend()

    ax[1].plot(epochs, data.get("gx_loss", []), label="Train Gx")
    ax[1].plot(epochs, data.get("gy_loss", []), label="Train Gy")
    ax[1].plot(epochs, data.get("val_gx_loss", []), label="Val Gx")
    ax[1].plot(epochs, data.get("val_gy_loss", []), label="Val Gy")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Loss")
    ax[1].legend()
    ax[1].set_title("Generator Losses")

    ax[2].plot(epochs, data.get("dx_loss", []), label="Train Dx")
    ax[2].plot(epochs, data.get("dy_loss", []), label="Train Dy")
    ax[2].plot(epochs, data.get("val_dx_loss", []), label="Val Dx")
    ax[2].plot(epochs, data.get("val_dy_loss", []), label="Val Dy")
    ax[2].set_xlabel("Epoch")
    ax[2].set_ylabel("Loss")
    ax[2].legend()
    ax[2].set_title("Discriminator Losses")

    ax[3].plot(epochs, data.get("cycle_loss", []), label="Train Cycle")
    ax[3].plot(epochs, data.get("val_cycle_loss", []), label="Val Cycle")
    ax[3].set_xlabel("Epoch")
    ax[3].set_ylabel("Loss")
    ax[3].legend()

    ax[4].plot(epochs, data.get("identity_loss", []), label="Train Identity")
    ax[4].plot(epochs, data.get("val_identity_loss", []), label="Val Identity")
    ax[4].set_xlabel("Epoch")
    ax[4].set_ylabel("Loss")
    ax[4].legend()
    
    fig.tight_layout()
    if save_to is not None:
        fig.savefig(save_to)
    else:
        plt.show()


def plot_train_cycle_gan(filepath, save_to=None) -> None:
    """Plot just the training cycle GAN loss stored in ``filepath``."""
    data = load_loss_log(filepath)
    if not data:
        raise FileNotFoundError(f"No loss log found at {filepath}")

    epochs = data.get("epoch", [])
    cycle_loss = data.get("cycle_loss", [])
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, cycle_loss, label="Train Cycle Loss", color="blue")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Cycle GAN Loss")
    ax.legend()
    fig.tight_layout()
    if save_to is not None:
        fig.savefig(save_to)
    else:
        plt.show()