import tempfile, shutil, json, torch, pytest
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

# ---- import your project modules ----------------------------------------
from models.encoder           import Encoder
from models.decoder           import Decoder
from utils.model_saver        import save_encoder_decoder


# -------------------------------------------------------------------------
# 1)  A tiny synthetic dataset (identity mapping with random missingness)
# -------------------------------------------------------------------------
class SyntheticMaskedDataset(Dataset):
    """
    x ~ N(0,1); about 20 % values made missing (mask=0).
    target == input (autoencoder); we return (x, x, mask)
    """
    def __init__(self, n_samples=256, dim=16, missing_prob=0.2):
        rng    = torch.Generator().manual_seed(0)
        data   = torch.randn(n_samples, dim, generator=rng)
        mask   = (torch.rand(n_samples, dim, generator=rng) > missing_prob).float()
        self.x = data
        self.m = mask
    def __len__(self):              return len(self.x)
    def __getitem__(self, idx):     return self.x[idx], self.x[idx], self.m[idx]


# -------------------------------------------------------------------------
# 2)  Minimal training util (like in your main script, but shorter)
# -------------------------------------------------------------------------
def quick_train(dataset, hidden_dim=32, epochs=3, lr=1e-3):
    loader   = DataLoader(dataset, batch_size=64, shuffle=True)
    enc      = Encoder(input_dim=dataset.x.shape[1], hidden_dim=hidden_dim)
    dec      = Decoder(hidden_dim=hidden_dim, output_dim=dataset.x.shape[1])
    device   = torch.device("cpu")
    enc,dec  = enc.to(device), dec.to(device)

    opt      = optim.Adam([*enc.parameters(), *dec.parameters()], lr=lr)
    criterion= nn.MSELoss()

    # track first and last epoch loss
    first_loss, last_loss = None, None

    enc.train(); dec.train()
    for epoch in range(epochs):
        tot = 0
        for xin, tgt, mask in loader:
            xin, tgt, mask = xin, tgt, mask
            opt.zero_grad()
            z     = enc(xin, mask)
            recon = dec(z)
            loss  = criterion(recon * mask, tgt * mask)
            loss.backward(); opt.step()
            tot += loss.item()
        epoch_loss = tot / len(loader)
        if epoch == 0: first_loss = epoch_loss
        last_loss = epoch_loss
    return enc, dec, first_loss, last_loss


# -------------------------------------------------------------------------
# 3)  Tests
# -------------------------------------------------------------------------
def test_loss_decreases_and_reloadable():
    ds                 = SyntheticMaskedDataset()
    enc, dec, L0, Lf   = quick_train(ds)

    # ---- assert loss decreased meaningfully ------------------------------
    assert Lf < 0.5 * L0, f"Loss did not fall enough (L0={L0:.4f}, Lf={Lf:.4f})"

    # ---- save to a temp dir ---------------------------------------------
    with tempfile.TemporaryDirectory() as tmpdir:
        save_encoder_decoder(enc, dec, tmpdir, metadata={"test": True})
        meta_path = next(p for p in (shutil.os.listdir(tmpdir)) if p.startswith("metadata"))
        with open(f"{tmpdir}/{meta_path}") as f:
            meta = json.load(f)

        # ---- reload ------------------------------------------------------
        enc2 = Encoder(meta["input_dim"], meta["hidden_dim"])
        dec2 = Decoder(meta["hidden_dim"], meta["output_dim"])
        enc2.load_state_dict(torch.load(meta["encoder_path"], map_location="cpu"))
        dec2.load_state_dict(torch.load(meta["decoder_path"], map_location="cpu"))
        enc2.eval(); dec2.eval()

        # feed the same mini‑batch: outputs must be identical
        x, t, m   = ds[0]
        with torch.no_grad():
            out1 = dec(enc(x.unsqueeze(0), m.unsqueeze(0)))
            out2 = dec2(enc2(x.unsqueeze(0), m.unsqueeze(0)))
        assert torch.allclose(out1, out2, atol=1e-6), "Reloaded model differs from original"
