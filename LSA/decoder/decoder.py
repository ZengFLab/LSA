import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import numpy as np
from scipy.sparse import issparse

from ..utils.utils import tensor_to_numpy,convert_to_tensor

class Decoder(nn.Module):
    def __init__(self, input_dim, latent_dim=128, hidden_dims=[256, 512]):
        super().__init__()

        self.latent_dim = latent_dim

        # decoder: z -> x
        layers = []
        in_dim = latent_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
            ])
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, input_dim))
        self.net = nn.Sequential(*layers)

    def get_device(self):
        return next(self.parameters()).device
    
    def forward(self, z):
        return self.net(z)
    
    def decode(self, z, batch_size=1024, show_progress=True):
        Z = z.astype(np.float32)
        device = self.get_device()

        dataset = TensorDataset(
            torch.from_numpy(Z)
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        X = []
        with tqdm(total=len(loader), disable=not show_progress, desc='', unit='batch') as pbar:
            for batch in loader:
                z_batch = batch[0].to(device)
                x_batch = self.forward(z_batch)
                X.append(x_batch.detach().cpu().numpy())
                pbar.update(1)
        X = np.vstack(X)
        return X

    # ===========================
    # 训练接口
    # ===========================
    def fit(
        self,
        adata,
        z_key="X_latent",
        epochs=100,
        batch_size=256,
        lr=1e-3,
        patience=10,
        device=None,
        show_progress=True,
    ):
        """
        decoder.fit(adata)
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.to(device)

        print(f"\n{'='*80}")
        print("Training Decoder (z -> x)")
        print(f"{'='*80}")
        print(f"latent_dim : {self.latent_dim}")
        print(f"epochs     : {epochs}")
        print(f"batch_size : {batch_size}")
        print(f"lr         : {lr}")

        # ---------- 数据准备 ----------
        if z_key not in adata.obsm:
            raise ValueError(f"adata.obsm must contain '{z_key}'")

        Z = adata.obsm[z_key].astype(np.float32)
        if issparse(adata.X):
            X = adata.X.toarray().astype(np.float32)
        else:
            X = adata.X.astype(np.float32)

        dataset = TensorDataset(
            torch.from_numpy(Z),
            torch.from_numpy(X),
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # ---------- 优化器 ----------
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        best_loss = float("inf")
        patience_counter = 0
        loss_history = []

        # ---------- 训练循环 ----------
        pbar = tqdm(range(epochs), disable=not show_progress, desc="Decoder")

        for epoch in pbar:
            self.train()
            total_loss = 0.0

            for z_batch, x_batch in loader:
                z_batch = z_batch.to(device)
                x_batch = x_batch.to(device)

                optimizer.zero_grad()

                x_recon = self.forward(z_batch)
                loss = F.mse_loss(x_recon, x_batch)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(loader)
            loss_history.append(avg_loss)

            pbar.set_postfix({
                "loss": f"{avg_loss:.6f}",
                "best": f"{best_loss:.6f}",
            })

            # ---------- early stopping ----------
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break

        print(f"\nTraining finished. Best loss = {best_loss:.6f}")
        return loss_history