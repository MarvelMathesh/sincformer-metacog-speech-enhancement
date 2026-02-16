"""
DCSE: Direct Conformer Speech Enhancement — Golden Architecture v2.

Redesigned from first principles after diagnosing catastrophic failures
in the previous multi-agent system (PA/CPEA/MSA/MAA/Memory).

ROOT CAUSES OF PREVIOUS FAILURE:
  1. Over-engineered 5-agent pipeline → ~50-layer serial gradient path
  2. CPEA contrastive pretraining broken (sigmoid→normalize→cosine≈constant)
  3. MSA fusion: 1026-dim input (768 garbage + 258 signal) → 256-dim bottleneck
  4. Identity init (bias=5.0→sigmoid≈0.993) + conservation loss = inescapable trap
  5. 5 competing losses + negative STOI proxy → model games loss, zero improvement

GOLDEN ARCHITECTURE:
    Noisy Waveform → STFT → LayerNorm → Conformer → Bounded Polar Mask → iSTFT
    Loss: SI-SNR (waveform) + L1 magnitude + Multi-Resolution STFT

Key design principles:
  - Direct STFT input (no SincNet/PA overhead — STFT is the right representation)
  - Single Conformer backbone (no agent wrappers that kill gradients)
  - Bounded polar complex mask: sigmoid magnitude + limited tanh phase
  - Clean triple loss (no conservation, no STOI proxy, no adversarial)
  - Default initialization (no identity trap — starts at ~50% attenuation)
  - All SNR levels from epoch 1 (no curriculum filtering important data)

Novel Contribution:
  Bounded Polar Complex Mask with physically-motivated constraints:
    - Magnitude ∈ [0,1] via sigmoid → can only ATTENUATE, never amplify noise
    - Phase ∈ [-π/6, π/6] via tanh → small correction, prevents distortion
  This inductive bias encodes the physics of speech enhancement.
"""

import os
import sys
import glob
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.conformer import ConformerBlock


# ============================================================================
# Loss Functions
# ============================================================================

def si_snr_loss(estimated, target):
    """Negative Scale-Invariant Signal-to-Noise Ratio (to minimize).

    SI-SNR is the gold standard for speech separation/enhancement quality.
    Reference: Le Roux et al., "SDR — Half-baked or Well Done?", ICASSP 2019.
    """
    target = target - target.mean(dim=-1, keepdim=True)
    estimated = estimated - estimated.mean(dim=-1, keepdim=True)

    dot = (estimated * target).sum(dim=-1, keepdim=True)
    s_energy = (target ** 2).sum(dim=-1, keepdim=True) + 1e-8
    s_target = dot * target / s_energy

    e_noise = estimated - s_target

    si_snr = 10 * torch.log10(
        (s_target ** 2).sum(dim=-1) /
        ((e_noise ** 2).sum(dim=-1) + 1e-8) + 1e-8
    )
    return -si_snr.mean()


class MultiResolutionSTFTLoss(nn.Module):
    """Multi-resolution STFT loss capturing structure at multiple time scales.

    Spectral convergence + log magnitude at 3 FFT sizes.
    Reference: Yamamoto et al., "Parallel WaveGAN", ICASSP 2020.
    """

    def __init__(self, fft_sizes=None, hop_sizes=None, win_sizes=None):
        super().__init__()
        self.fft_sizes = fft_sizes or [256, 512, 1024]
        self.hop_sizes = hop_sizes or [64, 128, 256]
        self.win_sizes = win_sizes or [256, 512, 1024]

    def _stft_mag(self, x, fft_size, hop_size, win_size):
        window = torch.hann_window(win_size, device=x.device)
        stft = torch.stft(x, n_fft=fft_size, hop_length=hop_size,
                          win_length=win_size, window=window,
                          return_complex=True)
        return torch.abs(stft)

    def forward(self, predicted, target):
        loss = torch.tensor(0.0, device=predicted.device)
        for fft_sz, hop_sz, win_sz in zip(
                self.fft_sizes, self.hop_sizes, self.win_sizes):
            pred_mag = self._stft_mag(predicted, fft_sz, hop_sz, win_sz)
            tgt_mag = self._stft_mag(target, fft_sz, hop_sz, win_sz)

            # Spectral convergence
            sc = torch.norm(tgt_mag - pred_mag, p='fro') / (
                torch.norm(tgt_mag, p='fro') + 1e-8)
            # Log magnitude loss
            lm = F.l1_loss(torch.log(pred_mag + 1e-8),
                           torch.log(tgt_mag + 1e-8))
            loss = loss + sc + lm
        return loss / len(self.fft_sizes)


# ============================================================================
# Data Pipeline
# ============================================================================

def _load_audio(filepath, target_sr=None):
    """Load audio file, resample to target_sr, mono."""
    target_sr = target_sr or config.SAMPLE_RATE
    try:
        import soundfile as sf
        audio, sr = sf.read(filepath, dtype='float32')
    except Exception:
        from scipy.io import wavfile
        sr, audio = wavfile.read(filepath)
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        else:
            audio = audio.astype(np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        try:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        except ImportError:
            ratio = target_sr / sr
            new_len = int(len(audio) * ratio)
            indices = np.linspace(0, len(audio) - 1, new_len)
            audio = np.interp(indices, np.arange(len(audio)), audio)
    return audio.astype(np.float32)


def _add_noise_at_snr(clean, noise, snr_db):
    """Mix clean speech with noise at specified SNR level."""
    if len(noise) < len(clean):
        noise = np.tile(noise, int(np.ceil(len(clean) / len(noise))))
    noise = noise[:len(clean)]
    clean_power = np.mean(clean ** 2) + 1e-10
    noise_power = np.mean(noise ** 2) + 1e-10
    scale = np.sqrt(clean_power / (noise_power * 10 ** (snr_db / 10.0)))
    return (clean + scale * noise).astype(np.float32)


class WaveformDataset(Dataset):
    """(noisy, clean) waveform pairs for speech enhancement training."""

    def __init__(self, clean_files, noise_signals, snr_levels, fs,
                 max_len=None):
        self.pairs = []
        self.max_len = max_len or fs * 4
        noise_keys = list(noise_signals.keys())

        for i, f in enumerate(clean_files):
            try:
                clean = _load_audio(f, fs)
                if len(clean) < config.FRAME_SIZE * 4:
                    continue
            except Exception:
                continue
            noise_key = noise_keys[i % len(noise_keys)]
            snr = snr_levels[i % len(snr_levels)]
            noisy = _add_noise_at_snr(clean, noise_signals[noise_key], snr)
            if len(clean) > self.max_len:
                clean = clean[:self.max_len]
                noisy = noisy[:self.max_len]
            self.pairs.append((noisy, clean))

        print(f"    Dataset: {len(self.pairs)} utterances, "
              f"max_len={self.max_len} ({self.max_len / fs:.1f}s)")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        noisy, clean = self.pairs[idx]
        pad = self.max_len - len(noisy)
        if pad > 0:
            noisy = np.pad(noisy, (0, pad))
            clean = np.pad(clean, (0, pad))
        return torch.from_numpy(noisy.copy()), torch.from_numpy(clean.copy())


# ============================================================================
# STFT Helpers
# ============================================================================

def batch_stft(waveform, fft_size, hop_size, frame_size):
    """Batched STFT → (batch, time, freq) real/imag."""
    window = torch.hann_window(frame_size, device=waveform.device)
    stft = torch.stft(waveform, n_fft=fft_size, hop_length=hop_size,
                      win_length=frame_size, window=window,
                      return_complex=True)
    return stft.real.transpose(1, 2), stft.imag.transpose(1, 2)


def batch_istft(stft_real, stft_imag, fft_size, hop_size, frame_size, length):
    """Batched iSTFT → (batch, samples)."""
    window = torch.hann_window(frame_size, device=stft_real.device)
    stft_complex = torch.complex(
        stft_real.transpose(1, 2), stft_imag.transpose(1, 2))
    return torch.istft(stft_complex, n_fft=fft_size, hop_length=hop_size,
                       win_length=frame_size, window=window, length=length)


# ============================================================================
# Model: Direct Conformer Speech Enhancer
# ============================================================================

class SpeechEnhancer(nn.Module):
    """Direct STFT → Conformer → Bounded Polar Complex Mask.

    Novel contribution: Bounded Polar Complex Mask with physical constraints.
      - Magnitude ∈ [0, 1] via sigmoid — can only attenuate, never amplify
      - Phase ∈ [-π/6, π/6] via tanh — small correction only
    This encodes the physics: enhancement removes noise (attenuation),
    phase distortion from additive noise is small at typical SNRs.

    Architecture:
        concat(STFT_real, STFT_imag)              (batch, T, 2×n_freq)
        → LayerNorm → Linear projection           (batch, T, d_model)
        → N × ConformerBlock (pre-norm, residual)  (batch, T, d_model)
        → LayerNorm                                (batch, T, d_model)
        → Magnitude head → sigmoid                 (batch, T, n_freq)
        → Phase head → tanh × π/6                  (batch, T, n_freq)
        → Polar-to-Cartesian → Complex multiply    (batch, T, n_freq)

    ~4.3M parameters (4 blocks, d_model=256).
    """

    def __init__(self, n_freq=None, d_model=256, num_blocks=4,
                 num_heads=4, d_ff=1024, kernel_size=31, dropout=0.15):
        super().__init__()
        self.n_freq = n_freq or (config.FFT_SIZE // 2 + 1)

        # Input: concat real + imag STFT
        self.input_norm = nn.LayerNorm(2 * self.n_freq)
        self.input_proj = nn.Linear(2 * self.n_freq, d_model)

        # Conformer backbone — captures local (conv) + global (attention)
        # temporal structure of speech
        self.blocks = nn.ModuleList([
            ConformerBlock(d_model, num_heads, d_ff, kernel_size, dropout)
            for _ in range(num_blocks)
        ])

        # Output heads with separate magnitude and phase pathways
        self.output_norm = nn.LayerNorm(d_model)
        self.mag_head = nn.Linear(d_model, self.n_freq)
        self.phase_head = nn.Linear(d_model, self.n_freq)

    def forward(self, noisy_real, noisy_imag):
        """
        Args:
            noisy_real: (batch, T, n_freq) — real part of noisy STFT.
            noisy_imag: (batch, T, n_freq) — imaginary part of noisy STFT.

        Returns:
            enhanced_real, enhanced_imag: Complex-masked enhanced STFT.
            mask_mag: Mask magnitude for monitoring (batch, T, n_freq).
        """
        # Concatenate real + imaginary as input features
        x = torch.cat([noisy_real, noisy_imag], dim=-1)

        # Input processing
        x = self.input_norm(x)
        x = self.input_proj(x)

        # Conformer blocks (each has internal residual connections)
        for block in self.blocks:
            x = block(x)

        # Output heads
        x = self.output_norm(x)

        # Bounded Polar Complex Mask
        #   Magnitude ∈ [0, 1]: can only attenuate (physically motivated)
        #   Phase ∈ [-π/6, π/6]: small correction (noise is primarily additive)
        mask_mag = torch.sigmoid(self.mag_head(x))
        mask_phase = torch.tanh(self.phase_head(x)) * (math.pi / 6)

        # Convert polar to Cartesian
        mask_real = mask_mag * torch.cos(mask_phase)
        mask_imag = mask_mag * torch.sin(mask_phase)

        # Apply complex mask: Ŝ = M̂ ⊙ Z
        enh_real = mask_real * noisy_real - mask_imag * noisy_imag
        enh_imag = mask_real * noisy_imag + mask_imag * noisy_real

        return enh_real, enh_imag, mask_mag

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# Conformer Pipeline
# ============================================================================

class ConformerPipeline:
    """Clean training/inference pipeline for DCSE.

    Training recipe:
      - AdamW (lr=5e-4, betas=(0.9, 0.98), weight_decay=0.01)
      - Linear warmup (5 epochs) + cosine annealing to 1% of peak LR
      - Gradient clipping: max_norm=5.0
      - AMP (automatic mixed precision)
      - All SNR levels from epoch 1
      - Save best model by validation loss
    """

    def __init__(self):
        self.fs = config.SAMPLE_RATE
        self.fft_size = config.FFT_SIZE
        self.hop_size = config.HOP_SIZE
        self.frame_size = config.FRAME_SIZE

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"\n  [GPU] {gpu_name} ({gpu_mem:.1f} GB)")
        else:
            self.device = torch.device('cpu')
            print("\n  [!] No GPU — running on CPU")

        self.use_amp = self.device.type == 'cuda'
        self.model = None
        os.makedirs(config.MODEL_DIR, exist_ok=True)

    # ── Data preparation ──────────────────────────────────────────────────

    def _find_speech_files(self, max_files=None):
        patterns = [
            os.path.join(config.TIMIT_DIR, '**', '*.WAV'),
            os.path.join(config.TIMIT_DIR, '**', '*.wav'),
        ]
        files = []
        for p in patterns:
            files.extend(glob.glob(p, recursive=True))
        files = sorted(set(files))
        if max_files and len(files) > max_files:
            np.random.seed(42)
            idx = np.random.choice(len(files), max_files, replace=False)
            files = [files[i] for i in sorted(idx)]
        return files

    def _load_noise_signals(self):
        noises = {}
        for noise_type, path in config.NOISE_FILES.items():
            if os.path.exists(path):
                try:
                    noises[noise_type] = _load_audio(path, self.fs)
                    print(f"    + Loaded {noise_type}: "
                          f"{len(noises[noise_type]) / self.fs:.1f}s")
                except Exception as e:
                    print(f"    x {noise_type}: {e}")
        if not noises:
            noises['white'] = np.random.randn(
                self.fs * 30).astype(np.float32) * 0.3
        return noises

    def prepare_data(self, max_train=None, max_test=None):
        print(f"\n{'=' * 60}")
        print(f"  Preparing waveform datasets...")
        print(f"{'=' * 60}")

        all_files = self._find_speech_files()
        if not all_files:
            raise RuntimeError(f"No speech files in {config.TIMIT_DIR}")
        print(f"  Found {len(all_files)} speech files")

        np.random.seed(42)
        indices = np.random.permutation(len(all_files))
        split = int(0.9 * len(all_files))
        train_files = [all_files[i] for i in indices[:split]]
        test_files = [all_files[i] for i in indices[split:]]

        if max_train:
            train_files = train_files[:max_train]
        if max_test:
            test_files = test_files[:max_test]

        print(f"  Train: {len(train_files)} | Test: {len(test_files)}")
        noises = self._load_noise_signals()

        train_ds = WaveformDataset(
            train_files, noises, config.SNR_LEVELS, self.fs)
        test_ds = WaveformDataset(
            test_files, noises, config.SNR_LEVELS, self.fs)
        return train_ds, test_ds

    # ── Training ──────────────────────────────────────────────────────────

    def train(self, train_ds, test_ds, epochs=None):
        total_epochs = epochs or 50

        # Create model
        self.model = SpeechEnhancer(
            n_freq=self.fft_size // 2 + 1,
            d_model=256,
            num_blocks=4,
            num_heads=4,
            d_ff=1024,
            kernel_size=31,
            dropout=0.15,
        ).to(self.device)

        param_count = self.model.count_parameters()
        print(f"\n  Model: {param_count:,} parameters -> {self.device}")

        # Losses
        mr_stft_loss = MultiResolutionSTFTLoss().to(self.device)

        # Optimizer: AdamW with proper betas for speech
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=5e-4,
            betas=(0.9, 0.98),
            weight_decay=0.01,
        )

        # LR schedule: linear warmup + cosine annealing
        warmup_epochs = max(1, min(5, total_epochs // 5))

        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            progress = (epoch - warmup_epochs) / max(
                1, total_epochs - warmup_epochs)
            return max(0.01, 0.5 * (1 + math.cos(math.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        scaler = torch.amp.GradScaler('cuda') if self.use_amp else None

        # Data loaders
        train_loader = DataLoader(
            train_ds, batch_size=8, shuffle=True,
            num_workers=0, pin_memory=self.device.type == 'cuda',
            drop_last=True)
        test_loader = DataLoader(
            test_ds, batch_size=8, shuffle=False,
            num_workers=0, pin_memory=self.device.type == 'cuda')

        print(f"\n{'=' * 60}")
        print(f"  DCSE Training: {total_epochs} epochs")
        print(f"  Loss: SI-SNR + L1 Mag + Multi-Resolution STFT")
        print(f"  Optimizer: AdamW (lr=5e-4, wd=0.01)")
        print(f"  Warmup: {warmup_epochs} epochs, then cosine decay")
        print(f"  Gradient clip: 5.0, AMP: {'ON' if self.use_amp else 'OFF'}")
        print(f"{'=' * 60}")

        best_val_loss = float('inf')

        for epoch in range(total_epochs):
            # Train
            train_loss, train_sisnr = self._train_epoch(
                train_loader, optimizer, mr_stft_loss, scaler)

            # Validate
            val_loss, val_sisnr = self._validate(test_loader, mr_stft_loss)

            scheduler.step()

            improved = val_loss < best_val_loss
            if improved:
                best_val_loss = val_loss
                self._save_best()

            lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch + 1:3d}/{total_epochs} | "
                  f"Train: {train_loss:.4f} (SI-SNR: {train_sisnr:+.2f}) | "
                  f"Val: {val_loss:.4f} (SI-SNR: {val_sisnr:+.2f}) | "
                  f"LR: {lr:.2e} {'*' if improved else ''}")

        print(f"\n  Best validation loss: {best_val_loss:.4f}")

    def _train_epoch(self, loader, optimizer, mr_stft_fn, scaler):
        self.model.train()
        total_loss = 0
        total_sisnr = 0
        n_batches = 0

        for noisy, clean in loader:
            noisy = noisy.to(self.device, non_blocking=True)
            clean = clean.to(self.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            noisy_real, noisy_imag = batch_stft(
                noisy, self.fft_size, self.hop_size, self.frame_size)
            clean_real, clean_imag = batch_stft(
                clean, self.fft_size, self.hop_size, self.frame_size)

            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    loss, neg_sisnr = self._compute_loss(
                        noisy_real, noisy_imag, clean, clean_real,
                        clean_imag, mr_stft_fn)

                if torch.isnan(loss) or torch.isinf(loss):
                    continue

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss, neg_sisnr = self._compute_loss(
                    noisy_real, noisy_imag, clean, clean_real,
                    clean_imag, mr_stft_fn)

                if torch.isnan(loss) or torch.isinf(loss):
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 5.0)
                optimizer.step()

            total_loss += loss.item()
            total_sisnr += (-neg_sisnr.item())  # positive SI-SNR
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        avg_sisnr = total_sisnr / max(n_batches, 1)
        return avg_loss, avg_sisnr

    def _compute_loss(self, noisy_real, noisy_imag, clean_wav,
                      clean_real, clean_imag, mr_stft_fn):
        """Combined SI-SNR + L1 magnitude + multi-resolution STFT loss."""
        # Forward
        enh_real, enh_imag, mask_mag = self.model(noisy_real, noisy_imag)

        # Align temporal dimensions
        T = min(enh_real.shape[1], clean_real.shape[1])
        enh_real = enh_real[:, :T]
        enh_imag = enh_imag[:, :T]
        clean_real_t = clean_real[:, :T]
        clean_imag_t = clean_imag[:, :T]

        # Reconstruct waveform
        enh_wav = batch_istft(
            enh_real, enh_imag,
            self.fft_size, self.hop_size, self.frame_size,
            length=clean_wav.shape[-1])

        # 1. SI-SNR loss (primary perceptual quality objective)
        loss_sisnr = si_snr_loss(enh_wav, clean_wav)

        # 2. L1 magnitude loss (spectral detail preservation)
        enh_mag = torch.sqrt(enh_real ** 2 + enh_imag ** 2 + 1e-8)
        clean_mag = torch.sqrt(clean_real_t ** 2 + clean_imag_t ** 2 + 1e-8)
        loss_mag = F.l1_loss(enh_mag, clean_mag)

        # 3. Multi-resolution STFT loss (robustness across scales)
        loss_stft = mr_stft_fn(enh_wav, clean_wav)

        # Total loss with balanced weights
        total = loss_sisnr + 0.5 * loss_mag + loss_stft

        return total, loss_sisnr

    @torch.no_grad()
    def _validate(self, loader, mr_stft_fn):
        self.model.eval()
        total_loss = 0
        total_sisnr = 0
        n_batches = 0

        for noisy, clean in loader:
            noisy = noisy.to(self.device, non_blocking=True)
            clean = clean.to(self.device, non_blocking=True)

            noisy_real, noisy_imag = batch_stft(
                noisy, self.fft_size, self.hop_size, self.frame_size)
            clean_real, clean_imag = batch_stft(
                clean, self.fft_size, self.hop_size, self.frame_size)

            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    loss, neg_sisnr = self._compute_loss(
                        noisy_real, noisy_imag, clean, clean_real,
                        clean_imag, mr_stft_fn)
            else:
                loss, neg_sisnr = self._compute_loss(
                    noisy_real, noisy_imag, clean, clean_real,
                    clean_imag, mr_stft_fn)

            if not (torch.isnan(loss) or torch.isinf(loss)):
                total_loss += loss.item()
                total_sisnr += (-neg_sisnr.item())
                n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        avg_sisnr = total_sisnr / max(n_batches, 1)
        return avg_loss, avg_sisnr

    # ── Model I/O ─────────────────────────────────────────────────────────

    def _save_best(self):
        path = os.path.join(config.MODEL_DIR, 'best_conformer.pt')
        torch.save({
            'model_state': self.model.state_dict(),
            'model_class': 'SpeechEnhancer',
        }, path)

    def save_model(self, filename='conformer_final.pt'):
        if self.model is None:
            return
        path = os.path.join(config.MODEL_DIR, filename)
        torch.save({
            'model_state': self.model.state_dict(),
            'model_class': 'SpeechEnhancer',
        }, path)
        print(f"  + Model saved: {path}")

    def load_model(self, path=None):
        if path is None:
            path = os.path.join(config.MODEL_DIR, 'conformer_final.pt')
            if not os.path.exists(path):
                path = os.path.join(config.MODEL_DIR, 'best_conformer.pt')

        checkpoint = torch.load(path, map_location=self.device,
                                weights_only=False)

        self.model = SpeechEnhancer(
            n_freq=self.fft_size // 2 + 1,
            d_model=256,
            num_blocks=4,
            num_heads=4,
            d_ff=1024,
            kernel_size=31,
            dropout=0.15,
        ).to(self.device)

        self.model.load_state_dict(checkpoint['model_state'])
        self.model.eval()
        print(f"  + Conformer loaded: {path}")

    # ── Inference ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def enhance_signal(self, noisy_signal):
        """Enhance a single noisy waveform.

        Args:
            noisy_signal: numpy array, shape (samples,).

        Returns:
            Enhanced numpy array, shape (samples,).
        """
        if self.model is None:
            raise RuntimeError("No model loaded.")

        self.model.eval()
        noisy_t = torch.from_numpy(
            noisy_signal.astype(np.float32)
        ).unsqueeze(0).to(self.device)

        noisy_real, noisy_imag = batch_stft(
            noisy_t, self.fft_size, self.hop_size, self.frame_size)

        if self.use_amp:
            with torch.amp.autocast('cuda'):
                enh_real, enh_imag, _ = self.model(noisy_real, noisy_imag)
        else:
            enh_real, enh_imag, _ = self.model(noisy_real, noisy_imag)

        enhanced = batch_istft(
            enh_real, enh_imag,
            self.fft_size, self.hop_size, self.frame_size,
            length=len(noisy_signal))

        return enhanced.squeeze(0).cpu().numpy()
