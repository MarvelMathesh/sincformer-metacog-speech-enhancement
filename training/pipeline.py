"""
GPU-Accelerated Training Pipeline.

Optimizations over the original sequential CPU pipeline:
  1. Parallel feature extraction via multiprocessing (CPU → N workers)
  2. Disk caching of preprocessed features (.npz) — extract once, reuse
  3. CUDA training with automatic mixed precision (AMP) for RTX 40xx
  4. Proper PyTorch DataLoader with pinned memory and prefetching
  5. GPU-resident DNN training with gradient scaling
  6. Feature normalization (z-score) to prevent NaN from RBM saturation
"""

import os
import sys
import glob
import hashlib
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ═════════════════════════════════════════════════════════════════════════════
# Utility functions
# ═════════════════════════════════════════════════════════════════════════════

def load_audio(filepath, target_sr=None):
    """Load audio file and resample to target sample rate.

    Tries soundfile first, then scipy.io.wavfile as fallback.
    """
    target_sr = target_sr or config.SAMPLE_RATE

    try:
        import soundfile as sf
        audio, sr = sf.read(filepath, dtype='float32')
    except Exception:
        from scipy.io import wavfile
        sr, audio = wavfile.read(filepath)
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0
        else:
            audio = audio.astype(np.float32)

    # Mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Resample if needed
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


def add_noise_at_snr(clean, noise, snr_db):
    """Mix clean speech with noise at a specific SNR level.

    Args:
        clean: Clean speech signal.
        noise: Noise signal (will be trimmed/repeated to match clean length).
        snr_db: Target SNR in dB.

    Returns:
        Noisy signal at the specified SNR.
    """
    # Match noise length to clean
    if len(noise) < len(clean):
        repeats = int(np.ceil(len(clean) / len(noise)))
        noise = np.tile(noise, repeats)
    noise = noise[:len(clean)]

    # Compute scaling factor
    clean_power = np.mean(clean ** 2) + 1e-10
    noise_power = np.mean(noise ** 2) + 1e-10
    snr_linear = 10 ** (snr_db / 10.0)
    scale = np.sqrt(clean_power / (noise_power * snr_linear))

    return (clean + scale * noise).astype(np.float32)


# ═════════════════════════════════════════════════════════════════════════════
# Worker function for parallel preprocessing (runs in separate process)
# ═════════════════════════════════════════════════════════════════════════════

def _process_single_utterance(args):
    """Process one utterance: load → mix → extract features → compute mask.

    This runs in a worker process for parallel preprocessing.

    Args:
        args: Tuple of (clean_path, noise_signal, snr_db, mask_type,
                         cache_dir, fs)

    Returns:
        Tuple (features_with_context, mask_target) as numpy arrays,
        or None if processing fails.
    """
    clean_path, noise_signal, snr_db, mask_type, cache_dir, fs = args

    # Check cache first
    cache_key = hashlib.md5(
        f"{clean_path}_{snr_db}_{mask_type}".encode()
    ).hexdigest()
    cache_file = os.path.join(cache_dir, f"{cache_key}.npz")

    if os.path.exists(cache_file):
        try:
            data = np.load(cache_file)
            return data['features'], data['mask']
        except Exception:
            pass  # Corrupted cache, recompute

    try:
        # Import locally to avoid pickling issues with multiprocessing
        from signal_processing.gammatone import GammatoneFilterbank
        from signal_processing.features import FeatureExtractor
        from masks.irm import compute_irm
        from masks.pcirm import (compute_pcirm,
                                  compute_correlation_coefficients,
                                  compute_phase_differences)
        from masks.opt_pcirm import compute_snr_boundaries, quantize_pcirm

        # Load clean speech
        clean = load_audio(clean_path, fs)

        if len(clean) < config.FRAME_SIZE * 2:
            return None

        # Mix with noise
        noisy = add_noise_at_snr(clean, noise_signal, snr_db)

        # Extract features from NOISY signal (this is what DNN sees)
        fe = FeatureExtractor(fs=fs)
        raw_features = fe.extract_frame_features(noisy)
        features = fe.add_context(raw_features)

        # Compute oracle mask from clean + noise  (target for DNN)
        gfb = GammatoneFilterbank(sample_rate=fs)
        clean_m, clean_p = gfb.get_tf_magnitudes(clean)
        noise_trimmed = noise_signal[:len(clean)]
        if len(noise_trimmed) < len(clean):
            noise_trimmed = np.pad(noise_trimmed,
                                   (0, len(clean) - len(noise_trimmed)))
        noise_m, noise_p = gfb.get_tf_magnitudes(noise_trimmed)
        noisy_m, noisy_p = gfb.get_tf_magnitudes(noisy)

        # Align frame counts
        min_f = min(clean_m.shape[1], noise_m.shape[1],
                    noisy_m.shape[1], features.shape[0])
        clean_m = clean_m[:, :min_f]
        noise_m = noise_m[:, :min_f]
        clean_p = clean_p[:, :min_f]
        noise_p = noise_p[:, :min_f]
        noisy_m = noisy_m[:, :min_f]
        noisy_p = noisy_p[:, :min_f]
        features = features[:min_f]

        # Compute oracle mask
        if mask_type == 'irm':
            mask = compute_irm(clean_m, noise_m)
        elif mask_type == 'pcirm':
            rho_s, rho_n = compute_correlation_coefficients(
                noisy_m, clean_m, noise_m)
            phi1, phi2 = compute_phase_differences(
                noisy_p, clean_p, noise_p)
            mask = compute_pcirm(clean_m, noise_m,
                                  rho_s, rho_n, phi1, phi2)
        elif mask_type == 'opt_pcirm':
            rho_s, rho_n = compute_correlation_coefficients(
                noisy_m, clean_m, noise_m)
            phi1, phi2 = compute_phase_differences(
                noisy_p, clean_p, noise_p)
            pcirm = compute_pcirm(clean_m, noise_m,
                                    rho_s, rho_n, phi1, phi2)
            step_values, _ = compute_snr_boundaries()
            mask = quantize_pcirm(pcirm, step_values)
        else:
            mask = compute_irm(clean_m, noise_m)

        # Mask shape: (channels, frames) → (frames, channels)
        mask_target = mask.T

        # Save to cache
        try:
            np.savez_compressed(cache_file,
                                features=features.astype(np.float32),
                                mask=mask_target.astype(np.float32))
        except Exception:
            pass

        return features.astype(np.float32), mask_target.astype(np.float32)

    except Exception as e:
        return None


# ═════════════════════════════════════════════════════════════════════════════
# PyTorch Dataset with feature normalization
# ═════════════════════════════════════════════════════════════════════════════

class SpeechEnhancementDataset(Dataset):
    """PyTorch Dataset wrapping preprocessed (features, mask) frame pairs.

    Each item is a single frame: (feature_vector, mask_vector).
    Features are z-score normalized to prevent NaN during training.
    """

    def __init__(self, features_list, masks_list,
                 feat_mean=None, feat_std=None):
        """
        Args:
            features_list: List of (num_frames, feat_dim) arrays.
            masks_list: List of (num_frames, mask_dim) arrays.
            feat_mean: Optional precomputed mean for normalization.
            feat_std: Optional precomputed std for normalization.
        """
        all_features = []
        all_masks = []

        for feat, mask in zip(features_list, masks_list):
            min_len = min(feat.shape[0], mask.shape[0])
            if min_len > 0:
                all_features.append(feat[:min_len])
                all_masks.append(mask[:min_len])

        if all_features:
            raw = np.concatenate(all_features, axis=0)
            raw_masks = np.concatenate(all_masks, axis=0)

            # Replace any NaN/Inf in raw features with 0
            raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
            raw_masks = np.nan_to_num(raw_masks, nan=0.0,
                                       posinf=1.0, neginf=0.0)

            # Compute or use provided normalization stats
            if feat_mean is None:
                self.feat_mean = raw.mean(axis=0).astype(np.float32)
                self.feat_std = raw.std(axis=0).astype(np.float32)
                # Prevent div by zero — use 1.0 for constant features
                self.feat_std[self.feat_std < 1e-6] = 1.0
            else:
                self.feat_mean = feat_mean
                self.feat_std = feat_std

            # Z-score normalize features
            normalized = (raw - self.feat_mean) / self.feat_std

            # Clip to prevent extreme outliers
            normalized = np.clip(normalized, -10.0, 10.0)

            self.features = torch.from_numpy(normalized.astype(np.float32))
            self.masks = torch.from_numpy(
                np.clip(raw_masks, 0.0, 1.0).astype(np.float32))
        else:
            self.features = torch.zeros(0, 1)
            self.masks = torch.zeros(0, 1)
            self.feat_mean = np.zeros(1, dtype=np.float32)
            self.feat_std = np.ones(1, dtype=np.float32)

        print(f"    Dataset: {len(self):,} frames, "
              f"feat_dim={self.features.shape[1] if len(self) > 0 else 0}, "
              f"mask_dim={self.masks.shape[1] if len(self) > 0 else 0}")
        if len(self) > 0:
            print(f"    Features: mean={self.features.mean():.4f}, "
                  f"std={self.features.std():.4f}, "
                  f"range=[{self.features.min():.2f}, "
                  f"{self.features.max():.2f}]")

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        return self.features[idx], self.masks[idx]


# ═════════════════════════════════════════════════════════════════════════════
# Main Pipeline
# ═════════════════════════════════════════════════════════════════════════════

class TrainingPipeline:
    """GPU-accelerated training pipeline for speech enhancement DNN.

    Key optimizations:
      - Parallel preprocessing with ProcessPoolExecutor
      - Disk caching of extracted features (never recompute)
      - Z-score feature normalization (prevents RBM/DNN NaN)
      - CUDA training with AMP (automatic mixed precision)
      - Adam optimizer (more robust than SGD for this problem)
      - NaN detection with automatic recovery
      - Pinned memory DataLoader for fast CPU→GPU transfer
    """

    def __init__(self, mask_type='pcirm', use_rbm_pretrain=True):
        self.mask_type = mask_type
        self.use_rbm_pretrain = use_rbm_pretrain
        self.fs = config.SAMPLE_RATE
        self.model = None
        self.feat_mean = None
        self.feat_std = None

        # Device selection
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory
            print(f"\n  ⚡ GPU: {gpu_name} "
                  f"({gpu_mem / 1e9:.1f} GB)")
        else:
            self.device = torch.device('cpu')
            print("\n  ⚠ No GPU detected — running on CPU")

        # Use AMP only on CUDA
        self.use_amp = self.device.type == 'cuda'

        # Cache directory for preprocessed features
        self.cache_dir = os.path.join(config.BASE_DIR, '.feature_cache',
                                       mask_type)
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(config.MODEL_DIR, exist_ok=True)

    def _find_speech_files(self, max_files=None):
        """Discover TIMIT speech files."""
        patterns = [
            os.path.join(config.TIMIT_DIR, '**', '*.WAV'),
            os.path.join(config.TIMIT_DIR, '**', '*.wav'),
        ]

        files = []
        for pattern in patterns:
            files.extend(glob.glob(pattern, recursive=True))

        # Deduplicate
        files = list(set(files))
        files.sort()

        if max_files and len(files) > max_files:
            np.random.seed(42)
            indices = np.random.choice(len(files), max_files, replace=False)
            files = [files[i] for i in sorted(indices)]

        return files

    def _load_noise_signals(self):
        """Load and cache noise signals from NOISEX-92."""
        noises = {}
        for noise_type, noise_path in config.NOISE_FILES.items():
            if os.path.exists(noise_path):
                try:
                    noises[noise_type] = load_audio(noise_path, self.fs)
                    print(f"    ✓ Loaded {noise_type}: "
                          f"{len(noises[noise_type])/self.fs:.1f}s")
                except Exception as e:
                    print(f"    ✗ Failed to load {noise_type}: {e}")

        if not noises:
            print("    ⚠ No noise files found. Using synthetic white noise.")
            noises['white'] = np.random.randn(
                self.fs * 30).astype(np.float32) * 0.3

        return noises

    def prepare_data(self, max_train=None, max_test=None, num_workers=None):
        """Prepare training and test data with parallel preprocessing.

        Args:
            max_train: Max training utterances.
            max_test: Max test utterances.
            num_workers: Number of parallel workers (default: CPU count - 1).

        Returns:
            Tuple (train_loader, test_loader) — PyTorch DataLoaders.
        """
        if num_workers is None:
            num_workers = max(1, min(os.cpu_count() - 1, 8))

        print(f"\n{'=' * 60}")
        print(f"  Preparing datasets...")
        print(f"  Workers: {num_workers} | Cache: {self.cache_dir}")
        print(f"{'=' * 60}")

        # Discover speech files
        all_files = self._find_speech_files()
        if not all_files:
            raise RuntimeError(
                f"No speech files found in {config.TIMIT_DIR}. "
                "Check your TIMIT dataset path.")
        print(f"  Found {len(all_files)} speech files")

        # Split into train/test
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

        # Load noise signals
        noises = self._load_noise_signals()

        # Build preprocessing jobs
        def build_jobs(files, noise_dict, snr_levels):
            jobs = []
            noise_keys = list(noise_dict.keys())
            for i, f in enumerate(files):
                noise_key = noise_keys[i % len(noise_keys)]
                snr = snr_levels[i % len(snr_levels)]
                jobs.append((
                    f, noise_dict[noise_key], snr,
                    self.mask_type, self.cache_dir, self.fs
                ))
            return jobs

        train_jobs = build_jobs(train_files, noises, config.SNR_LEVELS)
        test_jobs = build_jobs(test_files, noises, config.SNR_LEVELS)

        # Process in parallel
        train_data = self._parallel_preprocess(
            train_jobs, num_workers, "TRAIN")
        test_data = self._parallel_preprocess(
            test_jobs, num_workers, "TEST")

        if not train_data[0]:
            raise RuntimeError("No training data was produced. "
                               "Check your dataset and feature extraction.")

        # Create datasets — train set computes normalization stats,
        # test set reuses them
        pin = self.device.type == 'cuda'
        train_ds = SpeechEnhancementDataset(*train_data)
        test_ds = SpeechEnhancementDataset(
            *test_data,
            feat_mean=train_ds.feat_mean,
            feat_std=train_ds.feat_std,
        )

        # Store normalization stats for inference
        self.feat_mean = train_ds.feat_mean
        self.feat_std = train_ds.feat_std

        train_loader = DataLoader(
            train_ds,
            batch_size=config.DNN_BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            pin_memory=pin,
            drop_last=True,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=config.DNN_BATCH_SIZE * 2,
            shuffle=False,
            num_workers=0,
            pin_memory=pin,
        )

        # Store dims for model creation
        self.feature_dim = train_ds.features.shape[1]
        self.mask_dim = train_ds.masks.shape[1]

        return train_loader, test_loader

    def _parallel_preprocess(self, jobs, num_workers, label):
        """Run preprocessing jobs in parallel with progress bar."""
        features_list = []
        masks_list = []
        failed = 0

        t0 = time.time()

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(_process_single_utterance, job): i
                for i, job in enumerate(jobs)
            }

            with tqdm(total=len(jobs), desc=f"  Preparing {label} data",
                       unit="utt", ncols=80) as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    if result is not None:
                        feat, mask = result
                        features_list.append(feat)
                        masks_list.append(mask)
                    else:
                        failed += 1
                    pbar.update(1)

        elapsed = time.time() - t0
        total = len(features_list)
        speed = elapsed / max(total, 1)
        print(f"    ✓ {label}: {total} utterances in {elapsed:.1f}s "
              f"({speed:.2f}s/utt, {failed} failed)")

        return features_list, masks_list

    def train(self, train_loader, test_loader, epochs=None):
        """Train the DNN with GPU acceleration and AMP.

        Uses Adam optimizer (more robust than SGD for initial convergence)
        with gradient clipping and NaN detection.

        Args:
            train_loader: Training DataLoader.
            test_loader: Validation DataLoader.
            epochs: Number of epochs (default from config).
        """
        epochs = epochs or config.DNN_EPOCHS

        # ── Create model ──
        from models.dnn import SpeechEnhancementDNN
        self.model = SpeechEnhancementDNN(
            input_dim=self.feature_dim,
            hidden_dim=config.DNN_HIDDEN_UNITS,
            output_dim=self.mask_dim,
            num_hidden_layers=config.DNN_HIDDEN_LAYERS,
            dropout=config.DNN_DROPOUT,
        )

        # ── Initialize weights (Kaiming) — much better than random ──
        self._init_weights(self.model)

        # ── Optional RBM pre-training ──
        if self.use_rbm_pretrain:
            self._rbm_pretrain(train_loader)

        # ── Move model to GPU ──
        self.model = self.model.to(self.device)
        param_count = self.model.count_parameters()
        print(f"\n  Model: {param_count:,} parameters → {self.device}")

        # ── Verify no NaN in initial parameters ──
        nan_params = sum(1 for p in self.model.parameters()
                         if torch.isnan(p).any())
        if nan_params > 0:
            print(f"  ⚠ WARNING: {nan_params} parameters contain NaN "
                  "after init. Re-initializing...")
            self._init_weights(self.model)
            self.model = self.model.to(self.device)

        # ── Adam optimizer (robust convergence) ──
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.DNN_LEARNING_RATE,
            weight_decay=1e-5,
        )

        # Learning rate scheduler: reduce on plateau
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5,
            min_lr=1e-6,
        )

        criterion = nn.MSELoss()

        # AMP scaler for mixed precision
        scaler = torch.amp.GradScaler('cuda') if self.use_amp else None

        # ── Training loop ──
        print(f"\n{'=' * 60}")
        print(f"  Training: {epochs} epochs, batch={config.DNN_BATCH_SIZE}, "
              f"AMP={'ON' if self.use_amp else 'OFF'}, Optimizer=Adam")
        print(f"{'=' * 60}")

        best_val_loss = float('inf')
        nan_count = 0

        for epoch in range(1, epochs + 1):
            # ── Train one epoch ──
            train_loss = self._train_epoch(
                train_loader, optimizer, criterion, scaler)

            # ── NaN detection and recovery ──
            if np.isnan(train_loss) or np.isinf(train_loss):
                nan_count += 1
                print(f"  ⚠ Epoch {epoch}: NaN detected! "
                      f"(count={nan_count}/3)")
                if nan_count >= 3:
                    print("  ✗ Training diverged. Re-initializing model...")
                    self._init_weights(self.model)
                    self.model = self.model.to(self.device)
                    optimizer = torch.optim.Adam(
                        self.model.parameters(),
                        lr=config.DNN_LEARNING_RATE * 0.1,
                        weight_decay=1e-5,
                    )
                    nan_count = 0
                continue

            nan_count = 0  # Reset on successful epoch

            # ── Validate ──
            val_loss = self._validate(test_loader, criterion)

            # Track best
            improved = val_loss < best_val_loss
            if improved:
                best_val_loss = val_loss
                self._save_best()

            # LR scheduling
            scheduler.step(val_loss)

            current_lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch:3d}/{epochs} │ "
                  f"Train: {train_loss:.6f} │ "
                  f"Val: {val_loss:.6f} │ "
                  f"LR: {current_lr:.2e} "
                  f"{'★' if improved else ''}")

        print(f"\n  Best validation loss: {best_val_loss:.6f}")

    @staticmethod
    def _init_weights(model):
        """Initialize weights with Kaiming normal (good for ReLU networks)."""
        for module in model.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _train_epoch(self, loader, optimizer, criterion, scaler):
        """Train one epoch with optional AMP."""
        self.model.train()
        total_loss = 0
        num_batches = 0

        for features, masks in loader:
            features = features.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    predictions = self.model(features)
                    loss = criterion(predictions, masks)

                # Check for NaN loss before backward
                if torch.isnan(loss) or torch.isinf(loss):
                    continue

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                predictions = self.model(features)
                loss = criterion(predictions, masks)

                if torch.isnan(loss) or torch.isinf(loss):
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=5.0)
                optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def _validate(self, loader, criterion):
        """Validate on test set."""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        for features, masks in loader:
            features = features.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)

            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    predictions = self.model(features)
                    loss = criterion(predictions, masks)
            else:
                predictions = self.model(features)
                loss = criterion(predictions, masks)

            if not (torch.isnan(loss) or torch.isinf(loss)):
                total_loss += loss.item()
                num_batches += 1

        return total_loss / max(num_batches, 1)

    def _rbm_pretrain(self, train_loader):
        """RBM layer-wise pre-training on normalized data.

        Data is already z-score normalized by the dataset,
        so the RBM sigmoid won't saturate.
        """
        from models.rbm import pretrain_dnn_with_rbm

        print(f"\n{'=' * 60}")
        print(f"  RBM Pre-training (on normalized features)")
        print(f"{'=' * 60}")

        # Collect a subset of features for RBM (CPU, numpy)
        max_rbm_samples = 50000
        all_features = []
        for features, _ in train_loader:
            all_features.append(features.numpy())
            if sum(f.shape[0] for f in all_features) >= max_rbm_samples:
                break
        rbm_data = np.concatenate(all_features, axis=0)[:max_rbm_samples]

        # Scale to [0, 1] range for RBM (sigmoid visible units)
        # Data is z-scored so roughly in [-3, 3] — sigmoid squash to [0,1]
        from models.rbm import RBM
        rbm_data_01 = RBM.sigmoid(rbm_data)

        print(f"  RBM data: {rbm_data_01.shape[0]:,} samples, "
              f"dim={rbm_data_01.shape[1]}")
        print(f"  Range: [{rbm_data_01.min():.4f}, {rbm_data_01.max():.4f}], "
              f"mean={rbm_data_01.mean():.4f}")

        # Define layers: input → hidden1 → hidden2 → hidden3
        layer_sizes = [self.feature_dim]
        for _ in range(config.DNN_HIDDEN_LAYERS):
            layer_sizes.append(config.DNN_HIDDEN_UNITS)

        rbm_weights = pretrain_dnn_with_rbm(rbm_data_01, layer_sizes)

        # Verify weights are not degenerate
        for i, (w, vb, hb) in enumerate(rbm_weights):
            w_std = np.std(w)
            if w_std < 1e-6 or np.isnan(w_std):
                print(f"  ⚠ RBM layer {i+1} weights degenerate "
                      f"(std={w_std:.8f}). Skipping load.")
                return

        self.model.load_rbm_weights(rbm_weights)
        print("  ✓ RBM weights loaded into DNN")

    def _save_best(self):
        """Save the best model checkpoint with normalization stats."""
        path = os.path.join(config.MODEL_DIR,
                            f'best_{self.mask_type}.pt')
        torch.save({
            'model_state': self.model.state_dict(),
            'mask_type': self.mask_type,
            'feature_dim': self.feature_dim,
            'mask_dim': self.mask_dim,
            'feat_mean': self.feat_mean,
            'feat_std': self.feat_std,
        }, path)

    def save_model(self, filename=None):
        """Save final trained model."""
        if self.model is None:
            print("  ✗ No model to save.")
            return

        filename = filename or f'dnn_{self.mask_type}_final.pt'
        path = os.path.join(config.MODEL_DIR, filename)
        torch.save({
            'model_state': self.model.state_dict(),
            'mask_type': self.mask_type,
            'feature_dim': self.feature_dim,
            'mask_dim': self.mask_dim,
            'feat_mean': self.feat_mean,
            'feat_std': self.feat_std,
            'device': str(self.device),
        }, path)
        print(f"  ✓ Model saved: {path}")

    def load_model(self, path=None):
        """Load a trained model from checkpoint."""
        from models.dnn import SpeechEnhancementDNN

        if path is None:
            path = os.path.join(config.MODEL_DIR,
                                f'best_{self.mask_type}.pt')

        checkpoint = torch.load(path, map_location=self.device,
                                weights_only=False)
        self.feature_dim = checkpoint['feature_dim']
        self.mask_dim = checkpoint['mask_dim']
        self.feat_mean = checkpoint.get('feat_mean')
        self.feat_std = checkpoint.get('feat_std')

        self.model = SpeechEnhancementDNN(
            input_dim=self.feature_dim,
            hidden_dim=config.DNN_HIDDEN_UNITS,
            output_dim=self.mask_dim,
            num_hidden_layers=config.DNN_HIDDEN_LAYERS,
            dropout=config.DNN_DROPOUT,
        ).to(self.device)

        self.model.load_state_dict(checkpoint['model_state'])
        print(f"  ✓ Model loaded: {path}")

    @torch.no_grad()
    def enhance_signal(self, noisy_signal):
        """Enhance a noisy signal using the trained DNN.

        Pipeline: noisy → features → normalize → DNN → 64-ch mask
                  → map mask to STFT bins → apply to STFT → iSTFT → enhanced

        The DNN predicts a 64-channel gammatone mask. We map those 64 values
        to STFT frequency bins by interpolation from the gammatone center
        frequencies, apply the mask to the STFT magnitude (keeping noisy
        phase), and reconstruct via inverse STFT.

        Args:
            noisy_signal: 1D numpy array of the noisy speech signal.

        Returns:
            Enhanced signal as 1D numpy array (same length as input).
        """
        if self.model is None:
            raise RuntimeError("No model loaded. Call load_model() first.")

        from signal_processing.gammatone import erb_space
        from signal_processing.features import FeatureExtractor

        fs = self.fs
        self.model.eval()

        # ── Step 1: Extract features and run DNN ──
        fe = FeatureExtractor(fs=fs)
        raw_features = fe.extract_frame_features(noisy_signal)
        features = fe.add_context(raw_features)

        # Normalize features (same stats as training)
        if self.feat_mean is not None and self.feat_std is not None:
            features = (features - self.feat_mean) / self.feat_std
            features = np.clip(features, -10.0, 10.0)

        features = np.nan_to_num(features, nan=0.0,
                                  posinf=0.0, neginf=0.0)

        # Run DNN inference on GPU
        feat_tensor = torch.from_numpy(
            features.astype(np.float32)).to(self.device)

        batch_size = 2048
        mask_parts = []
        for i in range(0, feat_tensor.shape[0], batch_size):
            batch = feat_tensor[i:i + batch_size]
            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    pred = self.model(batch)
            else:
                pred = self.model(batch)
            mask_parts.append(pred.cpu().numpy())

        predicted_mask = np.concatenate(mask_parts, axis=0)  # (frames, 64)
        predicted_mask = np.clip(predicted_mask, 0.0, 1.0)

        # ── Step 2: Compute STFT of noisy signal ──
        frame_size = config.FRAME_SIZE
        hop_size = config.HOP_SIZE
        fft_size = config.FFT_SIZE
        signal_len = len(noisy_signal)

        window = np.hanning(frame_size).astype(np.float64)
        num_stft_bins = fft_size // 2 + 1

        # Compute STFT frames
        num_frames = (signal_len - frame_size) // hop_size + 1
        stft = np.zeros((num_frames, num_stft_bins), dtype=np.complex128)

        for n in range(num_frames):
            start = n * hop_size
            end = start + frame_size
            frame = noisy_signal[start:end].astype(np.float64) * window
            stft[n] = np.fft.rfft(frame, n=fft_size)

        # ── Step 3: Map 64-channel mask → STFT bins ──
        # Gammatone center frequencies
        center_freqs = erb_space(config.FREQ_LOW, config.FREQ_HIGH,
                                  config.NUM_CHANNELS)
        # STFT bin frequencies
        stft_freqs = np.linspace(0, fs / 2, num_stft_bins)

        # Align frame counts between DNN output and STFT
        min_frames = min(predicted_mask.shape[0], num_frames)

        # Interpolate: for each STFT frame, map 64 gammatone mask values
        # to the STFT frequency bins
        stft_mask = np.ones((min_frames, num_stft_bins), dtype=np.float64)

        for n in range(min_frames):
            stft_mask[n] = np.interp(
                stft_freqs, center_freqs, predicted_mask[n],
                left=predicted_mask[n, 0],
                right=predicted_mask[n, -1],
            )

        # ── Step 4: Apply mask to STFT and reconstruct ──
        enhanced = np.zeros(signal_len, dtype=np.float64)
        window_sum = np.zeros(signal_len, dtype=np.float64)

        for n in range(min_frames):
            masked_stft = stft[n] * stft_mask[n]
            frame = np.fft.irfft(masked_stft, n=fft_size)[:frame_size]

            start = n * hop_size
            end = start + frame_size
            enhanced[start:end] += frame * window
            window_sum[start:end] += window ** 2

        # Normalize by window sum (prevents amplitude scaling issues)
        window_sum[window_sum < 1e-8] = 1.0
        enhanced /= window_sum

        return enhanced[:signal_len].astype(np.float32)
