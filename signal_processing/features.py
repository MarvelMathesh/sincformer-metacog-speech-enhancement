"""
Feature Extraction Module — AMS, RASTA-PLP, MFCC, GFCC.

Extracts four feature sets from noisy speech for DNN input,
as described in Section 2.1.1 of the paper:
  - Amplitude Modulation Spectrogram (AMS)
  - RASTA-PLP (Relative Spectral Transform + PLP)
  - Mel-Frequency Cepstral Coefficients (MFCC)
  - Gammatone Frequency Cepstral Coefficients (GFCC)
"""

import numpy as np
from scipy.signal import lfilter
from scipy.signal.windows import hamming
from scipy.fftpack import dct

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from signal_processing.gammatone import GammatoneFilterbank


# ═══════════════════════════════════════════════════════════════════════════════
# AMS — Amplitude Modulation Spectrogram
# ═══════════════════════════════════════════════════════════════════════════════

def extract_ams(signal, fs=None, num_bands=None):
    """Extract AMS features from a single gammatone channel signal.
    
    Process: full-wave rectification → decimate → segment (128 overlap 64)
             → Hamming → 256-pt FFT → 15 triangular windows → sum
    
    Args:
        signal: 1D numpy array from one gammatone channel.
        fs: Sampling rate.
        num_bands: Number of triangular modulation filter bands.
    
    Returns:
        1D array of AMS features (num_bands values).
    """
    fs = fs or config.SAMPLE_RATE
    num_bands = num_bands or config.AMS_NUM_BANDS
    
    # Full-wave rectification
    rectified = np.abs(signal)
    
    # Decimate by factor of 8 (reduce to ~1kHz modulation sampling)
    decimate_factor = 8
    decimated = rectified[::decimate_factor]
    
    # Segment into overlapping frames
    seg_len = config.AMS_SEGMENTS
    overlap = config.AMS_OVERLAP
    hop = seg_len - overlap
    
    num_segs = max(1, (len(decimated) - seg_len) // hop + 1)
    
    ams_features = np.zeros(num_bands)
    
    for s in range(num_segs):
        start = s * hop
        end = start + seg_len
        if end > len(decimated):
            break
        
        segment = decimated[start:end]
        
        # Hamming window
        segment = segment * hamming(seg_len)
        
        # Zero-pad and 256-point FFT
        fft_out = np.abs(np.fft.rfft(segment, n=config.AMS_FFT_SIZE))
        
        # Apply 15 triangular-shaped windows between 15.6Hz and 400Hz
        mod_fs = fs / decimate_factor
        freq_bins = np.arange(len(fft_out)) * mod_fs / config.AMS_FFT_SIZE
        
        low_mod_freq = 15.6
        high_mod_freq = 400.0
        
        # Create triangular filter centers
        centers = np.linspace(low_mod_freq, high_mod_freq, num_bands + 2)
        
        for b in range(num_bands):
            # Triangular window
            lo, mid, hi = centers[b], centers[b + 1], centers[b + 2]
            weights = np.zeros(len(freq_bins))
            
            # Rising slope
            mask_rise = (freq_bins >= lo) & (freq_bins <= mid)
            weights[mask_rise] = (freq_bins[mask_rise] - lo) / (mid - lo + 1e-10)
            
            # Falling slope
            mask_fall = (freq_bins > mid) & (freq_bins <= hi)
            weights[mask_fall] = (hi - freq_bins[mask_fall]) / (hi - mid + 1e-10)
            
            ams_features[b] += np.sum(fft_out * weights)
    
    # Average over segments
    ams_features /= max(num_segs, 1)
    
    return ams_features


# ═══════════════════════════════════════════════════════════════════════════════
# RASTA-PLP — Relative Spectral Transform + PLP
# ═══════════════════════════════════════════════════════════════════════════════

def hz_to_bark(f):
    """Convert frequency in Hz to Bark scale."""
    return 6.0 * np.arcsinh(f / 600.0)


def bark_to_hz(z):
    """Convert Bark scale to Hz."""
    return 600.0 * np.sinh(z / 6.0)


def rasta_filter(signal):
    """Apply the RASTA bandpass filter.
    
    The RASTA filter removes slow channel variations and fast noise.
    Transfer function: H(z) = 0.1 * z^4 * (2 + z^-1 - z^-3 - 2*z^-4) / (1 - 0.98*z^-1)
    """
    # Numerator: [0.2, 0.1, 0, -0.1, -0.2]
    num = np.array([0.2, 0.1, 0.0, -0.1, -0.2])
    # Denominator: [1, -0.98]
    den = np.array([1.0, -0.98])
    
    return lfilter(num, den, signal)


def extract_rasta_plp(signal, fs=None, num_coeffs=None):
    """Extract RASTA-PLP features.
    
    Process: power spectrum → bark scale → log compress → RASTA filter
             → exponential → PLP analysis → cepstral coefficients
    
    Args:
        signal: 1D numpy array.
        fs: Sampling rate.
        num_coeffs: Number of PLP coefficients.
    
    Returns:
        1D array of RASTA-PLP features.
    """
    fs = fs or config.SAMPLE_RATE
    num_coeffs = num_coeffs or config.RASTA_NUM_COEFF
    
    frame_size = config.FRAME_SIZE
    hop_size = config.HOP_SIZE
    fft_size = config.FFT_SIZE
    
    # Ensure minimum signal length
    if len(signal) < frame_size:
        signal = np.pad(signal, (0, frame_size - len(signal)))
    
    # Frame the signal
    num_frames = (len(signal) - frame_size) // hop_size + 1
    
    # Create bark-scale filterbank
    num_bark_filters = 21  # Standard PLP uses ~21 critical bands
    bark_low = hz_to_bark(0)
    bark_high = hz_to_bark(fs / 2)
    bark_centers = np.linspace(bark_low, bark_high, num_bark_filters + 2)
    hz_centers = bark_to_hz(bark_centers)
    
    freq_bins = np.arange(fft_size // 2 + 1) * fs / fft_size
    
    # Build bark filterbank matrix
    bark_fb = np.zeros((num_bark_filters, fft_size // 2 + 1))
    for i in range(num_bark_filters):
        lo = hz_centers[i]
        mid = hz_centers[i + 1]
        hi = hz_centers[i + 2]
        
        rise = (freq_bins >= lo) & (freq_bins <= mid)
        fall = (freq_bins > mid) & (freq_bins <= hi)
        
        bark_fb[i, rise] = (freq_bins[rise] - lo) / (mid - lo + 1e-10)
        bark_fb[i, fall] = (hi - freq_bins[fall]) / (hi - mid + 1e-10)
    
    # Compute power spectrum for each frame
    bark_spectrum = np.zeros((num_bark_filters, num_frames))
    
    for n in range(num_frames):
        start = n * hop_size
        end = start + frame_size
        frame = signal[start:end] * hamming(frame_size)
        power = np.abs(np.fft.rfft(frame, n=fft_size)) ** 2
        bark_spectrum[:, n] = bark_fb @ power
    
    # Log compression
    log_spectrum = np.log(bark_spectrum + 1e-10)
    
    # RASTA filter (applied per bark band across time)
    rasta_spectrum = np.zeros_like(log_spectrum)
    for i in range(num_bark_filters):
        rasta_spectrum[i, :] = rasta_filter(log_spectrum[i, :])
    
    # Exponential (undo log for PLP)
    exp_spectrum = np.exp(rasta_spectrum)
    
    # Equal-loudness pre-emphasis (simplified)
    eq_loudness = np.ones(num_bark_filters)
    for i in range(num_bark_filters):
        f = bark_to_hz(bark_centers[i + 1])
        # ITU-R 468 noise weighting approximation
        eq_loudness[i] = f ** 2 / (f ** 2 + 1.6e5)
    
    exp_spectrum = exp_spectrum * eq_loudness[:, np.newaxis]
    
    # Intensity to loudness (cube root compression)
    loudness_spectrum = exp_spectrum ** (1.0 / 3.0)
    
    # DCT for cepstral coefficients (averaged over frames)
    mean_spectrum = np.mean(loudness_spectrum, axis=1)
    plp_coeffs = dct(mean_spectrum, type=2, norm='ortho')[:num_coeffs]
    
    return plp_coeffs


# ═══════════════════════════════════════════════════════════════════════════════
# MFCC — Mel-Frequency Cepstral Coefficients
# ═══════════════════════════════════════════════════════════════════════════════

def hz_to_mel(f):
    """Convert Hz to Mel scale."""
    return 2595.0 * np.log10(1 + f / 700.0)


def mel_to_hz(m):
    """Convert Mel scale to Hz."""
    return 700.0 * (10 ** (m / 2595.0) - 1)


def mel_filterbank(num_filters, fft_size, fs):
    """Create a Mel-scale filterbank matrix.
    
    Args:
        num_filters: Number of Mel filters (paper: 64).
        fft_size: FFT size (paper: 512).
        fs: Sampling rate.
    
    Returns:
        2D array of shape (num_filters, fft_size//2 + 1).
    """
    mel_low = hz_to_mel(0)
    mel_high = hz_to_mel(fs / 2)
    mel_points = np.linspace(mel_low, mel_high, num_filters + 2)
    hz_points = mel_to_hz(mel_points)
    
    bins = np.floor((fft_size + 1) * hz_points / fs).astype(int)
    
    fb = np.zeros((num_filters, fft_size // 2 + 1))
    for i in range(num_filters):
        for j in range(bins[i], bins[i + 1]):
            if j < fb.shape[1]:
                fb[i, j] = (j - bins[i]) / (bins[i + 1] - bins[i] + 1e-10)
        for j in range(bins[i + 1], bins[i + 2]):
            if j < fb.shape[1]:
                fb[i, j] = (bins[i + 2] - j) / (bins[i + 2] - bins[i + 1] + 1e-10)
    
    return fb


def extract_mfcc(signal, fs=None, num_coeffs=None):
    """Extract MFCC features.
    
    Process: pre-emphasis → 512-pt STFT + 20ms Hamming → power spectrum
             → 64-ch mel scale → log → DCT
    
    Args:
        signal: 1D numpy array.
        fs: Sampling rate.
        num_coeffs: Number of MFCC coefficients.
    
    Returns:
        1D array of mean MFCC features across frames.
    """
    fs = fs or config.SAMPLE_RATE
    num_coeffs = num_coeffs or config.MFCC_NUM_COEFF
    
    fft_size = config.MFCC_FFT_SIZE
    num_filters = config.MFCC_NUM_FILTERS
    frame_size = config.FRAME_SIZE
    hop_size = config.HOP_SIZE
    
    # Pre-emphasis
    pre_emph = np.append(signal[0], signal[1:] - 0.97 * signal[:-1])
    
    # Ensure minimum length
    if len(pre_emph) < frame_size:
        pre_emph = np.pad(pre_emph, (0, frame_size - len(pre_emph)))
    
    # Frame and window
    num_frames = (len(pre_emph) - frame_size) // hop_size + 1
    
    # Mel filterbank
    mel_fb = mel_filterbank(num_filters, fft_size, fs)
    
    # Process each frame
    mfcc_frames = np.zeros((num_frames, num_coeffs))
    
    for n in range(num_frames):
        start = n * hop_size
        end = start + frame_size
        frame = pre_emph[start:end] * hamming(frame_size)
        
        # Power spectrum via 512-pt FFT
        power = np.abs(np.fft.rfft(frame, n=fft_size)) ** 2
        
        # Mel-scale filtering
        mel_energy = mel_fb @ power
        
        # Log compression
        log_energy = np.log(mel_energy + 1e-10)
        
        # DCT
        mfcc_frames[n, :] = dct(log_energy, type=2, norm='ortho')[:num_coeffs]
    
    # Return mean across frames (per-utterance feature vector)
    return np.mean(mfcc_frames, axis=0)


# ═══════════════════════════════════════════════════════════════════════════════
# GFCC — Gammatone Frequency Cepstral Coefficients
# ═══════════════════════════════════════════════════════════════════════════════

def extract_gfcc(signal, fs=None, num_coeffs=None, gfb=None):
    """Extract GFCC features.
    
    Process: 64-ch GFTB → 100 Hz decimation (10ms frameshift) 
             → cubic root → DCT
    
    Args:
        signal: 1D numpy array.
        fs: Sampling rate.
        num_coeffs: Number of GFCC coefficients.
        gfb: Pre-initialized GammatoneFilterbank (optional).
    
    Returns:
        1D array of mean GFCC features across frames.
    """
    fs = fs or config.SAMPLE_RATE
    num_coeffs = num_coeffs or config.GFCC_NUM_COEFF
    
    # Apply gammatone filterbank
    if gfb is None:
        gfb = GammatoneFilterbank(sample_rate=fs)
    
    filtered = gfb.filter(signal)
    
    # Decimate to 100 Hz (10 ms frameshift)
    decimate_rate = config.GFCC_DECIMATE_RATE
    hop_samples = fs // decimate_rate  # 80 samples at 8kHz
    
    num_frames = len(signal) // hop_samples
    num_channels = filtered.shape[0]
    
    gfcc_frames = np.zeros((num_frames, num_coeffs))
    
    for n in range(num_frames):
        start = n * hop_samples
        end = min(start + hop_samples, filtered.shape[1])
        
        # Energy in each gammatone channel for this frame
        channel_energy = np.mean(np.abs(filtered[:, start:end]) ** 2, axis=1)
        
        # Cubic root compression
        compressed = np.sign(channel_energy) * np.abs(channel_energy) ** (1.0 / 3.0)
        
        # DCT for cepstral coefficients
        gfcc_frames[n, :] = dct(compressed, type=2, norm='ortho')[:num_coeffs]
    
    # Return mean across frames
    if num_frames > 0:
        return np.mean(gfcc_frames, axis=0)
    return np.zeros(num_coeffs)


# ═══════════════════════════════════════════════════════════════════════════════
# Feature Extractor — Unified interface
# ═══════════════════════════════════════════════════════════════════════════════

class FeatureExtractor:
    """Extract and concatenate all four feature types for DNN input.
    
    Feature vector per frame = [AMS | RASTA-PLP | MFCC | GFCC]
    
    With context frames, each input to the DNN covers multiple frames
    to capture temporal structure.
    """
    
    def __init__(self, fs=None):
        self.fs = fs or config.SAMPLE_RATE
        self.gfb = GammatoneFilterbank(sample_rate=self.fs)
        self.context = config.CONTEXT_FRAMES
    
    def extract_frame_features(self, signal):
        """Extract per-frame features from a signal.
        
        This produces a feature matrix where each row is a frame
        and columns are the concatenated feature vector.
        
        Args:
            signal: 1D numpy array of the noisy speech signal.
        
        Returns:
            2D array of shape (num_frames, feature_dim) — the concatenated
            [AMS | RASTA-PLP | MFCC | GFCC] features per frame.
        """
        fs = self.fs
        frame_size = config.FRAME_SIZE
        hop_size = config.HOP_SIZE
        
        if len(signal) < frame_size:
            signal = np.pad(signal, (0, frame_size - len(signal)))
        
        num_frames = (len(signal) - frame_size) // hop_size + 1
        
        # Apply gammatone filterbank once
        filtered = self.gfb.filter(signal)
        
        # ── Per-channel AMS ──
        ams_dim = config.AMS_NUM_BANDS
        ams_per_frame = np.zeros((num_frames, ams_dim))
        for n in range(num_frames):
            start = n * hop_size
            end = min(start + frame_size * 4, len(signal))  # Wider window for AMS
            if end - start < frame_size:
                continue
            seg = signal[start:end]
            ams_per_frame[n, :] = extract_ams(seg, fs)
        
        # ── RASTA-PLP (entire signal, then map to frames) ──
        plp_coeffs = extract_rasta_plp(signal, fs)
        rasta_per_frame = np.tile(plp_coeffs, (num_frames, 1))
        
        # ── MFCC per frame ──
        mfcc_dim = config.MFCC_NUM_COEFF
        mfcc_per_frame = np.zeros((num_frames, mfcc_dim))
        
        pre_emph = np.append(signal[0], signal[1:] - 0.97 * signal[:-1])
        mel_fb = mel_filterbank(config.MFCC_NUM_FILTERS,
                                config.MFCC_FFT_SIZE, fs)
        
        for n in range(num_frames):
            start = n * hop_size
            end = start + frame_size
            frame = pre_emph[start:end] * hamming(frame_size)
            power = np.abs(np.fft.rfft(frame, n=config.MFCC_FFT_SIZE)) ** 2
            mel_energy = mel_fb @ power
            log_energy = np.log(mel_energy + 1e-10)
            mfcc_per_frame[n, :] = dct(log_energy, type=2,
                                        norm='ortho')[:mfcc_dim]
        
        # ── GFCC per frame ──
        gfcc_dim = config.GFCC_NUM_COEFF
        gfcc_per_frame = np.zeros((num_frames, gfcc_dim))
        
        dec_hop = fs // config.GFCC_DECIMATE_RATE
        for n in range(num_frames):
            center = n * hop_size + frame_size // 2
            ch_start = max(0, center - dec_hop // 2)
            ch_end = min(filtered.shape[1], center + dec_hop // 2)
            
            if ch_end <= ch_start:
                continue
            
            channel_energy = np.mean(np.abs(filtered[:, ch_start:ch_end]) ** 2,
                                      axis=1)
            compressed = np.sign(channel_energy) * \
                         np.abs(channel_energy) ** (1.0 / 3.0)
            gfcc_per_frame[n, :] = dct(compressed, type=2,
                                        norm='ortho')[:gfcc_dim]
        
        # ── Concatenate all features ──
        features = np.hstack([ams_per_frame, rasta_per_frame,
                              mfcc_per_frame, gfcc_per_frame])
        
        return features
    
    def add_context(self, features):
        """Add context frames to feature matrix.
        
        For each frame, concatenate CONTEXT_FRAMES frames from each side.
        
        Args:
            features: 2D array (num_frames, feat_dim).
        
        Returns:
            2D array (num_frames, feat_dim * (2*context + 1)).
        """
        num_frames, feat_dim = features.shape
        ctx = self.context
        
        # Pad with edge frames
        padded = np.pad(features, ((ctx, ctx), (0, 0)), mode='edge')
        
        context_features = np.zeros((num_frames, feat_dim * (2 * ctx + 1)))
        for n in range(num_frames):
            context_features[n, :] = padded[n:n + 2 * ctx + 1, :].flatten()
        
        return context_features
    
    @property
    def raw_feature_dim(self):
        """Dimension of the raw (no context) feature vector."""
        return (config.AMS_NUM_BANDS + config.RASTA_NUM_COEFF +
                config.MFCC_NUM_COEFF + config.GFCC_NUM_COEFF)
    
    @property
    def feature_dim(self):
        """Total feature dimension with context."""
        return self.raw_feature_dim * (2 * self.context + 1)
