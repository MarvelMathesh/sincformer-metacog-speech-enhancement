"""
Gammatone Filterbank (GFTB) — 64-channel auditory filterbank.

Implements the gammatone filter from Patterson et al. (1998)
as described in the paper (Eq. 2-3):

    G_t(f, t) = sqrt(W) * t^(O-1) * e^(-2π*W*t) * cos(2π*Cf*t)   for t >= 0

where O=4 is filter order, Cf is center frequency, W is ERB bandwidth.
"""

import numpy as np
from scipy.signal import fftconvolve

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def erb_bandwidth(cf):
    """Compute Equivalent Rectangular Bandwidth (ERB) for a center frequency.
    
    ERB(f) = 24.7 * (4.37 * f/1000 + 1)   [Glasberg & Moore, 1990]
    """
    return 24.7 * (4.37 * cf / 1000.0 + 1.0)


def erb_space(low_freq, high_freq, num_channels):
    """Generate center frequencies equally spaced on the ERB scale.
    
    Args:
        low_freq: Lowest center frequency (Hz).
        high_freq: Highest center frequency (Hz).
        num_channels: Number of channels.
    
    Returns:
        Array of center frequencies (Hz), sorted low to high.
    """
    # ERB number scale
    erb_low = 9.265 * np.log(1 + low_freq / (24.7 * 9.265))
    erb_high = 9.265 * np.log(1 + high_freq / (24.7 * 9.265))
    
    erb_points = np.linspace(erb_low, erb_high, num_channels)
    
    # Convert back to frequency
    cfs = 24.7 * 9.265 * (np.exp(erb_points / 9.265) - 1)
    return cfs


def gammatone_impulse_response(cf, fs, duration=0.05, order=4):
    """Generate the impulse response of a single gammatone filter.
    
    G_t(f, t) = sqrt(W) * t^(O-1) * e^(-2π*W*t) * cos(2π*Cf*t)
    
    Args:
        cf: Center frequency (Hz).
        fs: Sampling rate (Hz).
        duration: Duration of impulse response (seconds).
        order: Filter order (default 4).
    
    Returns:
        1D array of the impulse response.
    """
    t = np.arange(0, duration, 1.0 / fs)
    W = erb_bandwidth(cf)
    
    # Paper Eq. 2
    b = 2 * np.pi * W * 1.019  # 1.019 is the ERB correction factor
    response = (t ** (order - 1)) * np.exp(-b * t) * np.cos(2 * np.pi * cf * t)
    
    # Normalize to unit energy
    response = response / (np.sqrt(np.sum(response ** 2)) + 1e-10)
    
    return response


class GammatoneFilterbank:
    """64-channel Gammatone Filterbank (50 Hz to 4000 Hz at 8 kHz sampling).
    
    Converts time-domain signals into the time-frequency (T-F) domain
    using auditory-inspired gammatone filters, as used in CASA literature.
    
    Usage:
        gfb = GammatoneFilterbank()
        tf_output = gfb.filter(signal)  # shape: (num_channels, num_samples)
    """
    
    def __init__(self, num_channels=None, freq_low=None, freq_high=None,
                 sample_rate=None, filter_order=None):
        self.num_channels = num_channels or config.NUM_CHANNELS
        self.freq_low = freq_low or config.FREQ_LOW
        self.freq_high = freq_high or config.FREQ_HIGH
        self.sample_rate = sample_rate or config.SAMPLE_RATE
        self.filter_order = filter_order or config.FILTER_ORDER
        
        # Compute center frequencies on ERB scale
        self.center_freqs = erb_space(self.freq_low, self.freq_high,
                                       self.num_channels)
        
        # Pre-compute impulse responses for each channel
        self.impulse_responses = []
        for cf in self.center_freqs:
            ir = gammatone_impulse_response(cf, self.sample_rate,
                                            order=self.filter_order)
            self.impulse_responses.append(ir)
    
    def filter(self, signal):
        """Apply the gammatone filterbank to an input signal.
        
        Implements Eq. 3: ns(i, t) = ns(t) * G_t(f, t)
        where * is linear convolution, i is channel index.
        
        Args:
            signal: 1D numpy array of the input signal.
        
        Returns:
            2D array of shape (num_channels, num_samples) — the T-F
            representation where row i is the output of channel i.
        """
        num_samples = len(signal)
        output = np.zeros((self.num_channels, num_samples))
        
        for i, ir in enumerate(self.impulse_responses):
            # Linear convolution (Eq. 3), then truncate to original length
            filtered = fftconvolve(signal, ir, mode='full')[:num_samples]
            output[i, :] = filtered
        
        return output
    
    def filter_to_frames(self, signal, frame_size=None, hop_size=None):
        """Apply filterbank and segment into frames.
        
        Returns the T-F representation ns(i, n) where
        i = channel index, n = frame index.
        
        Args:
            signal: 1D numpy array.
            frame_size: Samples per frame (default from config).
            hop_size: Hop size in samples (default from config).
        
        Returns:
            3D array of shape (num_channels, num_frames, frame_size).
        """
        frame_size = frame_size or config.FRAME_SIZE
        hop_size = hop_size or config.HOP_SIZE
        
        # First apply the filterbank
        filtered = self.filter(signal)
        
        # Segment into frames
        num_frames = (filtered.shape[1] - frame_size) // hop_size + 1
        frames = np.zeros((self.num_channels, num_frames, frame_size))
        
        for n in range(num_frames):
            start = n * hop_size
            end = start + frame_size
            frames[:, n, :] = filtered[:, start:end]
        
        return frames
    
    def get_tf_magnitudes(self, signal, frame_size=None, hop_size=None,
                          fft_size=None):
        """Get magnitude spectra in each channel and frame.
        
        This produces |X(i,n)|^2 for each T-F unit (i=channel, n=frame).
        
        Args:
            signal: 1D numpy array.
            frame_size: Samples per frame.
            hop_size: Hop size in samples.
            fft_size: FFT size (default from config).
        
        Returns:
            Tuple of (magnitudes, phases):
                magnitudes: (num_channels, num_frames) — power per T-F unit
                phases: (num_channels, num_frames) — phase angle per T-F unit
        """
        fft_size = fft_size or config.FFT_SIZE
        frames = self.filter_to_frames(signal, frame_size, hop_size)
        
        num_channels, num_frames, _ = frames.shape
        magnitudes = np.zeros((num_channels, num_frames))
        phases = np.zeros((num_channels, num_frames))
        
        for i in range(num_channels):
            for n in range(num_frames):
                spectrum = np.fft.rfft(frames[i, n, :], n=fft_size)
                magnitudes[i, n] = np.sum(np.abs(spectrum) ** 2)
                # Phase at the center frequency bin
                cf_bin = int(self.center_freqs[i] * fft_size / self.sample_rate)
                cf_bin = min(cf_bin, len(spectrum) - 1)
                phases[i, n] = np.angle(spectrum[cf_bin])
        
        return magnitudes, phases
