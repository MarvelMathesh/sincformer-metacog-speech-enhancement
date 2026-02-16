"""
Meddis Hair Cell Model — Auditory nerve simulation.

Converts gammatone filter responses to auditory nerve firing rate 
representations, simulating the transduction from basilar membrane 
motion to neural spikes as described in the paper.

Based on: Meddis, R. (1986). "Simulation of mechanical to neural 
transduction in the auditory receptor."
"""

import numpy as np


class MeddisHairCell:
    """Meddis inner hair cell model.
    
    Simulates the mechanical-to-neural transduction in the cochlea,
    converting continuous filter bank output into a firing rate
    representation suitable for feature extraction.
    
    Parameters from Meddis (1986):
        A  = 5.0   - Permeability constant
        B  = 300.0 - Permeability constant
        g  = 2000  - Replenishment rate
        y  = 5.05  - Reprocessing rate
        l  = 2500  - Loss rate
        r  = 6580  - Reprocessing from cleft
        x  = 66.31 - Return rate from reprocessing store
        h  = 50000 - Maximum firing rate
    """
    
    def __init__(self, sample_rate=8000):
        self.fs = sample_rate
        self.dt = 1.0 / sample_rate
        
        # Meddis model parameters
        self.A = 5.0
        self.B = 300.0
        self.g = 2000.0
        self.y = 5.05
        self.l = 2500.0
        self.r = 6580.0
        self.x = 66.31
        self.h = 50000.0
        
        # Derived steady-state values
        self.M = 1.0                                # Max transmitter
        self._compute_steady_state()
    
    def _compute_steady_state(self):
        """Compute steady-state reservoir levels for initialization."""
        # At zero input, the permeability is:
        k_ss = self.g * self.A / (self.A + self.B)
        
        # Steady-state transmitter available
        self.q0 = self.M * self.y * k_ss / (self.l * k_ss + self.y * (self.l + self.r))
        
        # Steady-state cleft concentration
        self.c0 = self.q0 * k_ss / self.y
        
        # Steady-state reprocessing store
        self.w0 = self.c0 * self.r / self.x
    
    def process(self, signal):
        """Process a 1D signal through the hair cell model.
        
        Args:
            signal: 1D numpy array — output from one gammatone channel.
        
        Returns:
            1D numpy array of firing rate probabilities, same length as input.
        """
        num_samples = len(signal)
        output = np.zeros(num_samples)
        
        # Initialize state variables
        q = self.q0     # Transmitter available
        c = self.c0     # Cleft concentration
        w = self.w0     # Reprocessing store
        
        dt = self.dt
        
        for t in range(num_samples):
            # Input-dependent permeability
            s_t = max(signal[t] + self.A, 0)
            k = s_t / (s_t + self.B)
            
            # Differential equations (Euler integration)
            # Replenishment
            dq = dt * (self.y * (self.M - q) + self.x * w - k * q)
            q = max(q + dq, 0)
            
            # Cleft transmitter
            dc = dt * (k * q - self.l * c - self.r * c)
            c = max(c + dc, 0)
            
            # Reprocessing store
            dw = dt * (self.r * c - self.x * w)
            w = max(w + dw, 0)
            
            # Firing rate probability
            output[t] = self.h * c
        
        return output
    
    def process_filterbank(self, filterbank_output):
        """Process the full gammatone filterbank output.
        
        Args:
            filterbank_output: 2D array of shape (num_channels, num_samples)
                from GammatoneFilterbank.filter().
        
        Returns:
            2D array of shape (num_channels, num_samples) — firing rate
            representation for each channel.
        """
        num_channels, num_samples = filterbank_output.shape
        output = np.zeros_like(filterbank_output)
        
        for i in range(num_channels):
            output[i, :] = self.process(filterbank_output[i, :])
        
        return output
    
    def process_to_frames(self, filterbank_output, frame_size=160, hop_size=80):
        """Process filterbank output and segment into frames.
        
        Produces the T-F representation ns(i, n) where:
            i = channel index
            n = frame index
        
        Each frame value is the average firing rate over the frame duration.
        
        Args:
            filterbank_output: 2D array (num_channels, num_samples).
            frame_size: Samples per frame.
            hop_size: Hop between frames.
        
        Returns:
            2D array of shape (num_channels, num_frames) — the T-F units.
        """
        haircell_output = self.process_filterbank(filterbank_output)
        
        num_channels, num_samples = haircell_output.shape
        num_frames = (num_samples - frame_size) // hop_size + 1
        
        tf_units = np.zeros((num_channels, num_frames))
        
        for n in range(num_frames):
            start = n * hop_size
            end = start + frame_size
            tf_units[:, n] = np.mean(haircell_output[:, start:end], axis=1)
        
        return tf_units
