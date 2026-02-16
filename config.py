"""
Configuration for the Speech Enhancement System.
All hyperparameters and paths are centralized here.
"""

import os

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TIMIT_DIR = os.path.join(BASE_DIR, "DARPA-TIMIT", "data")
NOISEX_DIR = os.path.join(BASE_DIR, "Noises", "NoiseX-92")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")
DEMO_DIR = os.path.join(BASE_DIR, "demo_output")

# ─── Audio ───────────────────────────────────────────────────────────────────
SAMPLE_RATE = 8000          # 8 kHz narrowband (paper spec)
FRAME_SIZE_MS = 20          # 20 ms frame
FRAME_SIZE = int(SAMPLE_RATE * FRAME_SIZE_MS / 1000)  # 160 samples
HOP_SIZE = FRAME_SIZE // 2  # 50% overlap → 80 samples
FFT_SIZE = 256              # FFT points for STFT
WINDOW = "hamming"

# ─── Gammatone Filterbank ────────────────────────────────────────────────────
NUM_CHANNELS = 64           # 64 gammatone channels
FREQ_LOW = 50               # Hz
FREQ_HIGH = 4000            # Nyquist for 8 kHz
FILTER_ORDER = 4            # O=4 (paper Eq. 2)

# ─── Feature Extraction ─────────────────────────────────────────────────────
AMS_SEGMENTS = 128          # AMS overlapping segments
AMS_OVERLAP = 64            # AMS overlap samples
AMS_FFT_SIZE = 256          # AMS FFT points
AMS_NUM_BANDS = 15          # Triangular-shaped windows (15.6–400 Hz)

MFCC_NUM_COEFF = 13         # Number of MFCC coefficients
MFCC_FFT_SIZE = 512         # 512-point STFT for MFCC
MFCC_NUM_FILTERS = 64       # 64-channel mel scale

GFCC_NUM_COEFF = 13         # Number of GFCC coefficients
GFCC_DECIMATE_RATE = 100    # Hz → 10 ms frameshift

RASTA_NUM_COEFF = 13        # Number of RASTA-PLP coefficients

# Total feature dimension per frame (will be auto-computed)
CONTEXT_FRAMES = 5          # Number of context frames on each side

# ─── Noise Types ─────────────────────────────────────────────────────────────
NOISE_TYPES = ["babble", "white", "factory1", "destroyerengine"]
NOISE_FILES = {
    "babble": os.path.join(NOISEX_DIR, "babble.wav"),
    "white": os.path.join(NOISEX_DIR, "white.wav"),
    "factory1": os.path.join(NOISEX_DIR, "factory1.wav"),       # SSN proxy
    "destroyerengine": os.path.join(NOISEX_DIR, "destroyerengine.wav"),  # Engine
}
SNR_LEVELS = [-5, 0, 5, 10]  # dB

# ─── Dataset Splits ──────────────────────────────────────────────────────────
MAX_TRAIN_UTTERANCES = 19200  # Paper: 19,200 training utterances
MAX_TEST_UTTERANCES = 1920    # Paper: 1,920 test utterances

# ─── DNN Architecture (Original Paper) ──────────────────────────────────────
DNN_INPUT_LAYERS = 5         # 5 input context frames
DNN_HIDDEN_LAYERS = 3        # 3 hidden layers
DNN_HIDDEN_UNITS = 1024      # 1024 ReLU units per hidden layer
DNN_DROPOUT = 0.2            # Dropout rate
DNN_MOMENTUM_INITIAL = 0.5   # Momentum for first 5 epochs
DNN_MOMENTUM_FINAL = 0.9     # Momentum after 5 epochs
DNN_MOMENTUM_SWITCH_EPOCH = 5
DNN_LEARNING_RATE = 0.001
DNN_EPOCHS = 50
DNN_BATCH_SIZE = 256

# ─── RBM Pre-training ───────────────────────────────────────────────────────
RBM_LEARNING_RATE = 0.01
RBM_EPOCHS = 10
RBM_BATCH_SIZE = 256
RBM_K_STEPS = 1             # CD-k steps

# ─── PSO Configuration ──────────────────────────────────────────────────────
PSO_NUM_PARTICLES = 30       # N particles
PSO_MAX_ITER = 100           # Maximum iterations
PSO_W = 0.7                  # Inertia weight
PSO_C1 = 1.5                 # Cognitive coefficient
PSO_C2 = 1.5                 # Social coefficient
PSO_BOUNDS = (0.0, 1.0)      # Search space bounds

# ─── OPT-PCIRM ──────────────────────────────────────────────────────────────
OPT_NUM_STEPS = 3            # M=3 attenuation steps
LOCAL_CRITERION_DB = -15     # LC = -15 dB (for IBM)

# ─── Conformer ──────────────────────────────────────────────────────────────
CONFORMER_NUM_BLOCKS = 6     # N=6 Conformer blocks
CONFORMER_D_MODEL = 256      # Model dimension
CONFORMER_NUM_HEADS = 4      # Multi-head attention heads
CONFORMER_FF_DIM = 1024      # Feed-forward dimension
CONFORMER_KERNEL_SIZE = 31   # Convolutional kernel size
CONFORMER_DROPOUT = 0.1

# ─── VQ Quantization ────────────────────────────────────────────────────────
VQ_NUM_CENTROIDS = 3         # M=3 learnable centroids
VQ_COMMITMENT_WEIGHT = 0.25  # β for commitment loss

# ─── Multi-Agent System ─────────────────────────────────────────────────────
CPEA_HIDDEN_SIZE = 128       # Bidirectional LSTM hidden size
CPEA_NUM_LAYERS = 2          # 2-layer BiLSTM
PA_ENCODER_CHANNELS = 256    # Perception Agent encoder channels
MAA_THRESHOLD_INIT = 0.5     # Initial uncertainty threshold τ

# ─── Training ───────────────────────────────────────────────────────────────
PERCEPTUAL_LOSS_WEIGHT = 10.0  # Force model to care about STOI intelligibility
ADVERSARIAL_LOSS_WEIGHT = 0.5  # Stronger GAN penalty for speech artifacts
COMMITMENT_LOSS_WEIGHT = 0.25

# ─── Evaluation ──────────────────────────────────────────────────────────────
STOI_EXTENDED = False        # Use extended STOI
PESQ_MODE = "nb"             # Narrowband PESQ

# ─── Curriculum Learning ────────────────────────────────────────────────────
CURRICULUM_STAGE1_EPOCHS = 15   # High-SNR + soft mask only
CURRICULUM_STAGE2_EPOCHS = 20   # Progressive low-SNR
CURRICULUM_STAGE3_EPOCHS = 15   # VQ activation + intelligibility loss
