"""
Speech Enhancement System — Main Entry Point.

Usage:
    python main.py train       # Train DNN with PCIRM on TIMIT+NOISEX-92
    python main.py test        # Evaluate trained model on test set
    python main.py demo        # Quick demo on synthetic data (no datasets needed)
    python main.py evaluate    # Full evaluation with all metrics

Supports both the original paper's pipeline (DNN + PCIRM/OPT-PCIRM)
and the modernized pipeline (Conformer + VQ).
"""

import argparse
import os
import sys
import numpy as np

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config


def demo(args):
    """Run a quick demonstration on synthetic data.
    
    Generates synthetic clean speech + noise, runs through the full pipeline:
    features → DNN → mask → enhanced speech, then prints metrics.
    """
    from signal_processing.gammatone import GammatoneFilterbank
    from signal_processing.haircell import MeddisHairCell
    from signal_processing.features import FeatureExtractor
    from masks.irm import compute_irm, apply_irm
    from masks.pcirm import (compute_pcirm, compute_correlation_coefficients,
                              compute_phase_differences)
    from masks.opt_pcirm import compute_snr_boundaries, quantize_pcirm
    from evaluation.stoi import compute_stoi
    from evaluation.pesq_eval import compute_pesq
    from evaluation.ssnr import compute_ssnr, compute_ssnr_improvement
    from training.pipeline import add_noise_at_snr
    
    print("=" * 70)
    print("  Speech Enhancement Demo — Synthetic Signal")
    print("=" * 70)
    
    fs = config.SAMPLE_RATE
    duration = 2.0  # seconds
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    
    # ── Generate synthetic "speech" (sum of formant-like sinusoids) ──
    clean = (
        0.5 * np.sin(2 * np.pi * 250 * t) +
        0.3 * np.sin(2 * np.pi * 500 * t) +
        0.2 * np.sin(2 * np.pi * 1000 * t) +
        0.15 * np.sin(2 * np.pi * 2000 * t) +
        0.1 * np.sin(2 * np.pi * 3000 * t)
    ).astype(np.float32)
    
    # Add envelope modulation (speech-like amplitude variation)
    envelope = np.abs(np.sin(2 * np.pi * 3 * t)) ** 0.5
    clean = clean * envelope
    clean = clean / np.max(np.abs(clean))
    
    # ── Generate noise ──
    noise = np.random.randn(len(clean)).astype(np.float32) * 0.3
    
    snr_levels = [0, 5, 10]
    
    for snr_db in snr_levels:
        print(f"\n{'─' * 60}")
        print(f"  SNR = {snr_db} dB")
        print(f"{'─' * 60}")
        
        noisy = add_noise_at_snr(clean, noise, snr_db)
        
        # ── Compute masks (oracle — using clean speech) ──
        gfb = GammatoneFilterbank(sample_rate=fs)
        
        clean_mags, clean_phases = gfb.get_tf_magnitudes(clean)
        noisy_mags, noisy_phases = gfb.get_tf_magnitudes(noisy)
        noise_trimmed = noise[:len(clean)]
        noise_mags, noise_phases = gfb.get_tf_magnitudes(noise_trimmed)
        
        # Align frames
        min_f = min(clean_mags.shape[1], noisy_mags.shape[1],
                    noise_mags.shape[1])
        clean_mags = clean_mags[:, :min_f]
        noisy_mags = noisy_mags[:, :min_f]
        noise_mags = noise_mags[:, :min_f]
        clean_phases = clean_phases[:, :min_f]
        noisy_phases = noisy_phases[:, :min_f]
        noise_phases = noise_phases[:, :min_f]
        
        # IRM (baseline)
        irm = compute_irm(clean_mags, noise_mags)
        enhanced_irm = apply_irm(noisy_mags, irm)
        
        # PCIRM
        rho_s, rho_n = compute_correlation_coefficients(
            noisy_mags, clean_mags, noise_mags
        )
        phi1, phi2 = compute_phase_differences(
            noisy_phases, clean_phases, noise_phases
        )
        pcirm = compute_pcirm(clean_mags, noise_mags,
                               rho_s, rho_n, phi1, phi2)
        enhanced_pcirm = noisy_mags * pcirm
        
        # OPT-PCIRM (fixed steps, no PSO for demo speed)
        step_values, _ = compute_snr_boundaries()
        opt_pcirm = quantize_pcirm(pcirm, step_values)
        enhanced_opt = noisy_mags * opt_pcirm
        
        # ── Reconstruct time-domain signals via overlap-add synthesis ──
        def reconstruct_from_mask(mask_values, noisy_sig, num_frames, gfb):
            """Simple overlap-add reconstruction."""
            frame_size = config.FRAME_SIZE
            hop = config.HOP_SIZE
            out = np.zeros_like(noisy_sig)
            weights = np.zeros_like(noisy_sig)
            
            for n in range(num_frames):
                s = n * hop
                e = min(s + frame_size, len(noisy_sig))
                avg_mask = np.mean(mask_values[:, n])
                out[s:e] += noisy_sig[s:e] * avg_mask
                weights[s:e] += 1
            
            weights = np.maximum(weights, 1)
            return out / weights
        
        enh_irm_td = reconstruct_from_mask(irm, noisy, min_f, gfb)
        enh_pcirm_td = reconstruct_from_mask(pcirm, noisy, min_f, gfb)
        enh_opt_td = reconstruct_from_mask(opt_pcirm, noisy, min_f, gfb)
        
        # ── Evaluate ──
        print(f"\n  {'Metric':<20} {'Noisy':>10} {'IRM':>10} {'PCIRM':>10} {'OPT-PCIRM':>10}")
        print(f"  {'─' * 60}")
        
        stoi_noisy = compute_stoi(clean, noisy, fs)
        stoi_irm = compute_stoi(clean, enh_irm_td, fs)
        stoi_pcirm = compute_stoi(clean, enh_pcirm_td, fs)
        stoi_opt = compute_stoi(clean, enh_opt_td, fs)
        print(f"  {'STOI':<20} {stoi_noisy:>10.4f} {stoi_irm:>10.4f} "
              f"{stoi_pcirm:>10.4f} {stoi_opt:>10.4f}")
        
        ssnr_noisy = compute_ssnr(clean, noisy, fs)
        ssnr_irm = compute_ssnr(clean, enh_irm_td, fs)
        ssnr_pcirm = compute_ssnr(clean, enh_pcirm_td, fs)
        ssnr_opt = compute_ssnr(clean, enh_opt_td, fs)
        print(f"  {'SSNR (dB)':<20} {ssnr_noisy:>10.2f} {ssnr_irm:>10.2f} "
              f"{ssnr_pcirm:>10.2f} {ssnr_opt:>10.2f}")
        
        pesq_noisy = compute_pesq(clean, noisy, fs)
        pesq_irm = compute_pesq(clean, enh_irm_td, fs)
        pesq_pcirm = compute_pesq(clean, enh_pcirm_td, fs)
        pesq_opt = compute_pesq(clean, enh_opt_td, fs)
        print(f"  {'PESQ':<20} {pesq_noisy:>10.2f} {pesq_irm:>10.2f} "
              f"{pesq_pcirm:>10.2f} {pesq_opt:>10.2f}")
        
        # Mask statistics
        print(f"\n  Mask stats:")
        print(f"    IRM      — mean={np.mean(irm):.3f}, "
              f"std={np.std(irm):.3f}")
        print(f"    PCIRM    — mean={np.mean(pcirm):.3f}, "
              f"std={np.std(pcirm):.3f}")
        print(f"    OPT-PCIRM— unique values={np.unique(opt_pcirm).round(4)}, "
              f"mean={np.mean(opt_pcirm):.3f}")
    
    print(f"\n{'=' * 70}")
    print("  Demo complete!")
    print(f"{'=' * 70}\n")


def train(args):
    """Train on TIMIT + NOISEX-92 data."""
    pipeline_type = getattr(args, 'pipeline', 'dnn')
    
    if pipeline_type == 'conformer':
        from training.conformer_pipeline import ConformerPipeline
        
        print("=" * 70)
        print("  Speech Enhancement — Conformer Training")
        print("=" * 70)
        
        pipeline = ConformerPipeline()
        train_ds, test_ds = pipeline.prepare_data(
            max_train=args.max_train,
            max_test=args.max_test,
        )
        pipeline.train(train_ds, test_ds, epochs=args.epochs)
        pipeline.save_model()
    else:
        from training.pipeline import TrainingPipeline
        
        print("=" * 70)
        print("  Speech Enhancement — DNN Training")
        print("=" * 70)
        
        pipeline = TrainingPipeline(
            mask_type=args.mask_type,
            use_rbm_pretrain=not args.no_rbm,
        )
        train_loader, test_loader = pipeline.prepare_data(
            max_train=args.max_train,
            max_test=args.max_test,
        )
        pipeline.train(train_loader, test_loader, epochs=args.epochs)
        pipeline.save_model()
    
    print("\nTraining complete!")


def evaluate(args):
    """Evaluate trained models with STOI, PESQ, SSNR across ALL noise types.

    Loads all available trained models (PCIRM, OPT-PCIRM, Conformer)
    and evaluates on held-out TIMIT utterances × 4 noise types × 4 SNRs.
    """
    import glob
    import torch
    from training.pipeline import TrainingPipeline, load_audio, add_noise_at_snr
    from evaluation.stoi import compute_stoi
    from evaluation.pesq_eval import compute_pesq
    from evaluation.ssnr import compute_ssnr
    from tqdm import tqdm

    print("=" * 70)
    print("  Speech Enhancement — Full Multi-Noise Evaluation")
    print("=" * 70)

    fs = config.SAMPLE_RATE
    snr_levels = config.SNR_LEVELS  # [-5, 0, 5, 10]
    max_eval = getattr(args, 'max_eval', 50)

    # ── Discover models ──
    mask_types = []
    pipelines = {}

    for mt in ['pcirm', 'opt_pcirm']:
        model_path = os.path.join(config.MODEL_DIR, f'dnn_{mt}_final.pt')
        best_path = os.path.join(config.MODEL_DIR, f'best_{mt}.pt')
        if os.path.exists(model_path) or os.path.exists(best_path):
            p = TrainingPipeline(mask_type=mt, use_rbm_pretrain=False)
            path = model_path if os.path.exists(model_path) else best_path
            p.load_model(path)
            pipelines[mt] = p
            mask_types.append(mt)
            print(f"  ✓ Found trained model: {mt}")
        else:
            print(f"  ✗ No model found: {mt} (skipping)")

    conformer_path = os.path.join(config.MODEL_DIR, 'conformer_final.pt')
    conformer_best = os.path.join(config.MODEL_DIR, 'best_conformer.pt')
    if os.path.exists(conformer_path) or os.path.exists(conformer_best):
        from training.conformer_pipeline import ConformerPipeline
        cp = ConformerPipeline()
        cp.load_model()
        pipelines['conformer'] = cp
        mask_types.append('conformer')
        print(f"  ✓ Found trained model: conformer (DCSE)")

    if not mask_types:
        print("\n  No trained models found! Train first with:")
        print("    python main.py train --mask-type pcirm")
        print("    python main.py train --pipeline conformer")
        return

    # ── Discover test utterances ──
    patterns = [
        os.path.join(config.TIMIT_DIR, '**', '*.WAV'),
        os.path.join(config.TIMIT_DIR, '**', '*.wav'),
    ]
    all_files = []
    for pattern in patterns:
        all_files.extend(glob.glob(pattern, recursive=True))
    all_files = sorted(set(all_files))

    np.random.seed(99)
    eval_files = np.random.choice(
        all_files, min(max_eval, len(all_files)), replace=False).tolist()

    # ── Load ALL noise signals ──
    noises = {}
    for noise_type, path in config.NOISE_FILES.items():
        if os.path.exists(path):
            try:
                noises[noise_type] = load_audio(path, fs)
                print(f"  + Loaded noise: {noise_type} "
                      f"({len(noises[noise_type]) / fs:.1f}s)")
            except Exception as e:
                print(f"  x {noise_type}: {e}")
    if not noises:
        noises['white'] = np.random.randn(
            fs * 30).astype(np.float32) * 0.3

    noise_names = list(noises.keys())
    methods = ['noisy'] + mask_types

    print(f"\n  Evaluating {len(eval_files)} utterances × "
          f"{len(noise_names)} noises × {len(snr_levels)} SNRs")
    print(f"  Methods: {', '.join(methods)}")

    # ── Grand results: results[noise][method][snr][metric] = [values] ──
    grand_results = {}
    for noise_name in noise_names:
        grand_results[noise_name] = {}
        for method in methods:
            grand_results[noise_name][method] = {
                snr: {'stoi': [], 'pesq': [], 'ssnr': []}
                for snr in snr_levels
            }

    # ── Evaluate per noise type ──
    for noise_name in noise_names:
        noise = noises[noise_name]
        total = len(eval_files) * len(snr_levels)
        desc = f"  {noise_name}"

        with tqdm(total=total, desc=desc, unit="utt", ncols=80) as pbar:
            for filepath in eval_files:
                try:
                    clean = load_audio(filepath, fs)
                    if len(clean) < config.FRAME_SIZE * 4:
                        pbar.update(len(snr_levels))
                        continue
                except Exception:
                    pbar.update(len(snr_levels))
                    continue

                for snr in snr_levels:
                    noisy_sig = add_noise_at_snr(clean, noise, snr)

                    # Noisy baseline
                    try:
                        r = grand_results[noise_name]['noisy'][snr]
                        r['stoi'].append(compute_stoi(clean, noisy_sig, fs))
                        r['pesq'].append(compute_pesq(clean, noisy_sig, fs))
                        r['ssnr'].append(compute_ssnr(clean, noisy_sig, fs))
                    except Exception:
                        pass

                    # Each model
                    for mt in mask_types:
                        try:
                            enhanced = pipelines[mt].enhance_signal(noisy_sig)
                            ml = min(len(clean), len(enhanced))
                            r = grand_results[noise_name][mt][snr]
                            r['stoi'].append(
                                compute_stoi(clean[:ml], enhanced[:ml], fs))
                            r['pesq'].append(
                                compute_pesq(clean[:ml], enhanced[:ml], fs))
                            r['ssnr'].append(
                                compute_ssnr(clean[:ml], enhanced[:ml], fs))
                        except Exception:
                            pass

                    pbar.update(1)

    # ── Print per-noise results ──
    for noise_name in noise_names:
        print(f"\n{'=' * 70}")
        print(f"  RESULTS — {noise_name} noise, {len(eval_files)} utterances")
        print(f"{'=' * 70}")

        for metric_name in ['STOI', 'PESQ', 'SSNR (dB)']:
            metric_key = metric_name.split()[0].lower()

            header = f"\n  {'SNR (dB)':<12}"
            for method in methods:
                header += f"  {method:>12}"
            print(header)
            print("  " + "-" * (12 + 14 * len(methods)))

            for snr in snr_levels:
                row = f"  {snr:>8d} dB"
                for method in methods:
                    vals = grand_results[noise_name][method][snr][metric_key]
                    if vals:
                        row += f"  {np.mean(vals):>12.4f}"
                    else:
                        row += f"  {'N/A':>12}"
                print(row)

            row = f"  {'Average':>11}"
            for method in methods:
                all_vals = []
                for snr in snr_levels:
                    all_vals.extend(
                        grand_results[noise_name][method][snr][metric_key])
                if all_vals:
                    row += f"  {np.mean(all_vals):>12.4f}"
                else:
                    row += f"  {'N/A':>12}"
            print(row)

    # ── Print grand summary (averaged over all noises) ──
    print(f"\n{'=' * 70}")
    print(f"  GRAND SUMMARY — averaged over {len(noise_names)} noise types")
    print(f"{'=' * 70}")

    for metric_name in ['STOI', 'PESQ', 'SSNR (dB)']:
        metric_key = metric_name.split()[0].lower()

        header = f"\n  {metric_name + ' ↑':<12}"
        for method in methods:
            header += f"  {method:>12}"
        print(header)
        print("  " + "-" * (12 + 14 * len(methods)))

        for snr in snr_levels:
            row = f"  {snr:>8d} dB"
            for method in methods:
                all_vals = []
                for noise_name in noise_names:
                    all_vals.extend(
                        grand_results[noise_name][method][snr][metric_key])
                if all_vals:
                    row += f"  {np.mean(all_vals):>12.4f}"
                else:
                    row += f"  {'N/A':>12}"
            print(row)

        row = f"  {'Average':>11}"
        for method in methods:
            all_vals = []
            for noise_name in noise_names:
                for snr in snr_levels:
                    all_vals.extend(
                        grand_results[noise_name][method][snr][metric_key])
            if all_vals:
                mean_v = np.mean(all_vals)
                std_v = np.std(all_vals)
                row += f"  {mean_v:>7.4f}±{std_v:.3f}"
            else:
                row += f"  {'N/A':>12}"
        print(row)

    print(f"\n{'=' * 70}")
    print("  Evaluation complete!")
    print(f"{'=' * 70}")


def info(args):
    """Print system information and configuration."""
    import torch
    
    print("=" * 70)
    print("  Speech Enhancement System — Configuration")
    print("=" * 70)
    
    print(f"\n  Sample Rate:        {config.SAMPLE_RATE} Hz")
    print(f"  Frame Size:         {config.FRAME_SIZE} samples")
    print(f"  Hop Size:           {config.HOP_SIZE} samples")
    print(f"  GFTB Channels:      {config.NUM_CHANNELS}")
    print(f"  DNN Hidden Layers:  {config.DNN_HIDDEN_LAYERS}")
    print(f"  DNN Hidden Units:   {config.DNN_HIDDEN_UNITS}")
    print(f"  DNN Dropout:        {config.DNN_DROPOUT}")
    print(f"  PSO Particles:      {config.PSO_NUM_PARTICLES}")
    print(f"  PSO Max Iters:      {config.PSO_MAX_ITER}")
    
    print(f"\n  PyTorch Version:    {torch.__version__}")
    print(f"  CUDA Available:     {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU:                {torch.cuda.get_device_name(0)}")
    
    print(f"\n  TIMIT Dir:          {config.TIMIT_DIR}")
    print(f"  NOISEX Dir:         {config.NOISEX_DIR}")
    print(f"  TIMIT exists:       {os.path.exists(config.TIMIT_DIR)}")
    print(f"  NOISEX exists:      {os.path.exists(config.NOISEX_DIR)}")


def main():
    parser = argparse.ArgumentParser(
        description='Speech Enhancement System: PSO-DNN with PCIRM/OPT-PCIRM'
    )
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Demo command
    demo_parser = subparsers.add_parser('demo',
        help='Quick demo on synthetic data (no datasets needed)')
    
    # Train command
    train_parser = subparsers.add_parser('train',
        help='Train on TIMIT + NOISEX-92')
    train_parser.add_argument('--pipeline', default='dnn',
        choices=['dnn', 'conformer'],
        help='Pipeline: dnn (original) or conformer (SOTA)')
    train_parser.add_argument('--mask-type', default='pcirm',
        choices=['irm', 'pcirm', 'opt_pcirm'],
        help='Mask type for DNN training')
    train_parser.add_argument('--epochs', type=int, default=None,
        help='Number of training epochs')
    train_parser.add_argument('--max-train', type=int, default=100,
        help='Max training utterances')
    train_parser.add_argument('--max-test', type=int, default=20,
        help='Max test utterances')
    train_parser.add_argument('--no-rbm', action='store_true',
        help='Skip RBM pre-training (DNN only)')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate',
        help='Full evaluation with all metrics')
    eval_parser.add_argument('--max-eval', type=int, default=50,
        help='Max test utterances to evaluate on (default: 50)')
    
    # Info command
    info_parser = subparsers.add_parser('info',
        help='Print system configuration')
    
    args = parser.parse_args()
    
    if args.command == 'demo':
        demo(args)
    elif args.command == 'train':
        train(args)
    elif args.command == 'evaluate':
        evaluate(args)
    elif args.command == 'info':
        info(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
