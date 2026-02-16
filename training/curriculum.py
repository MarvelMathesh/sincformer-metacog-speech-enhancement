"""
Curriculum Learning Scheduler.

    Stage 1: Train on high-SNR examples (5-10 dB) with soft mask targets only.
    Stage 2: Introduce low-SNR examples (-5 to 0 dB) progressively.
    Stage 3: Activate the quantization layer (VQ) and fine-tune with 
             intelligibility-weighted loss.

This gradually increases task difficulty, enabling better convergence
on challenging low-SNR conditions.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class CurriculumScheduler:
    """Three-stage curriculum learning scheduler.
    
    Controls which SNR levels are included in training and whether
    VQ quantization is active, based on the current epoch.
    
    Usage:
        scheduler = CurriculumScheduler()
        for epoch in range(total_epochs):
            stage_info = scheduler.get_stage(epoch)
            # Use stage_info['snr_levels'], stage_info['use_vq'], etc.
    """
    
    def __init__(self):
        self.stage1_epochs = config.CURRICULUM_STAGE1_EPOCHS
        self.stage2_epochs = config.CURRICULUM_STAGE2_EPOCHS
        self.stage3_epochs = config.CURRICULUM_STAGE3_EPOCHS
        self.total_epochs = self.stage1_epochs + self.stage2_epochs + self.stage3_epochs
    
    def get_stage(self, epoch):
        """Get curriculum stage information for the given epoch.
        
        Args:
            epoch: Current epoch number (0-indexed).
        
        Returns:
            Dict with keys:
                'stage': Current stage number (1, 2, or 3).
                'snr_levels': List of SNR levels to include.
                'use_vq': Whether VQ quantization is active.
                'use_soft_mask': Whether to use soft mask targets.
                'loss_type': Which loss function to use.
                'description': Human-readable stage description.
        """
        if epoch < self.stage1_epochs:
            return {
                'stage': 1,
                'snr_levels': [5, 10],
                'use_vq': False,
                'use_soft_mask': True,
                'loss_type': 'mse',
                'description': 'Stage 1: High-SNR + soft mask only',
            }
        elif epoch < self.stage1_epochs + self.stage2_epochs:
            # Progressively introduce lower SNR levels
            stage2_progress = (epoch - self.stage1_epochs) / self.stage2_epochs
            
            if stage2_progress < 0.33:
                snr_levels = [0, 5, 10]
            elif stage2_progress < 0.66:
                snr_levels = [-5, 0, 5, 10]
            else:
                snr_levels = [-5, 0, 5, 10]
            
            return {
                'stage': 2,
                'snr_levels': snr_levels,
                'use_vq': False,
                'use_soft_mask': True,
                'loss_type': 'mse+perceptual',
                'description': f'Stage 2: Progressive low-SNR (SNRs={snr_levels})',
            }
        else:
            return {
                'stage': 3,
                'snr_levels': [-5, 0, 5, 10],
                'use_vq': True,
                'use_soft_mask': False,
                'loss_type': 'perceptual+vq+adversarial',
                'description': 'Stage 3: VQ activation + intelligibility loss',
            }
    
    def print_schedule(self):
        """Print the full curriculum schedule."""
        print("=" * 60)
        print("Curriculum Learning Schedule")
        print("=" * 60)
        
        for epoch in range(self.total_epochs):
            stage = self.get_stage(epoch)
            if (epoch == 0 or 
                epoch == self.stage1_epochs or
                epoch == self.stage1_epochs + self.stage2_epochs):
                print(f"\n--- {stage['description']} ---")
                print(f"  Epochs: {epoch} - {epoch + [self.stage1_epochs, self.stage2_epochs, self.stage3_epochs][stage['stage']-1] - 1}")
                print(f"  SNR levels: {stage['snr_levels']}")
                print(f"  VQ active: {stage['use_vq']}")
                print(f"  Loss: {stage['loss_type']}")
