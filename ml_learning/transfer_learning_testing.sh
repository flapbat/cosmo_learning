#!/bin/bash
echo "Starting transfer learning testing at $(date)"

# Create results directory for cleanliness
mkdir -p transfer_learning_testing

# Consistent parameters
YAML="./projects/lsst_y1/xi_emulator_transfer_learning.yaml"
PROBE="cosmic_shear"
PRETRAINED_MODEL="./projects/lsst_y1/emulators/xi_low_accuracy"
LEARNING_RATE=1e-3
EPOCHS=500

echo "========================================"
echo "TRANSFER LEARNING TESTING"
echo "Consistent parameters:"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Epochs: $EPOCHS"
echo "  Training: 1000 high-accuracy samples"
echo "  Validation: 9000 high-accuracy samples"
echo "========================================"

# ============================================================================
# EXPERIMENT 1: Baseline (No Freezing)
# ============================================================================
echo ""
echo "=== EXPERIMENT 1: Baseline (No Freezing) ==="

python projects/lsst_y1/train_emulator.py \
    --yaml $YAML \
    --probe $PROBE \
    --transfer_learning True \
    --pretrained_model $PRETRAINED_MODEL \
    --freeze_strategy none \
    --learning_rate $LEARNING_RATE \
    --epochs $EPOCHS \
    --save_losses True \
    --save_testing_metrics True

mv losses.txt projects/lsst_y1/transfer_learning_testing/losses_none.txt
#mv testing_metrics.txt projects/lsst_y1/transfer_learning_testing/testing_metrics_none.txt

# ============================================================================
# EXPERIMENT 2: Late Freezing 1
# ============================================================================
echo ""
echo "=== EXPERIMENT 2: Late Freezing 1 (model.4 only) ==="

python projects/lsst_y1/train_emulator.py \
    --yaml $YAML \
    --probe $PROBE \
    --transfer_learning True \
    --pretrained_model $PRETRAINED_MODEL \
    --freeze_strategy late_1 \
    --learning_rate $LEARNING_RATE \
    --epochs $EPOCHS \
    --save_losses True \
    --save_testing_metrics True

mv losses.txt projects/lsst_y1/transfer_learning_testing/losses_early_1.txt
mv testing_metrics.txt projects/lsst_y1/transfer_learning_testing/testing_metrics_early_1.txt

# ============================================================================
# EXPERIMENT 3: Late Freezing 2
# ============================================================================
echo ""
echo "=== EXPERIMENT 3: Late Freezing 2 (model.3 + model.4) ==="

python projects/lsst_y1/train_emulator.py \
    --yaml $YAML \
    --probe $PROBE \
    --transfer_learning True \
    --pretrained_model $PRETRAINED_MODEL \
    --freeze_strategy late_2 \
    --learning_rate $LEARNING_RATE \
    --epochs $EPOCHS \
    --save_losses True \
    --save_testing_metrics True

mv losses.txt projects/lsst_y1/transfer_learning_testing/losses_early_2.txt
mv testing_metrics.txt projects/lsst_y1/transfer_learning_testing/testing_metrics_early_2.txt

# ============================================================================
# EXPERIMENT 4: Late Freezing 3 (model 2,3, 4)
# ============================================================================
echo ""
echo "=== EXPERIMENT 4: Late Freezing 3 (model 2, 3, 4) ==="

python projects/lsst_y1/train_emulator.py \
    --yaml $YAML \
    --probe $PROBE \
    --transfer_learning True \
    --pretrained_model $PRETRAINED_MODEL \
    --freeze_strategy late_3 \
    --learning_rate $LEARNING_RATE \
    --epochs $EPOCHS \
    --save_losses True \
    --save_testing_metrics True

mv losses.txt projects/lsst_y1/transfer_learning_testing/losses_late_1.txt
mv testing_metrics.txt projects/lsst_y1/transfer_learning_testing/testing_metrics_late_1.txt

# ============================================================================
# EXPERIMENT 5: Late Freezing 4 (model.4 + model.3)
# ============================================================================
echo ""
echo "=== EXPERIMENT 5: Late Freezing 4 (model 1, 2, 3, 4) ==="

python projects/lsst_y1/train_emulator.py \
    --yaml $YAML \
    --probe $PROBE \
    --transfer_learning True \
    --pretrained_model $PRETRAINED_MODEL \
    --freeze_strategy late_4 \
    --learning_rate $LEARNING_RATE \
    --epochs $EPOCHS \
    --save_losses True \
    --save_testing_metrics True

mv losses.txt projects/lsst_y1/transfer_learning_testing/losses_late_2.txt
mv testing_metrics.txt projects/lsst_y1/transfer_learning_testing/testing_metrics_late_2.txt

# ============================================================================
# EXPERIMENT 6: Input + Output Freezing (model.0 + model.4)
# ============================================================================
echo ""
echo "=== EXPERIMENT 6: Input + Output Freezing (model.0 + model.4) ==="

python projects/lsst_y1/train_emulator.py \
    --yaml $YAML \
    --probe $PROBE \
    --transfer_learning True \
    --pretrained_model $PRETRAINED_MODEL \
    --freeze_strategy input_output \
    --learning_rate $LEARNING_RATE \
    --epochs $EPOCHS \
    --save_losses True \
    --save_testing_metrics True

mv losses.txt projects/lsst_y1/transfer_learning_testing/losses_input_output.txt
mv testing_metrics.txt projects/lsst_y1/transfer_learning_testing/testing_metrics_input_output.txt

2+3 Freezing (model.1 + model.2 + model.3) ==="

python projects/lsst_y1/train_emulator.py \
    --yaml $YAML \
    --probe $PROBE \
    --transfer_learning True \
    --pretrained_model $PRETRAINED_MODEL \
    --freeze_strategy resnet_123 \
    --learning_rate $LEARNING_RATE \
    --epochs $EPOCHS \
    --save_losses True \
    --save_testing_metrics True

mv losses.txt projects/lsst_y1/transfer_learning_testing/losses_resnet_123.txt
mv testing_metrics.txt projects/lsst_y1/transfer_learning_testing/testing_metrics_resnet_123.txt
"""

echo ""
echo "All experiments completed at $(date)"
echo "Results saved in transfer_learning_testing/"
