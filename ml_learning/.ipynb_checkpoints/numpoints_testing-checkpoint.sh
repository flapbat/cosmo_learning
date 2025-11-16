#!/bin/bash
echo "Starting transfer learning testing at $(date)"

# Create results directory for cleanliness
mkdir -p transfer_learning_testing

# Consistent parameters
#YAML="./projects/lsst_y1/xi_emulator_transfer_learning.yaml"
PROBE="cosmic_shear"
PRETRAINED_MODEL="./projects/lsst_y1/emulators/xi_low_accuracy"
LEARNING_RATE=1e-3
EPOCHS=1000

echo "========================================"
echo "TRANSFER LEARNING TESTING"
echo "Consistent parameters:"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Epochs: $EPOCHS"
echo "  Training: 1000 high-accuracy samples"
echo "  Validation: 9000 high-accuracy samples"
echo "========================================"


echo ""
echo "=== EXPERIMENT 1: No Freeze 100 ==="

python projects/lsst_y1/train_emulator.py \
    --yaml "./projects/lsst_y1/xi_emul_100points.yaml.yaml" \
    --probe $PROBE \
    --transfer_learning False \
    --learning_rate $LEARNING_RATE \
    --epochs $EPOCHS \
    --save_losses True \
    --save_testing_metrics True

mv losses.txt projects/lsst_y1/transfer_learning_testing/losses_nofreeze100.txt
mv testing_metrics.txt projects/lsst_y1/transfer_learning_testing/testing_metrics_nofreeze100.txt

echo ""
echo "=== EXPERIMENT 2: Early2 100 ==="

python projects/lsst_y1/train_emulator.py \
    --yaml "./projects/lsst_y1/xi_emul_100points.yaml.yaml" \
    --probe $PROBE \
    --transfer_learning True \
    --pretrained_model $PRETRAINED_MODEL \
    --freeze_strategy early_2 \
    --learning_rate $LEARNING_RATE \
    --epochs $EPOCHS \
    --save_losses True \
    --save_testing_metrics True

mv losses.txt projects/lsst_y1/transfer_learning_testing/losses_early2_100.txt
mv testing_metrics.txt projects/lsst_y1/transfer_learning_testing/testing_metrics_early2_100.txt

echo ""
echo "=== EXPERIMENT 3: No Freeze 1000 ==="

python projects/lsst_y1/train_emulator.py \
    --yaml "./projects/lsst_y1/xi_emul_1000points.yaml.yaml" \
    --probe $PROBE \
    --transfer_learning False \
    --learning_rate $LEARNING_RATE \
    --epochs $EPOCHS \
    --save_losses True \
    --save_testing_metrics True

mv losses.txt projects/lsst_y1/transfer_learning_testing/losses_nofreeze1000.txt
mv testing_metrics.txt projects/lsst_y1/transfer_learning_testing/testing_metrics_nofreeze1000.txt

echo ""

echo ""
echo "=== EXPERIMENT 2: Early2 1000 ==="

python projects/lsst_y1/train_emulator.py \
    --yaml "./projects/lsst_y1/xi_emul_1000points.yaml.yaml" \
    --probe $PROBE \
    --transfer_learning True \
    --pretrained_model $PRETRAINED_MODEL \
    --freeze_strategy early_2 \
    --learning_rate $LEARNING_RATE \
    --epochs $EPOCHS \
    --save_losses True \
    --save_testing_metrics True

mv losses.txt projects/lsst_y1/transfer_learning_testing/losses_early2_1000.txt
mv testing_metrics.txt projects/lsst_y1/transfer_learning_testing/testing_metrics_early2_1000.txt

echo ""

echo ""
echo "=== EXPERIMENT 3: No Freeze 10000 ==="

python projects/lsst_y1/train_emulator.py \
    --yaml "./projects/lsst_y1/xi_emul_10000points.yaml.yaml" \
    --probe $PROBE \
    --transfer_learning False \
    --learning_rate $LEARNING_RATE \
    --epochs $EPOCHS \
    --save_losses True \
    --save_testing_metrics True

mv losses.txt projects/lsst_y1/transfer_learning_testing/losses_nofreeze10000.txt
mv testing_metrics.txt projects/lsst_y1/transfer_learning_testing/testing_metrics_nofreeze10000.txt

echo ""

echo ""
echo "=== EXPERIMENT 2: Early2 10000 ==="

python projects/lsst_y1/train_emulator.py \
    --yaml "./projects/lsst_y1/xi_emul_10000points.yaml.yaml" \
    --probe $PROBE \
    --transfer_learning True \
    --pretrained_model $PRETRAINED_MODEL \
    --freeze_strategy early_2 \
    --learning_rate $LEARNING_RATE \
    --epochs $EPOCHS \
    --save_losses True \
    --save_testing_metrics True

mv losses.txt projects/lsst_y1/transfer_learning_testing/losses_early2_10000.txt
mv testing_metrics.txt projects/lsst_y1/transfer_learning_testing/testing_metrics_early2_10000.txt

echo ""
echo "All experiments completed at $(date)"
echo "Results saved in transfer_learning_testing/"
