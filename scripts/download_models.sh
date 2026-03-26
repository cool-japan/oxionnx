#!/bin/bash
# Download ONNX model zoo test models.
# Usage: ./scripts/download_models.sh [output_dir]

MODEL_DIR="${1:-tests/models}"
mkdir -p "$MODEL_DIR"

echo "Downloading ONNX test models to $MODEL_DIR..."

# ResNet-18 (ONNX Model Zoo)
echo "  ResNet-18..."
curl -sL "https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet18-v1-7.onnx" \
    -o "$MODEL_DIR/resnet18.onnx" 2>/dev/null || echo "    (download failed)"

# MobileNetV2
echo "  MobileNetV2..."
curl -sL "https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-7.onnx" \
    -o "$MODEL_DIR/mobilenetv2.onnx" 2>/dev/null || echo "    (download failed)"

# Note: BERT, GPT-2, YOLOv8, Whisper models are larger and may need
# different sources. Users can add them manually.

echo "Done. Run: OXIONNX_MODEL_DIR=$MODEL_DIR cargo test model_zoo -- --nocapture"
