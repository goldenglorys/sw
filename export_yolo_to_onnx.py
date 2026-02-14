import tensorflow as tf
import tf2onnx
import onnx
from model.tiny_yolo import TinyYolo

print("Loading YOLO model...")
yolo = TinyYolo(classes=2, training=False)
model = yolo._gen_model()

# Load your trained weights
model.load_weights('data/model/yolov3_train_class2_final.weights.h5')
print("Weights loaded")

# Convert to ONNX
print("Converting to ONNX...")

# Get input/output specs
input_signature = [tf.TensorSpec(model.inputs[0].shape, tf.float32, name='input')]

onnx_model, _ = tf2onnx.convert.from_keras(
    model,
    input_signature=input_signature,
    opset=13,
    output_path='yolo_barcode_detector.onnx'
)

print("Model exported to: yolo_barcode_detector.onnx")

# Verify
onnx_model = onnx.load('yolo_barcode_detector.onnx')
onnx.checker.check_model(onnx_model)
print("ONNX model verified")