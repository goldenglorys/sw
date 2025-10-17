import cv2
from settings import Settings
from model.architecture import YoloV3Tiny, YoloLoss
from model.dataset import load_tfrecord_dataset, transform_images, transform_targets
from model.utils import freeze_all, draw_outputs
import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard,
)
from PIL import Image
from io import BytesIO
from tensorflow.keras.preprocessing.image import img_to_array
from absl import logging
import gc
import psutil

class GarbageCollectionCallback(tf.keras.callbacks.Callback):  
    def on_epoch_end(self, epoch, logs=None):  
        gc.collect()  

class ResourceMonitor(tf.keras.callbacks.Callback):  
    def on_epoch_end(self, epoch, logs=None):  
        mem = psutil.virtual_memory()  
        print(f"\nMemory usage: {mem.percent}% ({mem.used / 1e9:.2f}GB / {mem.total / 1e9:.2f}GB)")  

class TinyYolo:

    def __init__(self, channels=3, classes=1, training=False, **kwargs):
        self.channels = channels
        self.training = training
        self.classes = classes
        self.__dict__.update(Settings.model)  # Default settings
        self.__dict__.update(kwargs)  # Overrides

    def _gen_model(self):
        args = [
            "size",
            "channels",
            "anchors",
            "masks",
            "classes",
            "score_threshold",
            "iou_threshold",
            "max_boxes",
            "training",
        ]
        model = YoloV3Tiny(**{arg: getattr(self, arg) for arg in args})
        if not self.training:
            # model.load_weights(self.weights).expect_partial()
            model.load_weights(self.weights)
        return model

            
    def train(self, skip_transfer_learning=False, initial_epoch=0):
        self.training = True
        model = self._gen_model()

        # Retrieve train params
        train_dataset_dir = Settings.train["train_dataset"]
        val_dataset_dir = Settings.train["val_dataset"]
        classes = Settings.train["classes"]
        batch_size = Settings.train["batch_size"]
        learning_rate = Settings.train["learning_rate"]
        epochs = Settings.train["epochs"]
        checkpoints = Settings.train["checkpoints"]
        logs = Settings.train["logs"]
        pretrained_weights = Settings.train["pretrained_weights"]
        final_weights = Settings.train["final_weights"]
        run_eagerly = Settings.train["run_eagerly"]

        # Load train & val datasets
        train_dataset = load_tfrecord_dataset(
            train_dataset_dir, classes, self.size, self.max_boxes
        )
        train_dataset = train_dataset.cache()
        train_dataset = train_dataset.shuffle(buffer_size=512)
        train_dataset = train_dataset.batch(batch_size)
        train_dataset = train_dataset.map(
            lambda x, y: (
                transform_images(x, self.size),
                transform_targets(y, self.anchors, self.masks, self.size),
            )
        )
        train_dataset = train_dataset.prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE
        )

        val_dataset = load_tfrecord_dataset(
            val_dataset_dir, classes, self.size, self.max_boxes
        )
        val_dataset = val_dataset.batch(batch_size)
        val_dataset = val_dataset.map(
            lambda x, y: (
                transform_images(x, self.size),
                transform_targets(y, self.anchors, self.masks, self.size),
            )
        )

        # Pretrained weights (80 classes default pretrained classes)
        if not skip_transfer_learning:
            model_pretrained = TinyYolo(training=True, classes=80)._gen_model()
            model_pretrained.load_weights(pretrained_weights)
            # Only transfer the darknet backbone, not the detection heads
            model.get_layer("yolo_darknet").set_weights(
                model_pretrained.get_layer("yolo_darknet").get_weights()
            )
            freeze_all(model.get_layer("yolo_darknet"))

        # Train
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        loss = [
            YoloLoss(self.anchors[mask], classes=self.classes) for mask in self.masks
        ]
        model.compile(optimizer=optimizer, loss=loss, run_eagerly=run_eagerly)
        callbacks = [
            ReduceLROnPlateau(patience=3, verbose=1),
            EarlyStopping(patience=10, verbose=1),
            ModelCheckpoint(checkpoints, verbose=1, save_weights_only=True),
            TensorBoard(log_dir=logs),
            GarbageCollectionCallback(),
            ResourceMonitor()
        ]
        history = model.fit(
            train_dataset,
            epochs=epochs,
            initial_epoch=initial_epoch,
            callbacks=callbacks,
            validation_data=val_dataset,
        )
        # model.save_weights(final_weights)

    def predict(self, image, return_metadata=False):  
        class_names = Settings.class_names  
        model = self._gen_model()  
        img_raw = Image.open(image)  
        buf = BytesIO()  
        img_raw.save(buf, "JPEG", quality=50, optimize=True)  
        img_raw = Image.open(buf)  
        img = img_to_array(img_raw)  
        img = tf.expand_dims(img, 0)  
        img = transform_images(img, 416)  
        boxes, scores, classes, nums = model(img)  
        for i in range(nums[0]):  
            logging.info(  
                f'\t{np.array(scores[0][i])}, {np.array(boxes[0][i])}')  
        img = draw_outputs(np.array(img_raw),  
                            (boxes, scores, classes, nums), class_names)  
        
        if return_metadata:  
            return img, boxes, scores, classes, nums  
        return img

    def predict_array(self, img_array, return_metadata=False):
        """    
        Predict on a NumPy array (e.g., from camera frame)    
        
        Args:    
            img_array: NumPy array of shape (H, W, 3) in BGR or RGB format  
            return_metadata: If True, return (img, boxes, scores, classes, nums)  
        
        Returns:    
            Annotated image as NumPy array, or tuple if return_metadata=True  
        """    
        class_names = Settings.class_names    
        model = self._gen_model()    
        
        # Convert BGR to RGB if needed (OpenCV uses BGR)    
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:    
            img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)    
        else:    
            img_rgb = img_array    
        
        # Convert to tensor    
        img = tf.convert_to_tensor(img_rgb, dtype=tf.float32)    
        img = tf.expand_dims(img, 0)    
        img = transform_images(img, 416)    
        
        # Inference    
        boxes, scores, classes, nums = model(img)    
        
        # Draw outputs on original image    
        img_annotated = draw_outputs(img_rgb, (boxes, scores, classes, nums), class_names)    
        
        if return_metadata:  
            return img_annotated, boxes, scores, classes, nums  
        return img_annotated
