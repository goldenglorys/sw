import io
import tensorflow as tf  
from PIL import Image  
import os  
  
def extract_images_from_tfrecord(tfrecord_path, output_dir):  
    """Extract images from TFRecord file"""  
    os.makedirs(output_dir, exist_ok=True)  
      
    dataset = tf.data.TFRecordDataset(tfrecord_path)  
      
    for idx, record in enumerate(dataset):  
        example = tf.train.Example()  
        example.ParseFromString(record.numpy())  
          
        # Extract encoded image  
        encoded_jpg = example.features.feature['image/encoded'].bytes_list.value[0]  
          
        # Save as JPEG  
        img = Image.open(io.BytesIO(encoded_jpg))  
        img.save(f"{output_dir}/image_{idx:04d}.jpg")  
          
        # Extract bounding box info for reference  
        xmin = example.features.feature['image/object/bbox/xmin'].float_list.value[0]  
        ymin = example.features.feature['image/object/bbox/ymin'].float_list.value[0]  
        xmax = example.features.feature['image/object/bbox/xmax'].float_list.value[0]  
        ymax = example.features.feature['image/object/bbox/ymax'].float_list.value[0]  
          
        print(f"Image {idx}: bbox=[{xmin},{ymin},{xmax},{ymax}]")  
  
# Extract training images  
extract_images_from_tfrecord('data/all.tf_record', 'data/extracted_train')  
# Extract validation images    
extract_images_from_tfrecord('data/validation.tf_record', 'data/extracted_val')