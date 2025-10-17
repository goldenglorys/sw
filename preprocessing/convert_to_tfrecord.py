import tensorflow as tf  
import os  
from PIL import Image  
import io  
import csv  
  
def _bytes_feature(value):  
    if isinstance(value, type(tf.constant(0))):  
        value = value.numpy()  
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))  
  
def _float_feature(value):  
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))  
  
def convert_to_tfrecord(image_dir, annotations_csv, output_tfrecord, class_map):  
    """Convert images and annotations to TFRecord format."""  
    writer = tf.io.TFRecordWriter(output_tfrecord)  
      
    # Read all annotations  
    annotations_by_image = {}  
    with open(annotations_csv, 'r') as f:  
        reader = csv.reader(f)  
        for row in reader:  
            img_name = row[0]  
            if img_name not in annotations_by_image:  
                annotations_by_image[img_name] = []  
            annotations_by_image[img_name].append({  
                'xmin': float(row[1]),  
                'ymin': float(row[2]),  
                'xmax': float(row[3]),  
                'ymax': float(row[4]),  
                'class_id': int(row[5])  
            })  
      
    print(f"Processing {len(annotations_by_image)} images...")  
      
    processed = 0  
    for img_name, boxes in annotations_by_image.items():  
        # Try different extensions  
        img_path = None  
        for ext in ['.jpg', '.jpeg', '.JPG', '.JPEG']:  
            candidate = os.path.join(image_dir, img_name + ext)  
            if os.path.exists(candidate):  
                img_path = candidate  
                break  
          
        if not img_path:  
            print(f"Warning: Image not found for {img_name}")  
            continue  
          
        try:  
            img = Image.open(img_path)  
            buf = io.BytesIO()  
            img.save(buf, format='JPEG')  
              
            # Create feature dict for each box  
            for box in boxes:  
                class_text = class_map[box['class_id']]  
                  
                example = tf.train.Example(  
                    features=tf.train.Features(feature={  
                        "image/encoded": _bytes_feature(buf.getvalue()),  
                        "image/object/bbox/xmin": _float_feature(box['xmin']),  
                        "image/object/bbox/ymin": _float_feature(box['ymin']),  
                        "image/object/bbox/xmax": _float_feature(box['xmax']),  
                        "image/object/bbox/ymax": _float_feature(box['ymax']),  
                        "image/object/class/text": _bytes_feature(class_text.encode()),  
                    }))  
                writer.write(example.SerializeToString())  
              
            processed += 1  
            if processed % 100 == 0:  
                print(f"  Processed {processed} images...")  
                  
        except Exception as e:  
            print(f"Error processing {img_name}: {e}")  
            continue  
      
    writer.close()  
    print(f"Created {output_tfrecord} with {processed} images")  
  
def main():  
    class_map = {0: "Barcode", 1: "QR"}  
      
    print("Converting training set...")  
    convert_to_tfrecord(  
        image_dir="data/train_images",  
        annotations_csv="data/train_annotations.csv",  
        output_tfrecord="data/train_barcode_qr.tf_record",  
        class_map=class_map  
    )  
      
    print("\nConverting validation set...")  
    convert_to_tfrecord(  
        image_dir="data/val_images",  
        annotations_csv="data/val_annotations.csv",  
        output_tfrecord="data/val_barcode_qr.tf_record",  
        class_map=class_map  
    )  
      
    print("\nDone! TFRecord files created in data/ directory")  
  
if __name__ == "__main__":  
    main()