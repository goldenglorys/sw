import xml.etree.ElementTree as ET
import os
import shutil
import csv
from pathlib import Path
from collections import defaultdict
import random


def parse_xml_annotations(archive_dir):
    """Parse all XML files and extract annotations."""
    annotations = []
    class_counts = defaultdict(int)

    archive_path = Path(archive_dir)
    xml_files = list(archive_path.glob("*.xml"))

    print(f"Found {len(xml_files)} XML files")

    for xml_file in xml_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Try to find actual image file using XML filename
        img_candidates = [
            archive_path / (xml_file.stem + ext)
            for ext in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]
        ]
        actual_img = next((f for f in img_candidates if f.exists()), None)

        if not actual_img:
            print(f"Warning: No image found for {xml_file.name}")
            continue

        size = root.find("size")
        width = int(size.find("width").text)
        height = int(size.find("height").text)

        # Process each object in the image
        for obj in root.findall("object"):
            class_name = obj.find("name").text.lower()

            # Skip invalid classes
            if class_name not in ["barcode", "qr"]:
                print(
                    f"Warning: Skipping invalid class '{class_name}' in {xml_file.name}"
                )
                continue

            bbox = obj.find("bndbox")

            xmin = int(bbox.find("xmin").text)
            ymin = int(bbox.find("ymin").text)
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)

            # Normalize coordinates to [0, 1]
            xmin_norm = xmin / width
            ymin_norm = ymin / height
            xmax_norm = xmax / width
            ymax_norm = ymax / height

            # Map class names to indices
            class_id = 0 if class_name == "barcode" else 1

            annotations.append(
                {
                    "original_file": actual_img.name,
                    "xmin": xmin_norm,
                    "ymin": ymin_norm,
                    "xmax": xmax_norm,
                    "ymax": ymax_norm,
                    "class": class_name,
                    "class_id": class_id,
                }
            )

            class_counts[class_name] += 1

    print(f"\nClass distribution:")
    for cls, count in class_counts.items():
        print(f"  {cls}: {count}")

    return annotations


def stratified_split(annotations, train_ratio=0.9, seed=42):
    """Split annotations into train/val with stratification by class."""
    random.seed(seed)

    # Group by original filename
    image_annotations = defaultdict(list)
    for ann in annotations:
        image_annotations[ann["original_file"]].append(ann)

    # Separate images by dominant class
    barcode_images = []
    qr_images = []

    for filename, anns in image_annotations.items():
        # Count classes in this image
        class_counts = defaultdict(int)
        for ann in anns:
            class_counts[ann["class"]] += 1

        # Assign to dominant class
        if class_counts["barcode"] > class_counts.get("qr", 0):
            barcode_images.append(filename)
        else:
            qr_images.append(filename)

    print(f"\nImages by dominant class:")
    print(f"  Barcode images: {len(barcode_images)}")
    print(f"  QR images: {len(qr_images)}")

    # Shuffle and split each class
    random.shuffle(barcode_images)
    random.shuffle(qr_images)

    barcode_split = int(len(barcode_images) * train_ratio)
    qr_split = int(len(qr_images) * train_ratio)

    train_images = set(barcode_images[:barcode_split] + qr_images[:qr_split])
    val_images = set(barcode_images[barcode_split:] + qr_images[qr_split:])

    # Split annotations
    train_anns = [ann for ann in annotations if ann["original_file"] in train_images]
    val_anns = [ann for ann in annotations if ann["original_file"] in val_images]

    print(f"\nSplit results:")
    print(f"  Train: {len(train_images)} images, {len(train_anns)} annotations")
    print(f"  Val: {len(val_images)} images, {len(val_anns)} annotations")

    return train_anns, val_anns, train_images, val_images


def rename_and_copy_images(annotations, image_set, source_dir, dest_dir, split_name):
    """Rename images with clean naming convention and copy to destination."""
    os.makedirs(dest_dir, exist_ok=True)

    # Single counter for all images
    image_counter = 0

    # Create mapping from original filename to new filename
    filename_mapping = {}

    # Group annotations by original file
    image_annotations = defaultdict(list)
    for ann in annotations:
        if ann["original_file"] in image_set:
            image_annotations[ann["original_file"]].append(ann)

    copied = 0
    for original_filename in sorted(image_set):
        # Generate new filename with unified naming
        new_filename = f"image_{image_counter:04d}"
        image_counter += 1

        filename_mapping[original_filename] = new_filename

        # Copy and rename image
        src = os.path.join(source_dir, original_filename)
        if os.path.exists(src):
            # Preserve original extension
            ext = Path(original_filename).suffix
            dest = os.path.join(dest_dir, new_filename + ext)
            shutil.copy2(src, dest)
            copied += 1

    print(f"Copied and renamed {copied}/{len(image_set)} images to {dest_dir}")
    print(f"  Total images: {image_counter}")

    return filename_mapping


def save_csv_annotations(annotations, filename_mapping, output_file):
    """Save annotations to CSV format with renamed filenames."""
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        for ann in annotations:
            new_filename = filename_mapping.get(ann["original_file"])
            if new_filename:
                # Format: filename,xmin,ymin,xmax,ymax,class_id
                writer.writerow(
                    [
                        new_filename,  # No extension in CSV
                        ann["xmin"],
                        ann["ymin"],
                        ann["xmax"],
                        ann["ymax"],
                        ann["class_id"],
                    ]
                )
    print(
        f"Saved {len([a for a in annotations if a['original_file'] in filename_mapping])} annotations to {output_file}"
    )


def main():
    # Configuration
    archive_dir = "archive"
    data_dir = "data"

    # Create output directories
    train_img_dir = os.path.join(data_dir, "train_images")
    val_img_dir = os.path.join(data_dir, "val_images")

    print("=" * 60)
    print("STEP 1: Parsing XML annotations")
    print("=" * 60)
    annotations = parse_xml_annotations(archive_dir)

    print("\n" + "=" * 60)
    print("STEP 2: Splitting into train/validation")
    print("=" * 60)
    train_anns, val_anns, train_imgs, val_imgs = stratified_split(annotations)

    print("\n" + "=" * 60)
    print("STEP 3: Renaming and copying training images")
    print("=" * 60)
    train_mapping = rename_and_copy_images(
        train_anns, train_imgs, archive_dir, train_img_dir, "train"
    )

    print("\n" + "=" * 60)
    print("STEP 4: Renaming and copying validation images")
    print("=" * 60)
    val_mapping = rename_and_copy_images(
        val_anns, val_imgs, archive_dir, val_img_dir, "val"
    )

    print("\n" + "=" * 60)
    print("STEP 5: Saving CSV annotations")
    print("=" * 60)
    train_csv = os.path.join(data_dir, "train_annotations.csv")
    val_csv = os.path.join(data_dir, "val_annotations.csv")
    save_csv_annotations(train_anns, train_mapping, train_csv)
    save_csv_annotations(val_anns, val_mapping, val_csv)

    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)
    print(f"\nNext steps:")
    print(f"1. Run convert_to_tfrecord.py to create TFRecord files")
    print(f"2. Update settings.py with new class names and paths")
    print(f"3. Train with: TinyYolo(training=True, classes=2)")


if __name__ == "__main__":
    main()
