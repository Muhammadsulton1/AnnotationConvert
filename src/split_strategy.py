import os
import json
import shutil
import glob
import numpy as np
from tqdm import tqdm
from src.logger import logger
from src.strategy import ConvertStrategy


class DataSplitStrategy(ConvertStrategy):
    """
    Strategy for splitting datasets into train/val/test subsets
    Supports both YOLO format and COCO JSON annotations
    """

    def __init__(self,
                 split_ratios=(0.8, 0.1, 0.1),
                 image_extensions=('.jpg', '.jpeg', '.png'),
                 coco_image_dir=None,
                 random_seed=42):
        """
        Initialize split strategy

        Args:
            split_ratios (tuple): Train/val/test ratios (must sum to <=1)
            image_extensions (tuple): Image file extensions to look for
            coco_image_dir (str): Path to COCO images directory (required for COCO splits)
            random_seed (int): Random seed for reproducibility
        """
        self.split_ratios = split_ratios
        self.image_extensions = image_extensions
        self.coco_image_dir = coco_image_dir
        self.random_seed = random_seed
        self._validate_ratios()

    def _validate_ratios(self):
        """Ensure valid split ratios"""
        total = sum(self.split_ratios)
        if total > 1.0 + 1e-6:
            raise ValueError(f"Split ratios sum to {total:.2f} (should be <=1)")
        if len(self.split_ratios) != 3:
            raise ValueError("Must provide exactly 3 split ratios (train, val, test)")

    def convert(self, input_path, output_path):
        """
        Main conversion method
        Automatically detects YOLO (directory) or COCO (JSON file) format
        """
        logger.info(f"Starting dataset split with ratios {self.split_ratios}")

        if os.path.isdir(input_path):
            self._split_yolo(input_path, output_path)
        elif os.path.isfile(input_path) and input_path.endswith('.json'):
            self._split_coco(input_path, output_path)
        else:
            raise ValueError("Unsupported input format - must be YOLO directory or COCO JSON file")

    def _split_yolo(self, input_dir, output_dir):
        """Split YOLO format dataset"""
        logger.info("Processing YOLO format dataset")

        image_dir = os.path.join(input_dir, 'images')
        label_dir = os.path.join(input_dir, 'labels')
        valid_pairs = self._get_yolo_pairs(image_dir, label_dir)

        splits = self._create_splits(valid_pairs)

        self._copy_yolo_splits(splits, output_dir)

    def _get_yolo_pairs(self, image_dir, label_dir):
        """Validate and collect YOLO image-label pairs"""
        valid_pairs = []

        image_files = []
        for ext in self.image_extensions:
            image_files.extend(glob.glob(os.path.join(image_dir, f'*{ext}')))
        image_files = sorted(image_files)

        logger.info(f"Found {len(image_files)} images in {image_dir}")

        for img_path in tqdm(image_files, desc="Validating YOLO files"):
            base = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(label_dir, f"{base}.txt")

            if os.path.exists(label_path):
                valid_pairs.append((img_path, label_path))
            else:
                logger.warning(f"No label found for image: {os.path.basename(img_path)}")

        logger.info(f"Found {len(valid_pairs)} valid image-label pairs")
        return valid_pairs

    def _create_splits(self, items):
        """Create randomized splits based on ratios"""
        np.random.seed(self.random_seed)
        indices = np.random.permutation(len(items))

        train_ratio, val_ratio, test_ratio = self.split_ratios
        cum_ratios = np.cumsum([train_ratio, val_ratio, test_ratio])

        split_points = (len(items) * cum_ratios).astype(int)
        splits = {
            'train': indices[:split_points[0]],
            'val': indices[split_points[0]:split_points[1]],
            'test': indices[split_points[1]:split_points[2]]
        }

        logger.info(f"Split sizes: "
                    f"Train: {len(splits['train'])} "
                    f"Val: {len(splits['val'])} "
                    f"Test: {len(splits['test'])}")
        return splits

    @staticmethod
    def _copy_yolo_splits(splits, output_dir):
        """Copy YOLO files to split directories"""
        for split_name, indices in splits.items():
            split_dir = os.path.join(output_dir, split_name)
            os.makedirs(os.path.join(split_dir, 'images'), exist_ok=True)
            os.makedirs(os.path.join(split_dir, 'labels'), exist_ok=True)

            logger.info(f"Creating {split_name} split in {split_dir}")

    def _split_coco(self, json_path, output_dir):
        """Split COCO format dataset"""
        logger.info("Processing COCO format dataset")

        # Load COCO data
        with open(json_path) as f:
            coco_data = json.load(f)

        splits = self._create_splits(coco_data['images'])

        for split_name, indices in splits.items():
            split_dir = os.path.join(output_dir, split_name)
            os.makedirs(os.path.join(split_dir, 'images'), exist_ok=True)

            subset_images = [coco_data['images'][i] for i in indices]
            image_ids = {img['id'] for img in subset_images}
            subset_anns = [ann for ann in coco_data['annotations']
                           if ann['image_id'] in image_ids]

            new_coco = {
                'images': subset_images,
                'annotations': subset_anns,
                'categories': coco_data['categories']
            }

            ann_path = os.path.join(split_dir, 'annotations.json')
            with open(ann_path, 'w') as f:
                json.dump(new_coco, f, indent=4)

            if self.coco_image_dir:
                self._copy_coco_images(subset_images, split_dir)
            else:
                logger.warning("No image directory provided - skipping image copying")

    def _copy_coco_images(self, images, split_dir):
        """Copy COCO images to split directory"""
        image_dir = os.path.join(split_dir, 'images')
        os.makedirs(image_dir, exist_ok=True)

        for img in tqdm(images, desc=f"Copying images to {os.path.basename(split_dir)}"):
            src_path = os.path.join(self.coco_image_dir, img['file_name'])
            dst_path = os.path.join(image_dir, img['file_name'])

            if os.path.exists(src_path):
                shutil.copy2(src_path, dst_path)
            else:
                logger.warning(f"Missing source image: {src_path}")


if __name__ == "__main__":
    dataset_spliter = DataSplitStrategy(
        split_ratios=(0.8, 0.1, 0.1),
        coco_image_dir="/path/to/coco/images")

    dataset_spliter.convert(
        input_path="/path/to/yolo_dataset",
        output_path="/path/to/split_output")

    dataset_spliter.convert(
        input_path="/path/to/coco/annotations.json",
        output_path="/path/to/split_output")