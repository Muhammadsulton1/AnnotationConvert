from abc import ABC, abstractmethod
import os
import json
import tqdm
import numpy as np
import cv2
from src.logger import logger


class ConvertStrategy(ABC):
    """
        Абстрактный базовый класс для реализации стратегий конвертации данных.

        Реализует паттерн "Стратегия", позволяющий выбирать алгоритм конвертации
        во время выполнения программы. Классы-наследники должны реализовать
        конкретную логику преобразования данных в методе convert().

        Attributes:
            Нет публичных атрибутов
        """

    @abstractmethod
    def convert(self, input_path, output_path):
        pass


class Coco2YoloStrategy(ConvertStrategy):
    def __init__(self, categories):
        self.categories = categories

    def convert(self, input_path, output_path):
        # Create output directories
        labels_dir = os.path.join(output_path, "labels")
        images_dir = os.path.join(output_path, "images")
        os.makedirs(labels_dir, exist_ok=True)
        os.makedirs(images_dir, exist_ok=True)
        logger.info(f"Created output directories: {labels_dir} and {images_dir}")

        with open(input_path) as f:
            coco = json.load(f)

        logger.info(f"Loaded COCO dataset from {input_path}")
        logger.info(f"Number of images: {len(coco['images'])}, Number of annotations: {len(coco['annotations'])}")

        id2category = {c["id"]: c["name"] for c in coco["categories"]}
        images_with_annotations = 0

        for img in tqdm(coco["images"], desc="Converting COCO to YOLO"):
            img_id = img["id"]
            width = img["width"]
            height = img["height"]
            annotations = [a for a in coco["annotations"] if a["image_id"] == img_id]

            yolo_lines = []
            for ann in annotations:
                category_id = ann["category_id"]
                if ann.get("segmentation"):
                    for segment in ann["segmentation"]:
                        normalized = []
                        for i in range(0, len(segment), 2):
                            x = segment[i] / width
                            y = segment[i + 1] / height
                            normalized.extend([x, y])
                        yolo_lines.append(f"{category_id} " + " ".join(map(str, normalized)))
                else:
                    x, y, w, h = ann["bbox"]
                    x_center = (x + w / 2) / width
                    y_center = (y + h / 2) / height
                    w_norm = w / width
                    h_norm = h / height
                    yolo_lines.append(f"{category_id} {x_center} {y_center} {w_norm} {h_norm}")

            if yolo_lines:
                images_with_annotations += 1
                txt_path = os.path.join(labels_dir, f"{os.path.splitext(img['file_name'])[0]}.txt")
                with open(txt_path, "w") as f:
                    f.write("\n".join(yolo_lines))
            else:
                logger.debug(f"No annotations found for image: {img['file_name']}")

        logger.info(f"Conversion complete. Processed {len(coco['images'])} images total.")
        logger.info(f"{images_with_annotations} images had valid annotations.")


class Yolo2CocoStrategy(ConvertStrategy):
    def __init__(self, categories):
        self.categories = categories
        self.annotation_id = 1

    def convert(self, input_path, output_path):
        logger.info(f"Starting YOLO to COCO conversion from: {input_path}")
        logger.info(f"Output will be saved to: {output_path}")

        coco = {
            "images": [],
            "annotations": [],
            "categories": self.categories
        }

        # Collect all txt files
        txt_files = []
        for root, _, files in os.walk(input_path):
            for file in files:
                if file.endswith(".txt"):
                    txt_files.append(os.path.join(root, file))

        logger.info(f"Found {len(txt_files)} annotation files to process")
        if not txt_files:
            logger.warning("No annotation files found in input directory!")
            return

        processed_files = 0
        for txt_file in tqdm(txt_files, desc="Processing YOLO files"):
            base_name = os.path.splitext(os.path.basename(txt_file))[0]
            img_extensions = ['.jpg', '.jpeg', '.png']
            img_path = None

            # Find associated image file
            for ext in img_extensions:
                possible_path = os.path.join(os.path.dirname(txt_file), base_name + ext)
                if os.path.exists(possible_path):
                    img_path = possible_path
                    break

            if not img_path:
                logger.warning(f"Image for {base_name} not found, skipping")
                continue

            # Read image dimensions
            img = cv2.imread(img_path)
            if img is None:
                logger.warning(f"Failed to read image: {img_path}, skipping")
                continue

            height, width = img.shape[:2]
            image_id = len(coco["images"]) + 1

            # Add image entry
            coco["images"].append({
                "id": image_id,
                "file_name": os.path.basename(img_path),
                "width": width,
                "height": height
            })

            # Process annotations
            with open(txt_file) as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue

                try:
                    category_id = int(parts[0])
                    coords = list(map(float, parts[1:]))
                except ValueError as e:
                    logger.warning(f"Invalid line format in {txt_file}: {line.strip()}")
                    continue

                if len(coords) == 4:
                    x_center, y_center, w, h = coords
                    x_min = (x_center - w / 2) * width
                    y_min = (y_center - h / 2) * height
                    w_abs = w * width
                    h_abs = h * height

                    coco["annotations"].append({
                        "id": self.annotation_id,
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": [x_min, y_min, w_abs, h_abs],
                        "area": w_abs * h_abs,
                        "iscrowd": 0,
                        "segmentation": []
                    })
                else:
                    points = np.array(coords).reshape(-1, 2)
                    abs_points = points * [width, height]

                    x_coords = abs_points[:, 0]
                    y_coords = abs_points[:, 1]
                    x_min = min(x_coords)
                    y_min = min(y_coords)
                    x_max = max(x_coords)
                    y_max = max(y_coords)
                    w = x_max - x_min
                    h = y_max - y_min

                    contour = np.array(abs_points, dtype=np.float32).reshape(-1, 1, 2)
                    area = cv2.contourArea(contour)

                    coco["annotations"].append({
                        "id": self.annotation_id,
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": [x_min, y_min, w, h],
                        "area": area,
                        "iscrowd": 0,
                        "segmentation": [abs_points.flatten().tolist()]
                    })

                    self.annotation_id += 1

        with open(output_path, "w") as f:
            json.dump(coco, f, indent=4)


class Converter:
    def __init__(self, strategy: ConvertStrategy):
        self._strategy = strategy
        logger.info(f"Initialized converter with strategy: {strategy.__class__.__name__}")

    def set_strategy(self, strategy: ConvertStrategy):
        self._strategy = strategy
        logger.info(f"Updated conversion strategy to: {strategy.__class__.__name__}")

    def convert(self, input_path, output_path):
        logger.info(f"Starting conversion from {input_path} to {output_path}")
        self._strategy.convert(input_path, output_path)
        logger.info("Conversion process completed")


if __name__ == "__main__":
    categories = [
        {"id": 0, "name": "cat", "supercategory": "animal"},
        {"id": 1, "name": "dog", "supercategory": "animal"},
        {"id": 2, "name": "car", "supercategory": "vehicle"},
        {"id": 3, "name": "bus", "supercategory": "vehicle"}
    ]

    # Конвертация из YOLO в COCO
    converter = Converter(Yolo2CocoStrategy(categories))
    converter.convert("yolo_dataset", "coco_annotations.json")

    # Конвертация из COCO в YOLO
    converter.set_strategy(Coco2YoloStrategy(categories))
    converter.convert("coco_annotations.json", "yolo_dataset_output")

