import numpy as np
import cv2
import os
import json
import math
import itertools
import random
from dataclasses import dataclass
from typing import List, Dict


# relative to cell size
@dataclass
class RelatedBoxSizes:
    width: float
    height: float


@dataclass
class Category:
    id: int
    name: str


@dataclass
class GenConfig:
    img_size: int
    grid_sizes: List[int]
    images_count_per_grid: int
    categories: List
    category_ids_to_colors: Dict
    box_min_sizes: RelatedBoxSizes
    box_max_sizes: RelatedBoxSizes


def generate_rects_for_image(
        img_size: int,
        grid_size: int,
        box_min_sizes: RelatedBoxSizes,
        box_max_sizes: RelatedBoxSizes,
        category_ids: List,
) -> List:
    rects = []
    cell_size = img_size / grid_size
    for i, j in itertools.product(range(grid_size), repeat=2):
        center_x = cell_size // 2 + i * cell_size
        center_y = cell_size // 2 + j * cell_size

        min_width = int(box_min_sizes.width * cell_size)
        max_width = int(box_max_sizes.width * cell_size)
        width = random.randint(min_width, max_width)

        min_height = int(box_min_sizes.height * cell_size)
        max_height = int(box_max_sizes.height * cell_size)
        height = random.randint(min_height, max_height)

        sign = 1.0 if random.random() < 0.5 else -1.0
        angle = sign * math.pi * random.random()

        category_id = category_ids[random.randint(0, len(category_ids) - 1)]
        rects.append((center_x, center_y, width, height, angle, category_id))

    return rects


def draw_rect_on_image(image: np.ndarray, rotated_rects: List, category_ids_to_colors: Dict) -> np.ndarray:
    for rect in rotated_rects:
        center_x, center_y, width, height, angle, category_id = rect
        box = cv2.boxPoints((
            (center_x, center_y),
            (width, height),
            (angle * 180) // math.pi,
        ))
        box = np.int32(box)
        image = cv2.fillPoly(image, [box], category_ids_to_colors[category_id])

    return image


def run_generation(config: GenConfig, annotations_path: str, images_path: str):
    img_size = config.img_size
    sequentially_number = 0
    annotations = {"categories": [{"id": category.id, "name": category.name} for category in config.categories],
                   "images": [],
                   "annotations": [],
                   }

    for grid_size in config.grid_sizes:
        for _ in range(config.images_count_per_grid):
            rects = generate_rects_for_image(img_size, grid_size, config.box_min_sizes, config.box_max_sizes,
                                             list(config.category_ids_to_colors.keys()))
            image = np.zeros([img_size, img_size, 3], dtype=np.uint8)
            image.fill(255)
            image = draw_rect_on_image(image, rects, config.category_ids_to_colors)
            image_file_name = f'image_{sequentially_number}.png'
            annotations["images"].append(
                {
                    "id": sequentially_number,
                    "file_name": image_file_name,
                    "width": img_size,
                    "height": img_size,
                })
            annotations["annotations"].extend([
                {
                    "image_id": sequentially_number,
                    "bbox": rect[:5],
                    "category_id": rect[5],
                } for rect in rects])

            cv2.imwrite(os.path.join(images_path, image_file_name), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            sequentially_number += 1

    with open(os.path.join(annotations_path, 'train_annotations.json'), 'w') as outfile:
        json.dump(annotations, outfile)
