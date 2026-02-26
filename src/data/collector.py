from pathlib import Path
from typing import Dict, List

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def scan_dataset(data_dir: str) -> Dict[str, List[str]]:
    """
    掃描資料集目錄，返回每隻寵物的圖片路徑清單。
    預期結構：data_dir/{pet_id}/{image}.jpg
    """
    base = Path(data_dir)
    dataset = {}

    for pet_dir in sorted(base.iterdir()):
        if not pet_dir.is_dir():
            continue
        images = sorted([
            str(f) for f in pet_dir.iterdir()
            if f.suffix.lower() in IMAGE_EXTENSIONS
        ])
        if images:
            dataset[pet_dir.name] = images

    return dataset


def get_pet_ids(dataset: Dict[str, List[str]]) -> List[str]:
    """返回資料集中所有寵物 ID 列表"""
    return list(dataset.keys())
