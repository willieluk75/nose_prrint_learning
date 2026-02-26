import random
from typing import Dict, List, Tuple

Pair = Tuple[str, str, int]  # (img1, img2, label): 0=same, 1=different


def generate_pairs(
    dataset: Dict[str, List[str]],
    negative_ratio: int = 3,
    seed: int = 42,
) -> List[Pair]:
    """
    生成 positive（同一寵物）和 negative（不同寵物）pairs。

    Args:
        dataset: {pet_id: [image_path, ...]}
        negative_ratio: 每個 positive pair 對應的 negative pair 數量
        seed: 隨機種子
    """
    rng = random.Random(seed)
    pairs = []
    pet_ids = list(dataset.keys())

    for pet_id, images in dataset.items():
        if len(images) < 2:
            continue
        for i in range(len(images)):
            for j in range(i + 1, len(images)):
                pairs.append((images[i], images[j], 0))

    num_negatives = len(pairs) * negative_ratio
    neg_count = 0
    attempts = 0

    while neg_count < num_negatives and attempts < num_negatives * 10:
        attempts += 1
        pet1, pet2 = rng.sample(pet_ids, 2)
        img1 = rng.choice(dataset[pet1])
        img2 = rng.choice(dataset[pet2])
        pairs.append((img1, img2, 1))
        neg_count += 1

    rng.shuffle(pairs)
    return pairs
