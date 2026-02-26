from src.data.pair_generator import generate_pairs

def _make_dataset():
    return {
        "dog_001": ["img1.jpg", "img2.jpg", "img3.jpg"],
        "dog_002": ["img4.jpg", "img5.jpg", "img6.jpg"],
        "dog_003": ["img7.jpg", "img8.jpg", "img9.jpg"],
    }

def test_generate_pairs_has_both_types():
    pairs = generate_pairs(_make_dataset())
    labels = [p[2] for p in pairs]
    assert 0 in labels
    assert 1 in labels

def test_positive_pairs_same_pet():
    dataset = _make_dataset()
    pairs = generate_pairs(dataset)
    img_to_pet = {img: pet for pet, imgs in dataset.items() for img in imgs}
    for img1, img2, label in pairs:
        if label == 0:
            assert img_to_pet[img1] == img_to_pet[img2]

def test_negative_pairs_different_pets():
    dataset = _make_dataset()
    pairs = generate_pairs(dataset)
    img_to_pet = {img: pet for pet, imgs in dataset.items() for img in imgs}
    for img1, img2, label in pairs:
        if label == 1:
            assert img_to_pet[img1] != img_to_pet[img2]
