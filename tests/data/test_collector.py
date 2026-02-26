import tempfile
from pathlib import Path
from src.data.collector import scan_dataset, get_pet_ids

def _create_dummy_dataset(base_dir):
    for dog_id in ["dog_001", "dog_002", "dog_003"]:
        dog_dir = Path(base_dir) / dog_id
        dog_dir.mkdir(parents=True)
        for i in range(3):
            (dog_dir / f"{i:03d}.jpg").touch()

def test_scan_dataset_finds_all_pets():
    with tempfile.TemporaryDirectory() as tmpdir:
        _create_dummy_dataset(tmpdir)
        result = scan_dataset(tmpdir)
        assert len(result) == 3

def test_scan_dataset_finds_images():
    with tempfile.TemporaryDirectory() as tmpdir:
        _create_dummy_dataset(tmpdir)
        result = scan_dataset(tmpdir)
        for pet_id, images in result.items():
            assert len(images) == 3

def test_get_pet_ids():
    with tempfile.TemporaryDirectory() as tmpdir:
        _create_dummy_dataset(tmpdir)
        dataset = scan_dataset(tmpdir)
        ids = get_pet_ids(dataset)
        assert set(ids) == {"dog_001", "dog_002", "dog_003"}
