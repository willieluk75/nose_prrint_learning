# 寵物鼻紋辨認系統 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 建立狗鼻紋 1:1 身份驗證系統（Phase 1a），ML 核心使用 Transfer Learning + Siamese Network，後端使用 FastAPI + Supabase (pgvector)，架構預留 ArcFace 升級路線。

**Architecture:** MobileNetV2 backbone → 256-dim L2 normalized embedding → Supabase pgvector 儲存與查詢。FastAPI 提供 REST API，Web 和 Mobile 共用同一套後端。

**Tech Stack:** Python 3.10+, TensorFlow 2.13, FastAPI, Supabase (PostgreSQL + pgvector + Storage), Pytest

---

## 執行順序總覽

```
Task 1   專案初始化
Task 2   資料預處理模組
Task 3   資料增強模組
Task 4   資料集整理工具
Task 5   Pair Generator
Task 6   Backbone 模型
Task 7   Embedding Head
Task 8   Siamese Network + Contrastive Loss
Task 9   Supabase Schema Migration
Task 10  Supabase Service（DB + Storage 操作）
Task 11  Embedder + Matcher
Task 12  FastAPI 主應用程式
Task 13  FastAPI /pets 路由
Task 14  Trainer
Task 15  評估指標
Task 16  整合測試
Task 17  Jupyter Notebook
```

---

### Task 1: 專案初始化

**Files:**
- Create: `requirements.txt`
- Create: `.env.example`
- Create: `src/__init__.py`
- Create: `src/data/__init__.py`
- Create: `src/models/__init__.py`
- Create: `src/training/__init__.py`
- Create: `src/inference/__init__.py`
- Create: `src/evaluation/__init__.py`
- Create: `api/__init__.py`
- Create: `api/routers/__init__.py`
- Create: `api/schemas/__init__.py`
- Create: `api/services/__init__.py`
- Create: `tests/__init__.py`

**Step 1: 建立資料夾結構**

```bash
mkdir -p data/raw/dogs data/processed data/embeddings
mkdir -p src/data src/models src/training src/inference src/evaluation
mkdir -p api/routers api/schemas api/services
mkdir -p tests/data tests/models tests/inference tests/evaluation tests/api
mkdir -p notebooks models docs/plans supabase/migrations
```

**Step 2: 建立 requirements.txt**

```
tensorflow==2.13.0
numpy==1.24.3
opencv-python==4.8.0.76
scikit-learn==1.3.0
matplotlib==3.7.2
pillow==10.0.0
fastapi==0.103.0
uvicorn==0.23.2
python-multipart==0.0.6
pydantic==2.3.0
pydantic-settings==2.0.3
supabase==1.2.0
httpx==0.24.1
pytest==7.4.0
pytest-asyncio==0.21.1
jupyter==1.0.0
tqdm==4.66.1
python-dotenv==1.0.0
```

**Step 3: 建立 .env.example**

```
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-key
SUPABASE_SERVICE_KEY=your-service-role-key
MODEL_WEIGHTS_PATH=models/best_model.h5
EMBEDDING_DIM=256
SIMILARITY_THRESHOLD=0.85
```

**Step 4: 安裝依賴**

```bash
pip install -r requirements.txt
```

Expected: 所有套件安裝成功

**Step 5: 建立所有 `__init__.py`（空白檔案）**

**Step 6: Commit**

```bash
git init
git add .
git commit -m "chore: project scaffolding with FastAPI + Supabase dependencies"
```

---

### Task 2: 資料預處理模組

**Files:**
- Create: `src/data/preprocessor.py`
- Create: `tests/data/test_preprocessor.py`

**Step 1: 寫失敗測試**

```python
# tests/data/test_preprocessor.py
import numpy as np
import pytest
from src.data.preprocessor import preprocess_image, load_and_preprocess

def test_preprocess_image_output_shape():
    dummy_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    result = preprocess_image(dummy_img)
    assert result.shape == (224, 224, 3)

def test_preprocess_image_normalized():
    dummy_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    result = preprocess_image(dummy_img)
    assert result.min() >= 0.0
    assert result.max() <= 1.0

def test_preprocess_image_dtype():
    dummy_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    result = preprocess_image(dummy_img)
    assert result.dtype == np.float32
```

**Step 2: 確認測試失敗**

```bash
pytest tests/data/test_preprocessor.py -v
```
Expected: FAIL — `ModuleNotFoundError`

**Step 3: 實作 preprocessor.py**

```python
# src/data/preprocessor.py
import cv2
import numpy as np
from pathlib import Path

TARGET_SIZE = (224, 224)


def preprocess_image(img: np.ndarray) -> np.ndarray:
    """
    將 numpy array 圖片 resize 至 224x224 並正規化至 [0, 1]。

    Args:
        img: shape (H, W, 3), dtype uint8
    Returns:
        shape (224, 224, 3), dtype float32
    """
    resized = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0


def load_and_preprocess(image_path: str) -> np.ndarray:
    """
    從檔案路徑讀取圖片並預處理。

    Raises:
        FileNotFoundError: 檔案不存在
        ValueError: 無法讀取圖片
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return preprocess_image(img_rgb)
```

**Step 4: 確認測試通過**

```bash
pytest tests/data/test_preprocessor.py -v
```
Expected: 3 passed

**Step 5: Commit**

```bash
git add src/data/preprocessor.py tests/data/test_preprocessor.py
git commit -m "feat: add image preprocessor"
```

---

### Task 3: 資料增強模組

**Files:**
- Create: `src/data/augmentor.py`
- Create: `tests/data/test_augmentor.py`

**Step 1: 寫失敗測試**

```python
# tests/data/test_augmentor.py
import numpy as np
from src.data.augmentor import augment_image

def test_augment_output_shape():
    img = np.random.rand(224, 224, 3).astype(np.float32)
    result = augment_image(img)
    assert result.shape == (224, 224, 3)

def test_augment_output_range():
    img = np.random.rand(224, 224, 3).astype(np.float32)
    result = augment_image(img)
    assert result.min() >= 0.0
    assert result.max() <= 1.0

def test_augment_produces_variation():
    img = np.ones((224, 224, 3), dtype=np.float32) * 0.5
    results = [augment_image(img) for _ in range(10)]
    all_same = all(np.allclose(results[0], r) for r in results[1:])
    assert not all_same
```

**Step 2: 確認測試失敗**

```bash
pytest tests/data/test_augmentor.py -v
```

**Step 3: 實作 augmentor.py**

```python
# src/data/augmentor.py
import numpy as np
import cv2


def augment_image(img: np.ndarray) -> np.ndarray:
    """
    對已預處理的圖片（float32, [0,1]）進行隨機增強。
    增強：水平翻轉、亮度/對比調整、輕微旋轉、隨機裁剪再 resize
    """
    result = img.copy()

    if np.random.rand() > 0.5:
        result = np.fliplr(result)

    brightness = np.random.uniform(0.8, 1.2)
    result = np.clip(result * brightness, 0.0, 1.0)

    contrast = np.random.uniform(0.8, 1.2)
    result = np.clip((result - 0.5) * contrast + 0.5, 0.0, 1.0)

    angle = np.random.uniform(-15, 15)
    h, w = result.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    result = cv2.warpAffine(result, M, (w, h))

    scale = np.random.uniform(0.8, 1.0)
    crop_h, crop_w = int(h * scale), int(w * scale)
    top = np.random.randint(0, h - crop_h + 1)
    left = np.random.randint(0, w - crop_w + 1)
    cropped = result[top:top+crop_h, left:left+crop_w]
    result = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_AREA)

    return result.astype(np.float32)
```

**Step 4: 確認測試通過**

```bash
pytest tests/data/test_augmentor.py -v
```
Expected: 3 passed

**Step 5: Commit**

```bash
git add src/data/augmentor.py tests/data/test_augmentor.py
git commit -m "feat: add random image augmentation"
```

---

### Task 4: 資料集整理工具

**Files:**
- Create: `src/data/collector.py`
- Create: `tests/data/test_collector.py`

**Step 1: 寫失敗測試**

```python
# tests/data/test_collector.py
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
```

**Step 2: 確認測試失敗**

```bash
pytest tests/data/test_collector.py -v
```

**Step 3: 實作 collector.py**

```python
# src/data/collector.py
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
```

**Step 4: 確認測試通過**

```bash
pytest tests/data/test_collector.py -v
```
Expected: 3 passed

**Step 5: Commit**

```bash
git add src/data/collector.py tests/data/test_collector.py
git commit -m "feat: add dataset scanner"
```

---

### Task 5: Pair Generator

**Files:**
- Create: `src/data/pair_generator.py`
- Create: `tests/data/test_pair_generator.py`

**Step 1: 寫失敗測試**

```python
# tests/data/test_pair_generator.py
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
```

**Step 2: 確認測試失敗**

```bash
pytest tests/data/test_pair_generator.py -v
```

**Step 3: 實作 pair_generator.py**

```python
# src/data/pair_generator.py
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
```

**Step 4: 確認測試通過**

```bash
pytest tests/data/test_pair_generator.py -v
```
Expected: 3 passed

**Step 5: Commit**

```bash
git add src/data/pair_generator.py tests/data/test_pair_generator.py
git commit -m "feat: add positive/negative pair generator"
```

---

### Task 6: Backbone 模型

**Files:**
- Create: `src/models/backbone.py`
- Create: `tests/models/test_backbone.py`

**Step 1: 寫失敗測試**

```python
# tests/models/test_backbone.py
import numpy as np
import tensorflow as tf
from src.models.backbone import build_backbone

def test_backbone_output_shape():
    model = build_backbone()
    dummy_input = np.random.rand(2, 224, 224, 3).astype(np.float32)
    output = model(dummy_input, training=False)
    assert len(output.shape) == 2
    assert output.shape[0] == 2

def test_backbone_feature_dim():
    model = build_backbone()
    dummy_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
    output = model(dummy_input, training=False)
    assert output.shape[1] == 1280

def test_backbone_returns_keras_model():
    model = build_backbone()
    assert isinstance(model, tf.keras.Model)
```

**Step 2: 確認測試失敗**

```bash
pytest tests/models/test_backbone.py -v
```

**Step 3: 實作 backbone.py**

```python
# src/models/backbone.py
import tensorflow as tf


def build_backbone(trainable: bool = False) -> tf.keras.Model:
    """
    MobileNetV2 backbone，移除分類頭，輸出 (batch, 1280) 特徵向量。
    預設凍結所有層（trainable=False）。
    """
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = trainable

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    return tf.keras.Model(inputs=inputs, outputs=x, name="backbone")
```

**Step 4: 確認測試通過**

```bash
pytest tests/models/test_backbone.py -v
```
Expected: 3 passed（第一次執行會下載 MobileNetV2 權重）

**Step 5: Commit**

```bash
git add src/models/backbone.py tests/models/test_backbone.py
git commit -m "feat: add MobileNetV2 backbone"
```

---

### Task 7: Embedding Head

**Files:**
- Create: `src/models/embedding_head.py`
- Create: `tests/models/test_embedding_head.py`

**Step 1: 寫失敗測試**

```python
# tests/models/test_embedding_head.py
import numpy as np
import tensorflow as tf
from src.models.embedding_head import build_embedding_head

def test_embedding_head_output_shape():
    head = build_embedding_head(input_dim=1280, embedding_dim=256)
    dummy_input = np.random.rand(4, 1280).astype(np.float32)
    output = head(dummy_input, training=False)
    assert output.shape == (4, 256)

def test_embedding_head_l2_normalized():
    head = build_embedding_head(input_dim=1280, embedding_dim=256)
    dummy_input = np.random.rand(4, 1280).astype(np.float32)
    output = head(dummy_input, training=False)
    norms = np.linalg.norm(output.numpy(), axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)
```

**Step 2: 確認測試失敗**

```bash
pytest tests/models/test_embedding_head.py -v
```

**Step 3: 實作 embedding_head.py**

```python
# src/models/embedding_head.py
import tensorflow as tf


def build_embedding_head(
    input_dim: int = 1280,
    embedding_dim: int = 256,
) -> tf.keras.Model:
    """
    Embedding projection head，輸出 L2 normalized 向量。
    所有 embedding 在單位超球面上，使歐氏距離計算有效。
    """
    inputs = tf.keras.Input(shape=(input_dim,))
    x = tf.keras.layers.Dense(512, activation="relu")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(embedding_dim)(x)
    x = tf.keras.layers.Lambda(
        lambda v: tf.math.l2_normalize(v, axis=1),
        name="l2_normalize",
    )(x)

    return tf.keras.Model(inputs=inputs, outputs=x, name="embedding_head")
```

**Step 4: 確認測試通過**

```bash
pytest tests/models/test_embedding_head.py -v
```
Expected: 2 passed

**Step 5: Commit**

```bash
git add src/models/embedding_head.py tests/models/test_embedding_head.py
git commit -m "feat: add L2-normalized embedding head"
```

---

### Task 8: Siamese Network

**Files:**
- Create: `src/models/siamese.py`
- Create: `tests/models/test_siamese.py`

**Step 1: 寫失敗測試**

```python
# tests/models/test_siamese.py
import numpy as np
import tensorflow as tf
from src.models.siamese import build_siamese_model, contrastive_loss

def test_siamese_output_shape():
    model = build_siamese_model()
    img1 = np.random.rand(2, 224, 224, 3).astype(np.float32)
    img2 = np.random.rand(2, 224, 224, 3).astype(np.float32)
    distances = model([img1, img2], training=False)
    assert distances.shape == (2,)

def test_same_image_zero_distance():
    model = build_siamese_model()
    img = np.random.rand(1, 224, 224, 3).astype(np.float32)
    distances = model([img, img], training=False)
    assert distances.numpy()[0] < 0.01

def test_contrastive_loss_positive_pair():
    y_true = tf.constant([0.0])
    loss_small = contrastive_loss(y_true, tf.constant([0.1]))
    loss_large = contrastive_loss(y_true, tf.constant([0.9]))
    assert loss_small < loss_large

def test_contrastive_loss_negative_pair():
    y_true = tf.constant([1.0])
    loss_small = contrastive_loss(y_true, tf.constant([0.1]))
    loss_large = contrastive_loss(y_true, tf.constant([1.5]))
    assert loss_small > loss_large
```

**Step 2: 確認測試失敗**

```bash
pytest tests/models/test_siamese.py -v
```

**Step 3: 實作 siamese.py**

```python
# src/models/siamese.py
import tensorflow as tf
from src.models.backbone import build_backbone
from src.models.embedding_head import build_embedding_head


def contrastive_loss(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    margin: float = 1.0,
) -> tf.Tensor:
    """
    Contrastive Loss: L = (1-y)*D² + y*max(margin-D, 0)²
    y=0: same pet（距離應小）, y=1: different pet（距離應大）
    """
    y_true = tf.cast(y_true, tf.float32)
    d = tf.cast(y_pred, tf.float32)
    positive_loss = (1.0 - y_true) * tf.square(d)
    negative_loss = y_true * tf.square(tf.maximum(margin - d, 0.0))
    return tf.reduce_mean(positive_loss + negative_loss)


def build_siamese_model(embedding_dim: int = 256) -> tf.keras.Model:
    """
    Siamese Network，共享 backbone + embedding_head 權重。
    輸入兩張圖片，輸出歐氏距離。
    """
    backbone = build_backbone(trainable=False)
    embedding_head = build_embedding_head(input_dim=1280, embedding_dim=embedding_dim)

    input_a = tf.keras.Input(shape=(224, 224, 3), name="image_a")
    input_b = tf.keras.Input(shape=(224, 224, 3), name="image_b")

    emb_a = embedding_head(backbone(input_a))
    emb_b = embedding_head(backbone(input_b))

    distance = tf.keras.layers.Lambda(
        lambda x: tf.norm(x[0] - x[1], axis=1),
        name="euclidean_distance",
    )([emb_a, emb_b])

    return tf.keras.Model(inputs=[input_a, input_b], outputs=distance, name="siamese")
```

**Step 4: 確認測試通過**

```bash
pytest tests/models/test_siamese.py -v
```
Expected: 4 passed

**Step 5: Commit**

```bash
git add src/models/siamese.py tests/models/test_siamese.py
git commit -m "feat: add Siamese network with contrastive loss"
```

---

### Task 9: Supabase Schema Migration

**Files:**
- Create: `supabase/migrations/001_init.sql`

**Step 1: 建立 SQL migration**

```sql
-- supabase/migrations/001_init.sql

-- pgvector 擴充
CREATE EXTENSION IF NOT EXISTS vector;

-- 寵物主表
CREATE TABLE pets (
  id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  owner_id    UUID REFERENCES auth.users(id) ON DELETE CASCADE,
  name        TEXT NOT NULL,
  species     TEXT NOT NULL CHECK (species IN ('dog', 'cat')),
  breed       TEXT,
  created_at  TIMESTAMPTZ DEFAULT NOW(),
  updated_at  TIMESTAMPTZ DEFAULT NOW()
);

-- 鼻紋 Embedding 表
CREATE TABLE nose_embeddings (
  id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  pet_id       UUID NOT NULL REFERENCES pets(id) ON DELETE CASCADE,
  embedding    vector(256) NOT NULL,
  image_url    TEXT,
  created_at   TIMESTAMPTZ DEFAULT NOW()
);

-- 向量索引（加速 cosine similarity 搜尋）
CREATE INDEX nose_embeddings_embedding_idx
  ON nose_embeddings
  USING ivfflat (embedding vector_cosine_ops)
  WITH (lists = 100);

-- 相似度搜尋 RPC 函式
CREATE OR REPLACE FUNCTION match_nose_embedding(
  query_embedding vector(256),
  match_threshold FLOAT DEFAULT 0.85,
  match_count     INT DEFAULT 5
)
RETURNS TABLE (
  pet_id       UUID,
  embedding_id UUID,
  similarity   FLOAT
)
LANGUAGE sql STABLE AS $$
  SELECT
    ne.pet_id,
    ne.id AS embedding_id,
    1 - (ne.embedding <=> query_embedding) AS similarity
  FROM nose_embeddings ne
  WHERE 1 - (ne.embedding <=> query_embedding) > match_threshold
  ORDER BY ne.embedding <=> query_embedding
  LIMIT match_count;
$$;

-- updated_at 自動更新 trigger
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER pets_updated_at
  BEFORE UPDATE ON pets
  FOR EACH ROW EXECUTE FUNCTION update_updated_at();
```

**Step 2: 在 Supabase Dashboard 執行**

1. 登入 Supabase Dashboard → SQL Editor
2. 貼上 `001_init.sql` 內容並執行
3. 確認 `pets` 和 `nose_embeddings` 表已建立
4. 確認 `match_nose_embedding` function 已建立

**Step 3: Commit**

```bash
git add supabase/migrations/001_init.sql
git commit -m "feat: add Supabase pgvector schema migration"
```

---

### Task 10: Supabase Service

**Files:**
- Create: `api/services/supabase_service.py`
- Create: `tests/api/test_supabase_service.py`

**Step 1: 寫失敗測試（使用 mock）**

```python
# tests/api/test_supabase_service.py
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from api.services.supabase_service import SupabaseService


@pytest.fixture
def mock_supabase_service():
    """使用 mock 避免真實 Supabase 連線"""
    with patch("api.services.supabase_service.create_client") as mock_client:
        mock_client.return_value = MagicMock()
        service = SupabaseService(url="https://mock.supabase.co", key="mock-key")
        service.client = MagicMock()
        yield service


def test_register_pet_returns_pet_id(mock_supabase_service):
    mock_supabase_service.client.table.return_value.insert.return_value.execute.return_value.data = [
        {"id": "test-uuid-123"}
    ]
    pet_id = mock_supabase_service.register_pet(
        name="小白",
        species="dog",
        owner_id="owner-uuid",
    )
    assert pet_id == "test-uuid-123"


def test_save_embedding_calls_insert(mock_supabase_service):
    mock_supabase_service.client.table.return_value.insert.return_value.execute.return_value.data = [
        {"id": "emb-uuid"}
    ]
    embedding = np.random.rand(256).astype(np.float32)
    emb_id = mock_supabase_service.save_embedding(
        pet_id="pet-uuid",
        embedding=embedding,
        image_url="pet-nose-images/raw/dog_001/001.jpg",
    )
    assert emb_id == "emb-uuid"


def test_find_matching_pet_returns_result(mock_supabase_service):
    mock_supabase_service.client.rpc.return_value.execute.return_value.data = [
        {"pet_id": "matched-uuid", "embedding_id": "emb-uuid", "similarity": 0.92}
    ]
    embedding = np.random.rand(256).astype(np.float32)
    result = mock_supabase_service.find_matching_pet(embedding, threshold=0.85)
    assert result is not None
    assert result["pet_id"] == "matched-uuid"
    assert result["similarity"] == 0.92


def test_find_matching_pet_no_match(mock_supabase_service):
    mock_supabase_service.client.rpc.return_value.execute.return_value.data = []
    embedding = np.random.rand(256).astype(np.float32)
    result = mock_supabase_service.find_matching_pet(embedding)
    assert result is None
```

**Step 2: 確認測試失敗**

```bash
pytest tests/api/test_supabase_service.py -v
```

**Step 3: 實作 supabase_service.py**

```python
# api/services/supabase_service.py
import numpy as np
from typing import Optional, Dict, Any
from supabase import create_client, Client


class SupabaseService:
    """
    封裝所有 Supabase 操作：寵物 CRUD、embedding 儲存、相似度搜尋。
    """

    def __init__(self, url: str, key: str):
        self.client: Client = create_client(url, key)

    def register_pet(
        self,
        name: str,
        species: str,
        owner_id: Optional[str] = None,
        breed: Optional[str] = None,
    ) -> str:
        """在 pets 表建立新寵物記錄，返回 pet_id"""
        payload = {"name": name, "species": species}
        if owner_id:
            payload["owner_id"] = owner_id
        if breed:
            payload["breed"] = breed

        result = self.client.table("pets").insert(payload).execute()
        return result.data[0]["id"]

    def save_embedding(
        self,
        pet_id: str,
        embedding: np.ndarray,
        image_url: Optional[str] = None,
    ) -> str:
        """儲存鼻紋 embedding 至 nose_embeddings 表，返回 embedding_id"""
        payload = {
            "pet_id": pet_id,
            "embedding": embedding.tolist(),  # pgvector 接受 Python list
        }
        if image_url:
            payload["image_url"] = image_url

        result = self.client.table("nose_embeddings").insert(payload).execute()
        return result.data[0]["id"]

    def find_matching_pet(
        self,
        embedding: np.ndarray,
        threshold: float = 0.85,
        limit: int = 1,
    ) -> Optional[Dict[str, Any]]:
        """
        用 pgvector cosine similarity 搜尋最相似的寵物。

        Returns:
            最相似的結果 {"pet_id", "embedding_id", "similarity"}，
            或 None（無符合閾值的結果）
        """
        result = self.client.rpc(
            "match_nose_embedding",
            {
                "query_embedding": embedding.tolist(),
                "match_threshold": threshold,
                "match_count": limit,
            },
        ).execute()

        if not result.data:
            return None
        return result.data[0]

    def get_pet(self, pet_id: str) -> Optional[Dict[str, Any]]:
        """查詢寵物資料"""
        result = (
            self.client.table("pets")
            .select("*, nose_embeddings(count)")
            .eq("id", pet_id)
            .single()
            .execute()
        )
        return result.data

    def delete_pet(self, pet_id: str):
        """刪除寵物及關聯的所有 embedding（cascade）"""
        self.client.table("pets").delete().eq("id", pet_id).execute()

    def upload_image(
        self,
        bucket: str,
        path: str,
        image_bytes: bytes,
        content_type: str = "image/jpeg",
    ) -> str:
        """上傳圖片至 Supabase Storage，返回儲存路徑"""
        self.client.storage.from_(bucket).upload(
            path=path,
            file=image_bytes,
            file_options={"content-type": content_type},
        )
        return path
```

**Step 4: 確認測試通過**

```bash
pytest tests/api/test_supabase_service.py -v
```
Expected: 4 passed

**Step 5: Commit**

```bash
git add api/services/supabase_service.py tests/api/test_supabase_service.py
git commit -m "feat: add Supabase service for pets CRUD and embedding search"
```

---

### Task 11: Embedder + Matcher

**Files:**
- Create: `src/inference/embedder.py`
- Create: `src/inference/matcher.py`
- Create: `tests/inference/test_embedder.py`

**Step 1: 寫失敗測試**

```python
# tests/inference/test_embedder.py
import numpy as np
from src.inference.embedder import Embedder
from src.inference.matcher import compute_similarity, is_same_pet

def test_embedder_output_shape():
    embedder = Embedder(embedding_dim=256)
    dummy_img = np.random.rand(224, 224, 3).astype(np.float32)
    embedding = embedder.embed(dummy_img)
    assert embedding.shape == (256,)

def test_embedder_output_normalized():
    embedder = Embedder(embedding_dim=256)
    dummy_img = np.random.rand(224, 224, 3).astype(np.float32)
    embedding = embedder.embed(dummy_img)
    norm = np.linalg.norm(embedding)
    assert abs(norm - 1.0) < 1e-5

def test_compute_similarity_same_vector():
    v = np.random.rand(256).astype(np.float32)
    v = v / np.linalg.norm(v)
    assert abs(compute_similarity(v, v) - 1.0) < 1e-5

def test_is_same_pet_threshold():
    v = np.ones(256, dtype=np.float32) / np.sqrt(256)
    assert is_same_pet(v, v, threshold=0.8) == True
    assert is_same_pet(v, -v, threshold=0.8) == False
```

**Step 2: 確認測試失敗**

```bash
pytest tests/inference/test_embedder.py -v
```

**Step 3: 實作 embedder.py 和 matcher.py**

```python
# src/inference/embedder.py
import numpy as np
import tensorflow as tf
from src.models.backbone import build_backbone
from src.models.embedding_head import build_embedding_head


class Embedder:
    """從圖片生成 L2-normalized embedding 向量"""

    def __init__(self, weights_path: str = None, embedding_dim: int = 256):
        self.backbone = build_backbone(trainable=False)
        self.head = build_embedding_head(input_dim=1280, embedding_dim=embedding_dim)
        if weights_path:
            self.head.load_weights(weights_path)

    def embed(self, img: np.ndarray) -> np.ndarray:
        """
        將單張預處理圖片（224x224x3 float32）轉換為 embedding。
        Returns: shape (embedding_dim,), L2 normalized
        """
        img_batch = np.expand_dims(img, axis=0)
        features = self.backbone(img_batch, training=False)
        embedding = self.head(features, training=False)
        return embedding.numpy()[0]
```

```python
# src/inference/matcher.py
import numpy as np


def compute_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """
    計算兩個 L2-normalized embedding 的 cosine similarity。
    因為向量已 normalize，cosine similarity = dot product。
    Returns: float in [-1.0, 1.0]
    """
    return float(np.dot(emb1, emb2))


def is_same_pet(
    emb1: np.ndarray,
    emb2: np.ndarray,
    threshold: float = 0.85,
) -> bool:
    """相似度超過閾值則視為同一隻寵物"""
    return compute_similarity(emb1, emb2) >= threshold
```

**Step 4: 確認測試通過**

```bash
pytest tests/inference/test_embedder.py -v
```
Expected: 4 passed

**Step 5: Commit**

```bash
git add src/inference/embedder.py src/inference/matcher.py tests/inference/test_embedder.py
git commit -m "feat: add embedder and similarity matcher"
```

---

### Task 12: FastAPI 主應用程式

**Files:**
- Create: `api/main.py`
- Create: `api/services/embedding_service.py`
- Create: `tests/api/test_main.py`

**Step 1: 寫失敗測試**

```python
# tests/api/test_main.py
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock


def test_health_check():
    with patch("api.main.settings") as mock_settings:
        mock_settings.SUPABASE_URL = "https://mock.supabase.co"
        mock_settings.SUPABASE_KEY = "mock-key"
        from api.main import app
        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
```

**Step 2: 確認測試失敗**

```bash
pytest tests/api/test_main.py -v
```

**Step 3: 實作 embedding_service.py**

```python
# api/services/embedding_service.py
import numpy as np
from src.data.preprocessor import preprocess_image
from src.inference.embedder import Embedder


class EmbeddingService:
    """
    封裝圖片 → embedding 的 ML pipeline。
    供 FastAPI 路由使用。
    """

    def __init__(self, weights_path: str = None, embedding_dim: int = 256):
        self.embedder = Embedder(weights_path=weights_path, embedding_dim=embedding_dim)

    def image_bytes_to_embedding(self, image_bytes: bytes) -> np.ndarray:
        """
        將圖片 bytes（從 HTTP request 接收）轉換為 embedding 向量。

        Args:
            image_bytes: 圖片的原始 bytes
        Returns:
            shape (256,), L2 normalized float32
        """
        import cv2
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("Cannot decode image bytes")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        preprocessed = preprocess_image(img_rgb)
        return self.embedder.embed(preprocessed)
```

**Step 4: 實作 main.py**

```python
# api/main.py
from fastapi import FastAPI
from pydantic_settings import BaseSettings
from api.routers import pets


class Settings(BaseSettings):
    SUPABASE_URL: str = ""
    SUPABASE_KEY: str = ""
    SUPABASE_SERVICE_KEY: str = ""
    MODEL_WEIGHTS_PATH: str = ""
    EMBEDDING_DIM: int = 256
    SIMILARITY_THRESHOLD: float = 0.85

    class Config:
        env_file = ".env"


settings = Settings()

app = FastAPI(
    title="寵物鼻紋辨認 API",
    description="透過鼻紋辨認寵物身份，支援狗 (Phase 1) 和貓 (Phase 2)",
    version="1.0.0",
)

app.include_router(pets.router, prefix="/api/v1")


@app.get("/health")
def health_check():
    return {"status": "ok", "version": "1.0.0"}
```

**Step 5: 確認測試通過**

```bash
pytest tests/api/test_main.py -v
```
Expected: 1 passed

**Step 6: Commit**

```bash
git add api/main.py api/services/embedding_service.py tests/api/test_main.py
git commit -m "feat: add FastAPI app with health endpoint and embedding service"
```

---

### Task 13: FastAPI /pets 路由

**Files:**
- Create: `api/schemas/pet.py`
- Create: `api/routers/pets.py`
- Create: `tests/api/test_pets_router.py`

**Step 1: 寫失敗測試**

```python
# tests/api/test_pets_router.py
import pytest
import io
import numpy as np
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock


@pytest.fixture
def client():
    with patch("api.routers.pets.SupabaseService") as mock_sb, \
         patch("api.routers.pets.EmbeddingService") as mock_emb:

        mock_emb_instance = MagicMock()
        mock_emb_instance.image_bytes_to_embedding.return_value = (
            np.ones(256, dtype=np.float32) / np.sqrt(256)
        )
        mock_emb.return_value = mock_emb_instance

        mock_sb_instance = MagicMock()
        mock_sb_instance.register_pet.return_value = "pet-uuid-123"
        mock_sb_instance.save_embedding.return_value = "emb-uuid-456"
        mock_sb_instance.upload_image.return_value = "raw/pet-uuid-123/001.jpg"
        mock_sb_instance.find_matching_pet.return_value = {
            "pet_id": "pet-uuid-123",
            "embedding_id": "emb-uuid-456",
            "similarity": 0.94,
        }
        mock_sb_instance.get_pet.return_value = {
            "id": "pet-uuid-123",
            "name": "小白",
            "species": "dog",
        }
        mock_sb.return_value = mock_sb_instance

        from api.main import app
        yield TestClient(app)


def test_register_pet(client):
    image_bytes = b"fake-image-data"
    response = client.post(
        "/api/v1/pets/register",
        data={"name": "小白", "species": "dog"},
        files={"image": ("test.jpg", io.BytesIO(image_bytes), "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert "pet_id" in data
    assert data["name"] == "小白"


def test_verify_pet(client):
    image_bytes = b"fake-image-data"
    response = client.post(
        "/api/v1/pets/verify",
        files={"image": ("test.jpg", io.BytesIO(image_bytes), "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert "matched" in data
    assert "similarity" in data


def test_get_pet(client):
    response = client.get("/api/v1/pets/pet-uuid-123")
    assert response.status_code == 200
```

**Step 2: 確認測試失敗**

```bash
pytest tests/api/test_pets_router.py -v
```

**Step 3: 實作 schemas/pet.py**

```python
# api/schemas/pet.py
from pydantic import BaseModel
from typing import Optional


class PetRegisterResponse(BaseModel):
    pet_id: str
    name: str
    species: str
    embedding_id: str
    image_url: Optional[str] = None


class PetVerifyResponse(BaseModel):
    matched: bool
    pet_id: Optional[str] = None
    pet_name: Optional[str] = None
    similarity: Optional[float] = None
    threshold: float


class PetInfoResponse(BaseModel):
    id: str
    name: str
    species: str
    breed: Optional[str] = None
```

**Step 4: 實作 routers/pets.py**

```python
# api/routers/pets.py
import uuid
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Optional

from api.schemas.pet import PetRegisterResponse, PetVerifyResponse, PetInfoResponse
from api.services.supabase_service import SupabaseService
from api.services.embedding_service import EmbeddingService
from api.main import settings

router = APIRouter(prefix="/pets", tags=["pets"])

_supabase = None
_embedder = None


def get_supabase() -> SupabaseService:
    global _supabase
    if _supabase is None:
        _supabase = SupabaseService(url=settings.SUPABASE_URL, key=settings.SUPABASE_KEY)
    return _supabase


def get_embedder() -> EmbeddingService:
    global _embedder
    if _embedder is None:
        _embedder = EmbeddingService(
            weights_path=settings.MODEL_WEIGHTS_PATH or None,
            embedding_dim=settings.EMBEDDING_DIM,
        )
    return _embedder


@router.post("/register", response_model=PetRegisterResponse)
async def register_pet(
    name: str = Form(...),
    species: str = Form(...),
    breed: Optional[str] = Form(None),
    owner_id: Optional[str] = Form(None),
    image: UploadFile = File(...),
):
    """登記新寵物並儲存第一張鼻紋 embedding"""
    image_bytes = await image.read()

    embedding = get_embedder().image_bytes_to_embedding(image_bytes)
    pet_id = get_supabase().register_pet(name=name, species=species, owner_id=owner_id, breed=breed)

    image_path = f"raw/{pet_id}/{uuid.uuid4()}.jpg"
    image_url = get_supabase().upload_image("pet-nose-images", image_path, image_bytes)

    emb_id = get_supabase().save_embedding(pet_id=pet_id, embedding=embedding, image_url=image_url)

    return PetRegisterResponse(
        pet_id=pet_id,
        name=name,
        species=species,
        embedding_id=emb_id,
        image_url=image_url,
    )


@router.post("/verify", response_model=PetVerifyResponse)
async def verify_pet(image: UploadFile = File(...)):
    """上傳鼻紋照片，判斷是否為已登記寵物"""
    image_bytes = await image.read()
    embedding = get_embedder().image_bytes_to_embedding(image_bytes)

    match = get_supabase().find_matching_pet(
        embedding, threshold=settings.SIMILARITY_THRESHOLD
    )

    if match is None:
        return PetVerifyResponse(matched=False, threshold=settings.SIMILARITY_THRESHOLD)

    pet = get_supabase().get_pet(match["pet_id"])
    return PetVerifyResponse(
        matched=True,
        pet_id=match["pet_id"],
        pet_name=pet["name"] if pet else None,
        similarity=match["similarity"],
        threshold=settings.SIMILARITY_THRESHOLD,
    )


@router.post("/{pet_id}/embeddings")
async def add_embedding(pet_id: str, image: UploadFile = File(...)):
    """為現有寵物新增鼻紋樣本（提高辨識準確度）"""
    image_bytes = await image.read()
    embedding = get_embedder().image_bytes_to_embedding(image_bytes)

    image_path = f"raw/{pet_id}/{uuid.uuid4()}.jpg"
    image_url = get_supabase().upload_image("pet-nose-images", image_path, image_bytes)
    emb_id = get_supabase().save_embedding(pet_id=pet_id, embedding=embedding, image_url=image_url)

    return {"embedding_id": emb_id, "pet_id": pet_id}


@router.get("/{pet_id}", response_model=PetInfoResponse)
async def get_pet(pet_id: str):
    """查詢寵物資料"""
    pet = get_supabase().get_pet(pet_id)
    if pet is None:
        raise HTTPException(status_code=404, detail="Pet not found")
    return PetInfoResponse(**pet)


@router.delete("/{pet_id}")
async def delete_pet(pet_id: str):
    """刪除寵物及所有 embedding"""
    get_supabase().delete_pet(pet_id)
    return {"deleted": True, "pet_id": pet_id}
```

**Step 5: 確認測試通過**

```bash
pytest tests/api/test_pets_router.py -v
```
Expected: 3 passed

**Step 6: Commit**

```bash
git add api/schemas/pet.py api/routers/pets.py tests/api/test_pets_router.py
git commit -m "feat: add /pets REST API endpoints (register, verify, add embedding, get, delete)"
```

---

### Task 14: Trainer

**Files:**
- Create: `src/training/trainer.py`
- Create: `tests/models/test_trainer.py`

**Step 1: 寫失敗測試**

```python
# tests/models/test_trainer.py
import numpy as np
import pytest
from src.training.trainer import SiameseTrainer

def _make_dummy_pairs(n=8):
    imgs_a = np.random.rand(n, 224, 224, 3).astype(np.float32)
    imgs_b = np.random.rand(n, 224, 224, 3).astype(np.float32)
    labels = np.array([i % 2 for i in range(n)], dtype=np.float32)
    return imgs_a, imgs_b, labels

def test_trainer_initializes():
    trainer = SiameseTrainer(embedding_dim=64)
    assert trainer.model is not None

def test_trainer_runs_one_epoch(tmp_path):
    trainer = SiameseTrainer(embedding_dim=64)
    imgs_a, imgs_b, labels = _make_dummy_pairs()
    history = trainer.train(
        imgs_a=imgs_a, imgs_b=imgs_b, labels=labels,
        epochs=1, batch_size=4, save_dir=str(tmp_path),
    )
    assert "loss" in history

def test_trainer_saves_model(tmp_path):
    trainer = SiameseTrainer(embedding_dim=64)
    imgs_a, imgs_b, labels = _make_dummy_pairs()
    trainer.train(
        imgs_a=imgs_a, imgs_b=imgs_b, labels=labels,
        epochs=1, batch_size=4, save_dir=str(tmp_path),
    )
    assert len(list(tmp_path.glob("*.h5"))) > 0
```

**Step 2: 確認測試失敗**

```bash
pytest tests/models/test_trainer.py -v
```

**Step 3: 實作 trainer.py**

```python
# src/training/trainer.py
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Dict
from src.models.siamese import build_siamese_model, contrastive_loss


class SiameseTrainer:
    """Siamese Network 訓練器，支援 Early Stopping 和 checkpoint 儲存"""

    def __init__(self, embedding_dim: int = 256, learning_rate: float = 1e-3):
        self.model = build_siamese_model(embedding_dim=embedding_dim)
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=contrastive_loss,
        )

    def train(
        self,
        imgs_a: np.ndarray,
        imgs_b: np.ndarray,
        labels: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        validation_split: float = 0.2,
        save_dir: str = "models/",
    ) -> Dict:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        use_val = len(imgs_a) >= 10 and validation_split > 0
        callbacks = []

        if use_val:
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=10, restore_best_weights=True
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=str(save_path / "best_model.h5"),
                    monitor="val_loss",
                    save_best_only=True,
                ),
            ]

        history = self.model.fit(
            x=[imgs_a, imgs_b],
            y=labels,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split if use_val else 0.0,
            callbacks=callbacks,
            verbose=1,
        )

        self.model.save(str(save_path / "best_model.h5"))
        return history.history
```

**Step 4: 確認測試通過**

```bash
pytest tests/models/test_trainer.py -v
```
Expected: 3 passed

**Step 5: Commit**

```bash
git add src/training/trainer.py tests/models/test_trainer.py
git commit -m "feat: add Siamese trainer with early stopping"
```

---

### Task 15: 評估指標

**Files:**
- Create: `src/evaluation/metrics.py`
- Create: `tests/evaluation/test_metrics.py`

**Step 1: 寫失敗測試**

```python
# tests/evaluation/test_metrics.py
import numpy as np
from src.evaluation.metrics import compute_eer, compute_roc

def test_compute_eer_perfect_separation():
    similarities = np.array([0.9, 0.95, 0.1, 0.05])
    labels = np.array([1, 1, 0, 0])  # 1=same, 0=different
    eer, threshold = compute_eer(similarities, labels)
    assert eer < 0.1

def test_compute_eer_returns_tuple():
    similarities = np.random.rand(100)
    labels = (np.random.rand(100) > 0.5).astype(int)
    result = compute_eer(similarities, labels)
    assert len(result) == 2

def test_compute_roc_returns_arrays():
    similarities = np.random.rand(50)
    labels = (np.random.rand(50) > 0.5).astype(int)
    fpr, tpr, thresholds = compute_roc(similarities, labels)
    assert len(fpr) == len(tpr)
    assert np.all(fpr >= 0) and np.all(fpr <= 1)
```

**Step 2: 確認測試失敗**

```bash
pytest tests/evaluation/test_metrics.py -v
```

**Step 3: 實作 metrics.py**

```python
# src/evaluation/metrics.py
import numpy as np
from sklearn.metrics import roc_curve
from typing import Tuple


def compute_roc(
    similarities: np.ndarray,
    labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """計算 ROC 曲線。labels: 1=same pet, 0=different"""
    return roc_curve(labels, similarities)


def compute_eer(
    similarities: np.ndarray,
    labels: np.ndarray,
) -> Tuple[float, float]:
    """
    計算 Equal Error Rate (EER) 和對應閾值。
    EER 是 FAR = FRR 時的錯誤率，越低越好。
    """
    fpr, tpr, thresholds = compute_roc(similarities, labels)
    fnr = 1.0 - tpr
    eer_idx = np.argmin(np.abs(fpr - fnr))
    eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)
    return eer, float(thresholds[eer_idx])
```

**Step 4: 確認測試通過**

```bash
pytest tests/evaluation/test_metrics.py -v
```
Expected: 3 passed

**Step 5: Commit**

```bash
git add src/evaluation/metrics.py tests/evaluation/test_metrics.py
git commit -m "feat: add EER and ROC evaluation metrics"
```

---

### Task 16: 整合測試

**Files:**
- Create: `tests/test_integration.py`

**Step 1: 實作整合測試**

```python
# tests/test_integration.py
"""
端對端整合測試：ML pipeline + API layer（全部 mock Supabase）
"""
import numpy as np
import pytest
import io
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


@pytest.fixture
def full_client():
    """完整 mock：Supabase + ML model（避免下載 weights）"""
    with patch("api.routers.pets.get_supabase") as mock_get_sb, \
         patch("api.routers.pets.get_embedder") as mock_get_emb:

        mock_emb = MagicMock()
        mock_emb.image_bytes_to_embedding.return_value = (
            np.ones(256, dtype=np.float32) / np.sqrt(256)
        )
        mock_get_emb.return_value = mock_emb

        mock_sb = MagicMock()
        mock_sb.register_pet.return_value = "integrated-pet-uuid"
        mock_sb.save_embedding.return_value = "integrated-emb-uuid"
        mock_sb.upload_image.return_value = "raw/integrated-pet-uuid/001.jpg"
        mock_sb.find_matching_pet.return_value = {
            "pet_id": "integrated-pet-uuid",
            "embedding_id": "integrated-emb-uuid",
            "similarity": 0.96,
        }
        mock_sb.get_pet.return_value = {
            "id": "integrated-pet-uuid",
            "name": "小白",
            "species": "dog",
        }
        mock_get_sb.return_value = mock_sb

        from api.main import app
        yield TestClient(app)


def test_register_then_verify(full_client):
    """完整流程：登記 → 驗證"""
    # 1. 登記新寵物
    image_bytes = b"fake-nose-image"
    register_response = full_client.post(
        "/api/v1/pets/register",
        data={"name": "小白", "species": "dog"},
        files={"image": ("nose.jpg", io.BytesIO(image_bytes), "image/jpeg")},
    )
    assert register_response.status_code == 200
    pet_id = register_response.json()["pet_id"]
    assert pet_id == "integrated-pet-uuid"

    # 2. 驗證身份
    verify_response = full_client.post(
        "/api/v1/pets/verify",
        files={"image": ("nose2.jpg", io.BytesIO(image_bytes), "image/jpeg")},
    )
    assert verify_response.status_code == 200
    verify_data = verify_response.json()
    assert verify_data["matched"] == True
    assert verify_data["similarity"] > 0.85
    assert verify_data["pet_name"] == "小白"


def test_health_endpoint(full_client):
    response = full_client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_add_embedding(full_client):
    """為已登記寵物新增鼻紋樣本"""
    image_bytes = b"another-nose-image"
    response = full_client.post(
        "/api/v1/pets/integrated-pet-uuid/embeddings",
        files={"image": ("nose3.jpg", io.BytesIO(image_bytes), "image/jpeg")},
    )
    assert response.status_code == 200
    assert "embedding_id" in response.json()
```

**Step 2: 執行整合測試**

```bash
pytest tests/test_integration.py -v
```
Expected: 3 passed

**Step 3: 執行全部測試**

```bash
pytest tests/ -v --tb=short
```
Expected: 全部通過

**Step 4: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add end-to-end integration tests for register/verify flow"
```

---

### Task 17: Jupyter Notebook

**Files:**
- Create: `notebooks/01_data_exploration.ipynb`

**Step 1: 建立 Notebook cells**

**Cell 1 — 匯入**
```python
import sys
sys.path.append('..')
import numpy as np
import matplotlib.pyplot as plt
from src.data.collector import scan_dataset
from src.data.preprocessor import load_and_preprocess
from src.data.pair_generator import generate_pairs

DATA_DIR = "../data/raw/dogs"
```

**Cell 2 — 掃描資料集**
```python
dataset = scan_dataset(DATA_DIR)
print(f"共找到 {len(dataset)} 隻狗")
for pet_id, images in list(dataset.items())[:5]:
    print(f"  {pet_id}: {len(images)} 張照片")
```

**Cell 3 — 顯示範例圖片**
```python
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for i, (pet_id, images) in enumerate(list(dataset.items())[:2]):
    for j, img_path in enumerate(images[:4]):
        img = load_and_preprocess(img_path)
        axes[i][j].imshow(img)
        axes[i][j].set_title(f"{pet_id}")
        axes[i][j].axis('off')
plt.tight_layout()
plt.show()
```

**Cell 4 — Pair 統計**
```python
pairs = generate_pairs(dataset, negative_ratio=3)
pos = sum(1 for _, _, l in pairs if l == 0)
neg = sum(1 for _, _, l in pairs if l == 1)
print(f"Positive pairs: {pos} | Negative pairs: {neg} | Total: {len(pairs)}")
```

**Step 2: Commit**

```bash
git add notebooks/01_data_exploration.ipynb
git commit -m "docs: add data exploration notebook"
```

---

## 完成後：啟動 API Server

```bash
# 建立 .env（複製 .env.example 並填入 Supabase 資訊）
cp .env.example .env

# 啟動 FastAPI
uvicorn api.main:app --reload --port 8000

# 查看自動生成的 API 文件
# http://localhost:8000/docs
```

---

## Phase C 升級備忘

1. 建立 `src/models/arcface.py`
2. 在 `src/training/trainer.py` 加入 `ArcFaceTrainer`
3. backbone + embedding_head 權重直接載入繼續微調
4. Supabase schema 完全不需修改
