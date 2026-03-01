# 寵物鼻紋辨認系統 — 設計文件

**日期**：2026-02-26
**版本**：2.0（加入 Supabase + FastAPI）
**技術堆疊**：Python + TensorFlow/Keras + FastAPI + Supabase

---

## 專案概述

建立一套透過鼻紋辨認寵物身份的系統，分三個階段推進：

| 階段 | 內容 | 模型策略 |
|------|------|----------|
| Phase 1a | 狗鼻紋辨認資料庫 | Transfer Learning + Siamese (方案B) |
| Phase 1b | 狗鼻紋升級 | 換用 ArcFace Loss (方案C) |
| Phase 2 | 貓鼻孔紋理資料庫 | 同架構，新資料集 |
| Phase 3 | 手機 App | TFLite + REST API (FastAPI) |

**辨識類型**：1:1 驗證（判斷兩張照片是否為同一隻寵物），後期擴展至 1:N 搜尋

---

## 完整系統架構

```
┌──────────────┐    ┌──────────────┐
│  Mobile App  │    │   Web App    │
│  (Flutter)   │    │  (Next.js)   │
└──────┬───────┘    └──────┬───────┘
       │                   │
       └─────────┬─────────┘
                 │  HTTPS REST API
                 ▼
┌────────────────────────────────────┐
│         Backend API (FastAPI)      │
│                                    │
│  POST /api/v1/pets/register        │
│  POST /api/v1/pets/verify          │
│  POST /api/v1/pets/{id}/embeddings │
│  GET  /api/v1/pets/{id}            │
│  DELETE /api/v1/pets/{id}          │
│                                    │
│  ┌──────────────────────────────┐  │
│  │     ML Pipeline              │  │
│  │  MobileNetV2 → Embedding     │  │
│  │  → Siamese / ArcFace         │  │
│  └──────────────────────────────┘  │
└────────────────┬───────────────────┘
                 │
       ┌─────────┴──────────┐
       ▼                    ▼
┌─────────────┐    ┌─────────────────┐
│  Supabase   │    │ Supabase        │
│ PostgreSQL  │    │ Storage         │
│ (pgvector)  │    │ (pet images)    │
└─────────────┘    └─────────────────┘
```

---

## ML 核心架構

```
Input Image
    ↓
Preprocessing (crop, resize 224×224, normalize)
    ↓
Backbone (MobileNetV2, ImageNet pretrained, frozen)
    ↓
Embedding Head (Dense 512→256, BatchNorm, L2 normalize)
    ↓
256-dim Embedding Vector
    ↓
Phase B: Siamese Contrastive Loss (訓練)
Phase C: ArcFace Loss (升級)
    ↓
Supabase pgvector (儲存 / 查詢)
```

**Phase B → C 升級原則**：backbone + embedding_head 完全不變，只替換 loss function 和最後一層。

---

## API 端點設計

| 方法 | 端點 | 說明 |
|------|------|------|
| POST | `/api/v1/pets/register` | 登記新寵物 + 第一張鼻紋照片 |
| POST | `/api/v1/pets/verify` | 上傳照片，判斷是否為已登記寵物 |
| POST | `/api/v1/pets/{pet_id}/embeddings` | 為現有寵物新增鼻紋樣本 |
| GET | `/api/v1/pets/{pet_id}` | 查詢寵物資料 |
| DELETE | `/api/v1/pets/{pet_id}` | 刪除寵物及所有資料 |

### 請求 / 回應範例

**POST /api/v1/pets/register**
```json
// 入：multipart/form-data
{ "name": "小白", "species": "dog", "owner_id": "uuid", "image": <file> }

// 出：
{ "pet_id": "uuid", "name": "小白", "embedding_id": "uuid", "image_url": "..." }
```

**POST /api/v1/pets/verify**
```json
// 入：multipart/form-data
{ "image": <file> }

// 出：
{
  "matched": true,
  "pet_id": "uuid",
  "pet_name": "小白",
  "similarity": 0.94,
  "threshold": 0.85
}
```

---

## Supabase 資料庫設計

### Schema

```sql
-- 啟用 pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- 飼主（使用 Supabase Auth）
-- auth.users 表由 Supabase 自動管理

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
  pet_id       UUID REFERENCES pets(id) ON DELETE CASCADE,
  embedding    vector(256),
  image_url    TEXT,
  created_at   TIMESTAMPTZ DEFAULT NOW()
);

-- 向量索引（加速 1:N 相似度搜尋）
CREATE INDEX ON nose_embeddings
  USING ivfflat (embedding vector_cosine_ops)
  WITH (lists = 100);

-- 相似度搜尋 RPC 函式
CREATE OR REPLACE FUNCTION match_nose_embedding(
  query_embedding vector(256),
  match_threshold FLOAT DEFAULT 0.85,
  match_count     INT DEFAULT 5
)
RETURNS TABLE (
  pet_id      UUID,
  embedding_id UUID,
  similarity  FLOAT
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
```

### Supabase Storage

```
Bucket: pet-nose-images (public: false)
├── raw/
│   └── {pet_id}/
│       ├── {timestamp}_001.jpg
│       └── {timestamp}_002.jpg
└── processed/
    └── {pet_id}/
        └── {timestamp}_001.jpg
```

---

## 資料流

### 登記流程
```
上傳圖片
    ↓
FastAPI 接收
    ↓
preprocessor.py → 224×224 float32
    ↓
embedder.py → 256-dim vector
    ↓
Supabase Storage → 儲存原始圖片
    ↓
Supabase PostgreSQL → 儲存 pets + nose_embeddings
    ↓
返回 pet_id
```

### 驗證流程
```
上傳圖片
    ↓
FastAPI 接收
    ↓
preprocessor.py → 224×224 float32
    ↓
embedder.py → 256-dim vector
    ↓
Supabase RPC: match_nose_embedding()
    ↓
返回 { matched, pet_id, similarity }
```

---

## 專案結構

```
nose_print_learning/
├── api/
│   ├── main.py                     # FastAPI app 入口
│   ├── routers/
│   │   └── pets.py                 # /pets 路由
│   ├── schemas/
│   │   └── pet.py                  # Pydantic 資料模型
│   └── services/
│       ├── embedding_service.py    # ML pipeline 封裝
│       └── supabase_service.py     # Supabase 操作封裝
│
├── src/
│   ├── data/
│   │   ├── collector.py
│   │   ├── preprocessor.py
│   │   ├── augmentor.py
│   │   └── pair_generator.py
│   ├── models/
│   │   ├── backbone.py
│   │   ├── embedding_head.py
│   │   ├── siamese.py
│   │   └── arcface.py              # Phase C 預留
│   ├── training/
│   │   └── trainer.py
│   ├── inference/
│   │   ├── embedder.py
│   │   └── matcher.py
│   └── evaluation/
│       ├── metrics.py
│       └── visualizer.py
│
├── supabase/
│   └── migrations/
│       └── 001_init.sql            # pgvector schema
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_evaluation.ipynb
│
├── tests/
│   ├── data/
│   ├── models/
│   ├── inference/
│   ├── evaluation/
│   ├── api/
│   └── test_integration.py
│
├── models/                         # 訓練好的模型
├── .env.example                    # 環境變數範本
├── requirements.txt
└── README.md
```

---

## 評估指標

| 指標 | 說明 |
|------|------|
| EER | Equal Error Rate，FAR = FRR 時的錯誤率，越低越好 |
| AUC | ROC 曲線下面積，越接近 1 越好 |
| Threshold | 預設 0.85，可依應用場景調整 |

---

## Phase B → C 升級條件

滿足任一條件時升級：
- 資料集超過 100 隻寵物
- EER < 10% 但仍想提升精度
- 需要支援 1:N 搜尋（手機 App 前必須升級）

升級步驟：
1. 保留 backbone + embedding_head 權重
2. 建立 `src/models/arcface.py`
3. 修改 `src/training/trainer.py` 加入 `ArcFaceTrainer`
4. Supabase schema 不需修改

---

## 技術依賴

```
# ML
tensorflow==2.13.0
numpy==1.24.3
opencv-python==4.8.0.76
scikit-learn==1.3.0

# API
fastapi==0.103.0
uvicorn==0.23.2
python-multipart==0.0.6
pydantic==2.3.0
pydantic-settings==2.0.3

# Database
supabase==1.2.0

# Dev
pytest==7.4.0
httpx==0.24.1
jupyter==1.0.0
tqdm==4.66.1
python-dotenv==1.0.0
```
