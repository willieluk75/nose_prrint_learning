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
