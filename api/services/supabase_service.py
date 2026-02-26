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
            "embedding": embedding.tolist(),
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
        Returns: {"pet_id", "embedding_id", "similarity"} 或 None
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
