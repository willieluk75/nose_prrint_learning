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
