import uuid
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Optional

from api.schemas.pet import PetRegisterResponse, PetVerifyResponse, PetInfoResponse
from api.services.supabase_service import SupabaseService
from api.services.embedding_service import EmbeddingService
from api.config import settings

router = APIRouter(prefix="/pets", tags=["pets"])


def get_supabase() -> SupabaseService:
    return SupabaseService(url=settings.SUPABASE_URL, key=settings.SUPABASE_KEY)


def get_embedder() -> EmbeddingService:
    return EmbeddingService(
        weights_path=settings.MODEL_WEIGHTS_PATH or None,
        embedding_dim=settings.EMBEDDING_DIM,
    )


@router.post("/register", response_model=PetRegisterResponse)
async def register_pet(
    name: str = Form(...),
    species: str = Form(...),
    breed: Optional[str] = Form(None),
    owner_id: Optional[str] = Form(None),
    image: UploadFile = File(...),
):
    """Register a new pet and store the first nose print embedding."""
    image_bytes = await image.read()

    embedding = get_embedder().image_bytes_to_embedding(image_bytes)
    supabase = get_supabase()
    pet_id = supabase.register_pet(name=name, species=species, owner_id=owner_id, breed=breed)

    image_path = f"raw/{pet_id}/{uuid.uuid4()}.jpg"
    image_url = supabase.upload_image("pet-nose-images", image_path, image_bytes)

    emb_id = supabase.save_embedding(pet_id=pet_id, embedding=embedding, image_url=image_url)

    return PetRegisterResponse(
        pet_id=pet_id,
        name=name,
        species=species,
        embedding_id=emb_id,
        image_url=image_url,
    )


@router.post("/verify", response_model=PetVerifyResponse)
async def verify_pet(image: UploadFile = File(...)):
    """Upload a nose print image and determine if it matches a registered pet."""
    image_bytes = await image.read()
    embedding = get_embedder().image_bytes_to_embedding(image_bytes)

    supabase = get_supabase()
    match = supabase.find_matching_pet(embedding, threshold=settings.SIMILARITY_THRESHOLD)

    if match is None:
        return PetVerifyResponse(matched=False, threshold=settings.SIMILARITY_THRESHOLD)

    pet = supabase.get_pet(match["pet_id"])
    return PetVerifyResponse(
        matched=True,
        pet_id=match["pet_id"],
        pet_name=pet["name"] if pet else None,
        similarity=match["similarity"],
        threshold=settings.SIMILARITY_THRESHOLD,
    )


@router.post("/{pet_id}/embeddings")
async def add_embedding(pet_id: str, image: UploadFile = File(...)):
    """Add a new nose print sample to an existing pet (improves recognition accuracy)."""
    image_bytes = await image.read()
    embedding = get_embedder().image_bytes_to_embedding(image_bytes)

    supabase = get_supabase()
    image_path = f"raw/{pet_id}/{uuid.uuid4()}.jpg"
    image_url = supabase.upload_image("pet-nose-images", image_path, image_bytes)
    emb_id = supabase.save_embedding(pet_id=pet_id, embedding=embedding, image_url=image_url)

    return {"embedding_id": emb_id, "pet_id": pet_id}


@router.get("/{pet_id}", response_model=PetInfoResponse)
async def get_pet(pet_id: str):
    """Retrieve pet information."""
    pet = get_supabase().get_pet(pet_id)
    if pet is None:
        raise HTTPException(status_code=404, detail="Pet not found")
    return PetInfoResponse(**pet)


@router.delete("/{pet_id}")
async def delete_pet(pet_id: str):
    """Delete a pet and all associated embeddings."""
    get_supabase().delete_pet(pet_id)
    return {"deleted": True, "pet_id": pet_id}
