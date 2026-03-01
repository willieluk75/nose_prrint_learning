import { useState, useCallback } from 'react';
import { toast } from 'react-toastify';
import { registerPet } from '../services/api';

/**
 * useUpload - Hook for managing pet photo uploads
 */
export function useUpload() {
  const [uploads, setUploads] = useState([]);
  const [isUploading, setIsUploading] = useState(false);

  /**
   * Upload a single pet
   */
  const uploadPet = useCallback(async (pet) => {
    const uploadId = crypto.randomUUID();

    setUploads((prev) => [
      ...prev,
      {
        id: uploadId,
        petName: pet.name,
        species: pet.species,
        status: 'pending',
      },
    ]);

    try {
      setUploads((prev) =>
        prev.map((u) => (u.id === uploadId ? { ...u, status: 'uploading' } : u))
      );

      const formData = new FormData();
      formData.append('name', pet.name);
      formData.append('species', pet.species);
      if (pet.breed) formData.append('breed', pet.breed);
      if (pet.notes) formData.append('notes', pet.notes);
      formData.append('image', pet.image);

      const response = await registerPet(formData);

      setUploads((prev) =>
        prev.map((u) => (u.id === uploadId ? { ...u, status: 'success', response } : u))
      );

      return { success: true, response };
    } catch (error) {
      const errorMessage = error.message || '上傳失敗';
      setUploads((prev) =>
        prev.map((u) => (u.id === uploadId ? { ...u, status: 'error', error: errorMessage } : u))
      );
      return { success: false, error: errorMessage };
    }
  }, []);

  /**
   * Upload multiple pets
   */
  const uploadPets = useCallback(async (pets, options = {}) => {
    const { sequential = false, onProgress } = options;

    setIsUploading(true);
    setUploads([]);

    const results = [];

    if (sequential) {
      // Upload one by one
      for (const pet of pets) {
        const result = await uploadPet(pet);
        results.push(result);
        onProgress?.(results);
      }
    } else {
      // Upload all at once
      const promises = pets.map((pet) => uploadPet(pet));
      results.push(...(await Promise.all(promises)));
      onProgress?.(results);
    }

    setIsUploading(false);

    const failedCount = results.filter((r) => !r.success).length;
    if (failedCount > 0) {
      toast.error(`${failedCount} 個寵物上傳失敗`);
    } else {
      toast.success('所有寵物上傳成功！');
    }

    return results;
  }, [uploadPet]);

  /**
   * Retry failed uploads
   */
  const retryFailed = useCallback(() => {
    const failedUploads = uploads.filter((u) => u.status === 'error');

    if (failedUploads.length === 0) {
      return;
    }

    setUploads([]);
    // Note: This would need the original pet data to retry
    toast.warning('請重新上傳失敗的項目');
  }, [uploads]);

  /**
   * Clear uploads
   */
  const clearUploads = useCallback(() => {
    setUploads([]);
  }, []);

  /**
   * Remove specific upload from list
   */
  const removeUpload = useCallback((uploadId) => {
    setUploads((prev) => prev.filter((u) => u.id !== uploadId));
  }, []);

  return {
    uploads,
    isUploading,
    uploadPet,
    uploadPets,
    retryFailed,
    clearUploads,
    removeUpload,
  };
}
