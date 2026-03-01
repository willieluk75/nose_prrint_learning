// API Configuration
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api/v1';

/**
 * Upload a pet photo and register the pet
 * @param {Object} formData - FormData with name, species, breed, image
 * @returns {Promise<Object>} - Registration response
 */
export async function registerPet(formData) {
  const url = `${API_BASE_URL}/pets/register`;

  try {
    const response = await fetch(url, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
      throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
    }

    return await response.json();
  } catch (error) {
    if (error instanceof TypeError && error.message.includes('fetch')) {
      throw new Error('無法連接到伺服器，請檢查網路連線');
    }
    throw error;
  }
}

/**
 * Health check for the API
 * @returns {Promise<Object>}
 */
export async function healthCheck() {
  const response = await fetch(`${API_BASE_URL.replace('/api/v1', '')}/health`);
  return await response.json();
}
