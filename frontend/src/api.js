/**
 * FloraAI API Client
 * Configurable via VITE_API_BASE_URL environment variable.
 * Production Default: https://fatimatahseen-flower-recognitionai-api.hf.space
 */
export const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL ||
  'https://fatimatahseen-flower-recognitionai-api.hf.space'
).replace(/\/$/, '');

/**
 * Fetch health status of the backend API
 */
export async function getBackendHealth() {
  try {
    const res = await fetch(`${API_BASE_URL}/api/health`);
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.error || `Server responded with status ${res.status}`);
    }
    return await res.json();
  } catch (err) {
    throw new Error(`Unable to connect to FloraAI backend at ${API_BASE_URL}. Ensure the backend server is running.`);
  }
}

/**
 * Fetch model information & evaluation metrics
 */
export async function getModelInfo() {
  try {
    const res = await fetch(`${API_BASE_URL}/api/model_info`);
    if (!res.ok) {
      throw new Error(`Failed to load model info (status ${res.status})`);
    }
    return await res.json();
  } catch (err) {
    throw new Error(err.message || 'Error fetching model information.');
  }
}

/**
 * Predict flower species for an uploaded image file
 * @param {File} imageFile
 */
export async function predictFlowerImage(imageFile) {
  if (!imageFile) {
    throw new Error('Please select an image file to classify.');
  }

  // Client-side validation: format check
  const validTypes = ['image/jpeg', 'image/png', 'image/webp'];
  if (!validTypes.includes(imageFile.type) && !/\.(jpe?g|png|webp)$/i.test(imageFile.name)) {
    throw new Error('Unsupported format. Please upload a JPG, PNG, or WEBP image.');
  }

  // Client-side validation: file size limit (10MB)
  const maxSize = 10 * 1024 * 1024;
  if (imageFile.size > maxSize) {
    throw new Error('File size exceeds the 10 MB limit. Please choose a smaller image.');
  }

  const formData = new FormData();
  formData.append('image', imageFile);

  try {
    const res = await fetch(`${API_BASE_URL}/api/predict`, {
      method: 'POST',
      body: formData,
    });

    const data = await res.json().catch(() => ({}));

    if (!res.ok) {
      throw new Error(data.error || `Classification failed (${res.status})`);
    }

    return data;
  } catch (err) {
    if (err.message.includes('Failed to fetch') || err.name === 'TypeError') {
      throw new Error(`Connection to FloraAI backend failed. Please verify the backend is running at ${API_BASE_URL}.`);
    }
    throw err;
  }
}
