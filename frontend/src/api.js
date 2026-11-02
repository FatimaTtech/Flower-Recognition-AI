
/**
 * FloraAI - Hugging Face Gradio API Client
 * Connects to the trained ResNet-18 flower classification model.
 * Includes low-confidence rejection for unsupported images.
 */

import { Client, handle_file } from "@gradio/client";

const SPACE_ID = "FatimaTahseen/flower_recognitionai-api";

const MAX_FILE_SIZE = 10 * 1024 * 1024;

const CONFIDENCE_THRESHOLD = 70;

const VALID_TYPES = [
  "image/jpeg",
  "image/png",
  "image/webp",
];

const FLOWER_CLASSES = [
  "daisy",
  "dandelion",
  "rose",
  "sunflower",
  "tulip",
];

let clientPromise = null;

/**
 * Connect to the public Hugging Face Space.
 */
async function getClient() {
  if (!clientPromise) {
    clientPromise = Client.connect(SPACE_ID).catch((error) => {
      clientPromise = null;

      console.error("Hugging Face connection error:", error);

      throw new Error(
        "Could not connect to the FloraAI model. " +
        "Please check the Hugging Face Space and try again."
      );
    });
  }

  return clientPromise;
}

/**
 * Check backend connectivity.
 */
export async function getBackendHealth() {
  try {
    await getClient();

    return {
      status: "online",
      service: "FloraAI Flower Recognition API",
      model_loaded: true,
      model_name: "ResNet-18",
    };
  } catch (error) {
    console.error("Backend health check failed:", error);
    throw error;
  }
}

/**
 * Return model information.
 */
export async function getModelInfo() {
  return {
    model_name: "ResNet-18",
    architecture: "ResNet-18 with custom classification head",
    classes: FLOWER_CLASSES,
    accuracy: 87.21,
    macro_f1: 87.20,
    macro_precision: 87.25,
    macro_recall: 87.20,
    status: "loaded",
  };
}

/**
 * Validate the uploaded image.
 */
function validateImage(imageFile) {
  if (!imageFile) {
    throw new Error("Please select an image first.");
  }

  const validType =
    VALID_TYPES.includes(imageFile.type) ||
    /\.(jpe?g|png|webp)$/i.test(imageFile.name);

  if (!validType) {
    throw new Error(
      "Unsupported image format. Upload JPG, PNG, or WEBP."
    );
  }

  if (imageFile.size > MAX_FILE_SIZE) {
    throw new Error(
      "Image exceeds 10 MB. Please choose a smaller image."
    );
  }

  if (imageFile.size === 0) {
    throw new Error("The selected image is empty.");
  }
}

/**
 * Convert confidence values to percentages.
 */
function toPercentage(value) {
  const number = Number(value);

  if (!Number.isFinite(number)) {
    return 0;
  }

  const percentage =
    number <= 1 ? number * 100 : number;

  return Number(
    Math.max(0, Math.min(100, percentage)).toFixed(2)
  );
}

/**
 * Predict the flower using the Hugging Face model.
 */
export async function predictFlowerImage(imageFile) {
  validateImage(imageFile);

  try {
    const client = await getClient();

    const result = await client.predict("/predict", {
      image: handle_file(imageFile),
    });

    if (!result || !Array.isArray(result.data)) {
      throw new Error(
        "The model returned an invalid response."
      );
    }

    const [
      prediction,
      rawConfidence,
      labelResult,
    ] = result.data;

    if (!prediction) {
      throw new Error(
        "The model could not identify a flower."
      );
    }

    // Convert predicted confidence into a percentage.
    const confidence = toPercentage(rawConfidence);

    // Reject predictions below the confidence threshold.
    if (confidence < CONFIDENCE_THRESHOLD) {
      throw new Error(
        "This image does not appear to contain a supported flower. " +
        "Please upload a clear image of a daisy, dandelion, rose, " +
        "sunflower, or tulip."
      );
    }

    // Extract confidence scores for all flower classes.
    const confidences = Array.isArray(
      labelResult?.confidences
    )
      ? labelResult.confidences
      : [];

    // Get the three highest predictions.
    const topThree = confidences
      .slice()
      .sort(
        (a, b) =>
          Number(b.confidence) - Number(a.confidence)
      )
      .slice(0, 3)
      .map((item) => ({
        class_name: String(item.label),
        confidence: toPercentage(item.confidence),
      }));

    // Return results in the format expected by App.jsx.
    return {
      predicted_class: String(prediction),
      confidence_percentage: confidence,
      top_three: topThree,
    };

  } catch (error) {
    console.error("FloraAI prediction error:", error);

    throw new Error(
      error.message ||
      "Flower classification failed. Please try again."
    );
  }
}
