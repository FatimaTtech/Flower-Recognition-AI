
/**
 * BloomAI - Hugging Face Gradio API Client
 * Connects to the trained ResNet-18 flower classification model.
 */

import { Client, handle_file } from "@gradio/client";

const SPACE_ID = "FatimaTahseen/flower_recognitionai-api";

let clientPromise = null;

/**
 * Connect to the Hugging Face Gradio Space.
 * Reuse the connection instead of reconnecting for every request.
 */
async function getClient() {
  if (!clientPromise) {
    clientPromise = Client.connect(SPACE_ID).catch((error) => {
      clientPromise = null;
      throw error;
    });
  }

  return clientPromise;
}

/**
 * Check whether the Hugging Face backend is reachable.
 */
export async function getBackendHealth() {
  try {
    await getClient();

    return {
      status: "online",
      message: "BloomAI backend connected successfully.",
    };
  } catch (error) {
    throw new Error(
      "Unable to connect to the Hugging Face backend. Please try again later."
    );
  }
}

/**
 * Return the trained model information and evaluation metrics.
 */
export async function getModelInfo() {
  return {
    model_name: "ResNet-18",
    architecture: "ResNet-18 with custom classification head",
    classes: ["daisy", "dandelion", "rose", "sunflower", "tulip"],
    accuracy: 87.21,
    macro_f1: 87.20,
    macro_precision: 87.25,
    macro_recall: 87.20,
    status: "loaded",
  };
}

/**
 * Classify an uploaded flower image.
 * The Gradio API returns three outputs:
 * 1. Predicted flower
 * 2. Confidence score
 * 3. Top-three probability results
 *
 * @param {File} imageFile
 */
export async function predictFlowerImage(imageFile) {
  if (!imageFile) {
    throw new Error("Please select an image file to classify.");
  }

  const validTypes = [
    "image/jpeg",
    "image/png",
    "image/webp",
  ];

  if (
    !validTypes.includes(imageFile.type) &&
    !/\.(jpe?g|png|webp)$/i.test(imageFile.name)
  ) {
    throw new Error(
      "Unsupported format. Please upload a JPG, PNG, or WEBP image."
    );
  }

  const maxSize = 10 * 1024 * 1024;

  if (imageFile.size > maxSize) {
    throw new Error(
      "File size exceeds the 10 MB limit. Please choose a smaller image."
    );
  }

  try {
    const client = await getClient();

    const result = await client.predict("/predict", {
      image: handle_file(imageFile),
    });

    const [prediction, confidence, labelResult] = result.data;

    const topPredictions =
      labelResult?.confidences?.map((item) => ({
        label: item.label,
        confidence: item.confidence,
      })) || [];

    return {
      prediction,
      predicted_class: prediction,
      predicted_flower: prediction,
      confidence,
      top_predictions: topPredictions,
      top3: topPredictions,
      probabilities: topPredictions,
    };
  } catch (error) {
    console.error("BloomAI prediction error:", error);

    throw new Error(
      error.message ||
        "Flower classification failed. Please try again."
    );
  }
}
