
/**
 * FloraAI - Hugging Face Gradio API Client
 * Uses the trained ResNet-18 flower classification model.
 */

import { Client, handle_file } from "@gradio/client";

const SPACE_ID = "FatimaTahseen/flower_recognitionai-api";

let clientPromise = null;

async function getClient() {
  if (!clientPromise) {
    clientPromise = Client.connect(SPACE_ID).catch((error) => {
      clientPromise = null;
      throw error;
    });
  }

  return clientPromise;
}

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
    throw new Error(
      "Unable to connect to FloraAI Hugging Face backend. Please try again later."
    );
  }
}

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

export async function predictFlowerImage(imageFile) {
  if (!imageFile) {
    throw new Error("Please select an image file to classify.");
  }

  const validTypes = ["image/jpeg", "image/png", "image/webp"];

  if (
    !validTypes.includes(imageFile.type) &&
    !/\.(jpe?g|png|webp)$/i.test(imageFile.name)
  ) {
    throw new Error(
      "Unsupported format. Please upload a JPG, PNG, or WEBP image."
    );
  }

  if (imageFile.size > 10 * 1024 * 1024) {
    throw new Error(
      "File size exceeds the 10 MB limit. Please choose a smaller image."
    );
  }

  try {
    const client = await getClient();

    const result = await client.predict("/predict", {
      image: handle_file(imageFile),
    });

    const [prediction, rawConfidence, labelResult] = result.data;

    const toPercentage = (value) => {
      const number = Number(value);
      return Number.isFinite(number)
        ? Number((number <= 1 ? number * 100 : number).toFixed(2))
        : 0;
    };

    const topThree = (labelResult?.confidences || [])
      .slice()
      .sort((a, b) => Number(b.confidence) - Number(a.confidence))
      .slice(0, 3)
      .map((item) => ({
        class_name: String(item.label),
        confidence: toPercentage(item.confidence),
      }));

    return {
      predicted_class: String(prediction),
      confidence_percentage: toPercentage(rawConfidence),
      top_three: topThree,
    };
  } catch (error) {
    console.error("FloraAI prediction error:", error);

    throw new Error(
      error.message || "Flower classification failed. Please try again."
    );
  }
}
