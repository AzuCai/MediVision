import os
import cv2
import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel
import gradio as gr
from PIL import Image
import pydicom  # For DICOM support

# Disable symlinks warning
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Load CLIP model
model_name = "openai/clip-vit-base-patch32"  # Lightweight CLIP model
processor = CLIPProcessor.from_pretrained(model_name)

# Detect if CUDA is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CLIPModel.from_pretrained(model_name).to(device)
print(f"Using device: {device}")


# Define medical image descriptions using LLMs (simulated)
def get_medical_descriptions():
    """
    Generate textual descriptions for medical images using an LLM
    :return: List of medical conditions and their descriptions
    """
    conditions = ["pneumonia", "tumor", "fracture", "normal"]
    descriptions = {
        "pneumonia": "X-ray shows hazy or patchy areas in the lungs, indicating possible pneumonia.",
        "tumor": "MRI or CT shows abnormal masses or lesions, suggesting a tumor.",
        "fracture": "X-ray shows a break or discontinuity in bone structure, indicating a fracture.",
        "normal": "Image shows no abnormal findings, indicating a healthy condition."
    }
    return conditions, descriptions


conditions, description_dict = get_medical_descriptions()


# Process medical image (JPEG/PNG/DICOM)
def process_image(image):
    """
    Process an input medical image to extract visual features using CLIP
    :param image: File path (str) from Gradio File input
    :return: Image features tensor
    """
    try:
        print(f"Processing image: {image}, type: {type(image)}")
        if isinstance(image, str):
            if image.endswith('.dcm'):
                ds = pydicom.dcmread(image)
                print(f"DICOM pixel array shape: {ds.pixel_array.shape}")
                image_array = ds.pixel_array
                image = Image.fromarray(image_array).convert('RGB')
            else:
                image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            raise ValueError("Unsupported image input type")

        # Resize image to fit CLIP's expected input (224x224)
        image = image.resize((224, 224), Image.LANCZOS)
        inputs = processor(images=image, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
        return image_features
    except Exception as e:
        raise Exception(f"Error in process_image: {e}")


# Zero-shot medical image classification using CLIP
def recognize_medical_condition(image):
    """
    Recognize medical condition in an image using zero-shot learning
    :param image: File path (str) from Gradio File input
    :return: Predicted condition and description
    """
    try:
        image_features = process_image(image)
        text_inputs = processor(text=list(description_dict.values()), return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            text_features = model.get_text_features(**text_inputs)

        # Compute cosine similarity
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        similarity = (image_features @ text_features.T).softmax(dim=-1)

        # Get the most likely condition
        max_idx = similarity.argmax().item()
        condition = conditions[max_idx]
        return condition, description_dict[condition]
    except Exception as e:
        raise Exception(f"Error in recognize_medical_condition: {e}")


# Gradio interface with medical image processing
def gradio_medical(image_path=None):
    """
    Process medical image input for Gradio interface
    :param image_path: File path from Gradio File input (str or None)
    :return: Tuple of (condition, description)
    """
    print(f"Received image_path: {image_path}, type: {type(image_path)}")
    if image_path is None or image_path == "":
        return "Please upload a medical image (JPEG, PNG, or DICOM).", ""
    try:
        condition, description = recognize_medical_condition(image_path)
        return f"Predicted Condition: {condition}", f"Description: {description}"
    except Exception as e:
        print(f"Error processing image: {e}")
        return "Error processing image. Please try again.", ""


# Gradio interface setup
with gr.Blocks(theme=gr.themes.Soft(), css="""
    .gradio-container {
        background-color: #f0f4f8;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .title {
        color: #2c3e50;
        font-family: 'Arial', sans-serif;
        font-size: 24px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
    }
    .description {
        color: #7f8c8d;
        font-size: 16px;
        text-align: center;
        margin-bottom: 30px;
    }
    .gr-button {
        background-color: #3498db;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        font-size: 16px;
        cursor: pointer;
    }
    .gr-button:hover {
        background-color: #2980b9;
    }
    .gr-image, .gr-file {
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .gr-textbox {
        border-radius: 5px;
        padding: 10px;
        font-size: 16px;
        border: 1px solid #ddd;
    }
""") as interface:
    with gr.Column(elem_id="main-column"):
        gr.Markdown(
            """
            <div class="title">MediVision: Zero-Shot Medical Image Analysis</div>
            <div class="description">Upload a medical image (JPEG, PNG, or DICOM) for zero-shot analysis of conditions like pneumonia or tumors using LLMs and CLIP.</div>
            """,
            elem_id="header"
        )

        with gr.Row(equal_height=True):
            image_input = gr.File(label="Upload Medical Image", file_types=[".jpg", ".png", ".dcm"],
                                  elem_classes="gr-file")
            with gr.Column(scale=2):
                condition_output = gr.Textbox(label="Predicted Condition", elem_classes="gr-textbox")
                description_output = gr.Textbox(label="Medical Description", elem_classes="gr-textbox")

        image_input.change(
            fn=gradio_medical,
            inputs=[image_input],
            outputs=[condition_output, description_output]
        )

interface.launch(inbrowser=True, share=False)