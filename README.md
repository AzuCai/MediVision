# MediVision: Zero-Shot Medical Image Analysis

## Introduction
MediVision is an innovative tool that combines Large Language Models (LLMs) with vision capabilities for zero-shot medical image classification. Powered by CLIP (Contrastive Language-Image Pre-training) from OpenAI, it analyzes JPEG, PNG, and DICOM images to detect conditions like pneumonia, tumors, fractures, or normal states, providing textual descriptions. Built with PyTorch and Gradio, it supports GPU acceleration via CUDA and offers an intuitive web interface.

This project showcases my understanding of LLMs, their pre-training paradigms, and their synergy with vision models, applied to a practical medical imaging use case.

## Features
- **Zero-Shot Learning**: Leverages CLIP’s ability to generalize across unseen tasks using natural language prompts.
- **Supported Formats**: Processes JPEG, PNG, and DICOM medical images.
- **Conditions Detected**: Identifies pneumonia, tumors, fractures, and normal findings.
- **GPU Support**: Utilizes CUDA for efficient computation.
- **Interactive UI**: Built with Gradio for seamless user interaction.

## LLM Knowledge Points Applied
MediVision integrates several key concepts from LLMs, demonstrating my expertise in this domain:

1. **Contrastive Pre-training**:
   - CLIP is pre-trained on vast image-text pairs using a contrastive loss, aligning visual and textual embeddings in a shared latent space. This enables zero-shot classification by matching image features to text descriptions without fine-tuning.

2. **Natural Language as a Classifier**:
   - Instead of traditional label-based training, I use LLM-generated descriptions (e.g., "X-ray shows hazy areas indicating pneumonia") as prompts. CLIP computes similarity scores between these prompts and image features, showcasing LLMs’ role in flexible task specification.

3. **Text Embedding Generation**:
   - The CLIP model’s text encoder, derived from Transformer-based LLMs, processes medical condition descriptions into high-dimensional embeddings. This highlights my understanding of how LLMs encode semantic meaning for downstream tasks.

4. **Zero-Shot Generalization**:
   - By simulating LLM-like behavior with static descriptions (expandable to dynamic generation with models like GPT), MediVision demonstrates LLMs’ ability to generalize to new domains (e.g., medical imaging) without retraining.

5. **Integration with Vision**:
   - I leverage CLIP’s dual-encoder architecture (vision + language), a hallmark of multimodal LLMs, to bridge image analysis and textual reasoning, a cutting-edge application in AI research.

## Installation

### Prerequisites
- Python 3.9+
- NVIDIA GPU (optional, for CUDA support)
- CUDA Toolkit (e.g., 11.8 or 12.1, if using GPU)

### Setup
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/AzuCai/MediVision.git
   cd MediVision
   ```

2. **Create an Anaconda Environment**:
   ```bash
   conda create -n medivision python=3.9
   conda activate medivision
   ```
3. **Install Dependencies**：
   ```bash
   conda install pytorch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 cudatoolkit=11.8 -c pytorch -c conda-forge
   pip install opencv-python transformers gradio pydicom pillow numpy
   ```
4. **Verify Installation:**：
   ```bash
   python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
   ```
### Usage
1. **Run the Application**:
   ```bash
   conda activate medivision
   python main.py
   ```
2. **Open the provided URL in your browser (e.g., http://127.0.0.1:7860)**.
3. **Upload a medical image (JPEG, PNG, or DICOM)**.
4. **View the predicted condition and description, powered by CLIP’s LLM-vision synergy**.

## Future Enhancements
Integrate a full LLM (e.g., GPT) to dynamically generate condition descriptions.

Fine-tune CLIP on a medical dataset for improved accuracy.

Add multi-label classification using advanced LLM prompting techniques.

## Contributing
Contributions are welcome! Submit issues or pull requests to enhance functionality or optimize LLM integration.

## License
This project is licensed under the MIT License.

## Acknowledgments
Built with PyTorch, Transformers, and Gradio in an Anaconda environment.

Powered by CLIP, a pioneering model in LLM-vision research from OpenAI.
