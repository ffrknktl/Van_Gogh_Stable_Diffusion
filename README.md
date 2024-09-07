Van Gogh Style Transfer Using ComfyUI, Stable Diffusion, and ControlNet

This project allows users to transform their images into the style of Vincent Van Gogh using ComfyUI, a fine-tuned Stable Diffusion model, and ControlNet. The workflow is designed for ease of use, allowing users to load images and output stylized versions easily.

![licensed-image](https://github.com/user-attachments/assets/79bc42dd-7f59-44a3-b856-eef5d83a4371)


![ComfyUI_00001_](https://github.com/user-attachments/assets/817a6d03-99c8-4ece-9e8b-db474393cd52)



Project Overview

This project enables users to convert any uploaded image into the distinctive style of Vincent Van Gogh. Using a pre-trained model on Van Gogh artworks, the workflow can recreate his famous brush strokes, vibrant colors, and iconic style in modern images.

Technologies Used

ComfyUI: A node-based visual interface for integrating and running diffusion models.
Stable Diffusion: An open-source AI model used for generating images. 
ControlNet: A tool to guide the Stable Diffusion model, providing more control over the style and structure of generated images. 

Setup Process

Prerequisites

1. Python 3.8 or higher.

2. Git: Git is required to clone repositories. 

3. GPU-Enabled Machine: For optimal performance, it is recommended to have an NVIDIA GPU with CUDA support.

Step 1: Install ComfyUI

ComfyUI is a node-based user interface for working with diffusion models. To install ComfyUI, follow these steps:

1. Clone the ComfyUI repository:

   https://github.com/comfyanonymous/ComfyUI.git
 

3. Install the required Python libraries for ComfyUI:
  
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
   pip install xformers transformers opencv-python
 

Step 2: Download the Van Gogh Fine-Tuned Model

Download the fine-tuned Stable Diffusion model trained on Van Gogh artworks. 

1. Download the model manually from Hugging Face:

   - Go to the [Van Gogh Diffusion Model](https://huggingface.co/dallinmackay/Van-Gogh-diffusion).
   - Download the `van-gogh.ckpt` file along with the necessary configuration files.

2. Or clone using Git LFS** (if Git LFS is installed):

   git lfs install
   git clone https://huggingface.co/dallinmackay/Van-Gogh-diffusion
   cp van-gogh.ckpt /path/to/comfyui/checkpoints/
  

3. Move the downloaded model to ComfyUI’s `checkpoints` folder.   

Step 3: Download and Setup ControlNet

To guide the image generation with structural guidance (e.g., edge detection), ControlNet is used. Download the ControlNet model as follows:

1. Download the ControlNet model from Hugging Face:

   - Visit [ControlNet-v1-1](https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors).
   - Download the `.safetensors` file for ControlNet.

2. Move the ControlNet model to the ComfyUI ControlNet folder.


Step 4: Run ComfyUI

Once you have all the dependencies and models installed, you can start ComfyUI and load the workflow.
While ComfyUI is running, load the provided Van Gogh style transfer workflow:

1. In ComfyUI, click on `Load Workflow`.
2. Select the `vangogh-style-transfer-workflow.json` file from the project directory.
