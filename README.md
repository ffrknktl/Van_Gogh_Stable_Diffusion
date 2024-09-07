Van Gogh Style Transfer Using ComfyUI, Stable Diffusion, and ControlNet

This project allows users to transform their images into the style of Vincent Van Gogh using ComfyUI, a fine-tuned Stable Diffusion model, and ControlNet. The workflow is designed for ease of use, allowing users to load images and output stylized versions easily.

![atul-pandey-HlwQXlKkokk-unsplash](https://github.com/user-attachments/assets/a2c1b0b0-6b96-420e-bcf3-2ed1be3c892a)


![ComfyUI_00001_](https://github.com/user-attachments/assets/0748da0a-3ec7-4abb-bd53-7c8472304bf2)


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
3. Upload images from Load Image node and click Queue Prompt.


![image](https://github.com/user-attachments/assets/a0f7e60a-e357-4f0d-b44d-6e5851b33a52)


1. Load Image: This node allows the user to upload the image they want to test or transform. The workflow starts from here, where the image is passed to subsequent nodes for processing.

2. Upscale Image: This node increases the resolution of the input image to 512x512 pixels using the `nearest-exact` upscaling method. This resolution was selected because it matches the trained model's resolution, ensuring that the image can be processed correctly without losing important details. The `nearest-exact` method maintains sharp edges, which is essential when applying artistic transformations like the Van Gogh style.

3. Canny: The Canny edge detection node identifies the edges within the image, using a low threshold of 0.30 and a high threshold of 0.70. These values provide a balanced level of edge detection, capturing enough detail without overwhelming the transformation process. It helps to preserve the structure of the original image while applying artistic styles.

4. Preview Image: This node shows a preview of the image at different stages of processing. It's particularly useful to visualize how each step, especially edge detection and ControlNet, affects the final result.

5. Load Checkpoint: This node loads the pre-trained model, fine-tuned on Van Gogh's art style. The model used here is specifically trained to emulate Van Gogh’s style, such as thick brush strokes and vibrant colors. It serves as the core of the style transfer process.

6. Load ControlNet Model: The ControlNet model helps in conditioning the image transformation process by controlling specific aspects like structure. In this case, it assists in transforming the image while preserving key features like edges detected by the Canny node. The pre-selected ControlNet model `control_v11p_sd15_canny` is chosen because it works well with edge maps, ensuring that the structure from the original image remains intact.

7. Apply ControlNet: This node applies the ControlNet conditioning to the image with a strength of 0.80. The strength is set high enough to strongly guide the transformation, but not too high to overpower the Van Gogh style. It ensures a balanced influence between the edge structure and artistic transformation.

8. CLIP Text Encode (Prompt): The prompt “Vincent van Gogh style, thick brush strokes, vibrant colors, post-impressionism” encodes the desired style of transformation. This text prompt guides the model to apply Van Gogh’s artistic style, specifically focusing on his unique brushwork and color palette.

9. CLIP Text Encode (Negative Prompt): The negative prompt “worst quality, low quality, blurry, cropped, lowers” helps in avoiding unwanted artifacts like blurry or low-quality images. This ensures that the generated image remains sharp and clear.

10. KSampler: The KSampler node performs sampling in the latent space. It is set to 20 steps, which is a common choice for generating high-quality images without excessive processing time. The denoise strength is set to 0.80, allowing some level of noise for texture while keeping the image details intact. The Euler sampling method is chosen for its efficiency and quality balance.

11. VAE Decode (Tiled): This node decodes the latent image back into a visible image. The VAE (Variational Autoencoder) model is used to ensure that the generated image maintains the artistic features while keeping the general structure of the original.

12. Save Image: This node saves the final transformed image. It allows the user to export the output after the style transfer and transformation are complete.

13. Empty Latent Image: This node creates an empty latent image for initialization purposes. The latent image size is set to 512x512 to match the model's input requirements, ensuring that the input dimensions are consistent throughout the workflow.


