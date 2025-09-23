import torch
from torchvision.models.resnet import ResNet, Bottleneck
from torchvision.transforms import Compose, Normalize, ToTensor, Resize
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import random
import os

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        self.target_layer.register_forward_hook(self._save_activations)
        self.target_layer.register_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x):
        self.model.zero_grad()
        output = self.model(x)
        # Use the prediction for age estimation directly, not argmax
        output.backward()

        gradients = self.gradients.cpu().data.numpy()
        activations = self.activations.cpu().data.numpy()

        # Global average pooling of gradients
        weights = np.mean(gradients, axis=(2, 3))[0]
        heatmap = np.zeros(activations.shape[2:][::-1], dtype=np.float32)

        # Build heatmap
        for i, w in enumerate(weights):
            heatmap += w * activations[0, i, :, :]

        # Apply ReLU
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)

        return heatmap, output.item() # Return heatmap and the scalar prediction

def visualize_cam(image_path, heatmap, alpha=0.5):
    img = cv2.imread(image_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, alpha, heatmap, 1 - alpha, 0)
    return superimposed_img

# Load your trained model
model = torch.load('/content/drive/MyDrive/model_age_estimation.pt', weights_only=False).to(device)
model.eval()

# Define the target layer (you might need to adjust this based on your model's structure)
target_layer = model.model[5] # This is an example, adjust based on your model

# Create GradCAM object
grad_cam = GradCAM(model, target_layer)

# Load the test dataframe
df_test = pd.read_csv('/content/test_set.csv')

# Select a few random images from the test set
num_images_to_show = 5
random_image_files = random.sample(df_test['image_name'].tolist(), num_images_to_show)

# Use the same transformation as for validation/testing
transform = T.Compose([
    T.Resize((128, 128)),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

for image_file in random_image_files:
    image_path = os.path.join('/content/UTKFace', image_file) # Assuming UTKFace is in /content

    # Load and preprocess an image
    input_image = Image.open(image_path).convert('RGB')
    input_tensor = transform(input_image).unsqueeze(0).to(device)

    # Get target age
    target_age = df_test[df_test['image_name'] == image_file]['age'].iloc[0]


    # Generate heatmap and get prediction
    heatmap, predicted_age = grad_cam(input_tensor)


    # Visualize heatmap
    superimposed_image = visualize_cam(image_path, heatmap)

    # Display the original and superimposed images
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
    plt.title(f'Original Image\nTarget Age: {target_age}, Predicted Age: {predicted_age:.2f}')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(superimposed_image, cv2.COLOR_BGR2RGB))
    plt.title('Grad-CAM Heatmap')
    plt.axis('off')

    plt.show()
