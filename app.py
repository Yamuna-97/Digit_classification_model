import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import numpy as np

# Device
device = torch.device("cpu")

# Model Architecture (MUST match training exactly)
class DigitCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Flatten(),
            nn.Linear(64 * 3 * 3, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.network(x)

# Load model
model = DigitCNN()
model.load_state_dict(torch.load("digit_cnn_model.pth", map_location=device))
model.eval()

# Transform (MNIST style)
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# UI
st.title("✍️ Handwritten Digit Classifier")
st.write("Upload a digit image (white background, black digit)")

uploaded_file = st.file_uploader("Upload a digit image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Open image in grayscale
    image = Image.open(uploaded_file).convert("L")

    # Convert to numpy
    img_array = np.array(image)

    # Invert image (VERY IMPORTANT for MNIST)
    img_array = 255 - img_array

    # Convert back to PIL
    image = Image.fromarray(img_array)

    st.image(image, caption="Processed Image (Inverted)", width=150)

    # Transform
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        prob = F.softmax(output, dim=1)

        confidence, predicted = torch.max(prob, 1)

        # Top 3 predictions
        top_prob, top_class = torch.topk(prob, 3)

    st.success(f"Prediction: {predicted.item()}")
    st.info(f"Confidence: {confidence.item()*100:.2f}%")

    st.subheader("Top 3 Predictions")
    for i in range(3):
        st.write(
            f"{top_class[0][i].item()} → {top_prob[0][i].item()*100:.2f}%"
        )
