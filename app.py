import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import numpy as np

device = torch.device("cpu")

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

model = DigitCNN()
model.load_state_dict(torch.load("digit_cnn_model.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

st.title("✍️ Handwritten Digit Classifier")
st.write("Upload a digit image (white background, black digit)")

uploaded_file = st.file_uploader("Upload a digit image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")

    img_array = np.array(image)

    img_array = 255 - img_array

    image = Image.fromarray(img_array)

    st.image(image, caption="Processed Image (Inverted)", width=150)

    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        prob = F.softmax(output, dim=1)

        confidence, predicted = torch.max(prob, 1)

        # Top 3 predictions
        top_prob, top_class = torch.topk(prob, 3)

    st.success(f"Prediction: {predicted.item()}")
    st.info(f"Confidence: {confidence.item()*100:.2f}%")
