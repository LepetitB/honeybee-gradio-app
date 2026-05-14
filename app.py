
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import gradio as gr

# 1. Load config
with open("model_config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

MODEL_ARCH = config["model_arch"]
MODEL_PATH = config["model_file"]
IMG_SIZE = config["img_size"]

with open("class_names.json", "r", encoding="utf-8") as f:
    class_names = json.load(f)

num_classes = len(class_names)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 2. Build model
def build_model(model_arch, num_classes):
    if model_arch == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    elif model_arch == "mobilenet_v2":
        model = models.mobilenet_v2(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(f"Unsupported model_arch: {model_arch}")

    return model


model = build_model(MODEL_ARCH, num_classes)

# 3. Load trained weights
try:
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
except TypeError:
    checkpoint = torch.load(MODEL_PATH, map_location=device)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
else:
    state_dict = checkpoint

model.load_state_dict(state_dict)
model.to(device)
model.eval()

# 4. Image preprocessing
# ต้องตรงกับตอน train/test

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# 5. Prediction function
def predict(image):
    if image is None:
        return {}

    image = image.convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

    results = {
        class_names[i]: float(probabilities[i])
        for i in range(num_classes)
    }

    return results

# 6. Gradio app
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload honey bee image"),
    outputs=gr.Label(num_top_classes=num_classes, label="Prediction"),
    title="Honey Bee Species Classification",
    description="Upload a honey bee image and the model will predict the bee species."
)

if __name__ == "__main__":
    demo.launch()
