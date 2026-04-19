import torch
from torchvision.models import vgg16, VGG16_Weights
from torchvision import transforms
from PIL import Image
import tkinter as tk
from tkinter import filedialog

# ----------------------------
# 1. Load pretrained VGG16 model
# ----------------------------
weights = VGG16_Weights.IMAGENET1K_V1
model = vgg16(weights=weights)
model.eval()

print("VGG16 model loaded successfully!")

# ----------------------------
# 2. Open file picker to select image
# ----------------------------
root = tk.Tk()
root.withdraw()  # Hide the main Tk window

img_path = filedialog.askopenfilename(
    title="Select an Image",
    filetypes=[
        ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
        ("All files", "*.*")
    ]
)

if not img_path:
    print("No image selected. Exiting...")
    exit()

print(f"Selected image: {img_path}")

# ----------------------------
# 3. Load image
# ----------------------------
image = Image.open(img_path).convert("RGB")

# ----------------------------
# 4. Preprocess image
# ----------------------------
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

input_tensor = preprocess(image)
input_batch = input_tensor.unsqueeze(0)

# ----------------------------
# 5. Run model prediction
# ----------------------------
with torch.no_grad():
    output = model(input_batch)

probabilities = torch.nn.functional.softmax(output[0], dim=0)

# ----------------------------
# 6. Top 5 predictions
# ----------------------------
top5_prob, top5_catid = torch.topk(probabilities, 5)
categories = weights.meta["categories"]

print("\n--- Top 5 Predictions ---")
for i in range(5):
    label = categories[top5_catid[i].item()]
    confidence = top5_prob[i].item()
    print(f"{i+1}. {label}: {confidence:.4f}")
