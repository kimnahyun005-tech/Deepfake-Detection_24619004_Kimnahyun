import torch
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image
import argparse

# Load your trained model
# Load your trained model
def load_model(model_path="models/best_model.ckpt"): # <-- .pt를 .ckpt로 수정!
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    model = efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(0.4),
        torch.nn.Linear(in_features, 2)
    )

    # PyTorch Lightning 체크포인트(.ckpt) 껍질 벗기기
    checkpoint = torch.load(model_path, map_location="cpu")
    
    # 껍질(state_dict) 안에 있는 진짜 가중치만 꺼내기
    state_dict = checkpoint.get("state_dict", checkpoint) 

    # 뼈대에 붙어있는 불필요한 이름표(prefix) 제거
    clean_state_dict = {}
    for k, v in state_dict.items():
        clean_key = k.replace("model.", "").replace("backbone.", "")
        clean_state_dict[clean_key] = v

    model.load_state_dict(clean_state_dict)
    model.eval()
    return model

# Preprocess and classify image
def predict_image(image_path, model):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)[0]
        pred = torch.argmax(probs).item()

    label = "FAKE" if pred == 1 else "REAL"
    print(f"\n Prediction: {label}")
    print(f"Real: {probs[0]:.3f} | Fake: {probs[1]:.3f}")

# Run from terminal
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", help="Path to image file (.jpg/.png)")
    args = parser.parse_args()

    model = load_model()
    predict_image(args.image_path, model)
