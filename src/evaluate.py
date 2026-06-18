import os
import torch
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.models import resnet50, ResNet50_Weights, mobilenet_v3_large, MobileNet_V3_Large_Weights, vit_b_16, ViT_B_16_Weights
from PIL import Image

# 1. 사용자가 선택한 모델 아키텍처에 맞게 불러오기
def load_my_model(model_choice):
    if model_choice == "1":
        print(" -> [ResNet-50] 모델 가중치 불러오는 중...")
        weights = ResNet50_Weights.IMAGENET1K_V1
        model = resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = torch.nn.Sequential(torch.nn.Dropout(0.4), torch.nn.Linear(in_features, 2))
        model_path = "models/resnet50_best.ckpt"

    elif model_choice == "2":
        print(" -> [MobileNetV3-Large] 모델 가중치 불러오는 중...")
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1
        model = mobilenet_v3_large(weights=weights)
        in_features = model.classifier[3].in_features
        model.classifier[3] = torch.nn.Sequential(torch.nn.Dropout(0.4), torch.nn.Linear(in_features, 2))
        model_path = "models/mobilenetv3_best.ckpt"

    elif model_choice == "3":
        print(" -> [Vision Transformer] 모델 가중치 불러오는 중...")
        weights = ViT_B_16_Weights.IMAGENET1K_V1
        model = vit_b_16(weights=weights)
        in_features = model.heads.head.in_features
        model.heads.head = torch.nn.Sequential(torch.nn.Dropout(0.4), torch.nn.Linear(in_features, 2))
        model_path = "models/vit_best.ckpt"

    elif model_choice == "4":
        print(" -> [EfficientNet-B4] 모델 가중치 불러오는 중...")
        from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
        weights = EfficientNet_B4_Weights.IMAGENET1K_V1
        model = efficientnet_b4(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.4), 
            torch.nn.Linear(in_features, 2)
        )
        model_path = "models/efficientnet_b4_best.ckpt"

    elif model_choice == "5":
        print(" -> [EfficientNet-B5] 모델 가중치 불러오는 중...")
        from torchvision.models import efficientnet_b5, EfficientNet_B5_Weights
        weights = EfficientNet_B5_Weights.IMAGENET1K_V1
        model = efficientnet_b5(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = torch.nn.Sequential(torch.nn.Dropout(0.4), torch.nn.Linear(in_features, 2))
        model_path = "models/efficientnet_b5_best.ckpt"

    else:
        print(" -> [Baseline: EfficientNet-B0] 모델 가중치 불러오는 중...")
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        model = efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = torch.nn.Sequential(torch.nn.Dropout(0.4), torch.nn.Linear(in_features, 2))
        model_path = "models/best_model-v3.ckpt"

    if not os.path.exists(model_path):
        # 만약 확장자가 다르게 저장되었을 경우를 대비한 안전장치
        if model_choice == "0" and os.path.exists("models/best_model-v3.pt"):
            model_path = "models/best_model-v3.pt"
        else:
            print(f"⚠️ 에러: '{model_path}' 파일을 찾을 수 없습니다!")
            return None

    checkpoint = torch.load(model_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint) 

    clean_state_dict = {}
    for k, v in state_dict.items():
        clean_key = k.replace("model.", "").replace("backbone.", "")
        clean_state_dict[clean_key] = v

    model.load_state_dict(clean_state_dict)
    model.eval()
    return model

# 2. 폴더 안의 사진들을 한 번에 채점하는 함수
def evaluate_folder(folder_path, correct_label, model, transform):
    correct = 0
    total = 0
    if not os.path.exists(folder_path):
        print(f"⚠️ 경고: '{folder_path}' 폴더를 찾을 수 없습니다!")
        return 0, 0
        
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, filename)
            try:
                image = Image.open(img_path).convert("RGB")
                input_tensor = transform(image).unsqueeze(0)
                with torch.no_grad():
                    output = model(input_tensor)
                    pred = torch.argmax(output, dim=1).item() 
                if pred == correct_label:
                    correct += 1
                total += 1
            except Exception as e:
                pass
    return correct, total

if __name__ == "__main__":
    print("=" * 45)
    print(" 채점할 모델 번호를 선택하세요 ")
    print(" [0] Baseline (EfficientNet)")
    print(" [1] ResNet-50")
    print(" [2] MobileNetV3")
    print(" [3] Vision Transformer (ViT)")
    print(" [4] EfficientNet-B4")
    print(" [5] EfficientNet-B5")
    print("=" * 45)
    choice = input("번호 입력 후 엔터: ").strip()

    model = load_my_model(choice)
    
    if model is not None:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        real_folder = "data/test/real" 
        fake_folder = "data/test/fake" 
        
        real_correct, real_total = evaluate_folder(real_folder, 0, model, transform)
        fake_correct, fake_total = evaluate_folder(fake_folder, 1, model, transform)

        total_images = real_total + fake_total
        total_correct = real_correct + fake_correct

        if total_images == 0:
            print(" 사진을 찾을 수 없습니다. 경로를 다시 확인해주세요.")
        else:
            accuracy = (total_correct / total_images) * 100
            print("\n" + "=" * 45)
            print(f"REAL 원본 검증 : {real_total}장 중 {real_correct}장 정답맞춤")
            print(f"FAKE 조작 검증 : {fake_total}장 중 {fake_correct}장 정답맞춤")
            print("-" * 45)
            print(f" 총 검증 사진 : {total_images}장")
            print(f" 총 맞춘 개수 : {total_correct}장")
            print(f" 정확도(Accuracy) : {accuracy:.2f}%")
            print("=" * 45 + "\n")