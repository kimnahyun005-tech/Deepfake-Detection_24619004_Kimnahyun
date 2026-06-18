import os
import torch
import cv2
from PIL import Image
import numpy as np
from torchvision import transforms
from torchvision.models import efficientnet_b0

# === 1. Load model (우리가 학습한 v3 뇌에 맞게 구조 변경) ===
model = efficientnet_b0()
in_features = model.classifier[1].in_features
model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(0.4),
    torch.nn.Linear(in_features, 2)
)

# ⭐️ 방금 완성된 v3 파일로 이름 변경 및 껍질 벗기기
checkpoint = torch.load("models/best_model-v3.ckpt", map_location="cpu")
state_dict = checkpoint.get("state_dict", checkpoint)

clean_state_dict = {}
for k, v in state_dict.items():
    clean_key = k.replace("model.", "").replace("backbone.", "")
    clean_state_dict[clean_key] = v

model.load_state_dict(clean_state_dict)
model.eval()

# === 2. Preprocessing with optional noise (화질 훼손 시뮬레이션) ===
def distort(image, simulate=True):
    if simulate:
        image = image.resize((224, 224))
        arr = np.array(image).astype(np.uint8)

        # 50% 확률로 흐리게 만들기 (블러)
        if np.random.rand() < 0.5:
            arr = cv2.GaussianBlur(arr, (5, 5), 0)

        # 50% 확률로 카톡 전송처럼 화질 압축하기 (JPEG 노이즈)
        if np.random.rand() < 0.5:
            _, arr = cv2.imencode('.jpg', arr, [int(cv2.IMWRITE_JPEG_QUALITY), 40])
            arr = cv2.imdecode(arr, 1)

        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(arr)

    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])(image).unsqueeze(0)

# === 3. Run on test folder ===
def evaluate(folder, simulate_noise=True):
    print(f"\n📁 [{folder}] 폴더의 가혹 환경 테스트를 시작합니다...")
    if not os.path.exists(folder):
        print(f"❌ '{folder}' 폴더를 찾을 수 없습니다. 경로를 확인해주세요.")
        return

    correct = 0
    total = 0
    # 폴더 이름이 'fake'면 정답은 'Deepfake', 아니면 'Real'
    true_label = "Deepfake" if "fake" in folder.lower() else "Real"

    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        if not os.path.isfile(path):
            continue
        try:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                image = Image.open(path).convert("RGB")
                tensor = distort(image, simulate=simulate_noise)
            else:
                continue

            with torch.no_grad():
                out = model(tensor)
                prob = torch.softmax(out, dim=1)[0]
                conf, pred = torch.max(prob, dim=0)

            label = "Real" if pred.item() == 0 else "Deepfake"
            
            # 정답을 맞췄는지 체크
            if label == true_label:
                correct += 1
            total += 1
            
            print(f"{file:<30} ➤ 예측: {label:<9} ({conf.item()*100:.2f}%)")

        except Exception as e:
            pass
            
    if total > 0:
        print("-" * 45)
        print(f"🎯 훼손된 사진 방어력: {total}장 중 {correct}장 정답 ({(correct/total)*100:.2f}%)")
        print("-" * 45)

# Example use
if __name__ == "__main__":
    # ⭐️ 평가할 실제 폴더 경로 지정 (데이터가 있는 경로로 맞춰주세요)
    # 기존에 사용하셨던 검증 폴더의 fake(가짜) 사진들로 먼저 독하게 테스트해 봅니다!
    test_folder_path = "data/validation/fake" 
    
    evaluate(folder=test_folder_path, simulate_noise=True)