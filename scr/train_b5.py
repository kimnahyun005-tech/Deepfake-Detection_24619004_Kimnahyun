import torch
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b5, EfficientNet_B5_Weights
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

# 원본 main_trainer.py에서 Dataset 및 Detector 뼈대 라이브러리 가져오기
from main_trainer import HybridDeepfakeDataset, DeepfakeDetector  

if __name__ == "__main__":
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # === 🛠️ B5 스펙에 맞춘 데이터 증강 (Resize 크기를 456으로 확장) ===
    train_transform = transforms.Compose([
        transforms.Resize((456, 456)),  # EfficientNet-B5의 최적 입력 해상도는 456x456입니다.
        transforms.RandomCrop(456),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((456, 456)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # === 📂 데이터셋 절대 경로 설정 ===
    train_sources = [
        ("C:/Users/Konyang/Desktop/최종 베이스라인 코드/data/train/real", 0),
        ("C:/Users/Konyang/Desktop/최종 베이스라인 코드/data/train/fake", 1)
    ]

    val_sources = [
        ("C:/Users/Konyang/Desktop/최종 베이스라인 코드/data/validation/real", 0),
        ("C:/Users/Konyang/Desktop/최종 베이스라인 코드/data/validation/fake", 1)
    ]

    # === 📦 데이터로더 세팅 (B5의 엄청난 연산량으로 인해 batch_size를 2로 강제 제한) ===
    train_dataset = HybridDeepfakeDataset(train_sources, transform=train_transform)
    val_dataset = HybridDeepfakeDataset(val_sources, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=0)

    # === 🧠 Model Architecture (EfficientNet-B5 설정) ===
    print("-> EfficientNet-B5 사전 학습 가중치 다운로드 및 모델 생성 중...")
    weights = EfficientNet_B5_Weights.IMAGENET1K_V1
    backbone = efficientnet_b5(weights=weights)
    
    # B5의 최종 Classifier 레이어 채널 수(2048)를 자동으로 가져와 이진 분류 헤더로 수정
    in_features = backbone.classifier[1].in_features
    backbone.classifier = torch.nn.Sequential(
        torch.nn.Dropout(0.4),
        torch.nn.Linear(in_features, 2)
    )

    model = DeepfakeDetector(backbone, lr=cfg["learning_rate"])

    # === 💾 Callbacks 설정 ===
    checkpoint_callback = ModelCheckpoint(
        monitor=cfg.get("monitor_metric", "val_loss"),
        dirpath="models",
        filename="efficientnet_b5_best",  # 가중치 파일 이름 지정
        save_top_k=1,
        mode="min"
    )

    early_stop_callback = EarlyStopping(
        monitor=cfg.get("monitor_metric", "val_loss"),
        patience=cfg.get("early_stopping_patience", 7),
        mode="min"
    )

    # === 🚀 Trainer 실행 ===
    trainer = Trainer(
        max_epochs=30,  # 에러 방지를 위해 30회 고정
        accelerator="auto",
        devices=1,
        callbacks=[checkpoint_callback, early_stop_callback],
        log_every_n_steps=10
    )

    print("▶️ EfficientNet-B5 딥페이크 학습을 시작합니다!")
    trainer.fit(model, train_loader, val_loader)