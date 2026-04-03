import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import shutil
import zipfile
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

# --- Трансформации ---
def get_transforms(model_size):
    """Пайплайн трансформаций для инференса."""
    return A.Compose([
        A.Resize(model_size, model_size, interpolation=cv2.INTER_LINEAR),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

# --- Основная функция инференса ---
def run_inference(input_dir_path, model_path_str, temp_output_path, final_zip_path):
    input_dir = Path(input_dir_path)
    model_path = model_path_str
    
    # Временная папка в Colab для быстрой записи мелких файлов
    temp_masks_dir = Path(temp_output_path)
    temp_masks_dir.mkdir(parents=True, exist_ok=True)
    
    # Папка на Google Диске для итогового ZIP
    final_zip_dest = Path(final_zip_path).parent
    final_zip_dest.mkdir(parents=True, exist_ok=True)

    # Определение устройства
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Используемое устройство: {device}")
    if device.type == 'cpu':
        print("[!] ВНИМАНИЕ: Инференс на CPU будет очень медленным. "
              "Рекомендуется включить GPU в Настройках среды выполнения.")

    # Инициализация архитектуры
    print(f"[*] Инициализация модели {smp.UnetPlusPlus.__name__}...")
    model = smp.UnetPlusPlus(
        encoder_name="efficientnet-b0",
        encoder_weights=None, 
        in_channels=3,
        classes=1,
    )

    # Загрузка весов
    if not os.path.exists(model_path):
        print(f"[ERROR] Файл модели не найден по пути: {model_path}")
        return

    print(f"[*] Загрузка весов из: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(device)
    model.eval()

    # Параметры инференса (как в ноутбуке обучения)
    MODEL_SIZE = 256
    THRESHOLD = 0.44
    transforms = get_transforms(MODEL_SIZE)

    # Поиск изображений
    valid_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    if not input_dir.exists():
        print(f"[ERROR] Папка с картинками не найдена по пути: {input_dir}")
        return

    image_files = [f for f in input_dir.iterdir() if f.suffix.lower() in valid_extensions]
    
    print(f"[*] Найдено {len(image_files)} изображений для обработки.\n")

    if not image_files:
        print("[!] Нет изображений для обработки. Выходим.")
        return

    # Цикл инференса
    print(f"[*] Сохраняем временные маски в Colab: {temp_masks_dir}")
    with torch.no_grad():
        for idx, img_path in enumerate(image_files, 1):
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"[!] Ошибка чтения файла: {img_path}")
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Оригинальный размер для ресайза маски обратно
            original_h, original_w = image.shape[:2]

            # Предсказание
            augmented = transforms(image=image)
            img_tensor = augmented["image"].unsqueeze(0).to(device)
            logits = model(img_tensor)

            # Восстановление размера
            logits_original_size = F.interpolate(
                logits,
                size=(original_h, original_w),
                mode="bilinear",
                align_corners=False
            )

            # Бинаризация
            preds = (torch.sigmoid(logits_original_size) > THRESHOLD).float()
            mask_np = preds.squeeze().cpu().numpy()
            mask_img = (mask_np * 255).astype(np.uint8)

            # Сохранение во временную папку в формате PNG
            output_file = temp_masks_dir / f"{img_path.stem}.png"
            cv2.imwrite(str(output_file), mask_img)
            
            if idx % 50 == 0 or idx == len(image_files):
                print(f"  -> Обработано {idx}/{len(image_files)}")
    
    # --- Архивирование ---
    print(f"\n[*] Начинаем архивацию...")
    print(f"[*] Создаем ZIP архив на Google Диске: {final_zip_path}")
    
    try:
        # Создаем ZIP-файл напрямую на Google Диск
        with zipfile.ZipFile(final_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            cnt_added = 0
            for file_path in temp_masks_dir.glob('*.png'):
                # arcname=file_path.name сохраняет файл без вложенных папок
                zipf.write(file_path, arcname=file_path.name)
                cnt_added += 1
        print(f"[+] Успешно добавлено {cnt_added} масок в архив.")

        # --- Очистка временной папки ---
        print(f"[*] Очищаем временную папку в Colab...")
        shutil.rmtree(temp_masks_dir)
        print("[+] Готово.")
        
        print(f"\n{'='*60}\n"
              f" ИТОГ: Результаты инференса сохранены в ZIP архив на вашем "
              f"Google Диске:\n {final_zip_path}\n{'='*60}")

    except Exception as e:
        print(f"[ERROR] Ошибка при архивации или сохранении на Google Диск: {e}")
        print("Попробуйте проверить доступ к Диску или свободное место.")


# =====================================================================
# ТОЧКА ВХОДА И НАСТРОЙКИ (КОНФИГУРАЦИЯ ДЛЯ GOOGLE COLAB)
# =====================================================================
if __name__ == "__main__":
    
    # 1. Монтируем Google Диск (потребуется подтверждение в браузере)
    from google.colab import drive
    print("[*] Подключаем Google Диск...")
    # force_remount=True полезно, если вы перезапускаете ячейку и связь прервалась
    drive.mount('/content/drive', force_remount=True)
    print("[+] Диск подключен.\n")

    # 2. НАСТРОЙКИ ПУТЕЙ (в точности из вашего запроса)
    
    # Папка с тестовыми картинками
    INPUT_DIR = "/content/drive/MyDrive/Segmentation/testing/images" 
    
    # Путь к обученной модели .pth
    MODEL_PATH = "/content/drive/MyDrive/RoMeSegModel.pth"
    
    # Временная папка внутри Colab (не на Диске!) для промежуточного сохранения масок
    # Это работает намного быстрее, чем писать напрямую мелкие PNG на Диск
    TEMP_LOCAL_DIR = "/content/masks_tmp" 
    
    # Полный путь к итоговом ZIP файлу на Google Диске, в который мы все упакуем.
    # Файл будет сохранен внутри папки "pred" (которая создастся автоматически)
    FINAL_ZIP_ON_DRIVE = "/content/drive/MyDrive/pred/RoMe masks.zip"
    
    # 3. Запуск инференса
    run_inference(INPUT_DIR, MODEL_PATH, TEMP_LOCAL_DIR, FINAL_ZIP_ON_DRIVE)
