import os
import cv2
from pathlib import Path

tile_size=512

def tile_image(image_path, output_dir, tile_size=tile_size):
    os.makedirs(output_dir, exist_ok=True)
    
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)  # Hỗ trợ cả ảnh 16-bit hoặc đa kênh
    if img is None:
        print(f"⚠️ Could not read: {image_path}")
        return

    h, w = img.shape[:2]
    base_name = Path(image_path).stem
    count = 0

    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            tile = img[y:y+tile_size, x:x+tile_size]

            if tile.shape[0] == tile_size and tile.shape[1] == tile_size:
                out_path = os.path.join(output_dir, f"{base_name}_tile_{count:03d}.png")
                cv2.imwrite(out_path, tile)
                count += 1

    print(f"✅ {count} tiles saved from {image_path} to {output_dir}")

def tile_folder(input_folder, output_folder, tile_size=tile_size):
    os.makedirs(output_folder, exist_ok=True)
    supported_exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

    for file in os.listdir(input_folder):
        if file.lower().endswith(supported_exts):
            image_path = os.path.join(input_folder, file)
            tile_image(image_path, output_folder, tile_size)

# 🧪 Ví dụ sử dụng:
if __name__ == "__main__":
    input_folder = "raw_images"        # 📂 Chứa ảnh gốc .tif, .png, .jpg
    output_folder = "unlabel_data"       # 📂 Kết quả tile
    tile_folder(input_folder, output_folder)
