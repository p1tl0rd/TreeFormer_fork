import xml.etree.ElementTree as ET
from scipy.io import savemat
import os
import numpy as np

# Đường dẫn file XML từ CVAT export
XML_PATH = 'points_annotated/annotations.xml'
# Thư mục xuất file .mat
OUTPUT_DIR = 'points_annotated/mat_gt'
os.makedirs(OUTPUT_DIR, exist_ok=True)

tree = ET.parse(XML_PATH)
root = tree.getroot()

for image in root.iter('image'):
    name = image.attrib['name']
    locations = []

    for point in image.iter('points'):
        x_str, y_str = point.attrib['points'].split(',')
        x, y = float(x_str), float(y_str)
        locations.append([x, y])

    if locations:
        locations_np = np.round(np.array(locations)).astype(np.int32)
        # Bọc đúng kiểu TreeFormer
        image_info = [{'location': locations_np}]
        mat_data = {'image_info': image_info}

        out_path = os.path.join(OUTPUT_DIR, f'GT_{os.path.splitext(name)[0]}.mat')
        savemat(out_path, mat_data)
        print(f"✅ Saved: {out_path} ({len(locations)} points)")

print("🎉 Done.")
