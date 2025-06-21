# TreeFormer - Semi-Supervised Tree Counting with Transformers 🌳

TreeFormer là framework bán giám sát cho bài toán đếm cây từ ảnh RGB độ phân giải cao. Pipeline đầy đủ gồm các bước: tile ảnh lớn, gán nhãn điểm, tiền xử lý dữ liệu, huấn luyện và đánh giá mô hình.

## Cài đặt môi trường

Yêu cầu sử dụng Python 3.8. Khuyến nghị dùng Conda để quản lý môi trường:

conda create -n treeformer python=3.8 -y  
conda activate treeformer  
pip install -r requirements.txt

## Chuẩn bị dữ liệu

Bước 1: Cắt ảnh lớn thành tile 1024×1024 bằng lệnh:

python crop_tiles.py

Bước 2: Gán nhãn điểm (tọa độ tán cây) bằng CVAT hoặc makesense.ai. Sau khi gán nhãn, xuất file xuất file xml định dạng cvat

Bước 3: Chạy notebook tiền xử lý dữ liệu để tạo file .mat và density map và phân chia train/val/test:

Mở và chạy lần lượt các cell trong:

preprocess_data.ipynb

Sau khi chạy, dữ liệu sẽ được tổ chức lại theo chuẩn TreeFormer:

datasets/Sample/  
├── train_data/  
│   ├── images/  
│   └── ground_truth/  
├── valid_data/  
│   ├── images/  
│   └── ground_truth/  
├── test_data/  
│   ├── images/  
│   └── ground_truth/  
└── train_data_ul/  
    └── images/  ← chứa ảnh chưa gán nhãn

## Huấn luyện mô hình

Sau khi có đầy đủ dữ liệu đã chia và chuẩn hóa, chạy lệnh sau để huấn luyện:

python train.py --data-dir datasets/Sample --device 0 --batch-size 4 --batch-size-ul 4 --num-workers 16 --resume /home/drone/my_own_code/TreeFormer/checkpoints/ckpts/SEMI/Treeformer_test_12-1-input-256_reg-1_nIter-100_normCood-0/18_ckpt.tar

Trong quá trình train, model sẽ sử dụng cả ảnh đã gán nhãn (train_data) và ảnh chưa gán nhãn (train_data_ul) theo chiến lược bán giám sát.

## Đánh giá mô hình

Khi đã có model huấn luyện xong, bạn có thể chạy đánh giá trên tập test bằng:

python test.py --device 0 --batch-size 4 --model-path checkpoints/best_model.pth --data-path datasets/Sample

Kết quả bao gồm số đếm dự đoán, sai số MAE, MSE, và chỉ số R².

