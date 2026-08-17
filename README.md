# favit_lsda

`favit_lsda` là phiên bản FA-ViT được mở rộng bằng Latent Space Data
Augmentation (LSDA) cho generalized deepfake detection. Implementation dựa trên:

- [`fa_vit_remake`](../fa_vit_remake) cho GAM, LAM, FAL, manifest và protocol
  video-level Celeb-DF-v2;
- [LSDA, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Yan_Transcending_Forgery_Specificity_with_Latent_Space_Augmentation_for_Generalizable_Deepfake_CVPR_2024_paper.pdf);
- [`lsda_detector_example.py`](lsda_detector_example.py) được cung cấp trong thư mục này.

## Tổng quan pipeline

Repo giả định khuôn mặt đã được phát hiện, crop thành frame ảnh và lập manifest
ở bước preprocess của `fa_vit_remake`. Pipeline trong repo này bắt đầu từ các
frame và manifest đó:

```text
video FF++ / Celeb-DF-v2
          │
          ├─ detect + crop face + lấy frame (bước preprocess bên ngoài repo này)
          │
          ├─ FF++ train_pairs.csv ──> nhóm LSDA ──> train FA-ViT + LSDA
          │                                      └─> best.pt / last.pt
          │
          ├─ FF++ validation_frames.csv ─────────> chọn best.pt, early stopping
          │
          └─ Celeb-DF test_frames.csv ───────────> cross-dataset test một lần cuối
                                                    (AUC/accuracy cấp video)
```

Protocol đúng là train và chọn checkpoint hoàn toàn trên FF++, sau đó mới dùng
checkpoint đã cố định để test trên Celeb-DF-v2. Celeb-DF không được dùng để tính
gradient.

## Xử lý dữ liệu

### Manifest và vai trò của từng split

| Trường cấu hình | Cột bắt buộc | Vai trò |
| --- | --- | --- |
| `data.train_pairs` | `fake_path,real_path,method` | Tạo các group FF++ dùng cho backprop; `video_id,sample_index` có thể được giữ để truy vết |
| `data.validation_frames` | `path,label,video_id` | Validation FF++ cấp video để chọn checkpoint và early stopping |
| `data.ffpp_test_frames` | `path,label,video_id` | FF++ test độc lập cho `evaluate_ffpp.py` (không bắt buộc nếu truyền `--manifest`) |
| `data.celebdf_test_frames` | `path,label,video_id` | Cross-dataset evaluation trên Celeb-DF-v2 |

Đường dẫn ảnh trong manifest có thể là đường dẫn tuyệt đối hoặc tương đối với
`data.root`. Nhãn nhị phân dùng `0 = real`, `1 = fake`.

### Tạo group cho LSDA

`GroupedForgeryDataset` nhóm các dòng theo `real_path`. Một mẫu train có thứ tự
cố định:

```text
[real, Deepfakes, Face2Face, FaceSwap, NeuralTextures]
  0         1          2         3              4       <- domain label
  0         1          1         1              1       <- binary label
```

Chỉ các group có đủ bốn phương pháp giả mạo trong `model.forgery_methods` được
giữ lại; số group thiếu bị loại được in khi bắt đầu train. Mỗi lần đọc một group,
dataset chọn ngẫu nhiên một fake frame của từng method. Với cấu hình mặc định
`group_batch_size: 4`, tensor đầu vào có dạng `[4, 5, 3, 224, 224]`, tức 20 ảnh
cho mỗi optimizer step. Batch phải có ít nhất hai group để việc tìm hard example
theo tâm miền trong LSDA có ý nghĩa.

### Augmentation ảnh

Trong cùng một group, real và bốn fake dùng chung crop và horizontal flip để giữ
căn chỉnh hình học. Ảnh được crop và resize về `image_size`; các biến đổi
photometric/codec sau đó được lấy mẫu độc lập cho từng ảnh, gồm color jitter,
grayscale, Gaussian blur, hạ rồi nâng độ phân giải và JPEG recompression. Cuối
cùng ảnh được chuyển thành tensor và normalize từng kênh bằng mean/std `0.5`
(miền giá trị xấp xỉ `[-1, 1]`).

Các augmentation mạnh chỉ được bật cho `train_pairs`. Validation và test dùng
`FaceTransform` sạch, không bật flip, color jitter hay degradation.

> **Lưu ý tránh leakage:** cấu hình mẫu hiện chưa khai báo
> `data.validation_frames`. Trong trường hợp đó, `train.py` sẽ fallback sang
> `celebdf_test_frames`, đo Celeb-DF sau mỗi epoch và dùng AUC này để chọn
> `best.pt`/early stopping. Ảnh Celeb-DF vẫn không backprop, nhưng kết quả test đã
> ảnh hưởng đến model selection nên không còn là cross-test độc lập. Hãy thêm một
> manifest validation FF++ được tách theo **video nguồn** trước khi chạy thí nghiệm
> chính thức.

## Kiến trúc mô hình

### Sơ đồ khi huấn luyện

```text
group [real + 4 fake domains]
              │
              ▼
┌──────────────────────────────────────────────────────────────────┐
│ Shared FA-ViT encoder                                            │
│ ViT patch embedding → GAM trong attention blocks                 │
│ spatial CNN → LAM injection tại layers [0, 3, 6]                 │
└──────────────────────────────┬───────────────────────────────────┘
                               │ CLS token + patch map [D, 14, 14]
                  ┌────────────┴─────────────┐
                  ▼                          ▼
       ┌───────────────────┐       ┌─────────────────────────┐
       │ Student adapter   │       │ Domain teacher adapters │
       │ (dùng cả train và │       │ 1 real + 4 fake         │
       │ inference)        │       └────────────┬────────────┘
       └─────────┬─────────┘                    │ fake maps
                 │                    ┌─────────▼──────────┐
       mean pool patch map            │ LSDA augmentation │
                 + CLS                │ within + cross    │
                 │                    │ domain + fusion   │
        ┌────────▼────────┐           └─────────┬──────────┘
        │ Feature fusion  │                     │ target map
        └──────┬───────┬──┘            MSE distillation
               │       │                         │
       binary head     FAL              domain classification
       real / fake     features          + student GRL branch
```

### Các khối chính

1. **Shared FA-ViT encoder:** backbone
   `vit_base_patch16_224.augreg_in21k` tạo CLS token và patch map `14 x 14`.
   Global Adaptive Module (GAM) được chèn vào attention; nhánh spatial CNN và
   Local Adaptive Module (LAM) bổ sung đặc trưng cục bộ tại ba layer cấu hình.
2. **Student branch:** `ResidualLatentAdapter` biến đổi patch map bằng residual
   convolution có scale học được. Patch map sau adapter được mean-pool, ghép với
   CLS token rồi qua `feature_fusion` và binary head.
3. **Domain teacher branches:** một adapter cho real và một adapter riêng cho mỗi
   fake method tạo biểu diễn latent theo miền. Đây là các nhánh auxiliary nhẹ,
   không phải các mạng teacher pretrained độc lập.
4. **LSDA:** trên các fake teacher maps, Within-Domain (WD) chọn ngẫu nhiên một
   trong hard interpolation, centrifugal extrapolation, Gaussian perturbation,
   affine rotation hoặc difference transform. Cross-Domain (CD) Mixup trộn cặp
   fake domain bằng hệ số lấy từ phân phối Beta. WD, CD và latent gốc được fusion
   thành comprehensive target cho student.
5. **Domain invariance:** domain classifier học phân biệt real và từng fake
   method từ teacher maps. Một classifier khác nhận student fake features qua
   Gradient Reversal Layer (GRL); gradient đảo chiều buộc student giảm thông tin
   đặc thù của từng phương pháp giả mạo.

Khác với detector LSDA gốc, phiên bản này không chạy bốn EfficientNet teacher,
một ArcFace teacher và một student EfficientNet độc lập. Nó dùng một FA-ViT
encoder chung cùng các latent adapter nhẹ. Đây là thiết kế tích hợp LSDA vào
FA-ViT, không phải reproduction nguyên xi detector LSDA gốc.

### Sơ đồ khi inference

```text
frame ảnh → shared FA-ViT → student adapter → CLS + pooled patch fusion
                                                    │
                                                    ▼
                                             binary head → P(fake)
```

Teacher adapters, LSDA augmenter, domain classifiers và các auxiliary loss không
được gọi khi inference.

## Phương pháp huấn luyện

### Hàm loss

Với mỗi grouped batch, model tối ưu tổng loss:

```text
L = λbin·Lbalanced-CE
  + λdomain·Ldomain-CE
  + λinvariance·Linvariance-CE
  + λdistill·(MSEreal + MSEfake)
  + λFAL·LFAL
```

- `Lbalanced-CE` lấy trung bình loss của lớp real và fake với trọng số ngang nhau,
  tránh tỷ lệ một real/bốn fake làm lớp fake lấn át;
- `Ldomain-CE` giám sát các domain teacher bằng năm nhãn miền;
- `Linvariance-CE` đi qua GRL: classifier cố nhận biết bốn fake method, trong khi
  student encoder nhận gradient ngược để học đặc trưng bất biến theo method;
- `MSEreal` distill real teacher map sang student real map; `MSEfake` distill
  comprehensive LSDA target sang student fake maps;
- `LFAL` dùng vector trọng số lớp real của binary head làm prototype, kéo đặc
  trưng real lại gần prototype và đẩy fake ra xa theo cosine margin.

Trọng số thực tế lấy từ mục `loss` trong YAML, không mặc định đồng nhất với hệ số
trong paper. Cấu hình hiện tại dùng `λbin=1.0`, `λdomain=0.25`,
`λinvariance=0.1`, `λdistill=0.5` và `λFAL=0.25` sau warmup. Các loss LSDA,
invariance và FAL được ramp dần để detector học bài toán real/fake ổn định trước
khi nhận đầy đủ các ràng buộc auxiliary.

### Một epoch train

1. Dataloader shuffle các group và bỏ batch cuối nếu không đủ kích thước.
2. `forward_group` chạy encoder chung một lần cho toàn bộ real/fake trong batch,
   sau đó tách sang student và teacher/LSDA branches.
3. Tính năm thành phần loss, cộng theo trọng số của epoch rồi backprop.
4. Nếu chạy CUDA, AMP được bật theo cấu hình; gradient được clip bởi
   `max_grad_norm` trước optimizer step.
5. AdamW dùng learning rate nhỏ hơn cho backbone (`backbone_lr_multiplier`),
   warmup rồi cosine decay. Mặc định backbone chủ yếu bị freeze, chỉ train GAM,
   LayerNorm, CLS token, các module mới và số block cuối cấu hình bởi
   `unfreeze_last_blocks`.

### Validation, checkpoint và cross-test

Sau mỗi epoch, xác suất fake của các frame cùng `video_id` được lấy trung bình để
tạo xác suất cấp video; từ đó tính `video_auc` và `video_accuracy` (ngưỡng `0.5`).
`best.pt` được cập nhật khi validation video AUC tăng, `last.pt` luôn lưu trạng
thái mới nhất và early stopping dựa trên cùng validation AUC.

Khi `data.validation_frames` là FF++ validation hợp lệ, Celeb-DF loader tách biệt
và chỉ được chạy một lần sau khi model selection kết thúc. Script nạp lại
`best.pt`, đánh giá Celeb-DF và ghi `celebdf_test_metrics` vào checkpoint cũng như
`history.jsonl`.

## Cài đặt và train

```powershell
cd fa_vit_lsda
pip install -e ".[test]"
python train.py `
  --config configs/favit_lsda_ffpp_c23_celebdf.yaml `
  --device cuda:0
```

Khởi tạo các layer tương thích từ checkpoint `fa_vit_remake`:

```powershell
python train.py `
  --config configs/favit_lsda_ffpp_c23_celebdf.yaml `
  --init-favit ..\fa_vit_remake\outputs\favit_ffpp_c23\best.pt `
  --device cuda:0
```

Resume `favit_lsda`:

```powershell
python train.py `
  --config configs/favit_lsda_ffpp_c23_celebdf.yaml `
  --resume outputs\favit_lsda_ffpp_c23\last.pt `
  --device cuda:0
```

## Evaluate

Inference chỉ giữ shared FA-ViT, student adapter, feature fusion và binary head;
latent augmentation/domain teachers không chạy.

Đánh giá FF++ ở video level:

```powershell
python evaluate_ffpp.py `
  --config configs/favit_lsda_ffpp_c23_celebdf.yaml `
  --checkpoint outputs\favit_lsda_ffpp_c23\best.pt `
  --manifest E:\path\to\ffpp_c23_test_frames.csv `
  --level video `
  --device cuda:0
```

Nếu không truyền `--manifest`, script lần lượt tìm `data.ffpp_test_frames` rồi
`data.validation_frames` trong config.

Đánh giá Celeb-DF-v2 ở frame level (mặc định dùng
`data.celebdf_test_frames`):

```powershell
python evaluate_celebdf.py `
  --config configs/favit_lsda_ffpp_c23_celebdf.yaml `
  --checkpoint outputs\favit_lsda_ffpp_c23\best.pt `
  --level frame `
  --device cuda:0
```

`--level` nhận `frame` hoặc `video` và mặc định là `video`. Cả hai script trả về
JSON gồm `accuracy`, `f1_score`, `precision`, `recall` và `auc`. Ở video level,
xác suất fake của các frame cùng `video_id` được lấy trung bình trước khi tính
metric. AUC dùng xác suất liên tục; bốn metric còn lại dùng `--threshold 0.5`
(có thể thay đổi), với fake (`label=1`) là positive class.

## Test và ablation

```powershell
pytest
```

Ablation tối thiểu nên gồm:

1. FA-ViT: CE + FAL.
2. FA-ViT + domain adapters/domain loss.
3. FA-ViT + WD.
4. FA-ViT + CD.
5. `favit_lsda`: WD + CD + domain + distillation + FAL.

Đây là một kiến trúc nghiên cứu mới; mức cải thiện cần được xác nhận bằng cùng
seed, split, số frame và checkpoint-selection protocol.

## Protocol thí nghiệm cross-dataset

Để kết quả phản ánh khả năng tổng quát hóa thay vì target-domain leakage:

1. Tách FF++ thành train/validation theo **video nguồn**, không theo frame.
2. Dùng duy nhất FF++ train để backprop và FF++ validation để chọn `best.pt`.
3. Không điều chỉnh hyperparameter, epoch hay checkpoint theo Celeb-DF test AUC.
4. Cố định checkpoint rồi mới test trên Celeb-DF-v2, DFDC hoặc WildDeepfake.
5. Giữ cùng split, số frame và protocol cấp video giữa các phương pháp; chạy ít
   nhất ba seed và báo cáo mean/std AUC.

Checkpoint format cũ không thể `--resume` với kiến trúc mới; có thể dùng checkpoint
đó qua `--init-favit` để nạp các tensor tương thích rồi train một run mới.

Ablation đề xuất, mỗi cấu hình chạy ít nhất ba seed:

1. baseline cũ;
2. thêm class-balanced CE + image degradation augmentation;
3. thêm residual-gated LSDA + auxiliary ramp;
4. thêm student domain invariance;
5. full recipe với AdamW/cosine và hai backbone block cuối được fine-tune.

Không kết luận từ việc AUC tiếp tục tăng sau một epoch cụ thể. Tiêu chí quan trọng
là mean/std AUC trên target chưa thấy, với checkpoint chỉ được chọn từ source
validation.
