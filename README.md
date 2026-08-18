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
          │                                        (AUC cấp ảnh, bắt buộc)
          │
          └─ sau khi best.pt cố định, train.py tự chạy post-selection test:
             ├─ FF++ ffpp_test_frames.csv ────────> test nguồn cùng miền
             └─ Celeb-DF celebdf_test_frames.csv ─> cross-dataset test
                (cả hai đều cấp ảnh, một lần duy nhất, không ảnh hưởng selection)
```

Protocol đúng là train và chọn checkpoint hoàn toàn trên FF++, sau đó mới dùng
checkpoint đã cố định để test trên Celeb-DF-v2. Celeb-DF không được dùng để tính
gradient.

## Xử lý dữ liệu

### Manifest và vai trò của từng split

| Trường cấu hình | Cột bắt buộc | Vai trò |
| --- | --- | --- |
| `data.train_pairs` | `fake_path,real_path,method` | Tạo các group FF++ dùng cho backprop; `video_id,sample_index` có thể được giữ để truy vết |
| `data.validation_frames` | `path,label,video_id` | **Bắt buộc.** `train.py` báo lỗi và dừng ngay nếu thiếu trường này. Đây là tín hiệu chọn checkpoint duy nhất: mỗi epoch, `best.pt` được cập nhật theo AUC **cấp frame/image** trên chính manifest FF++ nguồn này (`evaluate_at_level(..., level="frame")`), không dùng Celeb-DF làm fallback |
| `data.ffpp_test_frames` | `path,label,video_id` | Test FF++ độc lập, chỉ chạy **một lần sau khi** model selection đã kết thúc (post-selection). Không ảnh hưởng tới việc chọn `best.pt`. Dùng cho `evaluate_ffpp.py` (không bắt buộc nếu truyền `--manifest`) |
| `data.celebdf_test_frames` | `path,label,video_id` | Cross-dataset evaluation trên Celeb-DF-v2, cũng chỉ chạy **một lần sau khi** model selection kết thúc (post-selection). Không bao giờ được dùng để chọn checkpoint |

Đường dẫn ảnh trong manifest có thể là đường dẫn tuyệt đối hoặc tương đối với
`data.root`. Nhãn nhị phân dùng `0 = real`, `1 = fake`.

> **Thay đổi hành vi:** ảnh nguồn không ở mode `RGB` (grayscale, RGBA, CMYK…)
> nay bị `FaceTransform` từ chối bằng `ValueError` kèm đường dẫn ảnh, thay vì
> được tự động convert sang RGB như trước. Lý do: artifact SRM/FFT/wavelet được
> tính trên đúng ba kênh của ảnh nguồn, nên một ảnh bị convert ngầm sẽ tạo
> artifact không phản ánh dữ liệu thật. Hãy convert sang RGB ở bước preprocess.

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

Nếu `model.artifact_mode` khác `rgb`, `FaceTransform` xây thêm tensor CNN artifact
**sau khi** đã áp dụng toàn bộ augmentation/degradation lên RGB (`build_cnn_input`
nhận RGB đã augment, không phải ảnh gốc). Vì vậy artifact SRM/FFT/wavelet luôn
phản ánh đúng ảnh mà nhánh CNN quan sát, kể cả khi bị nén JPEG hay hạ độ phân
giải. Xem chi tiết widths và nhánh CNN ở [Artifact CNN và late fusion](#artifact-cnn-và-late-fusion).

> **Bắt buộc `data.validation_frames`:** `train.py` yêu cầu trường này và báo lỗi
> ngay khi thiếu (`ValueError`) — không còn fallback sang `celebdf_test_frames` để
> chọn checkpoint. Đây là thay đổi có chủ đích để loại bỏ leakage: model selection
> chỉ được phép dựa trên AUC **cấp frame/image** đo trên manifest FF++ nguồn này.
> Hãy dùng một manifest validation FF++ được tách theo **video nguồn**.
> `data.ffpp_test_frames` và `data.celebdf_test_frames` chỉ được đánh giá **một
> lần, sau khi** `best.pt` đã cố định — không bao giờ ảnh hưởng đến việc chọn
> checkpoint.

## Kiến trúc mô hình

### Sơ đồ khi huấn luyện

```text
group [real + 4 fake domains]
              │
              ▼
┌──────────────────────────────────────────────────────────────────┐
│ Shared FA-ViT encoder (chỉ nhận RGB 3 kênh)                      │
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
        │ vit_feature_    │                     │ target map
        │ fusion          │            MSE distillation (real + fake)
        └────────┬────────┘                     │
                 │ vit_features        domain classification (teacher maps)
                 ├────────────────────► student GRL → invariance classifier
                 │
                 │   RGB đã augment (xem Augmentation ảnh)
                 │              │
                 │              ▼
                 │      build_cnn_input(mode)
                 │              │
                 │              ▼
                 │      ┌──────────────┐
                 │      │ ArtifactCNN  │  (độc lập — không chạm
                 │      └──────┬───────┘   GAM/LAM/LSDA/teacher)
                 │             │ cnn_features
                 └──────┬──────┘
                        ▼
                  late_fusion (concat + MLP)
                        │
               ┌────────┴────────┐
               ▼                 ▼
         binary head          FAL features
         real / fake
```

### Các khối chính

1. **Shared FA-ViT encoder:** backbone
   `vit_base_patch16_224.augreg_in21k` tạo CLS token và patch map `14 x 14`.
   Global Adaptive Module (GAM) được chèn vào attention; nhánh spatial CNN và
   Local Adaptive Module (LAM) bổ sung đặc trưng cục bộ tại ba layer cấu hình.
2. **Student branch:** `ResidualLatentAdapter` biến đổi patch map bằng residual
   convolution có scale học được. Patch map sau adapter được mean-pool, ghép với
   CLS token rồi qua `vit_feature_fusion`. Vector này chưa vào head ngay: nó
   được ghép với đặc trưng của `ArtifactCNN` rồi đi qua `late_fusion` trước
   binary head (xem mục 6 bên dưới).
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
6. **Artifact CNN và late fusion:** một `ArtifactCNN` độc lập (stem + hai block
   conv/BN/GELU + global average pool + linear projection) chạy song song với
   FA-ViT trên tensor RGB-plus-artifact (`model.artifact_mode`). Vector đặc trưng
   CNN được nối (`concat`) với CLS+patch feature của FA-ViT rồi qua `late_fusion`
   (MLP) trước khi vào binary head và FAL. Nhánh này hoàn toàn tách biệt khỏi
   latent adapter/teacher/LSDA của FA-ViT — augmentation latent-space và distill
   chỉ tác động lên nhánh ViT, không đụng tới `ArtifactCNN`.

Khác với detector LSDA gốc, phiên bản này không chạy bốn EfficientNet teacher,
một ArcFace teacher và một student EfficientNet độc lập. Nó dùng một FA-ViT
encoder chung cùng các latent adapter nhẹ. Đây là thiết kế tích hợp LSDA vào
FA-ViT, không phải reproduction nguyên xi detector LSDA gốc.

### Artifact CNN và late fusion

`model.artifact_mode` chọn tập artifact ghép vào RGB trước khi đưa vào
`ArtifactCNN`; `model.cnn_in_channels` phải khớp đúng width tương ứng, được
kiểm tra bởi `validate_model_config`/`build_model_from_config` (raise
`ValueError` nếu lệch) trước khi model được khởi tạo:

| `artifact_mode` | Artifact ghép thêm | `cnn_in_channels` |
| --- | --- | --- |
| `rgb` | không có | `3` |
| `rgb_srm` | SRM | `6` |
| `rgb_fft` | FFT log-magnitude | `6` |
| `rgb_wavelet` | Haar wavelet detail | `6` |
| `rgb_srm_fft` | SRM + FFT | `9` |
| `rgb_srm_wavelet` | SRM + wavelet | `9` |

Mỗi artifact được tính từ RGB **đã augment** (xem
[Augmentation ảnh](#augmentation-ảnh)), normalize về `[-1, 1]` theo từng ảnh rồi
`concat` theo kênh sau RGB gốc. Tensor RGB-plus-artifact này chỉ được
`ArtifactCNN` tiêu thụ; nhánh FA-ViT/GAM/LAM/LSDA phía trên vẫn chỉ nhận RGB ba
kênh như cũ.

### Sơ đồ khi inference

```text
frame ảnh (RGB đã augment/chuẩn hoá)
        │
        ├────────────────────────────┐
        ▼                            ▼
shared FA-ViT (RGB 3 kênh)   build_cnn_input(mode) → RGB+artifact
        │                            │
student adapter                ArtifactCNN
        │                            │
CLS + pooled patch feature     CNN feature vector
        └──────────────┬─────────────┘
                        ▼
                  late_fusion (concat + MLP)
                        │
                        ▼
                 binary head → P(fake)
```

Teacher adapters, LSDA augmenter, domain classifiers và các auxiliary loss
không được gọi khi inference — chỉ hai nhánh RGB FA-ViT và artifact CNN ở trên
cùng `late_fusion`/binary head chạy.

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

`data.validation_frames` là **bắt buộc**; `train.py` raise `ValueError` ngay khi
config thiếu trường này, không còn fallback sang Celeb-DF. Sau mỗi epoch, model
được đánh giá trên chính manifest này ở **cấp frame/image**
(`evaluate_at_level(..., level="frame")`) — không aggregate theo `video_id`.
`best.pt` được cập nhật khi AUC frame-level trên `validation_frames` tăng,
`last.pt` luôn lưu trạng thái mới nhất và early stopping dựa trên cùng AUC này.
Đây là tín hiệu chọn checkpoint duy nhất; Celeb-DF không bao giờ tham gia.

Sau khi vòng lặp train/early-stopping kết thúc (tức `best.pt` đã cố định), nếu
config có khai báo `data.ffpp_test_frames` và/hoặc `data.celebdf_test_frames`,
`train.py` nạp lại `best.pt` và đánh giá **một lần duy nhất** trên các manifest
này, ghi kết quả vào checkpoint (`ffpp_test_metrics`/`celebdf_test_metrics`) và
`history.jsonl`. Cả hai đều là test post-selection thuần tuý — không backprop,
không ảnh hưởng tới việc chọn checkpoint hay early stopping. Cả hai cũng chạy ở
**cấp ảnh** (`evaluate_at_level(..., level="frame")`), giống hệt tín hiệu chọn
checkpoint, để sáu experiment được so sánh trên cùng một thang đo image-level.
Nếu cần số liệu cấp video, chạy `evaluate_ffpp.py` / `evaluate_celebdf.py` với
`--level video` trên `best.pt`.

## Cài đặt và train

```powershell
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

### Sáu thí nghiệm CNN artifact có kiểm soát

`configs/favit_lsda_cnn_*.yaml` là sáu cấu hình dùng chung seed, optimizer,
schedule, augmentation, `image_size` và manifest (`train_pairs`,
`validation_frames`, `ffpp_test_frames`, `celebdf_test_frames`); chỉ
`model.artifact_mode`, `model.cnn_in_channels` và `output_dir` thay đổi giữa các
file, để so sánh riêng phần đóng góp của từng loại artifact:

| Config | `artifact_mode` | `cnn_in_channels` | `output_dir` |
| --- | --- | --- | --- |
| `configs/favit_lsda_cnn_rgb.yaml` | `rgb` | `3` | `outputs/favit_lsda_cnn_rgb` |
| `configs/favit_lsda_cnn_rgb_srm.yaml` | `rgb_srm` | `6` | `outputs/favit_lsda_cnn_rgb_srm` |
| `configs/favit_lsda_cnn_rgb_fft.yaml` | `rgb_fft` | `6` | `outputs/favit_lsda_cnn_rgb_fft` |
| `configs/favit_lsda_cnn_rgb_wavelet.yaml` | `rgb_wavelet` | `6` | `outputs/favit_lsda_cnn_rgb_wavelet` |
| `configs/favit_lsda_cnn_rgb_srm_fft.yaml` | `rgb_srm_fft` | `9` | `outputs/favit_lsda_cnn_rgb_srm_fft` |
| `configs/favit_lsda_cnn_rgb_srm_wavelet.yaml` | `rgb_srm_wavelet` | `9` | `outputs/favit_lsda_cnn_rgb_srm_wavelet` |

Train một thí nghiệm CNN artifact, khởi tạo từ checkpoint `fa_vit_remake`:

```powershell
python train.py `
  --config configs/favit_lsda_cnn_rgb_srm_fft.yaml `
  --init-favit ..\fa_vit_remake\outputs\favit_ffpp_c23\best.pt `
  --device cuda:0
```

### Chạy lần lượt từng case

Mỗi lệnh dưới đây ứng với một trong sáu config ở bảng trên, dùng chung
`--init-favit`. Sáu case độc lập với nhau — chạy theo thứ tự bất kỳ, chạy lại
một case không ảnh hưởng các case còn lại vì mỗi case ghi log/checkpoint/
`history.jsonl` vào `output_dir` riêng.

```powershell
# 1) rgb
python train.py --config configs/favit_lsda_cnn_rgb.yaml --init-favit ..\fa_vit_remake\outputs\favit_ffpp_c23\best.pt

# 2) rgb_srm
python train.py --config configs/favit_lsda_cnn_rgb_srm.yaml --init-favit ..\fa_vit_remake\outputs\favit_ffpp_c23\best.pt

# 3) rgb_fft
python train.py --config configs/favit_lsda_cnn_rgb_fft.yaml --init-favit ..\fa_vit_remake\outputs\favit_ffpp_c23\best.pt

# 4) rgb_wavelet
python train.py --config configs/favit_lsda_cnn_rgb_wavelet.yaml --init-favit ..\fa_vit_remake\outputs\favit_ffpp_c23\best.pt

# 5) rgb_srm_fft
python train.py --config configs/favit_lsda_cnn_rgb_srm_fft.yaml --init-favit ..\fa_vit_remake\outputs\favit_ffpp_c23\best.pt

# 6) rgb_srm_wavelet
python train.py --config configs/favit_lsda_cnn_rgb_srm_wavelet.yaml --init-favit ..\fa_vit_remake\outputs\favit_ffpp_c23\best.pt
```

Mỗi lệnh train xong là đủ số liệu so sánh: `outputs/favit_lsda_cnn_<mode>/best.pt`
và `history.jsonl` đã có sẵn `validation`, `ffpp_test_metrics`,
`celebdf_test_metrics` — cả ba đều ở cấp frame (xem
[Validation, checkpoint và cross-test](#validation-checkpoint-và-cross-test)).
Không cần chạy thêm `evaluate_ffpp.py`/`evaluate_celebdf.py`, trừ khi muốn số
liệu cấp video hoặc đánh giá lại trên manifest khác — ví dụ với case
`rgb_srm_fft`:

```powershell
python evaluate_ffpp.py --config configs/favit_lsda_cnn_rgb_srm_fft.yaml --checkpoint outputs\favit_lsda_cnn_rgb_srm_fft\best.pt --level video
python evaluate_celebdf.py --config configs/favit_lsda_cnn_rgb_srm_fft.yaml --checkpoint outputs\favit_lsda_cnn_rgb_srm_fft\best.pt --level video
```

Đổi `rgb_srm_fft` (trong `--config` và `--checkpoint`) thành tên case tương
ứng cho năm case còn lại. Chi tiết `--level`/định dạng kết quả xem
[Evaluate](#evaluate).

`--init-favit` chỉ copy các tensor FA-ViT trùng tên **và** trùng shape từ
checkpoint nguồn (`load_favit_initialization`); binary head luôn bị loại trừ,
còn `ArtifactCNN`, `late_fusion` và head được khởi tạo mới hoàn toàn — checkpoint
`fa_vit_remake` không có các nhánh này nên không kiểm tra `artifact_mode`/
`cnn_in_channels` ở bước này.

`--resume` thì ngược lại: nghiêm ngặt hơn có chủ đích
(`validate_checkpoint_artifacts`). Nó từ chối checkpoint có `architecture` khác
`"favit_lsda_cnn"` (checkpoint cũ trước khi có nhánh CNN) và từ chối checkpoint
có `artifact_mode`/`cnn_in_channels` khác với config hiện tại — một checkpoint
train với `rgb_srm` không thể `--resume` dưới config `rgb_fft`. Với checkpoint
không tương thích, dùng `--init-favit` để bắt đầu run mới thay vì `--resume`.

## Evaluate

Inference chỉ giữ shared FA-ViT, student adapter, `ArtifactCNN`, `late_fusion`
và binary head; latent augmentation/domain teachers không chạy (xem
[Sơ đồ khi inference](#sơ-đồ-khi-inference)).

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
5. Giữ cùng split, số frame và cùng thang đo (mặc định là **cấp ảnh**, khớp tín
   hiệu chọn checkpoint — xem [Validation, checkpoint và cross-test](#validation-checkpoint-và-cross-test))
   giữa các phương pháp; chạy ít nhất ba seed và báo cáo mean/std AUC.

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
