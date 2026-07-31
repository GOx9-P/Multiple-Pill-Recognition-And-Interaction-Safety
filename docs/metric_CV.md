# Metrics for CV Modules
## 0. Quy Ước Chung

Với segmentation, một viên thuốc được xem là phát hiện đúng nếu prediction match với ground truth theo điều kiện:

```text
IoU(predicted_instance, ground_truth_instance) >= 0.5
```

Trong đó:

```text
TP = true positive  = dự đoán đúng
FP = false positive = dự đoán thừa / nhầm
FN = false negative = bỏ sót
```

## 1. Bộ Metric Chính

| Module | Metric chính | Mục đích |
|---|---|---|
| Segmentation | Instance Recall | Đo có bỏ sót viên thuốc không. |
| Segmentation | Merge Error Rate | Đo có gộp nhiều viên thành một instance không. |
| Attribute Recognition | Macro F1 | Đo chất lượng nhận diện shape, color, dosage form, score line. |
| Imprint OCR | Candidate Recall@5 | Đo imprint đúng có còn nằm trong top 5 candidate không. |
| Imprint OCR | Character Error Rate | Đo OCR sai bao nhiêu ký tự. |
| CV Output | Valid JSON Rate | Đo output CV có đúng schema để module sau đọc không. |

## 2. Cách Tính Từng Metric

### 2.1. Instance Recall

Dùng cho module segmentation.

```text
Instance Recall = số viên thuốc phát hiện đúng / tổng số viên thuốc thật
```

Viết theo TP/FN:

```text
Instance Recall = TP / (TP + FN)
```

Ví dụ:

```text
Ảnh test có 100 viên thuốc thật.
Model phát hiện đúng 94 viên, bỏ sót 6 viên.

Instance Recall = 94 / 100 = 0.94 = 94%
```

Cách đọc:

```text
Recall càng cao càng tốt.
Với bài toán này, recall quan trọng hơn precision vì bỏ sót thuốc có thể làm mất cảnh báo tương tác thuốc.
```

### 2.2. Merge Error Rate

Dùng cho module segmentation trong ảnh có nhiều viên thuốc.

```text
Merge Error Rate = số ảnh có lỗi gộp viên / tổng số ảnh multi-pill
```

Một ảnh được tính là có lỗi gộp nếu model tạo ra một instance/mask bao phủ từ 2 viên thuốc thật trở lên.

Ví dụ:

```text
Tập benchmark có 40 ảnh multi-pill.
Trong đó 3 ảnh có lỗi gộp 2 viên thành 1 instance.

Merge Error Rate = 3 / 40 = 0.075 = 7.5%
```

Cách đọc:

```text
Merge Error Rate càng thấp càng tốt.
Lỗi này nguy hiểm vì làm sai số lượng viên thuốc và khiến OCR/attribute bị trộn giữa nhiều viên.
```

### 2.3. Macro F1

Dùng cho module attribute recognition.

Áp dụng riêng cho từng thuộc tính:

```text
shape_macro_f1
color_macro_f1
dosage_form_macro_f1
score_line_macro_f1
```

Với mỗi class:

```text
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2 * Precision * Recall / (Precision + Recall)
```

Sau đó lấy trung bình F1 của tất cả class:

```text
Macro F1 = (F1_class_1 + F1_class_2 + ... + F1_class_n) / n
```

Ví dụ với shape có 4 class:

```text
F1_round   = 0.92
F1_oval    = 0.86
F1_capsule = 0.81
F1_square  = 0.71

Shape Macro F1 = (0.92 + 0.86 + 0.81 + 0.71) / 4
               = 0.825 = 82.5%
```

Cách đọc:

```text
Macro F1 càng cao càng tốt.
Nó tốt hơn accuracy khi dataset mất cân bằng, vì class ít dữ liệu vẫn được tính ngang với class nhiều dữ liệu.
```

Ghi chú cho color:

```text
Color là multi-label.
Một viên có thể có nhiều màu, ví dụ blue-white.
Khi tính Macro F1 cho color, mỗi màu được xem như một nhãn nhị phân riêng:
white: có/không
blue: có/không
red: có/không
...
Sau đó lấy trung bình F1 của tất cả màu.
```

### 2.4. Candidate Recall@5

Dùng cho module imprint OCR sau bước normalize/sửa lỗi OCR.

```text
Candidate Recall@5 = số mẫu có imprint đúng nằm trong top 5 candidate / tổng số mẫu có imprint ground truth
```

Ví dụ:

```text
Có 100 viên thuốc có imprint thật.
Sau OCR + sửa lỗi, 87 viên có imprint đúng nằm trong top 5 candidate.

Candidate Recall@5 = 87 / 100 = 0.87 = 87%
```

Ví dụ candidate:

```json
[
  {"text": "A01", "score": 0.91},
  {"text": "AO1", "score": 0.83},
  {"text": "A0I", "score": 0.76},
  {"text": "AOI", "score": 0.68}
]
```

Nếu ground truth là `A01`, mẫu này được tính là đúng cho `Recall@5` vì `A01` nằm trong top 5.

Cách đọc:

```text
Candidate Recall@5 càng cao càng tốt.
Metric này phù hợp hơn exact match vì hệ thống không bắt OCR phải chắc chắn 100%, chỉ cần không làm mất đáp án đúng trước bước retrieval.
```

### 2.5. Character Error Rate

Dùng cho module imprint OCR.

CER đo mức sai khác giữa chuỗi OCR và chuỗi imprint thật theo từng ký tự.

```text
CER = LevenshteinDistance(predicted_text, ground_truth_text) / số ký tự của ground_truth_text
```

Trong đó Levenshtein Distance là số thao tác ít nhất để biến chuỗi dự đoán thành chuỗi đúng:

```text
substitution = thay ký tự
insertion    = thêm ký tự
deletion     = xóa ký tự
```

Ví dụ:

```text
Ground truth: A01
OCR output:   AO1

Sai 1 ký tự: O thay vì 0
CER = 1 / 3 = 0.333 = 33.3%
```

Ví dụ khác:

```text
Ground truth: A01
OCR output:   A001

Thừa 1 ký tự 0
CER = 1 / 3 = 33.3%
```

Cách đọc:

```text
CER càng thấp càng tốt.
Nó cho biết OCR sai nặng hay nhẹ, thay vì chỉ tính đúng/sai toàn bộ chuỗi.
```

### 2.6. Valid JSON Rate

Dùng cho output cuối của CV pipeline.

```text
Valid JSON Rate = số output đúng schema / tổng số output
```

Ví dụ:

```text
Pipeline xử lý 200 ảnh.
Có 198 output đúng schema JSON.
Có 2 output lỗi format hoặc thiếu field bắt buộc.

Valid JSON Rate = 198 / 200 = 0.99 = 99%
```

Cách đọc:

```text
Valid JSON Rate càng cao càng tốt.
Metric này không đo model thông minh hay không, nhưng đo khả năng tích hợp ổn định với module sau.
```

## 3. Target MVP Tham Khảo

| Module | Metric | Target tham khảo |
|---|---|---|
| Segmentation | Instance Recall | >= 95% |
| Segmentation | Merge Error Rate | <= 5% |
| Attribute - Shape | Macro F1 | >= 85% |
| Attribute - Color | Macro F1 | >= 80% |
| Attribute - Dosage Form | Macro F1 | >= 90% |
| Attribute - Score Line | Macro F1 | >= 75% |
| OCR | Candidate Recall@5 | >= 85% |
| OCR | Character Error Rate | <= 15% |
| CV Output | Valid JSON Rate | >= 99% |

## 4. Metric Phụ Chỉ Dùng Khi Debug

Các metric dưới đây không cần đưa vào bộ chính, chỉ dùng khi cần phân tích lỗi:

| Metric phụ | Khi nào dùng? |
|---|---|
| Mask mAP@50:95 | Khi phát hiện đúng số viên nhưng crop/mask vẫn xấu. |
| Confusion Matrix | Khi cần biết attribute nào hay bị nhầm với nhau. |
| Word Exact Match | Khi muốn biết OCR đọc đúng hoàn toàn chuỗi imprint bao nhiêu lần. |
| Field Missing Rate | Khi JSON hợp lệ nhưng hay thiếu field như `shape`, `color`, `imprint_candidates`. |
| False Positive Rate | Khi model hay nhận nhầm vật nền thành thuốc. |

## 5. Kết Luận

Bộ metric chính nên giữ gọn:

1. Segmentation: `Instance Recall`, `Merge Error Rate`
2. Attribute Recognition: `Macro F1`
3. Imprint OCR: `Candidate Recall@5`, `Character Error Rate`
4. CV Output: `Valid JSON Rate`

Như vậy hệ thống được đánh giá theo đúng các câu hỏi quan trọng:

1. Có phát hiện đủ viên thuốc không?
2. Có tách riêng từng viên không?
3. Có nhận diện đúng thuộc tính quan sát được không?
4. OCR có giữ lại imprint đúng trong candidate list không?
5. Output có đủ ổn định để module sau đọc không?
