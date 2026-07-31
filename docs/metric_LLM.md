# BÁO CÁO LỰA CHỌN 5 METRIC ĐÁNH GIÁ MODULE NHẬN DIỆN THUỐC

# 1. Identification Coverage và Precision của trạng thái IDENTIFIED
**Identification Coverage** đo tỷ lệ mẫu mà hệ thống có thể đưa ra kết luận `IDENTIFIED`.
**Precision** của trạng thái `IDENTIFIED` đo tỷ lệ kết luận đúng trong số những trường hợp hệ thống đã quyết định xác nhận thuốc.

coverage = {Số mẫu được trả IDENTIFIED} / {Tổng số mẫu cần nhận diện}

Precision_{IDENTIFIED} = {Số mẫu được IDENTIFIED đúng}/{Tổng số mẫu được trả IDENTIFIED}

Ví dụ :

```text
Có 100 mẫu thuốc thuộc database.
Hệ thống xác nhận được 80 mẫu.
20 mẫu còn lại là AMBIGUOUS hoặc UNKNOWN.
Có 50 mẫu được IDENTIFIED đúng
```
Khi đó: coverage = 80/100= 80%
        Precision_{IDENTIFIED} = 50/80
## 1.2. Ý nghĩa

Coverage thể hiện mức độ hữu dụng thực tế của hệ thống.

Một hệ thống có thể đạt Precision rất cao bằng cách chỉ xác nhận những trường hợp cực kỳ dễ và từ chối hầu hết các trường hợp còn lại.

Ví dụ:
```text
Precision_IDENTIFIED = 100%
Coverage = 10%
```

Điều này có nghĩa hệ thống luôn đúng khi xác nhận, nhưng chỉ nhận diện được 10 trong 100 mẫu.

Ngược lại:

```text
Precision_IDENTIFIED = 80%
Coverage = 100%
```

Hệ thống nhận diện được mọi mẫu nhưng tạo ra quá nhiều kết luận sai.

Do đó, Coverage phải luôn được xem cùng Precision.

## 1.3. Metric này trả lời câu hỏi gì?

> Trong thực tế, hệ thống có thể nhận diện chắc chắn được bao nhiêu phần trăm số thuốc đưa vào?
> Trong các thuốc được nhận diện chắc chắn thì có bao nhiêu cái được nhận diện đúng?

## 1.4. Vì sao cần dùng?

Precision cho biết trong các thuốc được nhận diện chắc chắn thì bao nhiêu cái đúng. Dùng precision vì trong bài toán này thì trường hợp đúng giả nó nguy hiểm hơn là sai giả.
Coverage giúp kiểm tra hệ thống có đang từ chối quá mức hay không.
Metric này đặc biệt hữu ích khi điều chỉnh:

```text
Ngưỡng Top-1
Ngưỡng margin
Ngưỡng OCR confidence
```

Khi tăng ngưỡng Safety Gate:

```text
Precision thường tăng
Coverage thường giảm
```

Khi giảm ngưỡng:

```text
Coverage thường tăng
Precision có thể giảm
```

Mục tiêu là tìm điểm cân bằng an toàn.

## 1.5. Mục tiêu đề xuất cho demo

Chỉ tính Coverage trên các thuốc:

* Có trong database.
* Có `cv_status = FEATURES_READY`.
* Có chất lượng OCR đủ dùng.

Mục tiêu:

Coverage ≥ 80%
Precision_IDENTIFIED ≥ 95%
Mục tiêu tốt: 98–100%

Không  tính các mẫu ảnh mờ hoặc thuốc ngoài database vào mục tiêu Coverage chính.

---

# 2. Top-1 Accuracy

## 2.1. Định nghĩa

Top-1 Accuracy đo tỷ lệ mẫu mà ứng viên được xếp ở vị trí đầu tiên đúng với thuốc thật.

Top1Accuracy = {số lần có thuốc thực đứng top 1} / {Tổng số mẫu thử hợp lệ}


Ví dụ:

```text
Có 50 lần thử hợp lệ.
Có 45 thuốc khi nhận diện thì đúng là nó đứng top 1.
```

Khi đó:
Top1Accuracy = 45/50 

## 2.2. Ý nghĩa

Metric này đo trực tiếp chất lượng của thuật toán nhận diện và xếp hạng thuốc.

Nó phản ánh khả năng kết hợp các thuộc tính:

```text
imprint
shape
color
dosage_form
score_line
logo_or_symbol
ocr_confidence
```
=> cho biết các kết quả của CV kết hợp đã hoàn hảo chưa.

Top-1 Accuracy được tính độc lập với việc Safety Gate có xác nhận kết quả hay không.

Ví dụ:

```text
Thuốc thật đứng Top-1
nhưng điểm chưa đủ cao
→ hệ thống vẫn có thể trả AMBIGUOUS.
```

Trong trường hợp đó, Top-1 Accuracy vẫn được tính là đúng, nhưng kết quả không được xác nhận.

## 2.3. Metric này trả lời câu hỏi gì?

Trong mỗi lần nhận một JSON từ phía CV, thuốc thật có được hệ thống xếp ở vị trí đầu tiên hay không?

Ở mức tổng thể, metric trả lời:

Trong 100 hoặc 1.000 lần nhận diện, bao nhiêu phần trăm số lần hệ thống đưa đúng thuốc lên đầu danh sách?

## 2.4. Vì sao cần dùng?

Đánh giá bộ xếp hạng. Cải tiến bộ xếp hạng
Đánh giấ gate safe đã tốt chưa hay bị cao hay thấp gì không.


#### ===> nên tính luôn cả tổng thể và cho riêng từng thuốc.

---

# 3. Unknown Rejection Rate

## 3.1. Định nghĩa

Unknown Rejection Rate đo khả năng hệ thống từ chối đúng các thuốc không tồn tại trong database.


UnknownRejectionRate = {Số thuốc ngoài database được trả UNKNOWN}/{Tổng số mẫu thuốc ngoài database}

Ví dụ:

```text
Database chỉ có 5 thuốc.
Có 20 mẫu thuốc không thuộc 5 thuốc này.
Hệ thống trả UNKNOWN đúng cho 18 mẫu.
```

Khi đó: UnknownRejectionRate = 18/20

## 3.2. Ý nghĩa

Thuật toán xếp hạng luôn có thể chọn một ứng viên Top-1, kể cả khi thuốc thật không tồn tại trong database.

Ví dụ database chỉ có:

```text
Acetaminophen
Ibuprofen
Aspirin
Naproxen
Diclofenac
```

Nhưng input thực tế là:

```text
Tramadol
```

Nếu không có cơ chế từ chối, hệ thống vẫn có thể chọn một trong năm thuốc có điểm gần nhất.

Unknown Rejection Rate đo khả năng hệ thống nhận ra:

```text
Không có ứng viên nào đủ phù hợp.
```

và trả về:

```text
UNKNOWN
```

## 3.3. Metric này trả lời câu hỏi gì?

> Khi gặp một thuốc không nằm trong database, hệ thống có biết từ chối thay vì nhận diện nhầm hay không?

## 3.4. Vì sao cần dùng?

Đây là metric rất quan trọng khi database demo chỉ chứa một số lượng thuốc giới hạn.

Nếu không đánh giá metric này, hệ thống có thể đạt Top-1 Accuracy cao trên năm thuốc đã biết nhưng lại nhận diện sai hầu hết thuốc bên ngoài database.

Tập test Unknown nên bao gồm:

* Thuốc có màu khác hoàn toàn.
* Thuốc có cùng màu với một thuốc trong database.
* Thuốc có cùng hình dạng.
* Thuốc có imprint gần giống.
* Input thiếu imprint.
* Input có các thuộc tính mâu thuẫn.

## 3.5. Mục tiêu đề xuất cho demo

```text
Unknown Rejection Rate ≥ 90%
Mục tiêu lý tưởng: 100%
```

---

# 4. ROC-AUC (Khả năng phân tách của mô hình lõi)

## 4.1. Định nghĩa

ROC-AUC (Area Under the Receiver Operating Characteristic Curve) đo lường khả năng phân tách của thuật toán xếp hạng lõi (XGBoost) giữa Mẫu Đúng (Label 1) và Mẫu Âm / Khó (Label 0), hoàn toàn độc lập với bất kỳ ngưỡng (threshold) Safety Gate nào.

`ROC-AUC = Xác suất mà mô hình gán điểm (P_match) cho một Mẫu Đúng ngẫu nhiên cao hơn một Mẫu Sai ngẫu nhiên.`

Ví dụ:
```text
AUC = 1.0 (100%): Mô hình phân tách hoàn hảo, điểm của thuốc đúng luôn cao hơn thuốc sai.
AUC = 0.5 (50%): Mô hình đoán bừa như tung đồng xu.
```

## 4.2. Ý nghĩa

Metric này đo lường trực tiếp "trí thông minh" bên trong của thuật toán XGBoost.
Trong khi `Top-1 Accuracy` chỉ quan tâm đến ứng viên đứng đầu, `ROC-AUC` quan tâm đến toàn bộ phổ điểm. 

Ví dụ, nếu thuốc thật được điểm 0.51 và thuốc giả được 0.49:
- Top-1 Accuracy vẫn tính là ĐÚNG (vì 0.51 > 0.49).
- Nhưng ROC-AUC sẽ bị sụt giảm vì khoảng cách quá mong manh, mô hình đang thiếu tự tin và rất dễ bị "lừa".

## 4.3. Metric này trả lời câu hỏi gì?

> Thuật toán cốt lõi có khả năng phân biệt rạch ròi, dứt khoát giữa thuốc đúng và những viên thuốc sai (nhưng có ngoại hình cực kỳ giống nhau) hay không?

## 4.4. Vì sao cần dùng?

Đây là vũ khí để bảo vệ tính hàn lâm và thuật toán của hệ thống:
- Nếu chỉ báo cáo `Top-1 Accuracy` hoặc `Coverage`, người nghe có thể cho rằng hệ thống chỉ hoạt động tốt ở bề nổi.
- Có `ROC-AUC`, chúng ta chứng minh được thuật toán XGBoost thực sự học được trọng số tối ưu từ dữ liệu, ép điểm các ca sai (Hard Negatives) xuống cực thấp và nâng điểm các ca đúng lên cực cao.

## 4.5. Mục tiêu đề xuất cho demo

```text
ROC-AUC ≥ 0.95 (95%)
Mục tiêu lý tưởng: 0.98 - 1.00
```

---


