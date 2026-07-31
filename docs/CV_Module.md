# MVP Architecture — Multi-Pill Recognition and Drug Interaction Warning

## 1. Objective and Safety Boundary

MVP xây dựng một hệ thống hỗ trợ nhận diện nhiều viên thuốc trong một ảnh và cảnh báo tương tác thuốc dựa trên dữ liệu có nguồn. Hệ thống được chia thành hai module lớn:

1. **Computer Vision Module (CV)**  
   Nhận ảnh đầu vào, phát hiện từng viên thuốc, tách mask/crop và trích xuất metadata thị giác: `shape`, `color`, `dosage_form`, `scoreline`, `imprint`, confidence và quality flags.

2. **Retrieval/RAG Module**  
   Nhận metadata từ CV, truy xuất ứng viên thuốc trong database có cấu trúc, xếp hạng candidate, chuẩn hóa hoạt chất, tra cứu tương tác thuốc và dùng LLM để trình bày báo cáo có căn cứ.

Ranh giới trách nhiệm:

```text
CV Module
    → Chỉ mô tả viên thuốc nhìn thấy như thế nào.

Retrieval/RAG Module
    → Dùng metadata thị giác để truy xuất thuốc có thể tương ứng,
      kiểm chứng candidate, lấy dữ liệu dược học và tạo báo cáo.
```

Nguyên tắc an toàn:

- CV không tự kết luận tên thuốc.
- LLM không tự đoán tên thuốc từ kiến thức nền.
- Việc định danh thuốc phải qua structured retrieval, không qua suy luận tự do.
- Hệ thống không buộc phải nhận diện mọi viên thuốc.
- Khi bằng chứng không đủ, trả `ambiguous`, `unknown`, `insufficient_visual_evidence` hoặc yêu cầu ảnh bổ sung.
- Chỉ thuốc `identified` mới được dùng cho kết luận DDI chắc chắn.
- Không tìm thấy DDI trong database không đồng nghĩa phối hợp thuốc an toàn.
- Kết quả chỉ có tính hỗ trợ, không thay thế bác sĩ hoặc dược sĩ.

Phạm vi MVP:

- Input chính là một ảnh RGB chứa một hoặc nhiều viên thuốc/viên nang đường uống.
- Ảnh có thể có ánh sáng không ổn định, viên tiếp xúc hoặc chồng lấp một phần.
- Người dùng có thể cung cấp thêm ảnh mặt còn lại, ảnh cận cảnh, country/market hoặc danh sách thuốc đang dùng.
- MVP chưa đảm bảo bao phủ mọi thuốc lưu hành toàn cầu.

---

## 2. System Architecture

```mermaid
%%{init: {
  "theme": "base",
  "flowchart": {
    "curve": "stepAfter",
    "htmlLabels": true,
    "nodeSpacing": 32,
    "rankSpacing": 48,
    "padding": 12
  },
  "themeVariables": {
    "fontFamily": "Arial, Helvetica, sans-serif",
    "fontSize": "13px",
    "primaryColor": "#FFFFFF",
    "primaryTextColor": "#111111",
    "primaryBorderColor": "#2F2F2F",
    "lineColor": "#2F2F2F",
    "secondaryColor": "#F5F5F5",
    "tertiaryColor": "#FAFAFA",
    "clusterBkg": "#FFFFFF",
    "clusterBorder": "#4A4A4A"
  }
}}%%

flowchart TB

    %% =========================================================
    %% GIAO DIỆN ĐẦU VÀO
    %% =========================================================


    %% =========================================================
    %% MODULE A — THỊ GIÁC MÁY TÍNH
    %% =========================================================
    subgraph CV["MODULE A — THỊ GIÁC MÁY TÍNH"]
        direction TB


        %% -----------------------------------------------------
        %% A1. ĐÁNH GIÁ CHẤT LƯỢNG ẢNH
        %% -----------------------------------------------------
        subgraph CV1["A1. Đánh giá chất lượng ảnh đầu vào"]
            direction LR

            IQ1["Đánh giá độ mờ<br/>và độ nét"]

            IQ2["Đánh giá ánh sáng<br/>thiếu sáng hoặc cháy sáng"]

            IQ3["Phát hiện phản sáng<br/>và vùng lóe sáng"]

            IQ4["Kiểm tra độ phân giải<br/>và kích thước viên trong ảnh"]

            IQG{"Chất lượng ảnh<br/>có đạt yêu cầu không?"}

            IQ1 --> IQG
            IQ2 --> IQG
            IQ3 --> IQG
            IQ4 --> IQG
        end


        %% -----------------------------------------------------
        %% A2. PHÁT HIỆN VÀ PHÂN VÙNG VIÊN THUỐC
        %% -----------------------------------------------------
        subgraph CV2["A2. Phát hiện và phân vùng từng viên thuốc"]
            direction TB

            SEG1["<b>YOLOv11-Seg</b><br/>Phân vùng không phụ thuộc loại thuốc<br/>Viên thuốc và nền<br/><br/>Dữ liệu huấn luyện: MEDISEG"]

            SEG2["Kết quả cho từng viên<br/>Bounding box<br/>Binary mask<br/>Mã định danh instance<br/>Điểm phát hiện"]

            SEG3["Kiểm tra kết quả phân vùng<br/>Mask quá nhỏ hoặc quá lớn<br/>Mask bị phân mảnh<br/>Nhiều viên bị gộp<br/>Khả năng không phải thuốc"]

            SEGG{"Kết quả phân vùng<br/>có hợp lệ không?"}

            SEG1 --> SEG2
            SEG2 --> SEG3
            SEG3 --> SEGG
        end


        %% -----------------------------------------------------
        %% A3. CHUẨN BỊ ẢNH CROP
        %% -----------------------------------------------------
        subgraph CV3["A3. Chuẩn bị ảnh crop cho từng viên"]
            direction TB

            CP1["Áp dụng binary mask<br/>lên ảnh gốc"]

            CP2["Crop theo bounding box<br/>và thêm khoảng đệm"]

            CP3["Chuẩn hóa nền<br/>trắng, xám hoặc trong suốt"]

            CPG{"Viên có trục chính<br/>ổn định không?"}

            CP4A["Căn chỉnh bằng PCA<br/>hoặc minimum-area rectangle"]

            CP4B["Giữ nguyên hướng<br/>của ảnh crop"]

            CP5["Đưa ảnh về<br/>kích thước chuẩn"]

            CP6["Tạo các biến thể xử lý<br/>Ảnh RGB<br/>Ảnh xám<br/>CLAHE<br/>Hiệu chỉnh gamma<br/>Ngưỡng thích nghi<br/>Làm sắc nét"]

            CP1 --> CP2
            CP2 --> CP3
            CP3 --> CPG

            CPG -->|Có| CP4A
            CPG -->|Không| CP4B

            CP4A --> CP5
            CP4B --> CP5

            CP5 --> CP6
        end


        %% -----------------------------------------------------
        %% A4. NHẬN DIỆN THUỘC TÍNH THỊ GIÁC
        %% -----------------------------------------------------
        subgraph CV4["A4. Nhận diện thuộc tính thị giác"]
            direction LR


            %% .................................................
            %% A4.1. HÌNH DẠNG VÀ DẠNG BÀO CHẾ
            %% .................................................
            subgraph SHAPE["A4.1. Hình dạng và dạng bào chế"]
                direction TB

                SH1["Ảnh viên thuốc<br/>đã áp dụng mask"]

                SH2["Trích xuất đặc trưng hình học<br/>Tỷ lệ dài rộng<br/>Độ tròn<br/>Độ đặc<br/>Độ lồi<br/>Đặc trưng contour"]

                SH3["Mô hình phân loại thuộc tính<br/>MobileNetV4, ResNet hoặc ConvNeXt<br/><br/>Dữ liệu huấn luyện:<br/>NIH Pill Image Dataset"]

                SH4["Kết quả thuộc tính<br/>Hình dạng<br/>Dạng bào chế<br/>Đường chia viên<br/>Điểm tin cậy"]

                SH1 --> SH2
                SH1 --> SH3

                SH2 --> SH4
                SH3 --> SH4
            end


            %% .................................................
            %% A4.2. MÀU SẮC
            %% .................................................
            subgraph COLOR["A4.2. Nhận diện màu sắc"]
                direction TB

                CL1["Ảnh RGB đã áp dụng mask"]

                CL2["Hiệu chỉnh sai lệch màu<br/>Cân bằng trắng hoặc Retinex"]

                CL3["Chỉ lấy pixel trong mask<br/>Loại vùng phản sáng và bóng mạnh"]

                CL4["Phân tích màu trong<br/>không gian HSV hoặc Lab"]

                CL5["Phân cụm màu<br/>Màu chính và màu phụ"]

                CL6["Kết quả màu sắc<br/>Màu chính<br/>Màu phụ<br/>Phân bố màu<br/>Cảnh báo ánh sáng<br/>Điểm tin cậy"]

                CL1 --> CL2
                CL2 --> CL3
                CL3 --> CL4
                CL4 --> CL5
                CL5 --> CL6
            end
        end


        %% -----------------------------------------------------
        %% A5. PHÁT HIỆN VÀ NHẬN DẠNG IMPRINT
        %% -----------------------------------------------------
        subgraph CV5["A5. Phát hiện và nhận dạng imprint bằng PaddleOCR"]
            direction TB

            OCR1["Đánh giá khả năng<br/>quan sát imprint"]

            OCRG1{"Có thể quan sát<br/>imprint không?"}

            OCR2["Tạo bốn hướng ảnh<br/>0°, 90°, 180°, 270°"]

            OCR3["Tăng cường chất lượng ảnh<br/>Ảnh gốc hoặc ảnh xám<br/>CLAHE<br/>Hiệu chỉnh gamma<br/>Ngưỡng thích nghi<br/>Top-hat hoặc bottom-hat"]

            OCR4["<b>PP-OCRv5 Text Detector</b><br/>Phát hiện vùng chứa chữ hoặc số"]

            OCRG2{"Có phát hiện được<br/>vùng imprint hợp lệ không?"}

            OCR5A["Crop các vùng imprint<br/>đã phát hiện"]

            OCR5B["Phương án dự phòng<br/>Dùng toàn bộ ảnh crop của viên"]

            OCR6["<b>PP-OCRv5 Text Recognizer</b><br/>Đọc chuỗi imprint<br/><br/>Bộ ký tự:<br/>A–Z, 0–9 và ký hiệu hợp lệ"]

            OCR7["Thu thập các kết quả OCR<br/>Chuỗi nhận dạng<br/>Điểm phát hiện<br/>Điểm nhận dạng<br/>Góc xoay<br/>Biến thể tiền xử lý<br/>Tọa độ vùng chữ"]

            OCR8["Ghép các vùng chữ theo vị trí<br/>Nhóm theo dòng<br/>Sắp xếp từ trái sang phải<br/>Ghép thành chuỗi hoàn chỉnh"]

            OCR9["Chuẩn hóa kết quả OCR<br/>Chuyển thành chữ hoa<br/>Chuẩn hóa khoảng trắng<br/>Giữ kết quả gốc có điểm cao"]

            OCR10["Xử lý có kiểm soát<br/>các ký tự dễ nhầm<br/>0, O, Q<br/>1, I, L<br/>5, S<br/>8, B<br/>2, Z<br/>6, G"]

            OCR11["Giữ một số giả thuyết imprint<br/>Tối đa 3 đến 5 ứng viên<br/>Chỉ giữ kết quả qua kiểm tra chất lượng"]

            OCRG3{"Bằng chứng OCR<br/>có đủ sử dụng không?"}

            OCR_OK["OCR đủ thông tin<br/>để tổng hợp metadata"]

            OCR_FAIL["OCR chưa đủ thông tin<br/>Chỉ giữ đặc trưng một phần"]

            OCR12["Không quan sát được imprint<br/>hoặc không đủ bằng chứng thị giác"]

            OCR1 --> OCRG1

            OCRG1 -->|Có| OCR2
            OCRG1 -->|Không| OCR12

            OCR2 --> OCR3
            OCR3 --> OCR4
            OCR4 --> OCRG2

            OCRG2 -->|Có| OCR5A
            OCRG2 -->|Không| OCR5B

            OCR5A --> OCR6
            OCR5B --> OCR6

            OCR6 --> OCR7
            OCR7 --> OCR8
            OCR8 --> OCR9
            OCR9 --> OCR10
            OCR10 --> OCR11
            OCR11 --> OCRG3

            OCRG3 -->|Có| OCR_OK
            OCRG3 -->|Không| OCR_FAIL
        end


        %% -----------------------------------------------------
        %% A6. TỔNG HỢP METADATA VÀ KIỂM TRA CHẤT LƯỢNG
        %% -----------------------------------------------------
        subgraph CV6["A6. Tổng hợp metadata và kiểm tra chất lượng CV"]
            direction TB

            VF1["Tổng hợp đặc trưng thị giác<br/>Thông tin phân vùng<br/>Hình dạng<br/>Màu sắc<br/>Dạng bào chế<br/>Đường chia viên<br/>Giả thuyết imprint<br/>Các cờ chất lượng"]

            VFG{"Bằng chứng thị giác<br/>có đủ không?"}

            VF2["cv_status = features_ready<br/>Đủ đặc trưng để truy xuất"]

            VF3["cv_status = partial_features<br/>Chỉ có một phần đặc trưng"]

            VF4["cv_status = insufficient_visual_evidence<br/>Không đủ bằng chứng thị giác"]

            VF5["cv_status = unknown_object<br/>Có thể không phải viên thuốc"]

            VF6["<b>STRUCTURED VISUAL METADATA JSON</b><br/><br/>Chỉ chứa bằng chứng thị giác<br/>Không chứa tên thuốc<br/>Không chứa hoạt chất<br/>Không chứa kết luận DDI"]

            VF1 --> VFG

            VFG -->|Đầy đủ| VF2
            VFG -->|Một phần| VF3
            VFG -->|Không đủ| VF4
            VFG -->|Không phải thuốc| VF5

            VF2 --> VF6
            VF3 --> VF6
        end


        %% -----------------------------------------------------
        %% LUỒNG NỘI BỘ MODULE CV
        %% -----------------------------------------------------
        IQG -->|Có| SEG1
        IQG -->|Không| VF4

        SEGG -->|Có| CP1
        SEGG -->|Không| VF4

        CP6 --> SH1
        CP6 --> CL1
        CP6 --> OCR1

        SH4 --> VF1
        CL6 --> VF1

        OCR_OK --> VF1
        OCR_FAIL --> VF3
        OCR12 --> VF3
    end


    %% =========================================================
    %% MODULE B — TRUY XUẤT / RAG
    %% =========================================================
    


    %% =========================================================
    %% ĐỊNH DẠNG PHONG CÁCH PAPER
    %% =========================================================
    classDef interface fill:#FFFFFF,stroke:#111111,stroke-width:2px,color:#111111;
    classDef process fill:#FFFFFF,stroke:#3A3A3A,stroke-width:1.4px,color:#111111;
    classDef model fill:#F2F2F2,stroke:#111111,stroke-width:2px,color:#111111;
    classDef decision fill:#FFFFFF,stroke:#111111,stroke-width:1.7px,color:#111111;
    classDef output fill:#EDEDED,stroke:#111111,stroke-width:1.7px,color:#111111;
    classDef warning fill:#FAFAFA,stroke:#7A0000,stroke-width:1.7px,color:#4A0000;
    classDef retrieval fill:#F5F5F5,stroke:#202020,stroke-width:1.7px,color:#111111;

    class UI_IN,UI_OUT interface;
    class SEG1,SH3,OCR4,OCR6 model;
    class IQG,SEGG,CPG,OCRG1,OCRG2,OCRG3,VFG decision;
    class SEG2,SH4,CL6,OCR7,OCR11,VF6 output;
    class VF4,VF5,OCR12,OCR_FAIL,RETAKE warning;
    class OCR_OK process;
    class R1,R2,R3,R4,R5,R6,R7 retrieval;


    %% =========================================================
    %% ĐỊNH DẠNG CÁC KHỐI MODULE
    %% =========================================================
    style CV fill:#FFFFFF,stroke:#111111,stroke-width:2px
    style CV1 fill:#FAFAFA,stroke:#666666,stroke-width:1px
    style CV2 fill:#FAFAFA,stroke:#666666,stroke-width:1px
    style CV3 fill:#FAFAFA,stroke:#666666,stroke-width:1px
    style CV4 fill:#FAFAFA,stroke:#666666,stroke-width:1px
    style CV5 fill:#FAFAFA,stroke:#666666,stroke-width:1px
    style CV6 fill:#FAFAFA,stroke:#666666,stroke-width:1px

    style SHAPE fill:#FFFFFF,stroke:#999999,stroke-width:1px
    style COLOR fill:#FFFFFF,stroke:#999999,stroke-width:1px

    style RAG fill:#FFFFFF,stroke:#111111,stroke-width:2px
```

Luồng dữ liệu rút gọn:

```text
Ảnh nhiều viên thuốc
    ↓
CV: segmentation từng instance
    ↓
CV: crop + alignment + mask
    ↓
CV: shape + color + dosage form + imprint candidates + confidence
    ↓
Structured visual metadata JSON
    ↓
Retrieval: imprint search + structured filters
    ↓
Ranking: candidate score + safety gate
    ↓
Drug metadata + ingredient normalization
    ↓
DDI structured lookup
    ↓
LLM trình bày báo cáo từ context đã truy xuất
```

---

## 3. CV Module

CV Module chuyển ảnh đầu vào thành metadata thị giác có cấu trúc cho từng viên thuốc. CV chỉ trả lời:

> Viên này nhìn thấy có hình dạng gì, màu gì, dạng gì, có scoreline/imprint nào, và các đặc trưng đó đáng tin đến mức nào?

CV không trả lời:

> Đây là thuốc gì?

### 3.0. Confidence and Field Policy

CV trả nhiều score, nhưng các score này không được mặc định là xác suất đúng. Trước khi được calibration bằng validation set thực tế, mọi confidence chỉ được xem là **relative model score**.

Mỗi field trong CV output phải có consumer rõ ràng:

```text
retrieval_key
    → dùng để tạo shortlist, ví dụ imprint candidates

rerank_evidence
    → dùng để chấm điểm trên shortlist, ví dụ shape/color/dosage_form

safety_gate
    → dùng để reject, hạ trạng thái hoặc yêu cầu ảnh bổ sung

debug_ui
    → dùng để audit hoặc hiển thị, không tham gia quyết định chính
```

Nếu một field chưa có consumer rõ ràng, không nên đưa vào API contract chính.

### 3.1. Segmentation

Mục tiêu:

- Đếm số instance viên thuốc.
- Tạo bounding box.
- Tạo binary mask riêng cho từng instance.
- Tách các viên tiếp xúc hoặc chồng lấp một phần.
- Tạo crop đã tách nền để phục vụ attribute recognition và OCR.

Model đề xuất:

- **YOLOv11-Seg**.
- Train bằng **MEDISEG** hoặc dataset segmentation tương đương.
- Bài toán class-agnostic:

```text
pill vs background
```

Image quality gate cần kiểm tra:

- Ảnh mờ do rung tay hoặc mất nét.
- Thiếu sáng hoặc cháy sáng.
- Phản chiếu mạnh trên bề mặt thuốc.
- Viên thuốc quá nhỏ trong ảnh.
- Che khuất nghiêm trọng.
- Nền có màu gần giống viên thuốc.

Segmentation quality gate cần kiểm tra:

- Mask quá nhỏ hoặc quá lớn bất thường.
- Mask bị phân mảnh thành nhiều component.
- Mask có khả năng gộp hai viên vào một instance.
- Contour có lõm mạnh, area/bbox ratio bất thường hoặc shape không nhất quán với crop.
- Object có khả năng không phải thuốc hoặc viên nang.

Output instance segmentation:

```json
{
  "instance_id": "pill_001",
  "bbox_xyxy": [142, 93, 326, 248],
  "segmentation": {
    "confidence": 0.96,
    "occlusion_estimate": 0.12,
    "possible_merged_instance": false,
    "possible_non_pill": false
  },
  "mask_path": "outputs/pill_001_mask.png",
  "crop_path": "outputs/pill_001_crop.png",
  "quality_flags": ["minor_glare"]
}
```

Crop preparation:

- Áp dụng mask lên ảnh gốc.
- Crop theo bounding box có padding.
- Resize về kích thước chuẩn.
- Căn chỉnh theo trục chính của contour bằng PCA hoặc minimum-area rectangle.
- Tạo các biến thể ảnh: `crop_rgb`, `crop_masked_rgb`, `crop_gray`, `crop_clahe`, `crop_gamma_corrected`, `crop_adaptive_threshold`, `crop_sharpened`.

Lưu ý: alignment theo trục dài không có nghĩa là chữ imprint đã đúng hướng. Nhánh OCR vẫn phải thử nhiều góc xoay.

Nếu `possible_merged_instance = true`, backend không được dùng crop đó để `identified` trừ khi có bằng chứng bổ sung rất mạnh. Trạng thái ưu tiên là `partial_features` hoặc `insufficient_visual_evidence`.

### 3.2. Attribute Recognition

Mục tiêu là nhận diện thuộc tính quan sát được, không định danh thuốc.

Attributes MVP:

```text
shape
color
dosage_form
scoreline
imprint_visibility
damage_or_occlusion
```

Dataset đề xuất:

- **NLM Pillbox / RxIMAGE / C3PI**: nguồn chính để train hoặc fine-tune attribute model vì có ảnh thuốc kèm metadata ngoại hình như shape, color, dosage form, imprint, scoreline và size.
- **NIH Pill Image Dataset**: nguồn bổ sung nếu map được nhãn shape/color phù hợp.
- **ePillID Dataset, CVPR Workshop 2020**: dùng để benchmark robustness và low-shot/fine-grained setting, không nên mặc định là nguồn label attribute chính nếu thiếu metadata shape/color/form.

Label policy:

- Chỉ train head khi label đủ sạch và có định nghĩa thống nhất.
- Shape/color/form có thể train trước.
- Scoreline, imprint visibility và damage/occlusion có thể là optional heads nếu chưa có annotation đáng tin.
- Dataset reference cần được kiểm tra domain gap với ảnh consumer-grade và ảnh tự chụp.

Multi-task model:

```text
Masked pill crop
    ↓
Backbone: ResNet-18 / ResNet-34 / MobileNetV4 / ConvNeXt-Tiny
    ↓
shape head
color head
dosage_form head
scoreline head
imprint_visibility head
damage_or_occlusion head
```

Loss huấn luyện:

```text
loss =
  λ_shape * CE(shape)
+ λ_color * CE_or_BCE(color)
+ λ_form * CE(dosage_form)
+ λ_scoreline * BCE(scoreline)
+ λ_imprint_visible * BCE(imprint_visibility)
+ λ_quality * BCE(damage_or_occlusion)
```

Shape labels MVP:

```text
round
oval
oblong
capsule
rectangle
triangle
diamond
other
```

Color labels MVP:

```text
white
red
blue
green
yellow
orange
pink
brown
gray
black
purple
multi_color
unknown
```

Color constancy:

- Chỉ lấy pixel trong mask.
- Loại vùng highlight và shadow.
- Dùng HSV/Lab để phân tích màu.
- Áp dụng white balance hoặc Retinex nếu ánh sáng lệch rõ.
- Với capsule hai màu, thử chia theo trục dài trước khi phân màu.
- Nếu có `lighting_warning`, color không được dùng làm hard filter.

Output attribute:

```json
{
  "shape": {
    "label": "oval",
    "confidence": 0.91,
    "alternatives": [
      {"label": "oblong", "confidence": 0.07}
    ]
  },
  "color": {
    "primary": "white",
    "secondary": null,
    "distribution": {
      "white": 0.72,
      "gray": 0.16,
      "yellow": 0.09
    },
    "confidence": 0.88,
    "lighting_warning": false
  },
  "dosage_form": {
    "label": "tablet",
    "confidence": 0.89
  },
  "scoreline": {
    "visible": true,
    "confidence": 0.81
  },
  "imprint_visibility": {
    "visible": true,
    "confidence": 0.86
  }
}
```

Score của attribute dùng để báo cho backend biết bằng chứng đó mạnh hay yếu. Ví dụ color confidence thấp hoặc có lighting warning thì backend không nên loại candidate chỉ vì màu không khớp.

Attribute usage policy:

```text
shape
    → rerank_evidence
    → safety_gate nếu mâu thuẫn mạnh và confidence cao

color
    → rerank_evidence
    → không bao giờ là primary identifier
    → bỏ hoặc giảm trọng số nếu lighting_warning = true

dosage_form
    → rerank_evidence
    → hard reject nếu tablet/capsule mâu thuẫn rõ và confidence cao

scoreline
    → rerank_evidence phụ

imprint_visibility
    → safety_gate cho OCR/retrieval

damage_or_occlusion
    → safety_gate
```

### 3.3. Imprint OCR

Imprint là tín hiệu phân biệt quan trọng nhất trong nhiều trường hợp. Kiến trúc OCR chính của hệ thống sử dụng **PaddleOCR PP-OCRv5 end-to-end**, trong đó cùng một framework đảm nhiệm hai phase:

```text
PP-OCRv5 Text Detector
    → tìm vùng có khả năng chứa imprint

PP-OCRv5 Text Recognizer
    → đọc chuỗi ký tự trong vùng imprint đã phát hiện
```

Cấu hình chính:

```text
Masked pill crop
    ↓
Basic contrast enhancement
    ↓
PaddleOCR PP-OCRv5
    ├── Text Detection
    └── Text Recognition
    ↓
Multi-angle OCR observations
    ↓
Limited normalization and Top-k imprint candidates
```

Lý do chọn PaddleOCR làm cấu hình chính:

- Cung cấp sẵn cả text detector và text recognizer trong cùng một pipeline.
- Có thể sử dụng pretrained model để xây baseline trước khi quyết định fine-tune.
- Detector và recognizer vẫn có thể được đánh giá hoặc fine-tune độc lập khi cần.
- Giảm độ phức tạp tích hợp so với việc ghép detector và recognizer từ hai framework khác nhau.
- Phù hợp với kiến trúc prototype vì có thể nâng cấp từng phase dựa trên kết quả benchmark.

#### 3.3.1. Phase 1 — Text Detection

Text detector nhận crop của từng viên thuốc và trả bounding box hoặc polygon của vùng có khả năng chứa imprint. Detector không quyết định chuỗi ký tự là gì.

Detector giúp loại bớt các vùng không liên quan như:

- Contour và cạnh viên thuốc.
- Scoreline.
- Bóng đổ và vùng phản sáng.
- Vết xước hoặc texture bề mặt.

Output detection mẫu:

```json
{
  "region_id": "region_01",
  "polygon": [[32, 24], [58, 22], [60, 39], [33, 41]],
  "detection_confidence": 0.84
}
```

Detector phải có fallback:

```text
Nếu detector tìm được text ROI đủ tin cậy
    → chạy recognizer trên từng ROI

Nếu detector không tìm thấy ROI
nhưng imprint_visibility đủ cao
    → chạy recognizer trực tiếp trên toàn crop đã mask

Nếu cả hai cách đều không tạo kết quả ổn định
    → trả partial_features hoặc insufficient_visual_evidence
```

#### 3.3.2. Phase 2 — Text Recognition

PP-OCRv5 recognizer nhận từng text ROI và dự đoán transcript imprint, ví dụ:

```text
ROI image → "L484"
```

Trong phiên bản đầu, hệ thống sử dụng recognizer pretrained để tạo baseline. Việc fine-tune được quyết định dựa trên benchmark:

```text
Detector tìm ROI tốt nhưng thường đọc sai ký tự
    → chỉ fine-tune recognizer

Detector thường xuyên bỏ sót vùng imprint
    → bổ sung annotation vùng chữ và fine-tune detector

Cả detector và recognizer đạt yêu cầu
    → giữ nguyên pretrained pipeline
```

Khi fine-tune recognizer, dữ liệu tối thiểu gồm:

```text
imprint ROI image
    +
ground-truth transcript
```

Charset nên được xây từ database imprint của thị trường mục tiêu. Charset cơ sở có thể gồm:

```text
A–Z
0–9
```

và bổ sung các ký hiệu thực sự xuất hiện như:

```text
- / . +
```

Charset chỉ cần tùy chỉnh khi fine-tune recognizer; không bắt buộc phải thay ngay trong baseline pretrained.

#### 3.3.3. Preprocessing and Multi-angle OCR

Imprint có thể khắc chìm, dập nổi, tương phản thấp hoặc nằm ở hướng bất kỳ. Hệ thống áp dụng preprocessing và rotation theo cơ chế cascade thay vì chạy mọi tổ hợp cố định.

```text
Tier 1
    0° và 180°
    original/gray + CLAHE
    PaddleOCR detection + recognition

Nếu kết quả chưa ổn định:
    ↓
Tier 2
    90° và 270°
    gamma correction + sharpening hoặc top-hat/bottom-hat

Nếu vẫn chưa đủ bằng chứng:
    → yêu cầu ảnh cận cảnh hoặc ảnh mặt còn lại
```

Preprocessing có thể gồm:

- CLAHE.
- Gamma correction.
- Sharpening nhẹ.
- Adaptive threshold.
- Morphological top-hat/bottom-hat.

Không nên chỉ dùng ảnh edge làm đầu vào duy nhất vì nét imprint khắc chìm có thể bị phá vỡ.

#### 3.3.4. OCR Observation Aggregation

Hệ thống không nên lấy duy nhất một chuỗi OCR từ một góc chụp. Thay vào đó, CV giữ một số quan sát tốt nhất từ các rotation và preprocessing khác nhau.

Ví dụ:

```json
[
  {"text": "AO1", "confidence": 0.81, "rotation": 0, "preprocessing": "clahe"},
  {"text": "A01", "confidence": 0.77, "rotation": 180, "preprocessing": "top_hat"}
]
```

Sau đó hệ thống tạo một tập nhỏ `normalized_candidates` bằng:

- Chuẩn hóa chữ hoa.
- Chuẩn hóa khoảng trắng.
- Giữ các raw observation có confidence cao.
- Áp dụng có giới hạn các nhóm ký tự thường bị nhầm:

```text
0 ↔ O ↔ Q
1 ↔ I ↔ L
5 ↔ S
8 ↔ B
2 ↔ Z
6 ↔ G
```

Ví dụ:

```json
[
  {"text": "A01", "score": 0.84, "source": "multi_angle_consensus"},
  {"text": "AO1", "score": 0.81, "source": "raw_ocr"}
]
```

`score` chỉ là relative OCR score, không phải xác suất thuốc đúng. Candidate expansion phải bị giới hạn:

```text
max_ocr_observations_per_pill = 8
max_imprint_candidates = 3 đến 5
min_text_detection_score = threshold theo validation set
min_ocr_confidence_for_candidate = threshold theo validation set
```

Không được đưa mọi chuỗi OCR từ mọi rotation vào retrieval. `ocr_observations` dùng cho debug/audit, còn `normalized_candidates` chỉ chứa các cách đọc đã qua quality gate.

Nếu detector trả nhiều text ROI, hệ thống có thể:

- Sort ROI theo trục `x` trong cùng một dòng.
- Group nhiều dòng theo trục `y`.
- Giữ transcript theo từng ROI và transcript đã ghép.
- Không tự sinh ký tự ngoài OCR raw hoặc nhóm confusion đã định nghĩa.

Nếu người dùng cung cấp ảnh mặt còn lại, CV vẫn xử lý từng ảnh độc lập nhưng phải giữ metadata để backend ghép:

```text
session_id
instance_token
side_hint: front | back | unknown
```

### 3.4. CV Output JSON

API contract giữa CV và Retrieval/RAG chỉ chứa bằng chứng thị giác, không chứa tên thuốc, hoạt chất hoặc Top-k candidate thuốc.

```json
{
  "request_id": "req_2026_001",
  "session_id": "sess_2026_001",
  "image_quality": {
    "status": "usable_with_warning",
    "blur_score": 0.21,
    "glare_detected": true
  },
  "pills": [
    {
      "instance_id": "pill_001",
      "instance_token": "pill_token_001",
      "side_hint": "unknown",
      "cv_status": "features_ready",
      "bbox_xyxy": [142, 93, 326, 248],
      "mask_path": "outputs/pill_001_mask.png",
      "crop_path": "outputs/pill_001_crop.png",
      "segmentation": {
        "confidence": 0.96,
        "occlusion_estimate": 0.18,
        "possible_merged_instance": false,
        "possible_non_pill": false
      },
      "shape": {
        "label": "oval",
        "confidence": 0.91,
        "alternatives": [
          {"label": "oblong", "confidence": 0.07}
        ]
      },
      "color": {
        "primary": "white",
        "secondary": null,
        "distribution": {
          "white": 0.72,
          "gray": 0.16,
          "yellow": 0.09
        },
        "confidence": 0.88,
        "lighting_warning": false
      },
      "dosage_form": {
        "label": "tablet",
        "confidence": 0.89
      },
      "scoreline": {
        "visible": true,
        "confidence": 0.81
      },
      "imprint_visibility": {
        "visible": true,
        "confidence": 0.86
      },
      "imprint": {
        "architecture": {
          "framework": "PaddleOCR",
          "detector": "PP-OCRv5",
          "recognizer": "PP-OCRv5",
          "mode": "end_to_end"
        },
        "text_regions": [
          {
            "region_id": "region_01",
            "polygon": [[32, 24], [58, 22], [60, 39], [33, 41]],
            "detection_confidence": 0.84
          }
        ],
        "raw": "A O1",
        "confidence": 0.72,
        "normalized_candidates": [
          {"text": "A 01", "score": 0.91, "source": "multi_angle_consensus"},
          {"text": "A O1", "score": 0.84, "source": "raw_ocr"},
          {"text": "A01", "score": 0.69, "source": "space_normalized"}
        ],
        "ocr_observations": [
          {
            "region_id": "region_01",
            "rotation_degrees": 0,
            "preprocessing": "clahe",
            "text": "A O1",
            "confidence": 0.72
          },
          {
            "region_id": "region_01",
            "rotation_degrees": 180,
            "preprocessing": "top_hat",
            "text": "A 01",
            "confidence": 0.68
          }
        ],
        "visible": true
      },
      "quality_flags": ["minor_glare"]
    }
  ]
}
```

Field usage summary:

```text
bbox_xyxy, mask_path, crop_path
    → UI/debug/audit

segmentation.confidence, occlusion_estimate, possible_merged_instance, possible_non_pill
    → safety_gate

shape, color, dosage_form, scoreline
    → rerank_evidence

imprint_visibility, imprint.confidence
    → safety_gate cho OCR/retrieval

imprint.normalized_candidates
    → retrieval_key chính

ocr_observations
    → debug/audit, không query trực tiếp toàn bộ

quality_flags
    → safety_gate và UI explanation
```

CV status:

```text
features_ready
partial_features
insufficient_visual_evidence
unknown_object
```

---

## 4. Retrieval/RAG Module

Retrieval/RAG Module nhận metadata thị giác từ CV và thực hiện:

1. Imprint candidate retrieval.
2. Candidate ranking and validation.
3. Drug metadata retrieval.
4. Drug normalization.
5. DDI structured lookup.
6. Grounded LLM report generation.

Tên Retrieval/RAG không có nghĩa LLM đoán tên thuốc. Bên trong module phải có structured retrieval layer rõ ràng:

```text
Structured Retrieval Layer
    → matching, lookup, ranking, validation

LLM Layer
    → chỉ diễn giải kết quả retrieval đã có nguồn
```

### 4.1. Imprint Candidate Retrieval

Retrieval không để LLM tự quyết định search theo gì. Structured retrieval code phải có quy trình cố định. Điểm quan trọng nhất:

```text
Imprint dùng để tạo shortlist.
Shape, color, dosage form và market dùng để rerank hoặc loại mâu thuẫn rõ.
Không tạo tổ hợp query imprint × color × shape.
```

Ví dụ nếu CV trả 4 imprint candidates và 3 color candidates, backend không tạo 12 query. Backend chỉ search theo imprint candidates, sau đó chấm điểm màu trên các record đã lấy ra.

Input:

- Imprint candidates có trọng số từ CV.
- Shape label và alternatives.
- Color distribution, color confidence và lighting warning.
- Dosage form quan sát được.
- Scoreline/imprint visibility.
- Country hoặc market nếu người dùng cung cấp.

Query mẫu:

```json
{
  "imprint_candidates": [
    {"text": "A 01", "score": 0.91},
    {"text": "A O1", "score": 0.84},
    {"text": "A01", "score": 0.69}
  ],
  "shape": {
    "label": "oval",
    "confidence": 0.91
  },
  "color": {
    "primary": "white",
    "distribution": {
      "white": 0.72,
      "gray": 0.16,
      "yellow": 0.09
    },
    "confidence": 0.88,
    "lighting_warning": false
  },
  "dosage_form": {
    "label": "tablet",
    "confidence": 0.89
  },
  "market": "US"
}
```

#### Retrieval stages

**Stage 0 — Evidence gating**

Trước khi query database, backend quyết định evidence nào được dùng cứng, mềm hoặc bỏ qua:

```text
possible_non_pill = true
    → unknown_object
    → không query drug database

possible_merged_instance = true
    → không cho identified từ crop này
    → yêu cầu tách viên hoặc ảnh bổ sung nếu cần

imprint visible + OCR confidence đủ dùng
    → dùng imprint làm retrieval key chính

imprint thiếu hoặc OCR quá thấp
    → không cho identified bằng shape/color/form đơn lẻ
    → chỉ tạo shortlist thận trọng hoặc yêu cầu ảnh bổ sung

color có lighting_warning
    → color chỉ là soft evidence

shape confidence thấp
    → giữ shape alternatives, không filter cứng

dosage_form confidence cao
    → có thể dùng làm hard reject nếu tablet/capsule mâu thuẫn rõ
```

**Stage 1 — Imprint-first candidate generation**

Backend chỉ query theo imprint candidates:

```text
usable_imprints = normalized_candidates
    filtered by min_candidate_score
    sorted by candidate_score desc
    limited to max_imprint_candidates

for each imprint_candidate in usable_imprints:
    search exact imprint
    search normalized imprint
    search weighted fuzzy imprint
```

Các record trả về được `union` và `deduplicate` theo `appearance_id` hoặc `product_id`. Nếu cùng một record match nhiều imprint candidates, giữ match có `imprint_match_score` cao nhất.

Backend không query trực tiếp toàn bộ `ocr_observations`. Trường này chỉ dùng để audit/debug hoặc để CV tạo `normalized_candidates`.

Không làm:

```text
for imprint in imprint_candidates:
    for color in color_candidates:
        query(imprint, color)
```

Lý do: color và shape là evidence phụ, nếu đưa vào query quá sớm có thể làm mất candidate đúng khi ảnh bị lệch màu hoặc shape confidence thấp.

**Stage 2 — Attribute scoring trên shortlist**

Sau khi có shortlist từ imprint index, backend mới tính attribute score cho từng record:

```text
shape_score(record)       = consistency(predicted_shape, record.shape)
color_score(record)       = overlap(predicted_color_distribution, record.colors)
dosage_form_score(record) = consistency(predicted_form, record.dosage_form)
market_score(record)      = availability(record.market, requested_market)
```

Color score là phép tra trực tiếp trên distribution, không tạo query mới. Ví dụ:

```text
Predicted colors:
white 0.72
gray 0.16
yellow 0.09

Record A color = white  → color_score = 0.72
Record B color = yellow → color_score = 0.09
```

**Stage 3 — Ranking**

MVP dùng công thức tuyến tính để dễ debug:

```text
final_score =
  0.65 * imprint_match_score
+ 0.15 * shape_score
+ 0.10 * color_score
+ 0.05 * dosage_form_score
+ 0.05 * market_score
```

Trong đó `imprint_match_score` được tính trên từng record:

```text
imprint_match_score =
    imprint_candidate.score
  × weighted_edit_similarity(imprint_candidate.text, record.imprint)
  × ocr_confidence
```

Ví dụ:

```text
OCR raw = AOI
Candidate imprint A01 score = 0.91
Database imprint A01 weighted_edit_similarity = 1.00
OCR confidence = 0.72

imprint_match_score = 0.91 × 1.00 × 0.72 = 0.655
```

`imprint_candidate.score` không phải xác suất thuốc đúng. Nó chỉ là độ tin cậy của cách đọc imprint. Xác suất nhận diện cuối cùng, nếu cần, phải được calibration riêng trên validation set.

**Stage 4 — Safety validation**

Ranking score không được tự động quyết định `identified`. Candidate phải qua rule cứng:

```text
Nếu CV trả insufficient_visual_evidence
    → insufficient_visual_evidence

Nếu không có candidate từ imprint và không có bằng chứng phụ đủ mạnh
    → unknown

Nếu thiếu imprint usable
    → tối đa probable_match, thường ambiguous hoặc unknown

Nếu dosage form mâu thuẫn rõ ràng
    → reject candidate

Nếu imprint mâu thuẫn rõ ràng
    → reject hoặc không cho identified

Nếu Top-1 và Top-2 quá gần nhau
    → ambiguous

Nếu Top-1 đủ cao và cách Top-2 đủ xa, không vi phạm rule cứng
    → identified
```

Trạng thái nhận diện:

```text
identified
probable_match
ambiguous
unknown
insufficient_visual_evidence
```

Chỉ `identified` mới được đưa vào DDI lookup chắc chắn.

#### Output ranking mẫu

```json
{
  "instance_id": "pill_001",
  "identification_status": "ambiguous",
  "candidate_generation": {
    "strategy": "imprint_first",
    "queried_imprints": ["A 01", "A O1", "A01"],
    "num_records_before_dedup": 18,
    "num_records_after_dedup": 7
  },
  "top_candidates": [
    {
      "product_id": "drug_1032",
      "product_name": "Candidate A",
      "final_score": 0.74,
      "evidence": {
        "best_imprint_candidate": "A01",
        "imprint_match_score": 0.655,
        "shape_score": 0.91,
        "color_score": 0.72,
        "dosage_form_score": 0.89,
        "market_score": 1.0,
        "hard_reject": false
      }
    },
    {
      "product_id": "drug_8471",
      "product_name": "Candidate B",
      "final_score": 0.68,
      "evidence": {
        "best_imprint_candidate": "AO1",
        "imprint_match_score": 0.598,
        "shape_score": 0.91,
        "color_score": 0.72,
        "dosage_form_score": 0.89,
        "market_score": 1.0,
        "hard_reject": false
      }
    }
  ],
  "required_action": "capture_reverse_side"
}
```

### 4.2. Candidate Ranking and Validation

Candidate ranking trong MVP là một pipeline deterministic, không phải RAG tự do:

```text
CV evidence
    → evidence gating
    → imprint-first retrieval
    → deduplicate candidate records
    → attribute scoring on shortlist
    → final ranking
    → safety validation
    → identification_status
```

Sau khi có validation set thực tế, công thức tuyến tính có thể được thay bằng logistic regression, isotonic regression hoặc learning-to-rank nhỏ. Tuy nhiên các rule cứng vẫn phải giữ vì mục tiêu an toàn là giảm false identification.

### 4.3. Drug Metadata Retrieval

Chỉ sau khi một candidate có `product_id` hợp lệ và vượt qua validation, hệ thống mới lấy metadata thuốc.

Dữ liệu cần lấy:

- Product name.
- Brand name.
- Generic name.
- Active ingredients.
- Strength.
- Dosage form.
- Route.
- Manufacturer.
- NDC hoặc mã tương đương.
- RxCUI hoặc mã chuẩn hóa.
- Country hoặc market.
- Source record và thời điểm cập nhật.

Schema mẫu:

```json
{
  "product_id": "drug_1032",
  "brand_name": "Example Brand",
  "generic_name": "example ingredient",
  "active_ingredients": [
    {
      "name": "example ingredient",
      "strength": "500 mg",
      "rxcui": "123456"
    }
  ],
  "dosage_form": "tablet",
  "route": "oral",
  "ndc": "00000-0000-00",
  "market": "US",
  "source": "canonical_pill_database",
  "last_updated": "2026-01-15"
}
```

Quy tắc:

- Không dùng LLM suy đoán hoạt chất từ tên gần giống.
- Không gộp các sản phẩm khác hàm lượng.
- Nếu không xác định được strength, trả `strength_unknown`.
- Mọi metadata cần giữ `source_id`, `source_version` và `last_updated`.

### 4.4. Drug Normalization and DDI Structured Lookup

Thuốc được chuẩn hóa về hoạt chất trước khi tra DDI.

Dạng chuẩn:

```text
active ingredient + strength + route + normalized identifier
```

Thuốc đa thành phần:

```text
Drug A = ingredient A1 + ingredient A2
Drug B = ingredient B1

DDI checks:
A1 × B1
A2 × B1
```

Duplicate ingredient:

- Hai sản phẩm khác tên thương mại nhưng cùng hoạt chất cần tạo cảnh báo `duplicate_ingredient`.
- Không chỉ dựa vào DDI pair thông thường.

DDI lookup phải dùng dữ liệu có cấu trúc như relational database, graph database hoặc API đã xác minh. Không dùng vector search hoặc LLM làm nguồn kết luận DDI chính.

DDI schema:

```json
{
  "ingredient_a": "ingredient_1",
  "ingredient_b": "ingredient_2",
  "severity": "major",
  "clinical_risk": "Increased bleeding risk",
  "mechanism": "Additive pharmacodynamic effect",
  "management": "Avoid combination or monitor closely",
  "source": "verified_ddi_database",
  "last_reviewed": "2026-01-15"
}
```

Quy tắc an toàn:

- Không có record không đồng nghĩa không có tương tác.
- MVP nên trả `no_interaction_found_in_current_database`, không trả `interaction_absent`.
- `probable_match`, `ambiguous`, `unknown` và `insufficient_visual_evidence` không được dùng cho kết luận DDI chắc chắn.

### 4.5. Grounded LLM Report

LLM chỉ nhận context đã được retrieval và validation.

Context gồm:

- Danh sách instance và metadata CV.
- Trạng thái nhận diện của từng viên.
- Candidate được chấp nhận.
- Hoạt chất chuẩn hóa.
- DDI records.
- Nguồn dữ liệu và thời điểm cập nhật.
- Các giới hạn và cảnh báo.

Strict prompt:

```text
Bạn là module trình bày dữ liệu y tế.
Chỉ sử dụng dữ liệu trong CONTEXT.
Không bổ sung tên thuốc, hoạt chất, mức độ tương tác hoặc khuyến nghị ngoài CONTEXT.
Nếu dữ liệu thiếu, hãy nói rõ hệ thống chưa đủ bằng chứng.
Không khuyên người dùng tự ngừng hoặc thay đổi liều thuốc.
Luôn khuyến nghị xác nhận với bác sĩ hoặc dược sĩ khi có cảnh báo nghiêm trọng hoặc nhận diện chưa chắc chắn.
```

Cấu trúc báo cáo:

1. Tóm tắt kết quả.
2. Danh sách viên thuốc.
3. Độ chắc chắn của từng viên.
4. Cảnh báo tương tác.
5. Hành động khuyến nghị.
6. Viên cần chụp lại hoặc xác nhận.
7. Nguồn dữ liệu và giới hạn hệ thống.

### 4.6. Minimum Database

Pill appearance table:

```json
{
  "appearance_id": "app_001",
  "product_id": "drug_1032",
  "market": "US",
  "dosage_form": "tablet",
  "imprint_front": "A 01",
  "imprint_back": "500",
  "shape": "oval",
  "primary_color": "white",
  "secondary_color": null,
  "score_count": 1,
  "size_mm": {
    "length": 14.5,
    "width": 7.2
  },
  "front_image_path": "...",
  "back_image_path": "...",
  "source": "...",
  "source_version": "...",
  "last_updated": "..."
}
```

Drug product table:

```json
{
  "product_id": "drug_1032",
  "brand_name": "...",
  "generic_name": "...",
  "active_ingredients": [
    {
      "name": "...",
      "strength": "...",
      "rxcui": "..."
    }
  ],
  "dosage_form": "tablet",
  "route": "oral",
  "manufacturer": "...",
  "ndc": "...",
  "market": "US",
  "source": "...",
  "last_updated": "..."
}
```

DDI table:

```json
{
  "ingredient_a_id": "rxcui_a",
  "ingredient_b_id": "rxcui_b",
  "severity": "major",
  "clinical_risk": "...",
  "mechanism": "...",
  "management": "...",
  "source": "...",
  "last_reviewed": "..."
}
```

### 4.7. Benchmark, Metrics and Calibration

Các score trong MVP phải được kiểm chứng bằng validation set thực tế. Trước calibration, score chỉ là relative model score, không phải xác suất đúng.

Benchmark được chia thành bốn tầng:

```text
1. Segmentation benchmark
2. CV attribute and imprint benchmark
3. Retrieval benchmark
4. End-to-end safety benchmark
```

#### 4.7.1. Segmentation benchmark for YOLOv11-Seg

Mục tiêu của benchmark này là đánh giá riêng khả năng phát hiện và phân vùng từng viên thuốc. Không dùng kết quả drug retrieval hoặc DDI trong benchmark segmentation.

Dataset yêu cầu:

```text
ảnh nhiều viên thuốc
    +
annotation từng instance:
    bbox
    instance mask hoặc polygon
```

Nguồn dữ liệu:

- **MEDISEG**: dùng làm nguồn train/validation/test chính nếu có instance mask đầy đủ.
- **Mini real-world benchmark tự chụp**: dùng để đo domain gap với ảnh người dùng thật.

Mini benchmark tự chụp nên có khoảng 100-300 ảnh, bao phủ:

```text
easy
    viên rời, nền đơn giản

touching
    các viên chạm nhau

overlap
    viên chồng lấp một phần

glare
    phản sáng mạnh

low_light
    thiếu sáng

similar_background
    nền gần màu viên thuốc

small_pills
    viên chiếm ít pixel

non_pill_objects
    vật thể giống thuốc như kẹo, vitamin, nút hoặc vật tròn
```

Metrics chính:

```text
mask mAP@50
mask mAP@50-95
box mAP@50
box mAP@50-95
instance recall
precision
F1
mean IoU
```

Metrics lỗi thực tế:

```text
merge_error_rate
    nhiều viên bị gộp thành một mask

split_error_rate
    một viên bị tách thành nhiều instance

missed_pill_rate
    viên thuốc bị bỏ sót

false_positive_non_pill_rate
    vật không phải thuốc bị detect nhầm

occlusion_bucket_performance
    hiệu năng theo mức overlap hoặc occlusion
```

Bảng báo cáo đề xuất:

```text
Case                  Mask mAP   Recall   Merge Error   Split Error   FP Non-pill
easy                  ...
touching              ...
overlap               ...
glare                 ...
low_light             ...
similar_background    ...
small_pills           ...
non_pill_objects      ...
```

Quy trình benchmark:

```text
1. Chuẩn hóa annotation sang YOLO segmentation format.
2. Split train/validation/test theo ảnh, tránh leakage cùng scene hoặc cùng video burst.
3. Train YOLOv11-Seg trên train split.
4. Chọn confidence threshold, IoU threshold và mask threshold trên validation split.
5. Báo cáo kết quả cuối trên test split.
6. Chạy thêm mini real-world benchmark tự chụp để đo domain gap.
```

Baseline so sánh:

```text
YOLOv8-Seg
YOLOv11-Seg
Mask R-CNN
SAM/SAM2-assisted segmentation, nếu dùng như post-processing hoặc upper-bound tham khảo
```

Trong MVP, so sánh YOLOv11-Seg với YOLOv8-Seg là đủ để chứng minh lựa chọn model.

#### 4.7.2. CV attribute and imprint benchmark

Attribute metrics:

- Shape: macro F1, per-class recall, confusion giữa `oval`, `oblong`, `capsule`.
- Color: macro F1, primary/secondary accuracy, performance theo lighting condition.
- Dosage form: accuracy và macro F1 cho `tablet`, `capsule`, `softgel`, `unknown`.
- Scoreline: F1 cho `none`, `single`, `cross`, `multiple`, `unknown` nếu có label.
- Imprint visibility: precision/recall/F1.
- Quality flags: precision/recall cho blur, glare, occlusion, possible merged instance và possible non-pill.

OCR metrics:

```text
text_region_detection_recall
text_region_detection_precision
Character Error Rate
exact_imprint_match_rate
Recall@1 / Recall@3 / Recall@5 của normalized_candidates
latency_per_pill
```

OCR benchmark cần tách riêng:

```text
pretrained PaddleOCR baseline
PaddleOCR + preprocessing cascade
PaddleOCR + multi-angle observation aggregation
fine-tuned recognizer, nếu có dữ liệu transcript đủ tốt
```

Điểm cần báo cáo:

- OCR có cải thiện sau preprocessing cascade không.
- OCR có sinh thêm false positive khi chạy nhiều góc không.
- Candidate đúng có nằm trong Top-k `normalized_candidates` không.
- Trường hợp OCR fail có được hạ xuống `partial_features` hoặc `insufficient_visual_evidence` đúng không.

#### 4.7.3. Retrieval benchmark

Retrieval benchmark đo khả năng đưa thuốc đúng vào Top-k candidate sau khi đã có CV metadata.

Metrics:

```text
Candidate Recall@1
Candidate Recall@5
Candidate Recall@10
Top-1 identification accuracy
False identification rate
Unknown detection rate
Expected Calibration Error
Top-1 vs Top-2 margin
```

Benchmark nên chạy ở hai chế độ:

```text
Oracle CV metadata
    dùng ground-truth imprint/shape/color/form
    → đo retrieval layer riêng

Predicted CV metadata
    dùng output thật từ CV
    → đo lỗi tích lũy end-to-end
```

Không được chỉ báo cáo Top-1 accuracy. Với hệ thống y tế, metric ưu tiên là:

```text
False Identification Rate
```

`identified` chỉ được chấp nhận khi:

- Top-1 đủ cao.
- Top-1 cách Top-2 đủ xa.
- Không vi phạm rule cứng.
- Evidence chính không đến từ fuzzy expansion score thấp.

#### 4.7.4. End-to-end safety benchmark

Benchmark end-to-end đánh giá toàn pipeline:

```text
image
    → segmentation
    → CV metadata
    → retrieval/ranking
    → identification_status
    → DDI lookup nếu có identified drugs
```

Test cases bắt buộc:

- Nhiều viên rời.
- Nhiều viên chạm nhau.
- Nhiều viên overlap.
- Một viên không đọc được imprint.
- Một viên có OCR nhầm `0/O`, `1/I`, `5/S`.
- Viên không có trong database.
- Vật thể không phải thuốc.
- Ảnh ánh sáng xấu.
- Một viên chưa xác định nhưng viên khác đã xác định.
- Hai thuốc khác tên nhưng cùng hoạt chất.

Metrics:

```text
end_to_end_identification_accuracy
false_identification_rate
ambiguous_rate
unknown_rate
ddi_pair_recall_on_identified_set
duplicate_ingredient_detection_accuracy
report_groundedness
```

End-to-end benchmark phải báo cáo riêng:

```text
coverage
    tỷ lệ viên được hệ thống dám identified

accuracy_on_accepted_cases
    accuracy chỉ trên các case identified

false_identification_rate
    tỷ lệ identified nhưng sai
```

MVP ưu tiên:

```text
false_identification_rate thấp
    hơn
coverage cao
```

#### 4.7.5. Calibration policy

Threshold cần chọn trên validation set:

- Detection confidence threshold.
- Mask threshold.
- `possible_merged_instance` threshold.
- `possible_non_pill` threshold.
- Attribute confidence threshold.
- OCR candidate threshold.
- Weighted edit cost threshold.
- Final ranking threshold.
- Top-1 vs Top-2 margin threshold.

Calibration methods có thể dùng:

```text
temperature scaling
isotonic regression
logistic regression calibration
reliability diagram
Expected Calibration Error
```

Nếu chưa calibration:

- UI/report không được trình bày score như xác suất y tế.
- Chỉ nên hiển thị trạng thái định tính như `high`, `medium`, `low` hoặc `identified`, `ambiguous`, `unknown`.
- Các quyết định safety vẫn dựa trên rule cứng và threshold thận trọng.
