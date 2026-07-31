# Project Report


**EN:** Recognizing Multiple Pills and Drug Interactions Using YOLOv8-Seg and LLM-RAG on Python for Medication Safety  
**VN:** Nhận diện Đa dược và Cảnh báo Tương tác thuốc sử dụng YOLOv8-Seg và LLM-RAG trên Python cho An toàn Dược lâm sàng

---

## 1. Abstract
Polypharmacy là nguyên nhân hàng đầu dẫn đến các sự cố y khoa liên quan đến Drug-Drug Interactions (DDI), đặc biệt ở người cao tuổi. Dự án này đề xuất một Hybrid AI Architecture kết hợp sức mạnh của Computer Vision và Natural Language Processing.

Để giải quyết đồng thời thách thức về môi trường chồng lấp phức tạp (Occlusion) và độ đa dạng chủng loại thuốc (>100 nhãn), khối Thị giác được thiết kế theo kiến trúc **Decoupled Vision Pipeline**: kết hợp mô hình Class-Agnostic Instance Segmentation (**YOLO11-Seg**) để định vị và bóc tách viền thực thể (`Foreground Mask`), tiếp nối bởi mạng **Fine-grained Pill Classifier** để phân loại chính xác mã định danh thuốc. Sau đó, dữ liệu đầu ra được chuẩn hóa thành cấu trúc văn bản và đưa vào **LLM** kết hợp kỹ thuật **RAG** để truy xuất cơ sở dữ liệu y khoa chuẩn, nhằm sinh ra các khuyến cáo lâm sàng chính xác bằng ngôn ngữ tự nhiên. Phương pháp này giải quyết triệt để rủi ro Hallucination của LLM trong y khoa, đồng thời đảm bảo yêu cầu Real-time Inference.

---

## 2. Introduction
**Clinical Problem Statement:** Bệnh nhân mãn tính thường sử dụng Dosette Boxes và loại bỏ bao bì gốc. Khi các viên thuốc bị rơi hoặc lẫn lộn, việc phân biệt bằng mắt thường dựa trên màu sắc và hình dáng là cực kỳ khó khăn. Uống nhầm thuốc hoặc phối hợp các thuốc có tương tác xấu (ví dụ: NSAIDs và Anticoagulants) có thể dẫn đến biến chứng lâm sàng nghiêm trọng.

**Limitations of Existing Approaches:**
1. Các mô hình Traditional Bounding Box Object Detection gặp sai số lớn (False Positives/False Negatives) khi các viên thuốc có hiện tượng chồng lấp (Occlusion / Overlap).
2. Việc sử dụng một mô hình Single-stage duy nhất vừa bóc tách vừa phân loại hàng trăm loại thuốc trong môi trường thực tế dễ bị suy giảm độ chính xác và khó mở rộng nhãn mới.
3. Các LLM (như GPT-4, Llama) có năng lực lý luận tốt nhưng thiếu căn cứ y khoa chuẩn xác, dễ phát sinh Hallucination nếu hoạt động độc lập.

**Project Objectives:** Thiết kế kiến trúc **Decoupled Vision-Reasoning** cho phép người dùng chỉ cần chụp một bức ảnh RGB đơn lẻ các viên thuốc, hệ thống sẽ tự động định danh chính xác từng viên và đưa ra khuyến cáo y khoa an toàn.

---

## 6. Practical & Commercial Applications

Hệ thống **Smart Pill Identifier** mang lại giá trị ứng dụng thực tiễn sâu sắc và tiềm năng thương mại hóa lớn trong 4 lĩnh vực trọng tâm của hệ sinh thái **HealthTech**:

### 6.1. Home Care Safety
*   **Clinical Problem:** Người cao tuổi mắc bệnh mãn tính (tiểu đường, tăng huyết áp, tim mạch) thường phải sử dụng 5–10 loại thuốc mỗi ngày. Khi bóc vỏ thuốc và sắp xếp vào các **Dosette Boxes** theo ngày/tuần, nếu lỡ tay làm rơi hoặc lẫn lộn các viên thuốc, bệnh nhân gần như không thể phân biệt bằng mắt thường. Việc uống nhầm liều hoặc phối hợp sai các hoạt chất kỵ nhau (ví dụ: `NSAIDs` kết hợp `Anticoagulants`) tiềm ẩn rủi ro xuất huyết nội tạng hoặc nguy hiểm đến tính mạng.
*   **AI Solution:** Hệ thống đóng vai trò như một **Safety Net** (chốt chặn an toàn cuối cùng). Người dùng chỉ cần sử dụng camera điện thoại quét qua vốc thuốc chuẩn bị uống, hệ thống thực hiện Real-time Inference, định danh chính xác từng viên và lập tức phát cảnh báo đỏ nếu phát hiện nguy cơ Drug-Drug Interactions (DDI).

### 6.2. Nursing Home Audit
*   **Clinical Problem:** Tại các viện dưỡng lão hoặc bệnh viện tuyến dưới, điều dưỡng viên phải thực hiện chia thuốc thủ công cho hàng chục đến hàng trăm bệnh nhân mỗi ngày. Áp lực khối lượng công việc lớn khiến tỷ lệ cấp phát nhầm thuốc tăng cao. Nếu phát nhầm thuốc dẫn đến sốc phản vệ hoặc tai biến lâm sàng, cơ sở y tế đối mặt với rủi ro pháp lý lớn, các vụ kiện tụng bồi thường hàng triệu USD và tước giấy phép hoạt động.
*   **AI Solution:** Hệ thống camera tích hợp AI được lắp đặt trực tiếp trên bàn chia thuốc. Trước khi điều dưỡng viên mang khay thuốc đi cấp phát, hệ thống tự động quét qua (`Audit`) để đối chiếu 100% danh sách viên thuốc trong khay với **Electronic Health Records (EHR)** của bệnh nhân, loại bỏ hoàn toàn sai sót do con người (`Human Error`).

### 6.3. InsurTech Economic Value
*   **Business Problem:** Các công ty bảo hiểm y tế và bảo hiểm nhân thọ hàng năm phải chi trả ngân sách khổng lồ cho các ca cấp cứu nhập viện phát sinh từ **Adverse Drug Events (ADE)** do bệnh nhân uống nhầm thuốc hoặc sử dụng sai chỉ định tại nhà.
*   **AI Solution:** Hệ thống giải quyết trực tiếp bài toán tối ưu chi phí bồi thường cho ngành **InsurTech**. Các công ty bảo hiểm có động lực lớn để tài trợ hoặc tích hợp miễn phí phần mềm này vào ứng dụng di động cho khách hàng sử dụng hàng ngày. Chi phí duy trì hệ thống AI thấp hơn hàng nghìn lần so với chi phí thanh toán viện phí cấp cứu phát sinh từ ADE.

### 6.4. Telemedicine Ecosystem Integration
*   **Clinical Problem:** Trong các buổi tư vấn khám bệnh từ xa (`Video Consultation`), bác sĩ thường yêu cầu bệnh nhân liệt kê các thuốc đang sử dụng tại nhà. Bệnh nhân (đặc biệt là người già) thường không nhớ tên khoa học của hoạt chất mà chỉ mô tả cảm tính (ví dụ: *"viên màu trắng tròn tròn"*), khiến bác sĩ gặp rào cản lớn khi kê đơn thuốc mới vì lo ngại tương tác thuốc.
*   **AI Solution:** Bệnh nhân chỉ cần đặt các viên thuốc đang sử dụng lên bàn và chụp một bức ảnh. Hệ thống lập tức bóc tách, nhận diện và trích xuất danh sách hoạt chất chuẩn xác dưới dạng **JSON payload** gửi thẳng lên màn hình quản lý của bác sĩ. Bác sĩ có đầy đủ cơ sở dữ liệu lâm sàng để tự tin kê đơn mới mà không rủi ro tương tác.

---

## 7. Conclusion
Dự án **Smart Pill Identifier** chứng minh sự hiệu quả của việc kết hợp các công nghệ AI hiện đại (**Instance Segmentation**, **Fine-grained Classification** và **Retrieval-Augmented Generation**) trong lĩnh vực **HealthTech**. Thay vì sử dụng một mô hình End-to-End khổng lồ, kiến trúc **Decoupled Vision-Reasoning** cùng chiến lược **Tách module CV (`Class-Agnostic Segmentation` + `Classifier`)** giúp tối ưu hóa tài nguyên phần cứng, dễ dàng mở rộng cơ sở dữ liệu lên hàng nghìn loại thuốc mà không gây nghẽn cổ chai gán nhãn, đảm bảo Low Latency và loại bỏ tuyệt đối rủi ro Hallucination, bảo vệ tính toàn vẹn của thông tin y tế.