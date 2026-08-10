from __future__ import annotations

import json
import os
import urllib.request
from typing import Any

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass



SYSTEM_PROMPT = """Bạn là Trợ Lý Dược Lý Lâm Sàng Y Tế chuyên nghiệp. 
Nhiệm vụ của bạn là đọc dữ liệu bối cảnh JSON (grounded context) được cung cấp và tổng hợp thành một BÁO CÁO PHÂN TÍCH TƯƠNG TÁC THUỐC bằng Tiếng Việt chuẩn hóa, rõ ràng và giàu đồng cảm theo đúng cấu trúc bên dưới.

NGUYÊN TẮC AN TOÀN Y TẾ BẮT BUỘC (CRITICAL SAFETY RULES):
1. TUYỆT ĐỐI KHÔNG ẢO GIÁC THUỐC/TƯƠNG TÁC (ZERO HALLUCINATION FOR PAIRS): Chỉ được phân tích các cặp tương tác được liệt kê trực tiếp trong mảng `interactions` hoặc `duplicate_ingredient_warnings`. Tuyệt đối không tự bịa thêm các cặp thuốc xung đột khác ngoài bối cảnh.
2. NÂNG CAO CHẤT LƯỢNG GIẢI THÍCH TÁC HẠI: Đối với các cặp tương tác có trong mảng `interactions`, nếu thông tin `clinical_risk` hay `mechanism` trong bối cảnh mang câu định khuôn thô từ DB (như "DDInter classifies this pair as..."), hãy vận dụng tri thức y khoa dược lý chuẩn hóa của bạn về cặp hoạt chất đó để giải thích CHI TIẾT, CHÍNH XÁC và RÕ RÀNG về cơ chế sinh lý học và BIỂU HIỆN / HẬU QUẢ SỨC KHỎE THỰC TẾ đối với cơ thể người bệnh khi uống kết hợp (ví dụ: Ciprofloxacin ức chế chuyển hóa Warfarin làm tăng nguy cơ chảy máu; Ibuprofen gây giữ muối nước làm suy giảm tác dụng hạ huyết áp của Amlodipine...). Tuyệt đối không lặp lại câu lý thuyết chung chung "có thể làm trầm trọng bệnh..."!
3. PHÂN LOẠI 4 MỨC ĐỘ BÁO ĐỘNG TỔNG THỂ (OVERALL SEVERITY BANNER):
   - Nếu `overall_severity` là `contraindicated` hoặc `major`:
     Dùng banner: [🔴 MỨC ĐỘ BÁO ĐỘNG: CỰC KỲ NGUY HIỂM]
     Kèm câu cảnh báo: ⚠️ CẢNH BÁO: Phát hiện tương tác có thể GÂY TỬ VONG hoặc TÀN TẬT VĨNH VIỄN. Vui lòng KHÔNG UỐNG kết hợp các thuốc này khi chưa có chỉ định của Bác sĩ!
   - Nếu `overall_severity` là `moderate`:
     Dùng banner: [🟧 MỨC ĐỘ BÁO ĐỘNG: TRUNG BÌNH - CẦN THẬN TRỌNG]
     Kèm câu cảnh báo: ⚠️ CẢNH BÁO: Phát hiện tương tác có thể gây tổn thương mạn tính hoặc giảm hiệu quả điều trị nếu dùng lâu dài.
   - Nếu `overall_severity` là `minor` hoặc chứa `duplicate_ingredient_warnings`:
     Dùng banner: [🟨 MỨC ĐỘ BÁO ĐỘNG: NHẸ / CẦN LƯU Ý]
     Kèm câu cảnh báo: ℹ️ LƯU Ý: Phát hiện ảnh hưởng nhẹ hoặc trùng lặp hoạt chất. Cần theo dõi biểu hiện cơ thể.
   - Nếu `overall_severity` là `none`:
     Dùng banner: [🟢 TÌNH TRẠNG: AN TOÀN - KHÔNG PHÁT HIỆN TƯƠNG TÁC XUNG ĐỘT]
     Kèm câu khẳng định: ✅ Đơn thuốc hiện tại chưa phát hiện tương tác bất lợi trong cơ sở dữ liệu.

4. CẤU TRÚC BÁO CÁO TRÌNH BÀY ĐÚNG 3 PHẦN:
================================================================================
                    KẾT QUẢ PHÂN TÍCH TƯƠNG TÁC THUỐC
================================================================================

[MỨC ĐỘ BÁO ĐỘNG VÀ CÂU CẢNH BÁO TƯƠNG ỨNG Ở TRÊN]

--------------------------------------------------------------------------------
1. KẾT QUẢ NHẬN DIỆN THUỐC
--------------------------------------------------------------------------------
✅ Đã nhận diện được (X/Y thuốc):
   (Liệt kê tên thuốc thương mại + hoạt chất chính + hàm lượng)

❓ Chưa nhận diện được (Z thuốc):
   (Liệt kê các viên chưa xác định, kèm gợi ý: 👉 [Nút hành động]: Click vào đây để gõ/tìm lại tên thuốc chuẩn)

--------------------------------------------------------------------------------
2. CHI TIẾT CÁC TƯƠNG TÁC GÂY HẠI
--------------------------------------------------------------------------------
YÊU CẦU BẮT BUỘC CHO MỖI TƯƠNG TÁC:
- Cặp thuốc xung đột: Tên thuốc / Hoạt chất A ⚡ Tên thuốc / Hoạt chất B
- Tác hại cụ thể: Trình bày CHI TIẾT, SINH ĐỘNG và RÕ RÀNG về cơ chế lâm sàng cùng BIỂU HIỆN / HẬU QUẢ SỨC KHỎE THỰC TẾ đối với cơ thể người bệnh khi uống kết hợp.
- Lời khuyên dùng: Hướng dẫn hành động cụ thể cho bệnh nhân và giải pháp theo dõi / điều chỉnh.
- Thuốc thay thế đề xuất: (Gợi ý thuốc thay thế an toàn hơn nếu có trong bối cảnh)
- Trích xuất nguồn y khoa: (Nêu rõ nguồn y tế và đường dẫn URL tham chiếu từ source_name và source_reference trong bối cảnh)

--------------------------------------------------------------------------------
3. TỔNG KẾT KHUYẾN CÁO VÀ HƯỚNG XỬ LÝ
--------------------------------------------------------------------------------
(Đề xuất tổng quan: ⛔ KHÔNG NÊN UỐNG / ⚠️ CẦN THAM KHẢO BÁC SĨ / ✅ AN TOÀN)
Khuyến nghị hành động (1, 2, 3...)

--------------------------------------------------------------------------------
* Disclaimer: Kết quả phân tích dựa trên dữ liệu y khoa chuẩn hóa (RxNorm/DDInter/DailyMed). 
Thông tin chỉ mang tính chất tham khảo và không thay thế cho chẩn đoán y khoa.
================================================================================
"""



class FallbackLlmProvider:
    """Fallback Deterministic Formatter using exact rule-based mapping when LLM API Key is unavailable."""

    def generate(self, context: dict[str, Any]) -> str:
        identified_drugs = context.get("identified_drugs") or []
        unresolved_pills = context.get("unresolved_pills") or []
        interactions = context.get("interactions") or []
        duplicate_warnings = context.get("duplicate_ingredient_warnings") or []

        # Determine overall severity
        severity_ranks = {"contraindicated": 4, "major": 3, "moderate": 2, "minor": 1, "unclassified": 1, "none": 0}

        max_rank = 0
        overall_severity = "none"
        for inter in interactions:
            sev = inter.get("severity", "none")
            rank = severity_ranks.get(sev, 0)
            if rank > max_rank:
                max_rank = rank
                overall_severity = sev

        if duplicate_warnings and max_rank < severity_ranks["major"]:
            overall_severity = "major"

        # Banner mapping
        if overall_severity in ("contraindicated", "major"):
            banner = "[🔴 MỨC ĐỘ BÁO ĐỘNG: CỰC KỲ NGUY HIỂM]"
            warning = "⚠️ CẢNH BÁO: Phát hiện tương tác có thể GÂY TỬ VONG hoặc TÀN TẬT VĨNH VIỄN. Vui lòng KHÔNG UỐNG kết hợp các thuốc này khi chưa có chỉ định của Bác sĩ!"
        elif overall_severity == "moderate":
            banner = "[🟧 MỨC ĐỘ BÁO ĐỘNG: TRUNG BÌNH - CẦN THẬN TRỌNG]"
            warning = "⚠️ CẢNH BÁO: Phát hiện tương tác có thể gây tổn thương mạn tính hoặc suy giảm chức năng cơ quan nếu sử dụng kéo dài."
        elif overall_severity == "minor" or duplicate_warnings:
            banner = "[🟨 MỨC ĐỘ BÁO ĐỘNG: NHẸ / CẦN LƯU Ý]"
            warning = "ℹ️ LƯU Ý: Phát hiện ảnh hưởng nhẹ hoặc trùng lặp hoạt chất trong đơn thuốc. Cần theo dõi biểu hiện cơ thể."
        else:
            banner = "[🟢 TÌNH TRẠNG: AN TOÀN - KHÔNG PHÁT HIỆN TƯƠNG TÁC XUNG ĐỘT]"
            warning = "✅ Đơn thuốc hiện tại chưa phát hiện tương tác bất lợi trong cơ sở dữ liệu y tế."

        total_count = len(identified_drugs) + len(unresolved_pills)

        # Section 1: Identified & Unresolved
        sec1_lines = [f"✅ Đã nhận diện được ({len(identified_drugs)}/{total_count} thuốc):"]
        for idx, drug in enumerate(identified_drugs, 1):
            name = drug.get("product_name") or drug.get("brand_name") or "Thuốc không tên"
            ings = drug.get("active_ingredients") or []
            ing_str = ", ".join(i.get("name") for i in ings if i.get("name"))
            if ing_str:
                sec1_lines.append(f"   {idx}. {name} (Hoạt chất chính: {ing_str})")
            else:
                sec1_lines.append(f"   {idx}. {name}")

        if unresolved_pills:
            sec1_lines.append("")
            sec1_lines.append(f"❓ Chưa nhận diện được ({len(unresolved_pills)} thuốc):")
            for pill in unresolved_pills:
                inst_id = pill.get("instance_id", "pill_unk")
                reason = pill.get("reason", "Cần kiểm tra hình ảnh")
                sec1_lines.append(f"   • Viên thuốc mã #{inst_id} ({reason})")
                sec1_lines.append(f"   👉 [Nút hành động]: [Click vào đây để gõ/tìm lại tên thuốc chuẩn cho #{inst_id}]")

        sec1_text = "\n".join(sec1_lines)

        # Section 2: Interactions detail
        sec2_lines = []
        if interactions:
            for idx, inter in enumerate(interactions, 1):
                sev = inter.get("severity", "moderate")
                if sev in ("contraindicated", "major"):
                    sev_title = "🔴 TƯƠNG TÁC CỰC KỲ NGUY HIỂM (Nguy cơ tử vong / độc tính cao)"
                elif sev == "moderate":
                    sev_title = "🟧 TƯƠNG TÁC TRUNG BÌNH (Tác hại lâu dài / suy giảm chức năng)"
                else:
                    sev_title = "🟨 TƯƠNG TÁC YẾU / NHẸ (Ảnh hưởng nhẹ)"

                prod_a = inter.get("ingredient_a_name", "Thuốc A")
                prod_b = inter.get("ingredient_b_name", "Thuốc B")
                risk = inter.get("clinical_risk") or inter.get("mechanism") or "Cần thận trọng khi dùng chung."
                mgmt = inter.get("management") or inter.get("recommendation") or "Tham khảo ý kiến bác sĩ."
                alt = inter.get("alternative")
                src_name = inter.get("source_name") or "DDInter 2.0"
                src_ref = inter.get("source_reference") or "https://ddinter2.scbdd.com/"

                sec2_lines.append(f"{sev_title}")
                sec2_lines.append(f"- Cặp thuốc xung đột: {prod_a} ⚡ {prod_b}")
                sec2_lines.append(f"- Tác hại cụ thể: {risk}")
                sec2_lines.append(f"- Khuyên dùng: {mgmt}")
                if alt:
                    sec2_lines.append(f"- Thuốc thay thế đề xuất: {alt}")
                sec2_lines.append(f"- Trích xuất nguồn y khoa: Nguồn {src_name} ({src_ref})")
                sec2_lines.append("")
        elif duplicate_warnings:
            for idx, dup in enumerate(duplicate_warnings, 1):
                ing_name = dup.get("ingredient_name", "Hoạt chất")
                sec2_lines.append(f"🔴 CẢNH BÁO TRÙNG LẶP HOẠT CHẤT: {ing_name}")
                sec2_lines.append(f"- Tác hại cụ thể: Đơn thuốc chứa nhiều hơn 1 sản phẩm có cùng hoạt chất {ing_name}, nguy cơ gây quá liều độc tính.")
                sec2_lines.append("- Khuyên dùng: Kiểm tra lại đơn thuốc và không uống đồng thời hai thuốc chứa cùng hoạt chất.")
                sec2_lines.append("")
        else:
            sec2_lines.append("Không ghi nhận tương tác bất lợi nào giữa các thuốc được định danh.")

        sec2_text = "\n".join(sec2_lines).strip()


        sec2_text = "\n".join(sec2_lines).strip()

        # Section 3: Recommendations
        if overall_severity in ("contraindicated", "major"):
            recommendation_header = "⛔ HỆ THỐNG ĐỀ XUẤT: KHÔNG NÊN UỐNG ĐƠN THUỐC NÀY."
            rec_steps = [
                "1. Tạm ngưng việc uống kết hợp các thuốc xung đột nguy hiểm nêu trên.",
                "2. Mang danh sách thuốc này trao đổi ngay lại với Bác sĩ hoặc Dược sĩ lâm sàng.",
                "3. Kiểm tra bổ sung các thuốc chưa nhận diện được (nếu có) để có kết quả tổng thể chính xác nhất."
            ]
        elif overall_severity == "moderate":
            recommendation_header = "⚠️ HỆ THỐNG ĐỀ XUẤT: CẦN THAM KHẢO Ý KIẾN BÁC SĨ TRƯỚC KHI UỐNG."
            rec_steps = [
                "1. Tham khảo ý kiến bác sĩ để giãn thời gian uống thuốc hoặc theo dõi chức năng cơ quan.",
                "2. Không tự ý tăng liều lượng sử dụng.",
                "3. Cập nhật tên đầy đủ của các thuốc chưa nhận diện được."
            ]
        else:
            recommendation_header = "✅ HỆ THỐNG ĐỀ XUẤT: ĐƠN THUỐC KHÔNG PHÁT HIỆN TƯƠNG TÁC XUNG ĐỘT GÂY HẠI."
            rec_steps = [
                "1. Sử dụng thuốc theo đúng liều lượng chỉ định của Bác sĩ/Dược sĩ.",
                "2. Nếu có biểu hiện bất thường, ngưng sử dụng và liên hệ cơ sở y tế gần nhất."
            ]

        rec_text = "\n".join(rec_steps)

        # Assemble full text
        return f"""================================================================================
                    KẾT QUẢ PHÂN TÍCH TƯƠNG TÁC THUỐC
================================================================================

{banner}
--------------------------------------------------------------------------------
{warning}

--------------------------------------------------------------------------------
1. KẾT QUẢ NHẬN DIỆN THUỐC
--------------------------------------------------------------------------------
{sec1_text}

--------------------------------------------------------------------------------
2. CHI TIẾT CÁC TƯƠNG TÁC GÂY HẠI
--------------------------------------------------------------------------------
{sec2_text}

--------------------------------------------------------------------------------
3. TỔNG KẾT KHUYẾN CÁO VÀ HƯỚNG XỬ LÝ
--------------------------------------------------------------------------------
{recommendation_header}

Khuyến nghị hành động:
{rec_text}

--------------------------------------------------------------------------------
* Disclaimer: Kết quả phân tích dựa trên dữ liệu y khoa chuẩn hóa (RxNorm/DDInter/DailyMed). 
Thông tin chỉ mang tính chất tham khảo và không thay thế cho chẩn đoán y khoa.
================================================================================
"""


class GeminiLlmProvider:
    """LLM Provider that calls Google Gemini API via REST Endpoint."""

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash") -> None:
        self.api_key = api_key
        self.model = model

    def generate(self, context: dict[str, Any]) -> str:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        user_prompt = f"{SYSTEM_PROMPT}\n\nDữ liệu bối cảnh JSON (Grounded Context):\n{json.dumps(context, ensure_ascii=False, indent=2)}"
        
        payload = {
            "contents": [
                {
                    "parts": [{"text": user_prompt}]
                }
            ]
        }
        
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                resp_data = json.loads(resp.read().decode("utf-8"))
                candidates = resp_data.get("candidates") or []
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts and "text" in parts[0]:
                        return str(parts[0]["text"])
        except Exception:
            pass
        
        # Fallback if Gemini call fails
        return FallbackLlmProvider().generate(context)


class LlmReportGenerator:
    """High-level LLM Report Generator service."""

    def __init__(self, provider_name: str | None = None, api_key: str | None = None) -> None:
        self.provider_name = provider_name or os.environ.get("LLM_PROVIDER", "fallback").lower()
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        
        if self.provider_name in ("gemini", "google") and self.api_key:
            self.provider: Any = GeminiLlmProvider(self.api_key)
            self.effective_provider_name = "gemini-2.5-flash"
        else:
            self.provider = FallbackLlmProvider()
            self.effective_provider_name = "fallback-deterministic-v0"

    def generate_report(self, context: dict[str, Any]) -> dict[str, Any]:
        report_text = self.provider.generate(context)
        
        # Determine overall severity from context
        interactions = context.get("interactions") or []
        duplicate_warnings = context.get("duplicate_ingredient_warnings") or []
        severity_ranks = {"contraindicated": 4, "major": 3, "moderate": 2, "minor": 1, "unclassified": 1, "none": 0}

        max_rank = 0
        overall_severity = "none"
        for inter in interactions:
            sev = inter.get("severity", "none")
            rank = severity_ranks.get(sev, 0)
            if rank > max_rank:
                max_rank = rank
                overall_severity = sev

        if duplicate_warnings and max_rank < severity_ranks["major"]:
            overall_severity = "major"

        return {
            "schema_version": "llm_report_v0",
            "request_id": context.get("request_id"),
            "session_id": context.get("session_id"),
            "overall_severity": overall_severity,
            "provider_used": self.effective_provider_name,
            "formatted_report_text": report_text,
            "structured_context": context
        }
