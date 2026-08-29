# Schema v1 — Module Input/Output

Tài liệu này chỉ định nghĩa input và output giữa các module.

---

## 1. Single-image CV Request

Entrypoint CV hiện tại chạy một ảnh cho mỗi request. Với nhiều ảnh, client tạo một request độc lập cho từng ảnh; `side_hint` không nằm trong input contract của runner.

```json
{
  "request_id": "req_001",
  "session_id": "sess_001",
  "image_id": "img_001",
  "image_path": "data/benchmark/real_world/sample_001.jpg"
}
```

---

## 2. Module 1 — Segmentation

### Input

```json
{
  "request_id": "req_001",
  "session_id": "sess_001",
  "image_id": "img_001",
  "image_path": "data/benchmark/real_world/sample_001.jpg"
}
```

### Output

```json
{
  "request_id": "req_001",
  "session_id": "sess_001",
  "image_id": "img_001",
  "image_quality": {
    "status": "usable_with_warning",
    "blur_score": 0.21,
    "glare_detected": true,
    "lighting_warning": false
  },
  "instances": [
    {
      "instance_id": "pill_001",
      "instance_token": "pill_token_001",
      "bbox_xyxy": [142, 93, 326, 248],
      "mask_path": "outputs/masks/req_001/img_001/pill_001_clean_mask.png",
      "color_crop_path": "outputs/crops/req_001/img_001/pill_001_color_crop.png",
      "shape_crop_path": "outputs/crops/req_001/img_001/pill_001_shape_crop.png",
      "ocr_crop_path": "outputs/crops/req_001/img_001/pill_001_ocr_crop.png",
      "crop_path": "outputs/crops/req_001/img_001/pill_001_color_crop.png",
      "segmentation": {
        "confidence": 0.96,
        "occlusion_estimate": 0.12,
        "possible_merged_instance": false,
        "possible_non_pill": false
      },
      "quality_flags": ["minor_glare"]
    }
  ]
}
```

`mask_path` là clean mask được căn cùng color/OCR crop. Shape dùng RGB ROI riêng; dilation chỉ mở rộng vùng lấy ROI, không tạo foreground mask mới cho shape classifier.

---

## 3. Module 2 — Attribute Recognition

### Input

```json
{
  "request_id": "req_001",
  "session_id": "sess_001",
  "image_id": "img_001",
  "instance_id": "pill_001",
  "instance_token": "pill_token_001",
  "crop_path": "outputs/crops/req_001/img_001/pill_001_color_crop.png",
  "color_crop_path": "outputs/crops/req_001/img_001/pill_001_color_crop.png",
  "shape_crop_path": "outputs/crops/req_001/img_001/pill_001_shape_crop.png",
  "mask_path": "outputs/masks/req_001/img_001/pill_001_clean_mask.png"
}
```

### Output

```json
{
  "request_id": "req_001",
  "session_id": "sess_001",
  "image_id": "img_001",
  "instance_id": "pill_001",
  "instance_token": "pill_token_001",
  "shape": {
    "label": "oval",
    "confidence": 0.91,
    "alternatives": [
      {"label": "round", "confidence": 0.07}
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
    "label": "unknown",
    "confidence": null,
    "source": "not_predicted_by_attribute"
  },
  "scoreline": {
    "label": "unknown",
    "visible": null,
    "confidence": null,
    "source": "not_predicted_by_attribute"
  },
  "logo_or_symbol": {
    "visible": null,
    "confidence": null,
    "source": "not_predicted_by_attribute"
  },
  "damage_or_occlusion": {
    "visible": null,
    "confidence": null,
    "source": "not_predicted_by_attribute"
  }
}
```

Model Attribute hiện tại chỉ có hai head đã train là `shape` và `color`. Các trường còn lại phải giữ `unknown/null`; không được biến giá trị mặc định thành dự đoán thật. Riêng `scoreline` sẽ được Module 3 OCR cập nhật bằng kết quả Hough consensus.

---

## 4. Module 3 — Imprint OCR

### Input

```json
{
  "request_id": "req_001",
  "session_id": "sess_001",
  "image_id": "img_001",
  "instance_id": "pill_001",
  "instance_token": "pill_token_001",
  "crop_path": "outputs/crops/req_001/img_001/pill_001_ocr_crop.png",
  "mask_path": "outputs/masks/req_001/img_001/pill_001_clean_mask.png"
}
```

### Output

```json
{
  "request_id": "req_001",
  "session_id": "sess_001",
  "image_id": "img_001",
  "instance_id": "pill_001",
  "instance_token": "pill_token_001",
  "scoreline": {
    "visible": true,
    "confidence": 0.77,
    "angle_degrees": 79.99,
    "orientation": "vertical",
    "line_xyxy": [618.0, 259.0, 675.0, 582.0],
    "support_count": 5,
    "rotation_degrees": 180,
    "preprocessing": "blackhat_bold",
    "source": "ocr_hough_consensus"
  },
  "imprint_visibility": {
    "visible": true,
    "confidence": 0.86
  },
  "imprint": {
    "visible": true,
    "raw": "A O1",
    "confidence": 0.72,
    "text_regions": [
      {
        "region_id": "region_01",
        "polygon": [[32, 24], [58, 22], [60, 39], [33, 41]],
        "detection_confidence": 0.84
      }
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
        "preprocessing": "blackhat_bold",
        "text": "A O1",
        "confidence": 0.68
      }
    ],
    "normalized_candidates": [
      {
        "text": "A O1",
        "score": 0.91,
        "source": "multi_angle_consensus",
        "evidence": ["mode=full_crop", "rot=0", "rot=180", "preprocessing=clahe", "preprocessing=blackhat_bold"]
      },
      {
        "text": "A O1",
        "score": 0.84,
        "source": "raw_ocr",
        "evidence": ["raw_ocr"]
      }
    ]
  }
}
```

`scoreline` luôn có trong output Module 3, kể cả khi `imprint.visible = false`. `line_xyxy`, `angle_degrees` và `orientation` luôn quy về hệ tọa độ của crop gốc; `rotation_degrees` và `preprocessing` chỉ ghi variant tạo ra evidence tốt nhất. Consensus chỉ chấp nhận các line có góc và vị trí hình học tương thích. Nếu không đủ evidence để xác nhận scoreline, Module 3 trả `visible = false`, `confidence = 0.0`, các trường góc/đường/rotation/preprocessing bằng `null`; `support_count` ghi kích thước nhóm consensus hình học lớn nhất.

---

## 5. Module 4 — CV Pipeline

Khi fusion theo cùng `instance_token`, Module 4 phải lấy toàn bộ object `scoreline` từ output Module 3. Placeholder `scoreline` của Module 2 không được chuyển tiếp hoặc dùng để ghi đè kết quả OCR.

### Input

```json
{
  "segmentation_output": {},
  "attribute_outputs": [],
  "ocr_outputs": []
}
```

### Output

```json
{
  "schema_version": "cv_output_v1",
  "request_id": "req_001",
  "session_id": "sess_001",
  "image_id": "img_001",
  "image_quality": {
    "status": "usable_with_warning",
    "blur_score": 0.21,
    "glare_detected": true,
    "lighting_warning": false
  },
  "pills": [
    {
      "instance_id": "pill_001",
      "instance_token": "pill_token_001",
      "side_hint": "unknown",
      "cv_status": "features_ready",
      "bbox_xyxy": [142, 93, 326, 248],
      "mask_path": "outputs/masks/req_001/img_001/pill_001_clean_mask.png",
      "crop_path": "outputs/crops/req_001/img_001/pill_001_color_crop.png",
      "segmentation": {
        "confidence": 0.96,
        "occlusion_estimate": 0.12,
        "possible_merged_instance": false,
        "possible_non_pill": false
      },
      "shape": {
        "label": "oval",
        "confidence": 0.91,
        "alternatives": [
          {"label": "round", "confidence": 0.07}
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
        "label": "unknown",
        "confidence": null,
        "source": "not_predicted_by_attribute"
      },
      "scoreline": {
        "visible": true,
        "confidence": 0.77,
        "angle_degrees": 79.99,
        "orientation": "vertical",
        "line_xyxy": [618.0, 259.0, 675.0, 582.0],
        "support_count": 5,
        "rotation_degrees": 180,
        "preprocessing": "blackhat_bold",
        "source": "ocr_hough_consensus"
      },
      "logo_or_symbol": {
        "visible": null,
        "confidence": null,
        "source": "not_predicted_by_attribute"
      },
      "damage_or_occlusion": {
        "visible": null,
        "confidence": null,
        "source": "not_predicted_by_attribute"
      },
      "imprint_visibility": {
        "visible": true,
        "confidence": 0.86
      },
      "imprint": {
        "visible": true,
        "raw": "A O1",
        "confidence": 0.72,
        "normalized_candidates": [
          {
            "text": "A O1",
            "score": 0.91,
            "source": "multi_angle_consensus",
            "evidence": ["mode=full_crop", "rot=0", "rot=180", "preprocessing=clahe", "preprocessing=blackhat_bold"]
          },
          {
            "text": "A O1",
            "score": 0.84,
            "source": "raw_ocr",
            "evidence": ["raw_ocr"]
          }
        ],
        "ocr_observations": [
          {
            "region_id": "region_01",
            "rotation_degrees": 0,
            "preprocessing": "clahe",
            "text": "A O1",
            "confidence": 0.72
          }
        ]
      },
      "quality_flags": ["minor_glare"]
    }
  ]
}
```

---

## 6. Module 5 — Retrieval and Ranking

### Input

```json
{
  "schema_version": "rag_request_v1",
  "request_id": "req_001",
  "session_id": "sess_001",
  "market": "US",
  "known_drug_names": [],
  "cv_output": {
    "schema_version": "cv_output_v1",
    "request_id": "req_001",
    "session_id": "sess_001",
    "image_id": "img_001",
    "image_quality": {
      "status": "usable_with_warning",
      "blur_score": 0.21,
      "glare_detected": true,
      "lighting_warning": false
    },
    "pills": [
      {
        "instance_id": "pill_001",
        "instance_token": "pill_token_001",
        "side_hint": "unknown",
        "cv_status": "features_ready",
        "bbox_xyxy": [142, 93, 326, 248],
        "crop_path": "outputs/crops/req_001/img_001/pill_001_color_crop.png",
        "segmentation": {
          "confidence": 0.96,
          "possible_merged_instance": false,
          "possible_non_pill": false
        },
        "shape": {
          "label": "oval",
          "confidence": 0.91
        },
        "color": {
          "primary": "white",
          "secondary": null,
          "distribution": {
            "white": 0.72,
            "gray": 0.16
          },
          "confidence": 0.88,
          "lighting_warning": false
        },
        "dosage_form": {
          "label": "unknown",
          "confidence": null,
          "source": "not_predicted_by_attribute"
        },
        "scoreline": {
          "visible": true,
          "confidence": 0.77,
          "angle_degrees": 79.99,
          "orientation": "vertical",
          "line_xyxy": [618.0, 259.0, 675.0, 582.0],
          "support_count": 5,
          "rotation_degrees": 180,
          "preprocessing": "blackhat_bold",
          "source": "ocr_hough_consensus"
        },
        "logo_or_symbol": {
          "visible": null,
          "confidence": null,
          "source": "not_predicted_by_attribute"
        },
        "damage_or_occlusion": {
          "visible": null,
          "confidence": null,
          "source": "not_predicted_by_attribute"
        },
        "imprint_visibility": {
          "visible": true,
          "confidence": 0.86
        },
        "imprint": {
          "visible": true,
          "raw": "A O1",
          "confidence": 0.72,
          "normalized_candidates": [
            {
              "text": "A O1",
              "score": 0.91,
              "source": "multi_angle_consensus",
              "evidence": ["mode=full_crop", "rot=0", "rot=180", "preprocessing=clahe", "preprocessing=blackhat_bold"]
            },
            {
              "text": "A O1",
              "score": 0.84,
              "source": "raw_ocr",
              "evidence": ["raw_ocr"]
            }
          ]
        },
        "quality_flags": ["minor_glare"]
      }
    ]
  }
}
```

### Output

```json
{
  "schema_version": "rag_identification_v1",
  "request_id": "req_001",
  "session_id": "sess_001",
  "pill_results": [
    {
      "instance_id": "pill_001",
      "instance_token": "pill_token_001",
      "identification_status": "ambiguous",
      "required_action": "capture_reverse_side",
      "candidate_generation": {
        "strategy": "imprint_first",
        "queried_imprints": ["A 01", "A O1"],
        "num_records_before_dedup": 18,
        "num_records_after_dedup": 7
      },
      "top_candidates": [
        {
          "rank": 1,
          "appearance_id": "app_001",
          "product_id": "drug_1032",
          "product_name": "Candidate A",
          "final_score": 0.74,
          "evidence": {
            "best_imprint_candidate": "A 01",
            "imprint_match_score": 0.655,
            "shape_score": 0.91,
            "color_score": 0.72,
            "dosage_form_score": null,
            "scoreline_score": 0.77,
            "market_score": 1.0,
            "top1_top2_margin": 0.06,
            "hard_reject": false,
            "hard_reject_reasons": []
          }
        }
      ],
      "accepted_product": null,
      "scope_warning": "top_candidates_too_close"
    }
  ]
}
```

---

## 7. Module 6 — Drug Metadata and DDI Lookup

### Input

```json
{
  "schema_version": "ddi_request_v0",
  "request_id": "req_001",
  "session_id": "sess_001",
  "identified_products": [
    {
      "instance_id": "pill_001",
      "product_id": "drug_1032"
    },
    {
      "instance_id": "pill_002",
      "product_id": "drug_8471"
    }
  ]
}
```

### Output

```json
{
  "schema_version": "ddi_output_v0",
  "request_id": "req_001",
  "session_id": "sess_001",
  "identified_drugs": [
    {
      "instance_id": "pill_001",
      "product_id": "drug_1032",
      "product_name": "Example Brand 500 mg",
      "brand_name": "Example Brand",
      "generic_name": "example ingredient",
      "dosage_form": "tablet",
      "route": "oral",
      "ndc": "00000-0000-00",
      "market": "US",
      "active_ingredients": [
        {
          "ingredient_id": "ing_001",
          "name": "example ingredient",
          "strength": "500 mg",
          "rxcui": "123456"
        }
      ],
      "source": {
        "source_name": "DailyMed",
        "source_reference": "set_id_or_url",
        "last_updated": "2026-01-15"
      }
    }
  ],
  "duplicate_ingredient_warnings": [
    {
      "ingredient_id": "ing_001",
      "ingredient_name": "acetaminophen",
      "source_instance_ids": ["pill_001", "pill_003"],
      "severity": "major",
      "warning": "duplicate_ingredient"
    }
  ],
  "interactions": [
    {
      "interaction_id": "ddi_001",
      "ingredient_a_id": "ing_001",
      "ingredient_b_id": "ing_002",
      "ingredient_a_name": "ingredient A",
      "ingredient_b_name": "ingredient B",
      "source_instance_ids": ["pill_001", "pill_002"],
      "severity": "major",
      "clinical_risk": "Increased bleeding risk",
      "mechanism": "Additive pharmacodynamic effect",
      "management": "Avoid combination or monitor closely",
      "source": {
        "source_name": "DDInter",
        "source_reference": "ddi_record_or_url",
        "last_reviewed": "2026-01-15"
      }
    }
  ],
  "overall_severity": "major",
  "scope_warnings": [
    "only_identified_drugs_checked",
    "no_interaction_found_does_not_mean_safe"
  ]
}
```

---

## 8. Module 7 — Report Context Builder

### Input

```json
{
  "schema_version": "context_builder_input_v0",
  "request_id": "req_001",
  "session_id": "sess_001",
  "cv_output": {},
  "rag_identification": {},
  "ddi_output": {}
}
```

### Output

```json
{
  "schema_version": "llm_context_v0",
  "request_id": "req_001",
  "session_id": "sess_001",
  "task": "format_grounded_medication_safety_report",
  "identified_drugs": [],
  "unresolved_pills": [
    {
      "instance_id": "pill_001",
      "identification_status": "ambiguous",
      "reason": "top_candidates_too_close",
      "required_action": "capture_reverse_side"
    }
  ],
  "interactions": [],
  "duplicate_ingredient_warnings": [],
  "scope_warnings": [
    "only_identified_drugs_checked",
    "no_interaction_found_does_not_mean_safe"
  ],
  "sources": []
}
```

---

## 9. Module 8 — Deterministic Safety Report Formatter

### Input

```json
{
  "schema_version": "llm_context_v0",
  "request_id": "req_001",
  "session_id": "sess_001",
  "task": "format_grounded_medication_safety_report",
  "identified_drugs": [],
  "unresolved_pills": [],
  "interactions": [],
  "duplicate_ingredient_warnings": [],
  "scope_warnings": [],
  "sources": []
}
```

### Output

```json
{
  "schema_version": "llm_report_v0",
  "request_id": "req_001",
  "session_id": "sess_001",
  "overall_severity": "major",
  "provider_used": "fallback-deterministic-v0",
  "formatted_report_text": "... báo cáo theo severity, interaction records, unresolved pills và rule có sẵn ...",
  "structured_context": {}
}
```

Tên schema `llm_context_v0` và `llm_report_v0`, cùng biến môi trường `LLM_PROVIDER`, được giữ để tương thích code hiện có. Runtime hiện tại phải dùng `LLM_PROVIDER=fallback`: formatter không gọi LLM/API ngoài, chỉ ánh xạ context đã kiểm chứng thành banner, danh sách thuốc, chi tiết DDI, khuyến nghị và disclaimer theo rule cố định.
