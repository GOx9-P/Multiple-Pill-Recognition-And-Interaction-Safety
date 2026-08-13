import torch
from sklearn.metrics import f1_score, accuracy_score  # Thêm accuracy_score nếu cần

def evaluate(model, shape_val_loader, color_val_loader, criterion_shape, criterion_color, device):
    """Hàm đánh giá độc lập trên tập Validation"""
    model.eval()
    val_loss = 0.0
    val_shape_loss = 0.0
    val_color_loss = 0.0
    
    all_s_preds, all_s_targets = [], []
    all_c_preds, all_c_targets = [], []
    
    with torch.no_grad():
        # Đánh giá nhánh Shape
        for s_imgs, s_labels in shape_val_loader:
            s_imgs = s_imgs.to(device)
            s_labels = s_labels.long().to(device)
            if s_labels.dim() > 1:
                s_labels = s_labels.squeeze(1)
                
            s_logits = model(s_imgs, task_type='shape')
            loss_s = criterion_shape(s_logits, s_labels)
            val_shape_loss += loss_s.item()
            
            all_s_preds.extend(torch.argmax(s_logits, dim=1).cpu().numpy())
            all_s_targets.extend(s_labels.cpu().numpy())
            
        # Đánh giá nhánh Color
        for c_imgs, c_labels in color_val_loader:
            c_imgs, c_labels = c_imgs.to(device), c_labels.float().to(device)
            
            c_logits = model(c_imgs, task_type='color')
            loss_c = criterion_color(c_logits, c_labels)
            val_color_loss += loss_c.item()
            
            c_preds_binary = (torch.sigmoid(c_logits) > 0.5).int()
            all_c_preds.extend(c_preds_binary.cpu().numpy())
            all_c_targets.extend(c_labels.int().cpu().numpy())
            
    avg_s_loss = val_shape_loss / max(len(shape_val_loader), 1)
    avg_c_loss = val_color_loss / max(len(color_val_loader), 1)
    avg_total_loss = avg_s_loss + avg_c_loss
    
    # --- Tính F1-Score ---
    val_shape_f1 = f1_score(all_s_targets, all_s_preds, average='macro', zero_division=0)
    val_color_f1 = f1_score(all_c_targets, all_c_preds, average='macro', zero_division=0)
    val_combined_f1 = (val_shape_f1 + val_color_f1) / 2.0
    
    # --- Tính Accuracy ---
    # Shape: Bài toán phân loại thông thường (Multi-class)
    val_shape_acc = accuracy_score(all_s_targets, all_s_preds)
    
    # Color: Tùy thuộc vào dạng bài toán của bạn
    # Trường hợp 1: Multi-label (mỗi ảnh có thể có nhiều màu cùng lúc) -> Dùng Subset Accuracy (khớp chính xác 100% các nhãn trên mỗi mẫu)
    val_color_acc = accuracy_score(all_c_targets, all_c_preds) 
    
    # Trường hợp 2: Nếu Color thực chất là Multi-class (chỉ chọn 1 màu duy nhất) nhưng dùng One-hot/Sigmoid, 
    # bạn cần argmax cho cả predictions và targets giống như nhánh Shape.
    
    val_combined_acc = (val_shape_acc + val_color_acc) / 2.0
    
    metrics = {
        "val_loss": round(avg_total_loss, 4),
        "shape_loss": round(avg_s_loss, 4),
        "color_loss": round(avg_c_loss, 4),
        "shape_f1": round(float(val_shape_f1), 4),
        "color_f1": round(float(val_color_f1), 4),
        "combined_f1": round(float(val_combined_f1), 4),
        "shape_acc": round(float(val_shape_acc), 4),
        "color_acc": round(float(val_color_acc), 4),
        "combined_acc": round(float(val_combined_acc), 4)
    }
    return metrics