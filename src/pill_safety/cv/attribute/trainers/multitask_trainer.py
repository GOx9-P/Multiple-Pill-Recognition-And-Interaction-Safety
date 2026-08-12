import torch
from tqdm import tqdm

class MultiTaskTrainer:
    def __init__(self, model, optimizer, criterion_shape, criterion_color, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion_shape = criterion_shape
        self.criterion_color = criterion_color
        self.device = device

    def set_training_strategy(self, strategy="head_tune"):
        """
        Chiến thuật huấn luyện:
        - 'head_tune': Đóng băng toàn bộ backbone, chỉ cho phép head học.
        - 'last_blocks_fine_tune': Đóng băng phần đầu backbone và head, chỉ train các block cuối của backbone.
        """
        if strategy == "head_tune":
            # Đóng băng toàn bộ backbone
            for param in self.model.backbone.parameters():
                param.requires_grad = False
            # Mở khóa 2 head
            for param in self.model.shape_head.parameters():
                param.requires_grad = True
            for param in self.model.color_head.parameters():
                param.requires_grad = True
            print("[Trainer] Chiến thuật: Train HEAD (Đã đóng băng Backbone)")

        elif strategy == "last_blocks_fine_tune":
            # Đóng băng toàn bộ trước
            for param in self.model.parameters():
                param.requires_grad = False
            
            # Đóng băng Head (không train head ở giai đoạn này)
            for param in self.model.shape_head.parameters():
                param.requires_grad = False
            for param in self.model.color_head.parameters():
                param.requires_grad = False

            # Chỉ mở khóa các layer/block cuối cùng của ResNet18 (ví dụ: layer4)
            if hasattr(self.model.backbone, 'layer4'):
                for param in self.model.backbone.layer4.parameters():
                    param.requires_grad = True
            print("[Trainer] Chiến thuật: Train LAST BLOCKS (Đã đóng băng Head và tầng đầu Backbone)")
        else:
            raise ValueError(f"Không hỗ trợ chiến thuật: {strategy}")

    def train_epoch(self, shape_loader, color_loader1, color_loader2):
        self.model.train()
        self.optimizer.zero_grad()
        
        shape_iter = iter(shape_loader)
        color_iter1 = iter(color_loader1)
        color_iter2 = iter(color_loader2)
        
        num_steps = len(shape_loader)
        total_shape_loss = 0.0
        total_color_loss = 0.0

        for _ in range(num_steps):
            # 1. Batch Shape (Tỷ lệ 1)
            try:
                shape_imgs, shape_labels = next(shape_iter)
            except StopIteration:
                shape_iter = iter(shape_loader)
                shape_imgs, shape_labels = next(shape_iter)
                
            shape_imgs, shape_labels = shape_imgs.to(self.device), shape_labels.to(self.device)
            shape_preds = self.model(shape_imgs, task_type='shape')
            loss_shape = self.criterion_shape(shape_preds, shape_labels)
            loss_shape.backward()
            total_shape_loss += loss_shape.item()

            # 2. Batch Color 1
            try:
                color_imgs_1, color_labels_1 = next(color_iter1)
            except StopIteration:
                color_iter1 = iter(color_loader1)
                color_imgs_1, color_labels_1 = next(color_iter1)
                
            color_imgs_1, color_labels_1 = color_imgs_1.to(self.device), color_labels_1.to(self.device)
            color_preds_1 = self.model(color_imgs_1, task_type='color')
            loss_color_1 = self.criterion_color(color_preds_1, color_labels_1)
            loss_color_1.backward()

            # 3. Batch Color 2 (Đảm bảo tỷ lệ 1 shape : 2 color)
            try:
                color_imgs_2, color_labels_2 = next(color_iter2)
            except StopIteration:
                color_iter2 = iter(color_loader2)
                color_imgs_2, color_labels_2 = next(color_iter2)
                
            color_imgs_2, color_labels_2 = color_imgs_2.to(self.device), color_labels_2.to(self.device)
            color_preds_2 = self.model(color_imgs_2, task_type='color')
            loss_color_2 = self.criterion_color(color_preds_2, color_labels_2)
            loss_color_2.backward()
            
            avg_color_loss = (loss_color_1.item() + loss_color_2.item()) / 2.0
            total_color_loss += avg_color_loss

            # Tích lũy gradient hoàn tất cho nhóm batch, thực hiện optimizer step
            self.optimizer.step()
            self.optimizer.zero_grad()

        return total_shape_loss / num_steps, total_color_loss / num_steps