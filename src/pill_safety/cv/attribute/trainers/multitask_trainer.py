"""Reusable training loop cho multi-task shape/color."""


class MultiTaskTrainer:
    """Huấn luyện shared backbone với batch shape và nhiều batch color mỗi step."""

    def __init__(self, model, optimizer, criterion_shape, criterion_color, device, lambda_color=1.0):
        self.model = model
        self.optimizer = optimizer
        self.criterion_shape = criterion_shape
        self.criterion_color = criterion_color
        self.device = device
        self.lambda_color = lambda_color

    def set_training_strategy(self, strategy, trainable_layers=None):
        """Freeze/unfreeze dung cac layer ResNet theo strategy cau hinh."""
        for parameter in self.model.parameters():
            parameter.requires_grad = False

        if strategy == "head_tune":
            trainable_modules = [self.model.shape_head, self.model.color_head]
        elif strategy == "last_blocks_finetune":
            selected = trainable_layers or ["layer3", "layer4", "classification_heads"]
            trainable_modules = []
            for layer_name in ("layer3", "layer4"):
                if layer_name in selected:
                    trainable_modules.append(getattr(self.model.backbone, layer_name))
            if "classification_heads" in selected:
                trainable_modules.extend([self.model.shape_head, self.model.color_head])
        else:
            raise ValueError(f"Unsupported training strategy: {strategy}")

        for module in trainable_modules:
            for parameter in module.parameters():
                parameter.requires_grad = True

        names = [name for name, parameter in self.model.named_parameters() if parameter.requires_grad]
        if not names:
            raise RuntimeError("No trainable parameters after applying training strategy.")
        return names

    @staticmethod
    def _next_batch(iterator, loader):
        """Lay batch tiep theo va quay lai dau loader khi da het epoch logic."""
        try:
            return next(iterator), iterator
        except StopIteration:
            iterator = iter(loader)
            return next(iterator), iterator

    def train_epoch(self, shape_loader, color_loader, steps_per_epoch, color_batches_per_step=2):
        """Chay 1 shape batch + N color batch, tich luy gradient va step mot lan."""
        if color_batches_per_step < 1:
            raise ValueError("color_batches_per_step must be at least 1.")

        self.model.train()
        shape_iterator = iter(shape_loader)
        color_iterator = iter(color_loader)
        totals = {"loss": 0.0, "shape_loss": 0.0, "color_loss": 0.0, "shape_samples": 0, "color_samples": 0}

        for _ in range(steps_per_epoch):
            self.optimizer.zero_grad(set_to_none=True)
            (shape_images, shape_labels), shape_iterator = self._next_batch(shape_iterator, shape_loader)
            shape_images = shape_images.to(self.device, non_blocking=True)
            shape_labels = shape_labels.long().to(self.device, non_blocking=True).reshape(-1)
            shape_loss = self.criterion_shape(self.model(shape_images, task_type="shape"), shape_labels)

            color_losses = []
            color_samples = 0
            for _ in range(color_batches_per_step):
                (color_images, color_labels), color_iterator = self._next_batch(color_iterator, color_loader)
                color_images = color_images.to(self.device, non_blocking=True)
                color_labels = color_labels.float().to(self.device, non_blocking=True)
                color_losses.append(self.criterion_color(self.model(color_images, task_type="color"), color_labels))
                color_samples += color_images.shape[0]

            mean_color_loss = sum(color_losses) / len(color_losses)
            total_loss = shape_loss + self.lambda_color * mean_color_loss
            total_loss.backward()
            self.optimizer.step()

            totals["loss"] += total_loss.item()
            totals["shape_loss"] += shape_loss.item()
            totals["color_loss"] += mean_color_loss.item()
            totals["shape_samples"] += shape_images.shape[0]
            totals["color_samples"] += color_samples

        return {
            "train_loss": totals["loss"] / steps_per_epoch,
            "shape_loss": totals["shape_loss"] / steps_per_epoch,
            "color_loss": totals["color_loss"] / steps_per_epoch,
            "shape_samples": totals["shape_samples"],
            "color_samples": totals["color_samples"],
        }
