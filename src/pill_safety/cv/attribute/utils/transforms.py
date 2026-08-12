from torchvision import transforms

def get_shape_transforms(image_size=224):
    """Pipeline chuẩn cho tập shape đã được augment offline (biến đổi hình học)"""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_color_transforms(image_size=224):
    """Pipeline chuẩn cho tập color đã được augment offline (giữ ổn định màu)"""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])