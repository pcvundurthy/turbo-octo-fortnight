import cv2

class CustomDataset:
    def __init__(self, image_paths, mask_paths=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        if image is None:
            raise FileNotFoundError(f"Image not found or couldn't be loaded: {self.image_paths[idx]}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE) // 255

        image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LINEAR)
        # mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)

        return image  # or (image, mask) if you enable masks
