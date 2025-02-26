import cv2
import argparse
import numpy as np
import kornia as K
import torch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Default yolo11 model training script.')
    parser.add_argument('--input', help='', type=str, required=True)
    parser.add_argument('--output', help='', type=str, required=True)
    parser.add_argument('--device', help='', type=str, default='cpu')

    assert torch.cuda.is_available()
    args = parser.parse_args()
    use_device = args.device

    device = torch.device(use_device)
    image = cv2.imread(args.input)
    imageTensor1: torch.Tensor = K.image_to_tensor(image, keepdim=False).to(device=device) / 255.
    imageTensor2: torch.Tensor = K.image_to_tensor(image, keepdim=False).to(device=device) / 255.
    imageTensor3: torch.Tensor = K.image_to_tensor(image, keepdim=False).to(device=device) / 255.

    images_torch_tensor = torch.cat((imageTensor1, imageTensor2, imageTensor3))

    kornia_transform = torch.nn.Sequential(
        K.augmentation.RandomHorizontalFlip(),
        K.augmentation.RandomVerticalFlip(),
        K.augmentation.RandomMotionBlur(3, 35., 0.5),
        K.augmentation.RandomRotation(degrees=45.0),
    )

    images_torch_tensor = kornia_transform(images_torch_tensor)

    for i in images_torch_tensor:
        resultImage: np.ndarray = K.tensor_to_image(i)
        print("Save shape: " + str(i.shape))
        cv2.imwrite("t.jpg", resultImage)
        break
