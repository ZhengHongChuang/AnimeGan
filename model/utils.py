
import torch
import numpy as np

# 定义 RGB 转 YUV 的矩阵
yuv_from_rgb = np.array([[0.299,       0.587,       0.114],
                         [-0.14714119, -0.28886916, 0.43601035],
                         [0.61497538,  -0.51496512, -0.10001026]])

# 定义特征提取的均值
feature_extract_mean = [123.68, 116.779, 103.939]

# 将输入的 RGB 图像缩放到 [-1, 1] 的范围
def rgbScaled(x):
    return (x + 1.0) / 2.0

# RGB 转 YUV 的函数
def rgb2yuv(x):
    x = rgbScaled(x)
    x = x.permute([0, 2, 3, 1])
    k_yuv_from_rgb = torch.from_numpy(yuv_from_rgb.T).to(x.dtype).to(x.device)
    yuv = torch.matmul(x, k_yuv_from_rgb)
    return yuv

# RGB 转灰度的函数
def color_2_gray(x):
    x = x.permute([0, 2, 3, 1])
    k_color_2_gray = torch.Tensor([[0.299], [0.587], [0.114]]).to(x.dtype).to(x.device)
    gray = torch.matmul(x, k_color_2_gray)
    gray = torch.cat([gray, gray, gray], dim=-1)
    gray = gray.permute([0, 3, 1, 2])
    return gray

# 计算 Gram 矩阵
def gram(x):
    x = x.permute([0, 2, 3, 1])
    shape = x.shape
    b = shape[0]
    c = shape[3]
    x = torch.reshape(x, [b, -1, c])
    return torch.matmul(x.permute(0, 2, 1), x) / (x.numel() // b)

# 准备特征提取的输入
def prepare_feature_extract(rgb):
    rgb_scaled = rgbScaled(rgb) * 255.0
    R, G, B = torch.chunk(rgb_scaled, 3, 1)
    feature_extract_input = torch.cat(
        [
            (B - feature_extract_mean[2]),
            (G - feature_extract_mean[1]),
            (R - feature_extract_mean[0]),
        ],
        dim=1
    )
    return feature_extract_input

# 计算图像的平均亮度
def calculate_average_brightness(img):
    R = img[..., 0].mean()
    G = img[..., 1].mean()
    B = img[..., 2].mean()
    brightness = 0.299 * R + 0.587 * G + 0.114 * B
    return brightness, B, G, R

# 调整目标图像的平均亮度到源图像的平均亮度
def adjust_brightness_from_src_to_dst(dst, src):
    brightness1, B1, G1, R1 = calculate_average_brightness(src)
    brightness2, B2, G2, R2 = calculate_average_brightness(dst)
    brightness_difference = brightness1 / brightness2
    dstf = dst * brightness_difference
    dstf = np.clip(dstf, 0, 255)
    dstf = np.uint8(dstf)
    return dstf
