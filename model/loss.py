import torch
from torch.nn import functional as F
from .utils import gram, rgb2yuv, prepare_feature_extract, color_2_gray
from  torch import autograd
from torch.autograd import Variable

def init_loss(model_backbone, real_images_color, generated):
    fake = generated
    real_feature_map = model_backbone(prepare_feature_extract(real_images_color))
    fake_feature_map = model_backbone(prepare_feature_extract(fake))
    loss = F.l1_loss(real_feature_map, fake_feature_map, reduction='mean')
    return loss * 1.5
def g_loss(model_backbone, real_images_color, style_images_gray, generated, generated_logit):
    fake = generated  
    real_feature_map = model_backbone(prepare_feature_extract(real_images_color))  # 使用model_backbone提取真实图像的特征映射
    fake_feature_map = model_backbone(prepare_feature_extract(fake))  # 使用model_backbone提取生成图像的特征映射
    fake_feature_map_gray = model_backbone(prepare_feature_extract(color_2_gray(fake)))  # 使用model_backbone提取生成图像灰度化后的特征映射
    anime_feature_map = model_backbone(prepare_feature_extract(style_images_gray))  # 使用model_backbone提取风格图像的特征映射

    c_loss = F.l1_loss(real_feature_map, fake_feature_map, reduction='mean')  # 计算内容损失
    s_loss = F.l1_loss(gram(anime_feature_map), gram(fake_feature_map_gray), reduction='mean')  # 计算风格损失

    real_images_color_yuv = rgb2yuv(real_images_color)  # 将真实图像转换为YUV色彩空间
    fake_yuv = rgb2yuv(fake)  # 将生成图像转换为YUV色彩空间
    color_loss = F.l1_loss(real_images_color_yuv[..., 0], fake_yuv[..., 0], reduction='mean') + \
                 F.smooth_l1_loss(real_images_color_yuv[..., 1], fake_yuv[..., 1], reduction='mean') + \
                 F.smooth_l1_loss(real_images_color_yuv[..., 2], fake_yuv[..., 2], reduction='mean')  # 计算颜色损失

    dh_input, dh_target = fake[:, :, :-1, :], fake[:, :, 1:, :]  # 用于计算TV Loss的输入和目标图像的水平方向差分
    dw_input, dw_target = fake[:, :, :, :-1], fake[:, :, :, 1:]  # 用于计算TV Loss的输入和目标图像的垂直方向差分
    tv_loss = F.mse_loss(dh_input, dh_target, reduction='mean') + \
              F.mse_loss(dw_input, dw_target, reduction='mean')  # 计算TV Loss

    fake_loss = torch.mean(torch.square(generated_logit - 1.0))  # 计算生成器的对抗损失

    return 1.5 * c_loss + 2.5 * s_loss + 10.0 * color_loss + 1.0 * tv_loss + 300.0 * fake_loss  # 返回生成器总体损失



def d_loss(generated_logit, anime_logit, anime_gray_logit, smooth_logit):
    
    real_loss = torch.mean(torch.square(anime_logit - 1.0))  # 计算真实图像的对抗损失
    gray_loss = torch.mean(torch.square(anime_gray_logit))  # 计算灰度图像的对抗损失
    fake_loss = torch.mean(torch.square(generated_logit))  # 计算生成图像的对抗损失
    real_blur_loss = torch.mean(torch.square(smooth_logit))  # 计算平滑图像的对抗损失
    
    return 300.0 * (1.7 * real_loss + 1.7 * fake_loss + 1.7 * gray_loss + 1.0* real_blur_loss)  # 返回判别器总体损失
