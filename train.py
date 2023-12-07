import os
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from data.datasets import GanDataset
from data.transforms import Compose
from model.backbone import build_vgg_backbones
from model.discriminator import D
from model.generator import G
from model.loss import init_loss, d_loss, g_loss
from model.utils import rgbScaled
from solver.build import lr_scheduler_discriminator, lr_scheduler_generator
from utils.checkpoint import ModelCheckpointer
from pathlib import Path
from tqdm import tqdm

def train():
    epoch_size = 500  # 设置训练总轮数
    output_dir = Path(__file__).parent / "out"
    data_dir = Path(__file__).parent / "datasets"
    tensorboard_dir = os.path.join(output_dir, 'tensorboard')
    writer = SummaryWriter(tensorboard_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_backbone = build_vgg_backbones().to(device).eval()
    model_generator = G(3).to(device).train()
    model_discriminator = D(3, 64, 2).to(device).train()

    optimizer_generator = torch.optim.Adam(model_generator.parameters(), 0.0002, betas=(0.5, 0.999))
    optimizer_discriminator = torch.optim.Adam(model_discriminator.parameters(), 0.0002, betas=(0.5, 0.999))

    scheduler_generator = lr_scheduler_generator(optimizer_generator, epoch_size)
    scheduler_discriminator = lr_scheduler_discriminator(optimizer_discriminator, epoch_size)

    train_dataset = GanDataset(dataDir=data_dir, split='train', transforms=Compose(is_train=True))
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)

    checkpointer_generator = ModelCheckpointer(model_generator, optimizer_generator, schedulers=scheduler_generator, save_dir=output_dir)
    checkpointer_discriminator = ModelCheckpointer(model_discriminator, optimizer_discriminator, schedulers=scheduler_discriminator, save_dir=output_dir)

    image_epoch = 0

    for epoch in tqdm(range(epoch_size), desc="Training", unit="epoch"):
        for iteration, (real_images, style_images, smooth_images, _) in enumerate(tqdm(train_dataloader, desc="Iteration", unit="batch", leave=False)):
        
            real_images_color = real_images[0]
            real_images_gray = real_images[1]
            style_images_color = style_images[0]
            style_images_gray = style_images[1]
            smooth_images_color = smooth_images[0]
            smooth_images_gray = smooth_images[1]

            if epoch <= 10:
                # FP
                real_images_color = real_images_color.to(device)
                generated = model_generator(real_images_color)
                loss_init = init_loss(model_backbone, real_images_color, generated)

                optimizer_generator.zero_grad()
                loss_init.backward()
                optimizer_generator.step()
                scheduler_generator.step()
                scheduler_discriminator.step()
            else:
                real_images_color = real_images_color.to(device)
                style_images_color = style_images_color.to(device)
                style_images_gray = style_images_gray.to(device)
                smooth_images_gray = smooth_images_gray.to(device)
                generated = model_generator(real_images_color)

                if iteration % 1 == 0:
                    # FP D
                    generated_logit = model_discriminator(generated.detach())
                    anime_logit = model_discriminator(style_images_color)
                    anime_gray_logit = model_discriminator(style_images_gray)
                    smooth_logit = model_discriminator(smooth_images_gray)
                    loss_d = d_loss(
                        generated_logit,
                        anime_logit,
                        anime_gray_logit,
                        smooth_logit
                    )

                    # BP D
                    optimizer_discriminator.zero_grad()
                    loss_d.backward()
                    optimizer_discriminator.step()

                    # 记录判别器损失
                    writer.add_scalar('train/D_loss', loss_d.item(), iteration)

                # FP G
                generated_logit = model_discriminator(generated)
                loss_g = g_loss(
                    model_backbone,
                    real_images_color,
                    style_images_gray,
                    generated,
                    generated_logit
                )

                # BP G
                optimizer_generator.zero_grad()
                loss_g.backward()
                optimizer_generator.step()
                scheduler_generator.step()
                scheduler_discriminator.step()

                # 记录生成器损失
                writer.add_scalar('train/G_loss', loss_g.item(), iteration)
    

            # loss记录
            if epoch != image_epoch:
                for i in range(0,8):
                    writer.add_image("images/real_color{}".format(i), rgbScaled(real_images_color[i]).clamp(0, 1))
                    writer.add_image("images/generated{}".format(i), rgbScaled(generated[i]).clamp(0, 1),global_step=epoch)
                image_epoch = epoch
        # 保存检查点
        if (epoch + 1) % 100 == 0 or epoch==0:
            checkpointer_generator.save("model_generator_epoch_{:03d}".format(epoch + 1))
            checkpointer_discriminator.save("model_discriminator_epoch_{:03d}".format(epoch + 1))

    writer.close()

if __name__ == "__main__":
    train()
