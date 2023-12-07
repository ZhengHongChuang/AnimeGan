from .lr_scheduler import WarmupMultiStepLR
def lr_scheduler_generator(optimizer, epoch_size):
    return WarmupMultiStepLR(
        optimizer,
        [step * epoch_size for step in (10,)],
        0.1,
        warmup_factor=1.0,
        warmup_iters=0,
        warmup_method="constant",
    )

def lr_scheduler_discriminator(optimizer, epoch_size):
    return WarmupMultiStepLR(
        optimizer,
        [step * epoch_size for step in (100,)],
        0.1,
        warmup_factor=1.0/3,
        warmup_iters=0,
        warmup_method="linear",
    )