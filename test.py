
"""
单个文件测试
"""
import cv2
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from model.generator import G
from data.transforms import Compose
from model.utils import *
def get_model(model_weight, device):
    model = G(3)
    checkpoint = torch.load(model_weight, map_location=torch.device("cpu"))
    model.load_state_dict( checkpoint.pop("models"))
    model.to(device)
    return model

    
if __name__ == '__main__':
    image_path = str(Path(__file__).parent / "out/source/8.jpg")
    output_path = str(Path(__file__).parent / "out/dstn/8.jpg")
    model_weight = Path(__file__).parent / "out/model_record/1_2.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(model_weight, device)
    model.eval()
    transform = Compose(False)
    image = cv2.imread(image_path)
    input = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input = Image.fromarray(input)
    input = transform([input])[0][0].unsqueeze(0)
    input = input.to(device)
    with torch.no_grad():
        pred = model(input).cpu()
    pred_img = (pred.squeeze() + 1.) / 2 * 255
    pred_img = pred_img.permute(1, 2, 0).numpy().clip(0, 255).astype(np.uint8)
    pred_img = adjust_brightness_from_src_to_dst(pred_img, image)
    pred_img = cv2.cvtColor(pred_img, cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, pred_img.shape[:-1][::-1])
    concat_img = np.concatenate((image, pred_img), 1)
    cv2.imwrite(output_path, concat_img)
