import torch
import numpy as np

from PIL import Image  # Python图像处理库, 图像处理功能
from io import BytesIO

from tools.utils.type import arr_type


def to_uint8(normalized_img: arr_type) -> arr_type:
    if isinstance(normalized_img, np.ndarray):
        return (np.clip(normalized_img * 255.0, 0.0, 255.0)).astype(np.uint8)
    return torch.clamp(normalized_img * 255.0, 0.0, 255.0).to(torch.uint8)


# 将输入的 numpy 数组转换为字节序列
def np_to_bytes(img_arr: np.ndarray) -> bytes:
    if Image is None:
        raise ValueError("`pillow` is needed for `np_to_bytes`")
    if img_arr.dtype != np.uint8:
        img_arr = to_uint8(img_arr)
    bytes_io = BytesIO()
    Image.fromarray(img_arr).save(bytes_io, format="PNG")
    return bytes_io.getvalue()


# 将输入的 numpy 数组进行归一化和转置操作
def get_normalized_arr_from_diffusion(img_arr: np.ndarray) -> np.ndarray:
    img_arr = 0.5 * (img_arr + 1.0)
    img_arr = img_arr.transpose([1, 2, 0])
    return img_arr


# 将diffusion的输出结果转换为字节序列
def get_bytes_from_diffusion(img_arr: np.ndarray) -> bytes:
    return np_to_bytes(get_normalized_arr_from_diffusion(img_arr))