import numpy as np

def str2bool(v):
    if v.lower() in ['true', 1]:
        return True
    elif v.lower() in ['false', 0]:
        return False
    else:
        raise print('Boolean value expected.')


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def preprocess(pil_img, seed):
    if seed > 0.9:
        img_aug = np.rot90(pil_img, 1, (1, 2)).copy()
    elif seed > 0.8 and seed <= 0.9 :
        img_aug = np.flip(pil_img).copy()
    elif seed > 0.7 and seed <= 0.8:
        img_aug = np.flip(pil_img, 2).copy()
    elif seed > 0.6 and seed <= 0.7:
        img_aug = np.flip(pil_img, 1).copy()
    elif seed > 0.5 and seed <= 0.6:
        img_aug = np.flip(pil_img, 0).copy()
    elif seed > 0.4 and seed <= 0.5:
        img_aug = np.flip(pil_img, (0, 1)).copy()
    elif seed > 0.3 and seed <= 0.4:
        img_aug = np.rot90(pil_img, 3, (1, 2)).copy()
    else:
        img_aug = pil_img

    img_aug = np.ascontiguousarray(img_aug)

    return img_aug