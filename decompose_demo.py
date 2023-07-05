import os
import numpy as np
import cv2
import glob
from networks.unet_ConvNeXt import convnext_tiny
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib
import torch
import random

def window(image, up=250, down=-250):
    image = image * 1000
    image = np.clip(image, a_min=down, a_max=up)
    return (image - down) * 255 / (up - down)


def plot_material(image, my_cmap, file_dir, material_name):
    plt.imshow(image, cmap=my_cmap, vmax=1, vmin=0)
    plt.axis('off')
    plt.savefig(file_dir + material_name + '.pdf', dpi=1200)
    # plt.show()


model = convnext_tiny(num_classes=4).cuda()
# change your weights here
model.load_state_dict(torch.load('./checkpoints/decompose_four.pth'))
# change your data here
data_dir = './demo_data'
train_fns_100 = sorted(glob.glob(os.path.join(data_dir, '100kv/*.npy')))
train_fns_140 = sorted(glob.glob(os.path.join(data_dir, '140kv/*.npy')))

mycmap1 = colors.LinearSegmentedColormap.from_list('mycmap1', ['#000000', '#00FFFF'])
mycmap2 = colors.LinearSegmentedColormap.from_list('mycmap2', ['#000000', '#0BFF16'])
mycmap3 = colors.LinearSegmentedColormap.from_list('mycmap3', ['#000000', '#C604C5'])
mycmap4 = colors.LinearSegmentedColormap.from_list('mycmap4', ['#000000', '#FF8D01'])

q = random.randint(0, 4)

kv_100 = np.load(train_fns_100[q])
kv_140 = np.load(train_fns_140[q])
image_input = np.stack((kv_100, kv_140), axis=0).astype(np.float32)
tensor_input = torch.from_numpy(image_input)
tensor_input = torch.unsqueeze(tensor_input, dim=0).cuda()

x_input = tensor_input.detach().cpu().numpy().reshape((2, 512, 512))

model.eval()
with torch.no_grad():
    output = model(tensor_input)

output = torch.squeeze(output, dim=0)
output = output.detach().cpu().numpy()

save_dir = './decompose_result/visualize/'
save_npy_dir = './decompose_result/data/'
os.makedirs(save_dir, exist_ok=True)
os.makedirs(save_npy_dir, exist_ok=True)

cv2.imwrite(save_dir + 'input_100.jpg', window(x_input[0]))
cv2.imwrite(save_dir + 'input_140.jpg', window(x_input[1]))

color_painted = False

if color_painted:
    plot_material(output[0], mycmap1, save_dir, 'Adipose')
    plot_material(output[3], mycmap2, save_dir, 'Air')
    plot_material(output[1], mycmap3, save_dir, 'Muscle')
    plot_material(output[2], mycmap4, save_dir, 'Iodine')

else:
    cv2.imwrite(save_dir + 'Adipose.jpg', 255 * output[0])
    cv2.imwrite(save_dir + 'Air.jpg', 255 * output[3])
    cv2.imwrite(save_dir + 'Muscle.jpg', 255 * output[1])
    cv2.imwrite(save_dir + 'Iodine.jpg', 255 * output[2])

np.save(save_npy_dir + 'Adipose.npy', output[0])
np.save(save_npy_dir + 'Air.npy', output[3])
np.save(save_npy_dir + 'Muscle.npy', output[1])
np.save(save_npy_dir + 'Iodine.npy', output[2])
