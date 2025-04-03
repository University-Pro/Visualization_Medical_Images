import os
import torch
import glob
import yaml
import matplotlib
import numpy as np
import nibabel as nb
from tqdm import tqdm
from matplotlib import gridspec
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from torchvision.utils import draw_segmentation_masks
import matplotlib.patches as patches  # 添加以绘制红色边框

# 全局设置字体和大小，使得图像更加美观，符合专业论文的风格
plt.rcParams.update({
    'font.size': 20,                 # 设置全局字体大小
    # 'font.family': 'Times New Roman' # 设置全局字体样式
})

# ACDC数据集可视化函数
def viz_acdc(config: dict):

    # 从配置文件中获取案例名称和对应的帧数
    case_names = config["acdc"]["case_names"]  # 获得所有的文件名，存储在数组中
    frame_nums = config["acdc"]["frame_nums"]  # 每个case的帧数不同
    iterables = zip(case_names, frame_nums)  # 将案例和帧数打包

    # 遍历每个案例并绘制可视化图像
    for case_name, frame_num in tqdm(
        iterables, total=len(case_names), desc="渲染ACDC可视化图像"
    ):
        frame_nums = np.random.choice(
            range(int(frame_num)),  # 随机选择若干帧用于可视化
            size=4,
            replace=False,
        )

        # 渲染并保存图像
        render_and_save_gridspec(
            case_name=case_name,
            frame_nums=frame_nums,
            data_folder=config["acdc"]["data_folder"], # 原来的数据集的文件夹
            pred_folders=config["acdc"]["pred_folders"], # 预测的分割结果的文件夹
            num_classes=config["acdc"]["num_classes"], # 类别数
            class_names=config["acdc"]["class_names"], # 分类名称
            colors=config["acdc"]["colors"], # 每种类别的颜色
            fig_save_dir=config["acdc"]["fig_save_dir"], # 保存图像的文件夹
            img_normalize=True, # 图像标准化
            transparency=0.65, # 透明度
        )

# Synapse数据集可视化函数
def viz_synapse(config: dict):
    case_names = config["synapse"]["case_names"]
    frame_nums = config["synapse"]["frame_nums"]  # 获取案例的帧数
    # 遍历每个案例并生成对应帧的可视化图像
    for case_name, frame_num in tqdm(
        zip(case_names, frame_nums),
        total=len(case_names),
        desc="渲染SYNAPSE可视化图像",
    ):
        frame_num = int(frame_num)
        frame_nums = np.random.choice(
            range(frame_num // 2, frame_num, 5),  # 随机选择指定范围内的帧
            size=4,
            replace=False,
        )
        # 渲染并保存图像
        render_and_save_gridspec(
            case_name=case_name,
            frame_nums=frame_nums,
            data_folder=config["synapse"]["data_folder"],
            pred_folders=config["synapse"]["pred_folders"],
            num_classes=config["synapse"]["num_classes"],
            class_names=config["synapse"]["class_names"],
            colors=config["synapse"]["colors"],
            fig_save_dir=config["synapse"]["fig_save_dir"],
            img_normalize=True,
            transparency=0.65,
        )

# BRATS数据集可视化函数
def viz_brats(config: dict):
    case_names = config["brats"]["case_names"]
    frame_n = int(config["brats"]["frame_nums"])
    # 遍历案例并生成对应帧的可视化图像
    for case_name in tqdm(case_names, desc="渲染BRATS可视化图像"):
        frame_nums = np.random.choice(
            range(30, 90, 8),  # 选择大脑从45%到75%的范围帧
            size=4,
            replace=False,
        )
        render_and_save_gridspec(
            case_name=case_name,
            frame_nums=frame_nums,
            data_folder=config["brats"]["data_folder"],
            pred_folders=config["brats"]["pred_folders"],
            num_classes=config["brats"]["num_classes"],
            class_names=config["brats"]["class_names"],
            colors=config["brats"]["colors"],
            fig_save_dir=config["brats"]["fig_save_dir"],
            img_normalize=True,
            transparency=0.65,
        )

# 叠加图像与分割掩膜函数
def overlay(
    image: np.ndarray,
    seg_mask: np.ndarray,
    num_classes: int,
    colors: list,
    normalize: bool = False,
    transparency: float = 0.5,
):
    img = image
    if normalize:
        eps = 0.000001
        img = (img - img.min()) / (img.max() - img.min() + eps) * 255.0
    img = img.astype(np.uint8)
    img = torch.tensor(img)
    img = torch.stack((img, img, img), dim=0)  # 将灰度图扩展为3通道RGB图
    msk = np.rint(seg_mask)  # 对掩膜进行四舍五入
    msk = torch.tensor(msk).long()
    msk = torch.nn.functional.one_hot(msk, num_classes)
    msk = torch.moveaxis(msk, 2, 0).bool()  # 转换维度
    msk = msk[1:, :, :]  # 忽略背景类
    out = draw_segmentation_masks(img, msk, alpha=transparency, colors=colors)
    out = torch.moveaxis(out, 0, 2).numpy()  # 将叠加后的图像转换回numpy格式
    return out

# 渲染并保存图像的函数,两行图片
def render_and_save_gridspec(
    data_folder: str,       # 原始数据文件夹路径
    pred_folders: list,     # 预测结果文件夹列表，每个文件夹对应一个模型
    frame_nums: list,       # 要渲染的帧索引列表
    case_name: str,         # 当前处理的病例名称
    num_classes: int,       # 分类数量（包括背景）
    class_names: list,      # 分类名称列表
    colors: list,           # 每个分类对应的颜色列表
    fig_save_dir: str,      # 保存绘图结果的目录
    img_normalize: bool,    # 是否对图像进行归一化
    transparency: float = 0.75,  # 叠加图像的透明度
):
    # 确保数据和保存路径存在
    assert os.path.exists(data_folder), f"{data_folder} 不存在"
    assert os.path.exists(fig_save_dir), f"{fig_save_dir} 不存在"
    
    # 构建保存图像的文件路径
    save_fig_fp = os.path.join(fig_save_dir, case_name.split(".nii.gz")[0] + ".pdf")
    # 构建原始数据文件路径
    data_fp = os.path.join(data_folder, case_name)
    # 读取原始图像数据
    data_volume = nb.load(data_fp).get_fdata()

    # 确保图片为两行显示，每行显示一半的帧数
    frame_nums_per_row = len(frame_nums) // 2
    frame_nums_split = [frame_nums[:frame_nums_per_row], frame_nums[frame_nums_per_row:]]

    # 动态设置列数为预测文件夹的数量
    cols = len(pred_folders)
    rows = 2  # 只设置2行

    # 创建图形对象，设置尺寸
    fig = plt.figure(figsize=(cols * 3, rows * 3))
    # 创建网格布局，调整子图之间的间距
    gs = gridspec.GridSpec(rows, cols, wspace=0.1, hspace=0.1)

    # 渲染网格图
    for row in range(rows):  # 遍历每一行
        frame_set = frame_nums_split[row]  # 每行分配对应的帧
        for col in range(cols):  # 遍历每一列（仅包括预测结果）
            for frame_ind in frame_set:  # 遍历当前帧
                axis = plt.subplot(gs[row, col])

                # 获取预测结果的文件夹路径
                pred_folder = pred_folders[col]  # 直接从第0列开始
                fp = os.path.join(pred_folder, case_name)
                seg_mask_volume = nb.load(fp).get_fdata()  # 读取分割结果数据
                mask = seg_mask_volume[:, :, frame_ind]
                image = data_volume[:, :, frame_ind]
                
                # 叠加原始图像和分割掩膜
                overlayed_img = overlay(
                    image=image,
                    seg_mask=mask,
                    num_classes=num_classes,
                    colors=colors,
                    normalize=img_normalize,
                    transparency=transparency,
                )
                axis.imshow(overlayed_img, cmap="bone")

                # 根据预测文件夹名称自动生成标题
                pred_name = pred_folder.split('/')[-1]

                if row == 0 and frame_ind == frame_set[0]:  # 只有第一行第一列才写上Title
                    axis.set_title(pred_name, fontsize=20, pad=10)

                # 如果是 Synapse 数据集的最后一类，为其添加红色边框
                if "synapse" in data_folder.lower():  # 检测是否是Synapse数据集
                    # 获取最后一类的分割掩膜
                    last_class_mask = mask == (num_classes - 1)  # num_classes-1是最后一类的索引
                    if last_class_mask.any():  # 如果最后一类存在
                        # 计算边框位置
                        y, x = np.where(last_class_mask)  # 获取最后一类掩膜的像素位置
                        rect = patches.Rectangle(
                            (x.min(), y.min()), x.max() - x.min(), y.max() - y.min(),
                            linewidth=2, edgecolor='r', facecolor='none'  # 红色边框
                        )
                        axis.add_patch(rect)  # 在当前子图中添加矩形边框

                axis.axis("off")  # 隐藏坐标轴

    # 生成图例
    legend_ax = plt.subplot(gs[rows-1, cols-1])  # 在网格的最后一个子图位置创建图例轴
    legend_artists = list()

    for cls_name, color in zip(class_names, colors):  # 遍历类别名称和颜色
        # 创建图例标记
        artist = Line2D(
            [0],
            [0],
            marker="s",             # 方形标记
            color="w",              # 边框颜色为白色
            markerfacecolor=color,  # 填充颜色为对应的类别颜色
            markersize=16,          # 设置标记大小
            label=cls_name,         # 设置标签为类别名称
        )
        legend_artists.append(artist)

    # 创建图例并放置在图外
    legend_ax.legend(
        handles=legend_artists,
        loc="lower center",
        ncol=len(class_names),      # 图例中一行显示的标记数量
        fontsize=16,                 # 增加字体大小
        bbox_to_anchor=(-3, -0.3),  # 图例相对于轴的位置,最好根据不同数据集进行调整
        frameon=False,              # 不显示图例边框
    )

    # 保存图像
    plt.savefig(
        save_fig_fp,
        bbox_inches="tight",           # 去除多余的空白边缘
        dpi=300,                       # 调整分辨率，设置为300dpi
        format="pdf"
    )
    
    plt.close()  # 关闭图像，释放内存

if __name__ == "__main__":
    # 设置随机种子，便于控制帧的选择
    np.random.seed(1234)

    # 读取Yaml数据
    with open("./pictures/Visualization_Meta.yaml") as f:
        config = yaml.safe_load(f)

    # ACDC数据集可视化
    # viz_acdc(config=config)

    # SYNAPSE数据集可视化
    viz_synapse(config=config)