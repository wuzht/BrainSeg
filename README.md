# BrainSeg
> Brain Segmentation

文件结构与描述：

```shell
BrainSeg
│
├── datasets			# 数据集文件夹
│   └── BrainS18		# BrainS18数据集文件夹
├── unet				# U-Net模型
│   ├── __init__.py
│   ├── unet_model.py	# 多种不同dropout设置的U-Net
│   └── unet_parts.py
├── exp					# 存放实验记录(训练log、模型参数等)
│   ├── 				# 每个文件夹是一次实验记录
│   └── ...
│
├── brains18.py		# BrainS18Dataset类，继承自torch.utils.data.Data
├── config.py		# 实验参数在这里设置
├── op.py			# Operation类，含数据模型加载、训练、测试等各种操作
├── viewer.py		# Viewer类，显示或保存实验结果的各种图表
│
├── dice_loss.py	# 图像分割评估方法dice_loss
├── main.ipynb		# 训练或测试模型
├── main.py			# 训练或测试模型
├── tools.py		# 一些常用工具
├── test_model.ipynb	# 测试模型
│
├── prepare_dataset.ipynb	# 数据集预处理，将nii.gz文件中的slices保存成png
├── print_curves.ipynb		# 设置正则表达式，读取训练log，查看相关curves
├── view_data.ipynb			# 用来预览一下数据
├── test.ipynb				# 不用管这个
│
└── README.md
```

