import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
import numpy as np


def arr2str(arr):
    s = '['
    for x in arr:
        s += '{:.3f}, '.format(x)
    return s[:-2] + ']'

def inc_rate(begin, end):
    return (end - begin) / begin * 100

def show_result(data, ids, operation):
    for i in ids:
        imgs, folder, slice_id = data[i]
        operation.predict(imgs[2], imgs[3], mode='Dropout', title='{} {}'.format(folder, slice_id))
# show_result(operation.val_data, [26], operation)

class Viewer(object):
    # cmap = plt.cm.get_cmap('Paired', 10)    # 10 discrete colors
    cmap_label = plt.cm.get_cmap('tab10', 10)    # 10 discrete colors
    # cmap = plt.cm.get_cmap('Set3', 10)    # 10 discrete colors
    cmap_heat  = plt.cm.get_cmap('jet')

    ent_vmax = 0.2
    var_vmax = 0.03

    def __init__(self):
        super().__init__()

    @staticmethod
    def print_dices(printer, total_dices, partitioned_dices):
        printer("")
        printer("Class     : {} [c1-c8 mean]".format(['{:3d}'.format(x) for x in range(0, len(total_dices))]))
        printer("Total dice: {} [{:.3f}]".format(arr2str(total_dices), total_dices[1:].mean()))
        printer("Bottomdice: {} [{:.3f}]".format(arr2str(partitioned_dices[0]), partitioned_dices[0][1:].mean()))
        printer("Mid   dice: {} [{:.3f}]".format(arr2str(partitioned_dices[1]), partitioned_dices[1][1:].mean()))
        printer("Up    dice: {} [{:.3f}]".format(arr2str(partitioned_dices[2]), partitioned_dices[2][1:].mean()))

        
    @staticmethod
    def show_fig_4(image, y_gt, y_pred, entropy, title, dices):
        fig, axs = plt.subplots(2,2, sharey=True, figsize=(10,8))
        axs[0][0].set_title("Original data")
        axs[0][1].set_title("Ground Truth")
        axs[1][0].set_title("Entropy [{:.3f}, {:.3f}]".format(entropy.min(), entropy.max()))
        axs[1][1].set_title("Prediction")
        plt.suptitle("({}) dice : {} [c1-c8 mean: {:.3f}]".format(title, arr2str(dices), dices[1:].mean()))

        ax00 = axs[0][0].imshow( image[0,...], aspect="auto")
        ax01 = axs[0][1].imshow( y_gt, cmap=Viewer.cmap_label, aspect="auto", vmin=0, vmax=9, interpolation='none')
        ax10 = axs[1][0].imshow( entropy[0,...],  aspect="auto", cmap=Viewer.cmap_heat, vmin=0, vmax=Viewer.ent_vmax, interpolation='none')
        ax11 = axs[1][1].imshow( y_pred, cmap=Viewer.cmap_label, aspect="auto", vmin=0, vmax=9, interpolation='none')
        
        fig.colorbar(ax00, ax=axs[0][0])
        fig.colorbar(ax01, ax=axs[0][1])
        fig.colorbar(ax10, ax=axs[1][0])
        fig.colorbar(ax11, ax=axs[1][1])


    @staticmethod
    def show_fig(image, y_gt, y_pred, entropy, variance, title, dices):
        fig, axs = plt.subplots(2,3, sharey=True, figsize=(16,8.5))
        axs[0][0].set_title("Original data")
        axs[1][0].set_title("Ground Truth")
        axs[1][1].set_title("Prediction")
        axs[0][1].set_title("Entropy [{:.3f}, {:.3f}]".format(entropy.min(), entropy.max()))
        axs[0][2].set_title("Variance [{:.3f}, {:.3f}]".format(variance.min(), variance.max()))
        plt.suptitle("({}) dice : {} [c1-c8 mean: {:.3f}]".format(title, arr2str(dices), dices[1:].mean()))

        ax00 = axs[0][0].imshow( image[0], aspect="auto", cmap='gray')
        ax10 = axs[1][0].imshow( y_gt, cmap=Viewer.cmap_label, aspect="auto", vmin=0, vmax=9, interpolation='none')
        ax11 = axs[1][1].imshow( y_pred, cmap=Viewer.cmap_label, aspect="auto", vmin=0, vmax=9, interpolation='none')
        ax01 = axs[0][1].imshow( entropy,  aspect="auto", cmap=Viewer.cmap_heat, vmin=0.0, vmax=Viewer.ent_vmax, interpolation='none')
        ax02 = axs[0][2].imshow( variance, aspect="auto", cmap=Viewer.cmap_heat, vmin=0.0, vmax=Viewer.var_vmax, interpolation='none')
        
        fig.colorbar(ax00, ax=axs[0][0])
        fig.colorbar(ax10, ax=axs[1][0])
        fig.colorbar(ax01, ax=axs[0][1])
        fig.colorbar(ax11, ax=axs[1][1])
        fig.colorbar(ax02, ax=axs[0][2])

    @staticmethod
    def save_figs(result_dir, image, y_gt, entropy, y_pred, variance, title, dices):
        # 创建文件夹
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        fig, axs = plt.subplots(nrows=1,ncols=5, sharey=True, figsize=(21,4))
        fig.tight_layout() # 调整整体空白
        plt.subplots_adjust(wspace=0, hspace=0) # 调整子图间距
        axs[0].set_title("Original data")
        axs[1].set_title("Ground Truth")
        axs[3].set_title("Entropy [{:.3f}, {:.3f}]".format(entropy.min(), entropy.max()))
        axs[2].set_title("Prediction")
        axs[4].set_title("Variance [{:.3f}, {:.3f}]".format(variance.min(), variance.max()))
        # plt.suptitle("({}) dice : {} [c1-c8 mean: {:.3f}]".format(title, arr2str(dices), dices[1:].mean()))
        for i in range(5):
            axs[i].axis('off') 

        ax00 = axs[0].imshow( image[0], aspect="auto", cmap='gray')
        ax10 = axs[1].imshow( y_gt, cmap=Viewer.cmap_label, aspect="auto", vmin=0, vmax=9, interpolation='none')
        ax01 = axs[3].imshow( entropy,  aspect="auto", cmap=Viewer.cmap_heat, vmin=0.0, vmax=Viewer.ent_vmax, interpolation='none')
        ax11 = axs[2].imshow( y_pred, cmap=Viewer.cmap_label, aspect="auto", vmin=0, vmax=9, interpolation='none')
        # ax02 = axs[4].imshow( variance, aspect="auto", cmap=Viewer.cmap_heat, vmin=0.0, vmax=0.33, interpolation='none')
        ax02 = axs[4].imshow( variance, aspect="auto", cmap=Viewer.cmap_heat, vmin=0.0, vmax=Viewer.var_vmax, interpolation='none')
        fig.colorbar(ax00, ax=axs[0])
        fig.colorbar(ax10, ax=axs[1])
        fig.colorbar(ax01, ax=axs[3])
        fig.colorbar(ax11, ax=axs[2])
        fig.colorbar(ax02, ax=axs[4])

        name = title
        plt.savefig(os.path.join(result_dir, name))

    @staticmethod
    def save_figs_foo(result_dir, image, y_gt, y_pred, entropy, variance, title, dices):
        # 创建文件夹
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        fig, axs = plt.subplots(nrows=1,ncols=5, sharey=True, figsize=(21,4))
        fig.tight_layout() # 调整整体空白
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        for i in range(5):
            axs[i].axis('off') 

        print('entropy min max:', entropy.min(), entropy.max())
        print('variance min max:', variance.min(), variance.max())

        axs[0].imshow( image[0], aspect="auto", cmap='gray')
        axs[1].imshow( y_gt, cmap=Viewer.cmap_label, aspect="auto", vmin=0, vmax=9, interpolation='none')
        axs[2].imshow( y_pred, cmap=Viewer.cmap_label, aspect="auto", vmin=0, vmax=9, interpolation='none')
        axs[3].imshow( entropy,  aspect="auto", cmap=Viewer.cmap_heat, vmin=0.0, vmax=Viewer.ent_vmax, interpolation='none')
        axs[4].imshow( variance, aspect="auto", cmap=Viewer.cmap_heat, vmin=0.0, vmax=Viewer.var_vmax, interpolation='none')
        fig.savefig(os.path.join(result_dir, title+'.png'), format='png', transparent=True, dpi=600, pad_inches=0)
        plt.close()

    @staticmethod
    def save_figs_many(result_dir, images, y_gts, y_preds, entropys, variances, title):
        # 创建文件夹
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        num = len(images)

        fig, axs = plt.subplots(nrows=5,ncols=num, sharey=True, figsize=(18,11))
        fig.tight_layout() # 调整整体空白
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0.01, wspace=0.01)
        for i in range(5):
            for j in range(num):
                axs[i][j].axis('off')
        
        for i, (image, y_gt, y_pred, entropy, variance) in enumerate(zip(images, y_gts, y_preds, entropys, variances)):
            print(i, 'entropy min max:', entropy.min(), entropy.max())
            print(i, 'variance min max:', variance.min(), variance.max())
            axs[0][i].imshow( image[0], aspect="auto", cmap='gray')
            axs[1][i].imshow( y_gt, cmap=Viewer.cmap_label, aspect="auto", vmin=0, vmax=9, interpolation='none')
            axs[2][i].imshow( y_pred, cmap=Viewer.cmap_label, aspect="auto", vmin=0, vmax=9, interpolation='none')
            axs[3][i].imshow( entropy,  aspect="auto", cmap=Viewer.cmap_heat, vmin=0.0, vmax=Viewer.ent_vmax, interpolation='none')
            axs[4][i].imshow( variance, aspect="auto", cmap=Viewer.cmap_heat, vmin=0.0, vmax=Viewer.var_vmax, interpolation='none')
        fig.savefig(os.path.join(result_dir, title+'.png'), format='png', transparent=True, dpi=600, pad_inches=0)
        # plt.close()


    @staticmethod
    def draw_scatter(xs1, ys1, xlabel1, ylabel1, xs2, ys2, xlabel2, ylabel2):
        xs = np.stack(arrays=(xs1, xs2))
        ys = np.stack(arrays=(ys1, ys2))
        xlabel = [xlabel1, xlabel2]
        ylabel = [ylabel1, ylabel2]
        scale = 120

        fig, axs = plt.subplots(nrows=1,ncols=2, sharey=False, figsize=(11,5))
        # fig.tight_layout() # 调整整体空白

        for j in range(2):
            for i, (x, y) in enumerate(zip(xs[j], ys[j])):
                color = np.array(Viewer.cmap_label(i))
                color = color.reshape(1, len(color))
                axs[j].scatter(x=x, y=y, c=color, s=scale, label=i, alpha=1)
                axs[j].annotate(s='  {}'.format(i), xy=(x, y))
            axs[j].set_xlabel(xlabel[j], size=12)
            axs[j].set_ylabel(ylabel[j], size=12)
            axs[j].grid(True)
            axs[j].set_xlim([0, 1.1 * np.max(xs[j])])
            # axs[j].set_ylim(top=1.0)
            axs[j].ticklabel_format(style='sci', axis='both', scilimits=(0, 0), useMathText=True)
        

    @staticmethod
    def draw_scatter2(xs1, ys1, xlabel1, ylabel1, xs2, ys2, xlabel2, ylabel2):
        xs = np.stack(arrays=(xs1, xs2))
        ys = np.stack(arrays=(ys1, ys2))
        xlabel = [xlabel1, xlabel2]
        ylabel = [ylabel1, ylabel2]
        scale = 120

        fig, axs = plt.subplots(nrows=1,ncols=2, sharey=False, figsize=(11,5))
        # fig.tight_layout() # 调整整体空白

        for j in range(2):
            for i, (x, y) in enumerate(zip(xs[j], ys[j])):
                color = np.array(Viewer.cmap_label(i+1))
                color = color.reshape(1, len(color))
                axs[j].scatter(x=x, y=y, c=color, s=scale, label=i+1, alpha=1)
                axs[j].annotate(s='  {}'.format(i+1), xy=(x, y))
            axs[j].set_xlabel(xlabel[j], size=12)
            axs[j].set_ylabel(ylabel[j], size=12)
            axs[j].grid(True)
            axs[j].set_xlim([0, 1.1 * np.max(xs[j])])
            # axs[j].set_ylim(top=1.0)
            axs[j].ticklabel_format(style='sci', axis='both', scilimits=(0, 0), useMathText=True)

    
    @staticmethod
    def draw_legend():
        colors = [Viewer.cmap_label(c)[:3] for c in range(10)]
        handles = [Rectangle(xy=(0,0),width=1,height=1, color=c) for c in colors]
        labels = [c for c in range(10)]

        plt.tight_layout() # 调整整体空白
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0.01, wspace=0.01)
        plt.figure(figsize = (15,0.5))
        plt.legend(handles,labels,mode='expand',ncol=10,fontsize='xx-large')
        plt.axis('off')
        plt.show()



# dices = np.array([0.995, 0.846, 0.887, 0.875, 0.677, 0.804, 0.941, 0.903, 0.935])
# ents = np.array([0.0029177, 0.04104619, 0.06392846, 0.02828205, 0.09408152, 0.04068947, 0.02724613, 0.02925691, 0.04272162])
# vars = np.array([6.69320525e-5, 7.21790792e-4, 1.84312820e-3, 5.68796767e-4, 2.46039101e-3, 1.35828184e-3, 4.56128301e-4, 6.97190573e-4, 1.06489208e-3])