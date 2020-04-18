import matplotlib.pyplot as plt
import os


def arr2str(arr):
    s = '['
    for x in arr:
        s += '{:.3f}, '.format(x)
    return s[:-2] + ']'


class Viewer(object):
    # cmap = plt.cm.get_cmap('Paired', 10)    # 10 discrete colors
    cmap_label = plt.cm.get_cmap('tab10', 10)    # 10 discrete colors
    cmap_heat  = plt.cm.get_cmap('jet')
    # cmap = plt.cm.get_cmap('Set3', 10)    # 10 discrete colors

    def __init__(self):
        super().__init__()
        
    @staticmethod
    def show_fig_4(image, y_gt, entropy, y_pred, title, dices):
        # show me the result
        fig, axs = plt.subplots(2,2, sharey=True, figsize=(10,8))
        axs[0][0].set_title("Original data")
        axs[0][1].set_title("Ground Truth")
        axs[1][0].set_title("Entropy [{:.3f}, {:.3f}]".format(entropy.min(), entropy.max()))
        axs[1][1].set_title("Prediction")
        plt.suptitle("({}) dice : {} [c1-c8 mean: {:.3f}]".format(title, arr2str(dices), dices[1:].mean()))

        ax00 = axs[0][0].imshow( image[0,...], aspect="auto")
        ax01 = axs[0][1].imshow( y_gt, cmap=Viewer.cmap_label, aspect="auto", vmin=0, vmax=9, interpolation='none')
        ax10 = axs[1][0].imshow( entropy[0,...],  aspect="auto", cmap=Viewer.cmap_heat, vmin=0, vmax=2, interpolation='none')
        ax11 = axs[1][1].imshow( y_pred, cmap=Viewer.cmap_label, aspect="auto", vmin=0, vmax=9, interpolation='none')
        
        fig.colorbar(ax00, ax=axs[0][0])
        fig.colorbar(ax01, ax=axs[0][1])
        fig.colorbar(ax10, ax=axs[1][0])
        fig.colorbar(ax11, ax=axs[1][1])


    @staticmethod
    def show_fig(image, y_gt, entropy, y_pred, variance, title, dices):
        fig, axs = plt.subplots(2,3, sharey=True, figsize=(16,8.5))
        axs[0][0].set_title("Original data")
        axs[1][0].set_title("Ground Truth")
        axs[0][1].set_title("Entropy [{:.3f}, {:.3f}]".format(entropy.min(), entropy.max()))
        axs[1][1].set_title("Prediction")
        axs[0][2].set_title("Variance [{:.3f}, {:.3f}]".format(variance.min(), variance.max()))
        plt.suptitle("({}) dice : {} [c1-c8 mean: {:.3f}]".format(title, arr2str(dices), dices[1:].mean()))
        
        # cmap = plt.cm.get_cmap('Paired', 10)    # 10 discrete colors
        # cmap = plt.cm.get_cmap('tab10', 10)    # 10 discrete colors
        # cmap = plt.cm.get_cmap('Set3', 10)    # 10 discrete colors

        ax00 = axs[0][0].imshow( image[0], aspect="auto", cmap='gray')
        ax10 = axs[1][0].imshow( y_gt, cmap=Viewer.cmap_label, aspect="auto", vmin=0, vmax=9, interpolation='none')
        ax01 = axs[0][1].imshow( entropy,  aspect="auto", cmap=Viewer.cmap_heat, vmin=0.0, vmax=2.0, interpolation='none')
        ax11 = axs[1][1].imshow( y_pred, cmap=Viewer.cmap_label, aspect="auto", vmin=0, vmax=9, interpolation='none')
        ax02 = axs[0][2].imshow( variance, aspect="auto", cmap=Viewer.cmap_heat, vmin=0.0, vmax=0.35, interpolation='none')
        
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
        ax01 = axs[3].imshow( entropy,  aspect="auto", cmap=Viewer.cmap_heat, vmin=0.0, vmax=2.0, interpolation='none')
        ax11 = axs[2].imshow( y_pred, cmap=Viewer.cmap_label, aspect="auto", vmin=0, vmax=9, interpolation='none')
        # ax02 = axs[4].imshow( variance, aspect="auto", cmap=Viewer.cmap_heat, vmin=0.0, vmax=0.33, interpolation='none')
        ax02 = axs[4].imshow( variance, aspect="auto", cmap=Viewer.cmap_heat, vmin=0.0, vmax=0.15, interpolation='none')
        fig.colorbar(ax00, ax=axs[0])
        fig.colorbar(ax10, ax=axs[1])
        fig.colorbar(ax01, ax=axs[3])
        fig.colorbar(ax11, ax=axs[2])
        fig.colorbar(ax02, ax=axs[4])

        name = title
        plt.savefig(os.path.join(result_dir, name))