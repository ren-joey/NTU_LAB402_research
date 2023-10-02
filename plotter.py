
import matplotlib
import matplotlib.pyplot as plt


def img_plotter (imgs, titles=None, colormode='gray', dpi=200, fontsize=8):
    matplotlib.rc('font', **{'size': fontsize})
    amount = len(imgs)
    plt.figure(dpi=dpi)
    for i in range(amount):
        plt.subplot(1, amount, i + 1)

        if titles is not None:
            plt.title(titles[i])

        plt.axis('off')
        if colormode == 'gray':
            plt.imshow(imgs[i], cmap='gray', vmin=0, vmax=255)
        elif colormode == 'rgb':
            plt.imshow(imgs[i])
    plt.show()
    plt.close()

