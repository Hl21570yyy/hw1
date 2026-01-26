import matplotlib.pyplot as plt
import numpy as np

def plot_data_distribution(y_train, y_test):
    #一列两行两个子图，一个train一个test
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    
    # 训练集分布
    unique, counts = np.unique(y_train, return_counts=True)
    axes[0].bar(unique, counts, color='steelblue')
    axes[0].set_xlabel('Class Label')
    axes[0].set_ylabel('Number of Samples')
    axes[0].set_title('Training Data Distribution')
    axes[0].set_xticks(range(10))
    
    # 测试集分布
    unique, counts = np.unique(y_test, return_counts=True)
    axes[1].bar(unique, counts, color='coral')
    axes[1].set_xlabel('Class Label')
    axes[1].set_ylabel('Number of Samples')
    axes[1].set_title('Test Data Distribution')
    axes[1].set_xticks(range(10))
    
    plt.tight_layout()
    plt.savefig('data_distribution.png', dpi=150)
    plt.show()


def plot_random_images(X_train, y_train, num_images=15):
    """可视化随机图像"""
    #fig是指整个图像，axes是子图数组
    fig, axes = plt.subplots(3, 5, figsize=(10, 6))
    indices = np.random.choice(len(X_train), num_images, replace=False)
    
    for i, ax in enumerate(axes.flat):
        ax.imshow(X_train[indices[i]], cmap='gray')
        ax.set_title(f'Label: {y_train[indices[i]]}')
        ax.axis('off')
    
    plt.suptitle('Random Sample Images', fontsize=14)
    plt.tight_layout()
    plt.savefig('random_images.png', dpi=150)
    plt.show()
