import numpy as np
from download import data_loader
from vis_data import plot_data_distribution, plot_random_images
from nn import NearestNeighborClassifier

def main():
    # ===== 1. 下载数据 =====
    
    print("=" * 50)
    print("步骤 1: 下载数据集")
    print("=" * 50)
    X_train, y_train, X_test, y_test = data_loader()
    
    # ===== 2. 可视化数据分布 =====
    print("\n" + "=" * 50)
    print("步骤 2: 可视化数据分布")
    print("=" * 50)
    plot_data_distribution(y_train, y_test)
    
    # ===== 3. 可视化随机图像 =====
    print("\n" + "=" * 50)
    print("步骤 3: 可视化随机图像")
    print("=" * 50)
    plot_random_images(X_train, y_train)
    
    # ===== 4. 实现最近邻分类器 =====
    print("\n" + "=" * 50)
    print("步骤 4: 最近邻分类")
    print("=" * 50)
    
    train_size = 5000   
    test_size = 500     
    
    X_train_sub = X_train[:train_size]
    y_train_sub = y_train[:train_size]
    X_test_sub = X_test[:test_size]
    y_test_sub = y_test[:test_size]
    
    # 测试 L1 距离
    print("\n--- 曼哈顿距离 ---")
    clf_l1 = NearestNeighborClassifier(distance_metric='L1')
    clf_l1.fit(X_train_sub, y_train_sub)
    accuracy_l1 = clf_l1.score(X_test_sub, y_test_sub)
    print(f"L1 准确率: {accuracy_l1 * 100:.2f}%")
    
    # 测试 L2 距离
    print("\n--- 欧几里得距离 ---")
    clf_l2 = NearestNeighborClassifier(distance_metric='L2')
    clf_l2.fit(X_train_sub, y_train_sub)
    accuracy_l2 = clf_l2.score(X_test_sub, y_test_sub)
    print(f"L2 准确率: {accuracy_l2 * 100:.2f}%")
    
   

if __name__ == "__main__":
    main()
