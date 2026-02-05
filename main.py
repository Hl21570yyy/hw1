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
    
    # ===== 4. 完整对比实验 =====
    print("\n" + "=" * 60)
    print("步骤 4: 完整对比实验（4组预处理 × 3种距离）")
    print("=" * 60)
    
    train_size = 5000
    test_size = 500
    
    X_train_sub = X_train[:train_size]
    y_train_sub = y_train[:train_size]
    X_test_sub = X_test[:test_size]
    y_test_sub = y_test[:test_size]
    
    print(f"训练集: {train_size}, 测试集: {test_size}\n")
    
    # 4组预处理配置
    preprocess_configs = [
        {"name": "无预处理",        "use_whiten": False, "use_l2norm": False},
        {"name": "只Whitening",     "use_whiten": True,  "use_l2norm": False},
        {"name": "只L2归一化",      "use_whiten": False, "use_l2norm": True},
        {"name": "Whitening+L2",    "use_whiten": True,  "use_l2norm": True},
    ]
    
    # 3种距离
    distance_metrics = ['L1', 'L2', 'cosine']
    
    # 存储所有结果
    all_results = {}
    
    # 遍历4组预处理
    for config in preprocess_configs:
        print("=" * 60)
        print(f"【{config['name']}】")
        print(f"    Whitening: {config['use_whiten']}, L2归一化: {config['use_l2norm']}")
        print("=" * 60)
        
        all_results[config['name']] = {}
        
        # 遍历3种距离
        for metric in distance_metrics:
            print(f"\n  --- {metric} 距离 ---")
            
            clf = NearestNeighborClassifier(
                distance_metric=metric,
                use_whiten=config['use_whiten'],
                use_l2norm=config['use_l2norm']
            )
            clf.fit(X_train_sub, y_train_sub)
            acc = clf.score(X_test_sub, y_test_sub, verbose=False)
            
            all_results[config['name']][metric] = acc
            print(f"  {metric} 准确率: {acc * 100:.2f}%")
        
        print()
    
    # ===== 5. 结果汇总表格 =====
    print("\n" + "=" * 60)
    print("结果汇总")
    print("=" * 60)
    
    # 表头
    print(f"{'预处理方式':<18} {'L1':>10} {'L2':>10} {'cosine':>10}")
    print("-" * 50)
    
    # 每行数据
    best_acc = 0
    best_config = ""
    
    for config_name, metrics in all_results.items():
        l1_acc = metrics['L1'] * 100
        l2_acc = metrics['L2'] * 100
        cos_acc = metrics['cosine'] * 100
        
        print(f"{config_name:<18} {l1_acc:>9.2f}% {l2_acc:>9.2f}% {cos_acc:>9.2f}%")
        
        # 找最佳
        for metric, acc in metrics.items():
            if acc > best_acc:
                best_acc = acc
                best_config = f"{config_name} + {metric}"
    
    print("-" * 50)
    print(f"最佳组合: {best_config} ({best_acc * 100:.2f}%)")
    



if __name__ == "__main__":
    main()
