import numpy as np

class NearestNeighborClassifier:
    def __init__(self, distance_metric='L2'):
        self.distance_metric = distance_metric
        self.X_train = None
        self.y_train = None
    
    def fit(self, X_train, y_train):
        """存储训练数据"""
        # 将图像展平为一维向量，为了更简单轻易的去比较因为一纬数组肯定要比二维方便比较一点
        self.X_train = X_train.reshape(X_train.shape[0], -1).astype(np.float32)
        self.y_train = y_train
        print(f"训练完成，存储了 {len(self.X_train)} 个样本")
    
    def predict_one(self, x):
        """预测单个样本"""
        #flatten同样展平
        x = x.flatten().astype(np.float32)
        
        if self.distance_metric == 'L1':
            # 曼哈顿距离
            distances = np.sum(np.abs(self.X_train - x), axis=1)
        else:
            # 欧几里得距离 (L2)
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
        
        # 找到最近的邻居，argmin选择最小的
        nearest_idx = np.argmin(distances)
        return self.y_train[nearest_idx]
    
    def predict(self, X_test, verbose=True):
        """预测多个样本"""
        predictions = []
        n_samples = len(X_test)
        
        for i, x in enumerate(X_test):
            pred = self.predict_one(x)
            predictions.append(pred)
            
            if verbose and (i + 1) % 100 == 0:
                print(f"已预测: {i + 1}/{n_samples}")
        
        return np.array(predictions)
    
    def score(self, X_test, y_test):
        """计算准确率"""
        predictions = self.predict(X_test)
        accuracy = np.mean(predictions == y_test)
        return accuracy
