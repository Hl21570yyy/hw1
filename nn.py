import numpy as np


class NearestNeighborClassifier:
    def __init__(self, distance_metric='L2', use_whiten=False, use_l2norm=False):
        """
        参数:
        - distance_metric: 'L1', 'L2', 'cosine'
        - use_whiten: 是否做 Whitening（标准化）
        - use_l2norm: 是否做 L2 归一化
        """
        self.distance_metric = distance_metric
        self.use_whiten = use_whiten
        self.use_l2norm = use_l2norm
        
        self.X_train = None
        self.y_train = None
        
        # Whitening 参数（在 fit 时计算，给 predict 用）
        self.mu = None
        self.sigma = None
    
    def fit(self, X_train, y_train):
        """存储训练数据 + 拟合预处理参数"""
        # 1. 展平
        X = X_train.reshape(X_train.shape[0], -1).astype(np.float32)
        
        # 2. Whitening
        if self.use_whiten:
            self.mu = X.mean(axis=0, keepdims=True)           # (1, D) 每个特征的均值
            self.sigma = X.std(axis=0, keepdims=True) + 1e-8  # (1, D) 每个特征的标准差
            X = (X - self.mu) / self.sigma
        
        # 3. L2 归一化 
        if self.use_l2norm:
            norms = np.linalg.norm(X, axis=1, keepdims=True)  # (N, 1) 每个样本的模长
            X = X / (norms + 1e-8)
        #两种区别就是一个是对特征处理归一化，一种是对行处理归一化，行和列的区别
        self.X_train = X
        self.y_train = y_train
    
    def _preprocess_one(self, x):
        """对单个测试样本应用相同的预处理"""
        x = x.flatten().astype(np.float32)
        
        if self.use_whiten:
            x = (x - self.mu.flatten()) / self.sigma.flatten()
        
        if self.use_l2norm:
            x = x / (np.linalg.norm(x) + 1e-8)
        
        return x
    
    def predict_one(self, x):
        """预测单个样本"""
        x = self._preprocess_one(x)
        
        if self.distance_metric == 'L1':
            # 曼哈顿距离
            distances = np.sum(np.abs(self.X_train - x), axis=1)
        
        elif self.distance_metric == 'L2':
            # 欧几里得距离
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
        
        elif self.distance_metric == 'cosine':
            # 余弦距离
            dot_product = np.dot(self.X_train, x)
            norm_train = np.linalg.norm(self.X_train, axis=1)
            norm_x = np.linalg.norm(x)
            cosine_sim = dot_product / (norm_train * norm_x + 1e-8)
            distances = 1 - cosine_sim
        
        else:
            raise ValueError(f"Unknown metric: {self.distance_metric}")

        nearest_idx = np.argmin(distances)
        return self.y_train[nearest_idx]
    
    def predict(self, X_test, verbose=True):
        """预测多个样本"""
        predictions = []
        n_samples = len(X_test)
        
        for i, x in enumerate(X_test):
            pred = self.predict_one(x)
            predictions.append(pred)
            
        
        return np.array(predictions)
    
    def score(self, X_test, y_test, verbose=True):
        """计算准确率"""
        predictions = self.predict(X_test, verbose=verbose)
        accuracy = np.mean(predictions == y_test)
        return accuracy
