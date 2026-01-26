from tensorflow.keras.datasets import mnist
def data_loader(): 
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    print(f"训练集 X: {X_train.shape}") 
    print(f"测试集 X: {X_test.shape}")
    print(f"测试集 y: {y_test.shape}") 
    print(f"训练集 y: {y_train.shape}")
    return X_train, y_train, X_test, y_test 
