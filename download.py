from tensorflow.keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(f"训练集: {X_train.shape}") 
print(f"训练集: {X_test.shape}")
print(f"训练集: {y_test.shape}") 
print(f"训练集: {y_train.shape}")
