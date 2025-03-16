### Tham số toàn cục
epochs = 2 # Số lần huấn luyện toàn bộ tập dữ liệu
batch_size = 128 # Số mẫu dữ liệu được dùng cho một lần tính Gradient và cập nhật trọng số
learning_rate = 0.001 # Tốc độ học: Kiểm soát sự cập nhật trọng số

### 1. Nhập các thư viện cần thiết
import numpy as np
from tensorflow import keras
from tensorflow.keras import models, layers, datasets, losses, optimizers
from keras.preprocessing.image import load_img, img_to_array

### 2. Nhập tập dữ liệu huấn luyện và kiểm thử
mnist = datasets.mnist

### 3. Tiền xử lý dữ liệu
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
input_shape = x_train.shape[1:]
num_classes = 10

### 4. Xây dựng mô hình
# model = models.Sequential([
#   layers.Flatten(input_shape=input_shape),
#   layers.Dense(units=128, activation='relu'),
#   layers.Dense(units=num_classes, activation='softmax')
# ])

model = models.Sequential([
  keras.Input(shape=input_shape),
  layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu"),
  layers.MaxPooling2D(pool_size=(2, 2)),
  layers.Flatten(),
  layers.Dense(units=num_classes, activation='softmax')
])

loss_fn = losses.SparseCategoricalCrossentropy()
optimizer = optimizers.Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

### Load the saved model
model.load_weights("savedWeights_CNN_MNIST.h5")

# region Mở rộng
from utils import *
classes  = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
num_rows, num_cols = 4, 2
image_index = np.random.randint(low=0, high=x_test.shape[0], size=num_rows*num_cols)
PlotTrainingImages(x_test[image_index], y_test[image_index], classes, num_rows, num_cols)
predictions = model(x_test[image_index], training=False)
PlotPredictedResults(np.squeeze(x_test[image_index]), y_test[image_index], classes, predictions, num_rows, num_cols)
# endregion

# region Images from Disk
path = "1.png"; true_value = int(path[0])
img_array = img_to_array(load_img('Test images/'+path, color_mode='grayscale')) # rgb
img = (255.0 - img_array) / 255.0

prediction = model(np.expand_dims(img,0), training=False)

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1); PlotImage(img, true_value, classes, prediction)
plt.subplot(1, 2, 2); PlotBar(true_value, np.squeeze(prediction))
plt.tight_layout(); plt.show(block=False)
# endregion
b=1