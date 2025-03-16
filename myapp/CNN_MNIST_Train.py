### 1. Nhập các thư viện cần thiết
import numpy as np
from tensorflow import keras
from tensorflow.keras import models, layers, datasets, losses, optimizers

def CNN_MNIST_Train_func(): # epochs,batch_size,learning_rate
  ### Tham số toàn cục
  epochs = 50  # Số lần huấn luyện toàn bộ tập dữ liệu
  batch_size = 128  # Số mẫu dữ liệu được dùng cho một lần tính Gradient và cập nhật trọng số
  learning_rate = 0.001  # Tốc độ học: Kiểm soát sự cập nhật trọng số

  ### 2. Nhập tập dữ liệu huấn luyện và kiểm thử
  data = datasets.mnist

  ### 3. Tiền xử lý dữ liệu
  (x_train, y_train), (x_test, y_test) = data.load_data()
  x_train, x_test = x_train / 255.0, x_test / 255.0

  x_train = x_train[:128]; x_test = x_test[:32]
  y_train = y_train[:128]; y_test = y_test[:32]

  x_train = np.expand_dims(x_train, -1)
  x_test = np.expand_dims(x_test, -1)
  input_shape = x_train.shape[1:]
  num_classes = 10

  ### 4. Xây dựng mô hình
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
  # model.summary()

  ### 5. Huấn luyện mô hình
  history = model.fit(x=x_train, y=y_train, validation_data=(x_test,y_test), epochs=epochs, batch_size=batch_size)

  ### 6. Đánh giá mô hình
  model.evaluate(x=x_test, y=y_test, verbose=2)

  # region Save weights
  # model.save("savedWeights.h5")
  # endregion

  # region Mở rộng
  # from utils import *
  # classes  = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
  # num_rows, num_cols = 4, 2
  # image_index = np.random.randint(low=0, high=x_test.shape[0], size=num_rows*num_cols)
  # predictions = model(x_test[image_index], training=False)
  # PlotPredictedResults(np.squeeze(x_test[image_index]), y_test[image_index], classes, predictions, num_rows, num_cols)
  # endregion

  # region Plot the accuracy and the loss of the model
  # accPlot = np.zeros((2,epochs)); lossPlot = np.zeros((2,epochs))
  # accPlot[0] = np.array(history.history['accuracy'])
  # accPlot[1] = np.array(history.history['val_accuracy'])
  # lossPlot[0] = np.array(history.history['loss'])
  # lossPlot[1] = np.array(history.history['val_loss'])
  # epochs_range = range(epochs)

  # plt.figure()
  # plt.subplot(1, 2, 1)
  # plt.plot(epochs_range, accPlot[0], label='Training Accuracy')
  # plt.plot(epochs_range, accPlot[1], label='Test Accuracy')
  # plt.legend(loc='lower right'); plt.title('Training and Test Accuracy')
  #
  # plt.subplot(1, 2, 2)
  # plt.plot(epochs_range, lossPlot[0], label='Training Loss')
  # plt.plot(epochs_range, lossPlot[1], label='Test Loss')
  # plt.legend(loc='upper right'); plt.title('Training and Test Loss')
  # plt.tight_layout(); plt.show(block=False)
  # endregion
  # b = 1

  epochs_range = list(range(1, epochs+1))
  train_loss = list(history.history['loss'])
  test_loss = list(history.history['val_loss'])
  # train_accuracy = list(history.history['accuracy'])
  # test_accuracy = list(history.history['val_accuracy'])
  train_accuracy = list(np.array(history.history['accuracy'])*100)
  test_accuracy = list(np.array(history.history['val_accuracy'])*100)
  return epochs_range, train_loss, test_loss, train_accuracy, test_accuracy

# epochs_range, train_loss, test_loss, train_accuracy, test_accuracy = CNN_MNIST_Train_func() # epochs, batch_size,learning_rate