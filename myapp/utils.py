import numpy as np
import matplotlib.pyplot as plt

cmap = plt.cm.gray_r # plt.cm.gray
def PlotTrainingImages(x_train, y_train, classes, num_rows, num_cols):
  # image_index = np.random.randint(low=0, high=x_train.shape[0], size=25)
  plt.figure()
  for i in range(num_rows*num_cols):
    plt.subplot(num_rows, num_cols, i + 1); plt.xticks([]); plt.yticks([]); plt.grid(False)
    plt.imshow(x_train[i], cmap=cmap)
    plt.xlabel(classes[y_train[i]])

  plt.tight_layout(); plt.show(block=False)

def PlotBar(true_label, prediction):
  plt.xticks(range(len(prediction))); plt.yticks([]); plt.grid(False) # plt.xticks(range(10), class_names, rotation=45)

  thisplot = plt.bar(range(len(prediction)), prediction, color="#777777"); plt.ylim([0, 1])
  predicted_label = np.argmax(prediction)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')
  pass

def PlotImage(img, true_label, classes, prediction):
  # classes = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
  plt.xticks([]); plt.yticks([]); plt.grid(False)
  plt.imshow(img, cmap=cmap)

  predicted_label = np.argmax(prediction)
  if predicted_label == true_label: color = "blue"
  else: color = "red"

  plt.xlabel(f"{classes[predicted_label]} {100*np.max(prediction):0.2f}% ({classes[true_label]})", color=color)

def PlotPredictedResults(x_test, y_test, classes, predictions, num_rows, num_cols):
  num_images = x_test.shape[0]; num_rows = num_rows; num_cols = num_cols
  plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
  for i in range(num_images):
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    PlotImage(x_test[i], y_test[i], classes, predictions[i])

    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
    PlotBar(y_test[i], predictions[i])

  plt.tight_layout(); plt.show(block=False)
  pass