# 1. Import necessary libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# 2. Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# 3. Normalize data ( Scale pixel values to 0-1 )
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# 4. Reshape data (CNN accepts 3D : height, width, channels)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
# -1 means figure this dimension out automatically
# 1 is the number of channels

# 5. Build a simple CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# 6. Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 7. Train the model
history = model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=64, #Faster training
    validation_data=(x_test, y_test),
    verbose=1 # shows progress bar
)

# 8. Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print("Test accuracy:",round(test_acc*100,2),"%")

# 9. Prediction Example
prediction = model.predict(x_test[:1])  #get prediction probabilities
prediction_label = prediction.argmax() #find the most likely class

plt.imshow(x_test[0].reshape(28, 28), cmap='gray')
plt.title("Predicted Label:" + str(prediction_label))
plt.axis('off')
plt.show()