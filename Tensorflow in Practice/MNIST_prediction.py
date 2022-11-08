# %%
import tensorflow as tf
print(tf.__version__)

# %%
fmnist = tf.keras.datasets.fashion_mnist
fmnist

# %%
(training_images, training_labels), (testing_images, testing_labels) = fmnist.load_data()

# %%
import numpy as np
import matplotlib.pyplot as plt
# You can put between 0 to 59999 here
index = 0
# Set no.of characters per row when printing
np.set_printoptions(linewidth=320)
# Print the Label and the image
print("Label : {val}".format(val=training_labels[index]))
print("\nImage Pixel Array\n : {val}".format(val = training_images[index]))
# Visulaize the Image
plt.imshow(training_images[index])

# %%
# Normalize the Pixel Values:
training_images = training_images/255.0
testing_images = testing_images/ 255.0

# %%
"""model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))"""

"""model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(512, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))"""

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(1024, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

# %%
model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(training_images, training_labels, epochs=10)

# %%
model.evaluate(testing_images, testing_labels)

# %%
classifications = model.predict(testing_images)
print(classifications[0])
# Get the Index with Highest Probability.
print(np.argmax(classifications[0]))
print(testing_labels[0])

# %%
import numpy as np

# Declare the smple inputs and convert them into Tensors
inputs = np.array([[1.0 ,3.0, 4.0, 2.0]]) # This is not 1D Array
inputs = tf.convert_to_tensor(inputs)
#print(inputs)
print(inputs.numpy())

# Feed the inputs to a Softmax Activation Function
outputs = tf.keras.activations.softmax(inputs)
print(outputs.numpy())
# Get the sum of all values after the Softmax.
sum = tf.reduce_sum(outputs)
print(sum)
# Get the Index with Highest Probability.
pred = np.argmax(outputs)
print(pred)


