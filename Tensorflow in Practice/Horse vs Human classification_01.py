# %%
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator as IDG

train_horse_dir = os.path.join('horse-or-human/horses')
train_human_dir = os.path.join('horse-or-human/humans')
val_horse_dir = os.path.join('validation-horse-or-human/horses')
val_human_dir = os.path.join('validation-horse-or-human/humans')

# %%
class cb(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('val_accuracy')>0.87:
            print('\nTraining is stopped as we reached desired validation accuracy')
            self.model.stop_training = True

callback = cb()


# %%
print('Total training horse images : ', len(os.listdir(train_horse_dir)))
print("Total training human images : ", len(os.listdir(train_human_dir)))
print('Total Validation Horse images : ', len(os.listdir(val_horse_dir)))
print("Total validation Human images : ", len(os.listdir(val_human_dir)))

# %%
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(300, 300, 3)))
model.add(tf.keras.layers.MaxPool2D(2,2))
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPool2D(2, 2))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPool2D(2, 2))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPool2D(2, 2))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPool2D(2, 2))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.summary()

# %%
model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
train_data_gen = IDG(rescale = 1./255)
path = "horse-or-human"
path2 = 'validation-horse-or-human'
train_generator = train_data_gen.flow_from_directory(path, target_size = (300, 300), batch_size = 128, class_mode = 'binary')
val_generator = train_data_gen.flow_from_directory(path2, target_size=(300, 300), batch_size=32, class_mode='binary')
history = model.fit(train_generator, steps_per_epoch=1024//128, epochs=15, verbose=1, validation_data=val_generator, validation_steps=8)


# %%
import numpy as np
from keras.preprocessing import image

path1 = 'testing images/Gurram/rider-gee5c714ee_1920.jpg'
img = image.load_img(path1, target_size=(300, 300))
x = image.img_to_array(img)
x = x/255.
x = np.expand_dims(x, axis=0)
images = np.vstack([x])
classes = model.predict(images, batch_size=10)
#print(classes[0])
if classes[0]>0.5:
    print("It is a Human.")
else:
    print("It is a Horse.")


