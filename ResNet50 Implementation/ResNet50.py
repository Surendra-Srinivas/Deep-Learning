from keras. preprocessing.image import ImageDataGenerator
from keras.layers import Activation, BatchNormalization, Conv2D, Add
from keras.layers import Input, ZeroPadding2D, MaxPooling2D, AveragePooling2D
from keras.layers import Flatten, Dense
from keras.models import Model
from keras import activations
from tensorflow.keras.optimizers import Adam
from keras.regularizers import l2

train_data_gen = ImageDataGenerator(rescale=1./255)
validation_data_gen = ImageDataGenerator(rescale=1./255)

train_generator = train_data_gen.flow_from_directory('/Users/surendrasrinivas/Downloads/Emotion Detection/train',target_size=(48,48),batch_size=64,color_mode="grayscale",class_mode='categorical')
validation_generator = train_data_gen.flow_from_directory('/Users/surendrasrinivas/Downloads/Emotion Detection/test',target_size=(48,48),batch_size=64,color_mode="grayscale",class_mode='categorical')

#class_types = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def res_identity(x, filters):
    x_skip = x
    f1, f2 = filters
    # Block 1
    x  = Conv2D(f1, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation(activations.relu)(x)
    # Block 2 - Bottle Neck
    x = Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation(activations.relu)(x)
    # Third Block
    x = Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Add()([x, x_skip])
    x = Activation(activations.relu)(x)

    return x

def res_conv(x, s, filters):
    x_skip = x
    f1, f2 = filters
    # First Block
    x = Conv2D(f1, kernel_size=(1, 1), strides = (s, s), padding='valid', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation(activations.relu)(x)
    # Second Block - Bottle Neck
    x = Conv2D(f1, kernel_size=(3, 3), strides= (1, 1), padding = 'same', kernel_regularizer= l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation(activations.relu)(x)
    # Third Block
    x = Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer= l2(0.001))(x)
    x = BatchNormalization()(x)
    # Shortcut
    x_skip = Conv2D(f2, kernel_size=(1, 1), strides=(s, s), padding='valid', kernel_regularizer= l2(0.001))(x)
    x_skip = BatchNormalization()(x_skip)
    # Add 
    x = Add()([x, x_skip])
    x = Activation(activations.relu)(x)

    return x

def resnet50():
    input_im = Input(shape=(48, 48, 1))
    x = ZeroPadding2D(padding=(3, 3))(input_im)
    
    x = Conv2D(64, kernel_size=(7, 7), strides=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Activation(activations.relu)(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = res_conv(x, s=1, filters=(64, 256))
    x = res_identity(x, filters=(64, 256))
    x = res_identity(x, filters=(64, 256))

    x = res_conv(x , s = 1, filters=(128, 512)) # s=2
    x = res_identity(x, filters=(128, 512))
    x = res_identity(x, filters=(128, 512))
    x = res_identity(x, filters=(128, 512))

    x = res_conv(x, s = 1, filters=(256, 1024)) # s=1
    x = res_identity(x, filters=(256, 1024))
    x = res_identity(x, filters=(256, 1024))
    x = res_identity(x, filters=(256, 1024))
    x = res_identity(x, filters=(256, 1024))
    x = res_identity(x, filters=(256, 1024))

    x = res_conv(x, s= 1, filters=(512, 2048)) #s=2
    x = res_identity(x, filters=(512, 2048))
    x = res_identity(x, filters=(512, 2048))

    x = AveragePooling2D((2, 2), padding = 'same')(x)
    x = Flatten()(x)
    x = Dense(7, activation='softmax', kernel_initializer='he_normal')(x)

    model = Model(inputs = input_im, outputs= x, name='ResNet50')

    return model
    
model = resnet50()
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate =1e-5), metrics = ['accuracy'])
final = model.fit(train_generator,steps_per_epoch=28709 // 64, epochs=10, validation_data= validation_generator, validation_steps=7178 // 64) 


