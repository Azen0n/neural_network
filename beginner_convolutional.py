import tensorflow as tf
import matplotlib.pyplot as plt

# 50 000 картинок в обучающей выборке и 10 000 в тестовой
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

# Нормализация значений пикселей
train_images = train_images / 255.0
test_images = test_images / 255.0

# Метки классов
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Функция активации, одна на все слои
activation_function = 'relu'

#
number_of_layers = 3
number_of_epochs = 10

#
filters = 32
filters2 = 64

#
kernel_size = (3, 3)

#
pool_size = (2, 2)

#
model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Conv2D(filters, kernel_size, activation=activation_function, input_shape=(32, 32, 3)))
for _ in range(number_of_layers - 1):
    model.add(tf.keras.layers.MaxPooling2D(pool_size))
    model.add(tf.keras.layers.Conv2D(filters2, kernel_size, activation=activation_function))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(filters2, activation=activation_function))
model.add(tf.keras.layers.Dense(len(class_names)))

model.summary()


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=number_of_epochs,
                    validation_data=(test_images, test_labels))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print(test_acc)
