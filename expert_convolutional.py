import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

# 50 000 картинок в обучающей выборке и 10 000 в тестовой
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

# Нормализация
train_images = train_images / 255.0
test_images = test_images / 255.0

# "Пакетная обработка"
train_ds = tf.data.Dataset.from_tensor_slices(
    (train_images, train_labels)).shuffle(10000).batch(32)

# Перемешивание датасета
test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(32)


# Модель tf.keras
class MyModel(Model):
    def __init__(self, filters, kernel_size, activation_function):
        super(MyModel, self).__init__()
        self.conv1 = Conv2D(filters, kernel_size, activation=activation_function)
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10)

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)


# https://www.tensorflow.org/tutorials/quickstart/advanced
######################################################################################
filters = 32                                                                        ##
kernel_size = 3                                                                     ##
activation_function = 'relu'                                                        ##
######################################################################################

# Create an instance of the model
model = MyModel(filters, kernel_size, activation_function)

# Функция потерь и оптимизатор для обучения
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()

# Метрики для измерения потерь и точности модели
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


# Обучение модели с tf.GradientTape
@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)


# Тест модели
@tf.function
def test_step(images, labels):
    # training=False is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(images, training=False)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)


number_of_epochs = 5

for epoch in range(number_of_epochs):
    # Reset the metrics at the start of the next epoch
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

    for images, labels in train_ds:
        train_step(images, labels)

    for images, labels in test_ds:
        test_step(images, labels)

    print(
        f'Epoch {epoch + 1}, '
        f'Loss: {train_loss.result()}, '
        f'Accuracy: {train_accuracy.result() * 100}, '
        f'Test Loss: {test_loss.result()}, '
        f'Test Accuracy: {test_accuracy.result() * 100}'
    )
