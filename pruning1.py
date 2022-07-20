import tensorflow as tf
import numpy as np

from tensorflow import keras
import tensorflow_model_optimization as tfmot


def main():
    (train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()

    # Normalize the input image so that each pixel value is between 0 and 1.
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # Define the model architecture.
    model = keras.Sequential([
        keras.layers.InputLayer(input_shape=(28, 28)),
        keras.layers.Flatten(),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(10)
    ])

    # Train the digit classification model
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.fit(
        train_images,
        train_labels,
        epochs=4,
        validation_split=0.1,
    )

    _, baseline_model_accuracy = model.evaluate(
        test_images, test_labels, verbose=0)

    print('Baseline test accuracy:', baseline_model_accuracy)

    # PRUNING
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam()
    epochs = 2
    batch_size = 128
    num_images = train_images.shape[0]
    # batches = np.ceil(num_images / batch_size).astype(np.int32)
    batches = 2
    end_step = batches * epochs

    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(target_sparsity=0.5, begin_step=0, end_step=-1,
                                                                  frequency=1)
    }

    model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)

    # Non-boilerplate.
    model_for_pruning.optimizer = optimizer
    step_callback = tfmot.sparsity.keras.UpdatePruningStep()
    step_callback.set_model(model_for_pruning)
    log_callback = tfmot.sparsity.keras.PruningSummaries(
        log_dir="log_dir")  # Log sparsity and other metrics in Tensorboard.
    log_callback.set_model(model_for_pruning)

    step_callback.on_train_begin()  # run pruning callback
    for _ in range(epochs):
        log_callback.on_epoch_begin(epoch=-1)  # run pruning callback
        for batch in range(batches):
            step_callback.on_train_batch_begin(batch=-1)  # run pruning callback

            with tf.GradientTape() as tape:
                logits = model_for_pruning(train_images[batch * batch_size: (batch + 1) * batch_size], training=True)
                loss_value = loss(train_labels[batch * batch_size: (batch + 1) * batch_size], logits)
                grads = tape.gradient(loss_value, model_for_pruning.trainable_variables)
                optimizer.apply_gradients(zip(grads, model_for_pruning.trainable_variables))

        step_callback.on_epoch_end(batch=-1)  # run pruning callback

    model_for_pruning.compile(optimizer='adam',
                              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                              metrics=['accuracy'])

    _, model_for_pruning_accuracy = model_for_pruning.evaluate(
        test_images, test_labels, verbose=0)

    print('Baseline test accuracy:', baseline_model_accuracy)
    print('Pruned test accuracy:', model_for_pruning_accuracy)


if __name__ == '__main__':
    main()



