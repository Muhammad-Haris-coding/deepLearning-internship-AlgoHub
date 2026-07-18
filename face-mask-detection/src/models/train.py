"""
Model Training
--------------
Train and evaluate the Face Mask Detection model using MobileNetV2.

Author : Afaq Ahmad Khan
Project: Face Mask Detection
"""

from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
)

from src.config import (
    IMAGE_SIZE,
    LEARNING_RATE,
    EPOCHS,
    MODEL_PATH,
)

from src.data.preprocess import get_datasets
from src.utils.visualization import plot_training_history




# ==========================================================
# Build Model
# ==========================================================

def build_model() -> Model:
    """
    Create MobileNetV2 transfer learning model.
    """

    base_model = MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )

    # Freeze pretrained weights
    base_model.trainable = False

    inputs = layers.Input(shape=(*IMAGE_SIZE, 3))

    # ---------------------------
    # Data Augmentation
    # ---------------------------

    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.10)(x)
    x = layers.RandomZoom(0.10)(x)

    # ---------------------------
    # Feature Extractor
    # ---------------------------

    x = base_model(x, training=False)

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dropout(0.30)(x)

    outputs = layers.Dense(
        1,
        activation="sigmoid",
    )(x)

    model = Model(inputs, outputs)

    model.compile(

        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),

        loss="binary_crossentropy",

        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )

    return model


# ==========================================================
# Callbacks
# ==========================================================

def get_callbacks():

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    callbacks = [

        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),

        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=2,
            verbose=1,
        ),

        ModelCheckpoint(
            filepath=MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),

    ]

    return callbacks


# ==========================================================
# Train
# ==========================================================

def train():

    train_ds, valid_ds, test_ds, class_names = get_datasets()

    print("\nClasses:", class_names)

    model = build_model()

    model.summary()

    history = model.fit(

        train_ds,

        validation_data=valid_ds,

        epochs=EPOCHS,

        callbacks=get_callbacks(),

    )
    plot_training_history(history)

    print("\nTraining Finished.")
    
    return model, history, test_ds

    


# ==========================================================
# Evaluate
# ==========================================================

def evaluate(
    model: Model,
    test_ds,
):

    print("\nEvaluating...\n")

    results = model.evaluate(
        test_ds,
        verbose=1,
    )

    print("\nTest Metrics")

    for name, value in zip(
        model.metrics_names,
        results,
    ):
        print(f"{name:<12}: {value:.4f}")


# ==========================================================
# Main
# ==========================================================

def main():

    model, history, test_ds = train()

    evaluate(
        model,
        test_ds,
    )

    print(f"\nSaved Model:\n{MODEL_PATH}")


if __name__ == "__main__":
    main()