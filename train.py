# train.py
import numpy as np
import tensorflow as tf
import os
import json
from model import build_cnn
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report

# Load preprocessed dataset
def load_data():
    X_train = np.load("models/X_train.npy")
    X_val = np.load("models/X_val.npy")
    X_test = np.load("models/X_test.npy")
    y_train = np.load("models/y_train.npy")
    y_val = np.load("models/y_val.npy")
    y_test = np.load("models/y_test.npy")
    return X_train, X_val, X_test, y_train, y_val, y_test

def main():
    os.makedirs("models", exist_ok=True)
    with open("metadata.json") as f:
        meta = json.load(f)
    labels = meta["labels"]
    n_classes = len(labels)

    X_train, X_val, X_test, y_train, y_val, y_test = load_data()
    input_shape = X_train.shape[1:]

    model = build_cnn(input_shape, n_classes)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()

    # Compute class weights to handle imbalance
    class_weights = compute_class_weight(
        "balanced", classes=np.unique(y_train), y=y_train
    )
    class_weights = {int(i): w for i, w in enumerate(class_weights)}
    print("Computed class weights:", class_weights)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            "models/baby_cry_best.keras",
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=12,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        class_weight=class_weights,
        callbacks=callbacks,
        shuffle=True,
    )

    # Evaluate on test set
    loss, acc = model.evaluate(X_test, y_test, verbose=2)
    print(f"\nTest Loss: {loss:.4f} | Test Accuracy: {acc:.4f}")

    # Save model and label map
    model.save("models/baby_cry_model_final.keras")
    with open("models/labels.json", "w") as f:
        json.dump(labels, f)
    np.save("models/train_history.npy", history.history)
    print("\n✅ Model saved successfully to models/baby_cry_model_final.keras")

    # Optional: print classification report
    preds = model.predict(X_test)
    y_pred = np.argmax(preds, axis=1)
    print("\n📊 Classification Report:\n")
    print(classification_report(y_test, y_pred, target_names=labels, digits=4))

if __name__ == "__main__":
    main()
