import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tqdm.keras import TqdmCallback
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from data_preprocessing import load_and_preprocess_data
from model import create_model

BASE_PATH = ""  # Please change this to the path of the dataset

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_and_preprocess_data(BASE_PATH)

    # Implement k-fold cross-validation
    n_splits = 5
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train)):
        print(f"Training fold {fold + 1}/{n_splits}")

        X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

        model = create_model((X_train.shape[1], X_train.shape[2]), y_train.shape[1])

        # Learning rate scheduler
        lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

        # Early stopping
        early_stopping = EarlyStopping(patience=10, restore_best_weights=True)

        history = model.fit(
            X_train_fold, y_train_fold,
            validation_data=(X_val_fold, y_val_fold),
            epochs=100,
            batch_size=64,
            callbacks=[
                lr_scheduler,
                early_stopping,
                TqdmCallback(verbose=1)
            ],
            verbose=0
        )

        # Evaluate on validation set
        val_loss = model.evaluate(X_val_fold, y_val_fold, verbose=0)
        fold_scores.append(val_loss)

        print(f"Fold {fold + 1} validation loss: {val_loss}")

    print(f"Average validation loss: {np.mean(fold_scores)}")

    # Train final model on all training data
    print("Training final model on all data...")
    final_model = create_model((X_train.shape[1], X_train.shape[2]), y_train.shape[1])
    final_history = final_model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=64,
        callbacks=[
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
            EarlyStopping(patience=10, restore_best_weights=True),
            TqdmCallback(verbose=1)
        ],
        verbose=0
    )

    print("Saving model...")
    final_model.save('lob_full_model.keras')
    print("Full model saved successfully in .keras format.")

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(final_history.history['loss'], label='Training Loss')
    plt.plot(final_history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    if 'lr' in final_history.history:
        plt.subplot(1, 2, 2)
        plt.plot(final_history.history['lr'], label='Learning Rate')
        plt.title('Learning Rate')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.legend()
    else:
        print("Learning rate not found in history. Skipping LR plot.")

    plt.tight_layout()
    plt.savefig('training_history.png')
    print("Training history plot saved as training_history.png")