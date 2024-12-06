from sklearn.model_selection import train_test_split # type: ignore
from keras.callbacks import EarlyStopping, ReduceLROnPlateau #type: ignore
from keras.models import Sequential # type: ignore
from keras.layers import Dense, Dropout, BatchNormalization # type: ignore


def neural_network(oof_predictions, y_train_resampled):
    X_nn_train, X_nn_val, y_nn_train, y_nn_val = train_test_split(oof_predictions, y_train_resampled, test_size=0.2, random_state=42)
    num_classes = 8
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

    nn_model = Sequential([
        Dense(256, activation='relu', input_dim=oof_predictions.shape[1]),
        BatchNormalization(),
        Dropout(0.5),

        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),

        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        Dense(num_classes, activation='softmax')
    ])

    nn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return nn_model, X_nn_train, X_nn_val, y_nn_val, y_nn_train, early_stopping, reduce_lr