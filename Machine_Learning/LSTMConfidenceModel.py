import numpy as np
import tensorflow as tf
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import time


class MultiTimeframeLSTM:

    def __init__(self, lookback=60, seed=42):
        self.lookback = lookback
        self.seed = seed
        self.scaler = RobustScaler()
        self.model = None

        np.random.seed(seed)
        tf.random.set_seed(seed)

    def create_sequences(self, X, y):
        Xs, ys = [], []
        for i in range(len(X) - self.lookback):
            Xs.append(X[i:(i + self.lookback)])
            ys.append(y[i + self.lookback])
        return np.array(Xs), np.array(ys)

    def train_and_predict(self, df, verbose=0):
        try:
            # 1. Select Features (Ensure Context Features are included if present)
            feature_cols = ['RSI', 'MACD', 'ATR', 'Dist_MA50', 'Rel_Str_Peers', 'Rel_Str_Partners']
            available_cols = [c for c in feature_cols if c in df.columns]

            if len(available_cols) < 3:
                print("Not enough features for training")
                return None

            X = df[available_cols].values
            y = df['Target'].values

            # Scale
            X_scaled = self.scaler.fit_transform(X)
            X_seq, y_seq = self.create_sequences(X_scaled, y)

            if len(X_seq) < 100: return None

            # Split (Last 15% for test)
            split = int(len(X_seq) * 0.85)
            X_train, X_test = X_seq[:split], X_seq[split:]
            y_train, y_test = y_seq[:split], y_seq[split:]

            # --- ADVANCED WEIGHTING LOGIC ---
            # 1. Time Decay: Recent data is 3.0x more important than old data
            t = np.linspace(0, 1, len(y_train))
            time_weights = np.exp(t * 3.0)
            time_weights /= time_weights.mean()  # Normalize

            # 2. Class Balance: Handle rare volatility spikes
            classes = np.unique(y_train)
            if len(classes) > 1:
                cw = compute_class_weight('balanced', classes=classes, y=y_train)
                cw_dict = dict(zip(classes, cw))
                sample_cw = np.array([cw_dict[c] for c in y_train])
            else:
                sample_cw = np.ones(len(y_train))

            # Combine Weights
            final_weights = time_weights * sample_cw

            # Build Model
            model = Sequential([
                Bidirectional(LSTM(128, return_sequences=True), input_shape=(self.lookback, len(available_cols))),
                BatchNormalization(),
                Dropout(0.3),

                Bidirectional(LSTM(64)),
                BatchNormalization(),
                Dropout(0.3),

                Dense(32, activation='swish'),
                Dense(1, activation='sigmoid')
            ])

            model.compile(optimizer=tf.keras.optimizers.Adam(0.0005),
                          loss='binary_crossentropy', metrics=['accuracy'])

            # Train with weights
            model.fit(
                X_train, y_train,
                epochs=35,
                batch_size=32,
                validation_data=(X_test, y_test),
                sample_weight=final_weights,  # <--- Applied Here
                callbacks=[
                    EarlyStopping(patience=5, restore_best_weights=True),
                    ReduceLROnPlateau(factor=0.5, patience=3)
                ],
                verbose=verbose
            )
            self.model = model

            # Evaluate
            preds = model.predict(X_test, verbose=0)
            pred_classes = (preds > 0.5).astype(int)
            acc = accuracy_score(y_test, pred_classes)

            # Predict Tomorrow
            last_seq = X_scaled[-self.lookback:].reshape(1, self.lookback, len(available_cols))
            prob = model.predict(last_seq, verbose=0)[0][0]

            return prob, acc, {}

        except Exception as e:
            print(f"LSTM Training Error: {e}")
            return None

    # Helper for existing calls (if any)
    def predict_next(self, df):
        pass