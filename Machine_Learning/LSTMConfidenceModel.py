import numpy as np
import tensorflow as tf
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Bidirectional, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

class MultiTimeframeLSTM:
    def __init__(self, lookback_short=20, lookback_medium=40, lookback_long=60):
        self.lookback_short = lookback_short
        self.lookback_medium = lookback_medium
        self.lookback_long = lookback_long
        self.scaler = RobustScaler()
        self.model = None

    def build_model(self, input_shape):
        """Build enhanced LSTM with more capacity"""
        model = Sequential([
            Input(shape=input_shape),
            Bidirectional(LSTM(128, return_sequences=True)),
            BatchNormalization(),
            Dropout(0.3),
            Bidirectional(LSTM(64, return_sequences=True)),
            BatchNormalization(),
            Dropout(0.3),
            LSTM(32),
            BatchNormalization(),
            Dropout(0.2),
            Dense(32, activation='swish'),
            Dropout(0.2),
            Dense(16, activation='swish'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(0.0003),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        self.model = model

    def prepare_sequences(self, X_scaled, y, lookback):
        """Prepare sequences for LSTM"""
        X_seq, y_seq = [], []
        for i in range(lookback, len(X_scaled) - 1):
            X_seq.append(X_scaled[i-lookback:i])
            y_seq.append(y[i])
        return np.array(X_seq), np.array(y_seq)

    def train_and_predict(self, df, use_class_weights=True):
        """Train model and return predictions with confidence"""
        if len(df) < self.lookback_long + 100:
            return None

        y = df['Target'].values
        X_scaled = self.scaler.fit_transform(df.drop('Target', axis=1).values)

        # Use medium lookback (40 days)
        X_seq, y_seq = self.prepare_sequences(X_scaled, y, self.lookback_medium)

        # Train/test split
        split = int(len(X_seq) * 0.85)
        X_train, y_train = X_seq[:split], y_seq[:split]
        X_test, y_test = X_seq[split:], y_seq[split:]

        # Calculate class weights if requested
        class_weight = None
        if use_class_weights:
            class_counts = np.bincount(y_train)
            total = len(y_train)
            class_weight = {
                0: total / (2 * class_counts[0]),
                1: total / (2 * class_counts[1])
            }

        self.build_model((X_train.shape[1], X_train.shape[2]))

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6)
        ]

        self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=50,
            batch_size=32,
            verbose=0,
            callbacks=callbacks,
            class_weight=class_weight
        )

        # Get predictions and probabilities
        test_probs = self.model.predict(X_test, verbose=0).flatten()
        test_preds = (test_probs > 0.5).astype(int)

        # Get last prediction
        last_seq = X_scaled[-self.lookback_medium:].reshape(1, self.lookback_medium, X_scaled.shape[1])
        next_prob = self.model.predict(last_seq, verbose=0)[0][0]

        # Calculate metrics
        acc = accuracy_score(y_test, test_preds) * 100

        # Detailed diagnostics
        diagnostics = {
            'class_balance_train': np.bincount(y_train),
            'class_balance_test': np.bincount(y_test),
            'confusion_matrix': confusion_matrix(y_test, test_preds),
            'test_probs': test_probs,
            'test_preds': test_preds,
            'test_actuals': y_test,
            'classification_report': classification_report(y_test, test_preds, output_dict=True)
        }

        return next_prob, acc, diagnostics


class ConfidenceFilter:
    @staticmethod
    def analyze_by_confidence(diagnostics, confidence_threshold=0.6):
        """Analyze performance by confidence levels"""
        probs = diagnostics['test_probs']
        preds = diagnostics['test_preds']
        actuals = diagnostics['test_actuals']

        # Calculate confidence (distance from 0.5)
        confidence = np.abs(probs - 0.5) * 2

        # Filter high-confidence predictions
        high_conf_mask = confidence >= confidence_threshold

        if high_conf_mask.sum() == 0:
            return None

        high_conf_preds = preds[high_conf_mask]
        high_conf_actuals = actuals[high_conf_mask]
        high_conf_acc = accuracy_score(high_conf_actuals, high_conf_preds) * 100

        # Low confidence predictions
        low_conf_mask = confidence < confidence_threshold
        low_conf_preds = preds[low_conf_mask]
        low_conf_actuals = actuals[low_conf_mask]
        low_conf_acc = accuracy_score(low_conf_actuals, low_conf_preds) * 100 if low_conf_mask.sum() > 0 else 0

        return {
            'total_predictions': len(probs),
            'high_conf_count': high_conf_mask.sum(),
            'high_conf_pct': (high_conf_mask.sum() / len(probs)) * 100,
            'high_conf_accuracy': high_conf_acc,
            'low_conf_count': low_conf_mask.sum(),
            'low_conf_accuracy': low_conf_acc,
            'confidence_threshold': confidence_threshold,
            'mean_confidence': confidence.mean() * 100,
            'max_confidence': confidence.max() * 100
        }

    @staticmethod
    def get_confidence_tiers(diagnostics):
        """Analyze performance across multiple confidence tiers"""
        probs = diagnostics['test_probs']
        preds = diagnostics['test_preds']
        actuals = diagnostics['test_actuals']

        confidence = np.abs(probs - 0.5) * 2

        tiers = [
            (0.0, 0.2, 'Very Low'),
            (0.2, 0.4, 'Low'),
            (0.4, 0.6, 'Medium'),
            (0.6, 0.8, 'High'),
            (0.8, 1.0, 'Very High')
        ]

        results = []
        for min_conf, max_conf, label in tiers:
            mask = (confidence >= min_conf) & (confidence < max_conf)
            if mask.sum() == 0:
                continue

            tier_preds = preds[mask]
            tier_actuals = actuals[mask]
            tier_acc = accuracy_score(tier_actuals, tier_preds) * 100

            results.append({
                'tier': label,
                'range': f'{min_conf*100:.0f}-{max_conf*100:.0f}%',
                'count': mask.sum(),
                'pct': (mask.sum() / len(probs)) * 100,
                'accuracy': tier_acc
            })

        return results
