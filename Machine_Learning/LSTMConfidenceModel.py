import numpy as np
import tensorflow as tf
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Bidirectional, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import time
from typing import Optional, Tuple, Dict


class MultiTimeframeLSTM:
    def __init__(self, lookback_short=20, lookback_medium=40, lookback_long=60, seed=42):
        """
        Initialize LSTM model with reproducibility

        Args:
            lookback_short: Short-term lookback window
            lookback_medium: Medium-term lookback window
            lookback_long: Long-term lookback window
            seed: Random seed for reproducibility
        """
        self.lookback_short = lookback_short
        self.lookback_medium = lookback_medium
        self.lookback_long = lookback_long
        self.seed = seed
        self.scaler = RobustScaler()
        self.model = None
        self.training_history = None

        # Set seeds for reproducibility
        np.random.seed(seed)
        tf.random.set_seed(seed)

    def validate_data_shape(self, X: np.ndarray, y: np.ndarray) -> Tuple[bool, str]:
        """Validate data shapes before training"""
        if len(X) == 0 or len(y) == 0:
            return False, "Empty data arrays"

        if len(X) != len(y):
            return False, f"X and y have different lengths: {len(X)} vs {len(y)}"

        if len(X) < self.lookback_medium + 100:
            return False, f"Insufficient data: {len(X)} samples (need at least {self.lookback_medium + 100})"

        return True, ""

    def build_model(self, input_shape):
        """Build enhanced LSTM with gradient clipping and regularization"""

        model = Sequential([
            Input(shape=input_shape),

            # First LSTM layer
            Bidirectional(LSTM(128, return_sequences=True)),
            BatchNormalization(),
            Dropout(0.3),

            # Second LSTM layer
            Bidirectional(LSTM(64, return_sequences=True)),
            BatchNormalization(),
            Dropout(0.3),

            # Third LSTM layer
            LSTM(32),
            BatchNormalization(),
            Dropout(0.2),

            # Dense layers
            Dense(32, activation='swish'),
            Dropout(0.2),
            Dense(16, activation='swish'),
            Dense(1, activation='sigmoid')
        ])

        # Compile with gradient clipping
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.0003,
            clipnorm=1.0  # Gradient clipping to prevent exploding gradients
        )

        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )

        self.model = model
        print(f"✓ Model built: {model.count_params():,} parameters")

        return model

    def prepare_sequences(self, X_scaled, y, lookback):
        """Prepare sequences for LSTM with validation"""
        if len(X_scaled) < lookback + 1:
            raise ValueError(f"Not enough data for lookback {lookback}")

        X_seq, y_seq = [], []
        for i in range(lookback, len(X_scaled) - 1):
            X_seq.append(X_scaled[i - lookback:i])
            y_seq.append(y[i])

        return np.array(X_seq), np.array(y_seq)

    def train_and_predict(self, df, use_class_weights=True, verbose=1):
        """
        Train model and make prediction with comprehensive diagnostics

        Args:
            df: DataFrame with features and Target column
            use_class_weights: Whether to use class weights for imbalanced data
            verbose: Verbosity level (0=silent, 1=progress bar, 2=one line per epoch)

        Returns:
            Tuple of (next_probability, accuracy, diagnostics) or None if failed
        """
        start_time = time.time()

        try:
            # Validate input
            if 'Target' not in df.columns:
                raise ValueError("DataFrame must have 'Target' column")

            if len(df) < self.lookback_long + 100:
                print(f"⚠ Insufficient data: {len(df)} samples (need at least {self.lookback_long + 100})")
                return None

            # Prepare data
            y = df['Target'].values
            X = df.drop('Target', axis=1).values

            print(f"Training with {X.shape[1]} features and {len(X)} samples")

            # Split data
            split_idx = int(len(X) * 0.85)
            X_train_raw = X[:split_idx]
            X_test_raw = X[split_idx:]
            y_train_raw = y[:split_idx]
            y_test_raw = y[split_idx:]

            print(f"✓ Train: {len(X_train_raw)} samples | Test: {len(X_test_raw)} samples")

            # Scale data
            self.scaler.fit(X_train_raw)
            X_train_scaled = self.scaler.transform(X_train_raw)
            X_test_scaled = self.scaler.transform(X_test_raw)

            # Create sequences
            X_train, y_train = self.prepare_sequences(
                X_train_scaled, y_train_raw, self.lookback_medium
            )
            X_test, y_test = self.prepare_sequences(
                X_test_scaled, y_test_raw, self.lookback_medium
            )

            # Validate shapes
            is_valid, error_msg = self.validate_data_shape(X_train, y_train)
            if not is_valid:
                raise ValueError(f"Data validation failed: {error_msg}")

            # Check class balance
            class_counts = np.bincount(y_train)
            print(f"✓ Class balance - Low Vol: {class_counts[0]} | High Vol: {class_counts[1]}")

            # Calculate class weights if needed
            class_weight = None
            if use_class_weights and len(class_counts) == 2:
                total = len(y_train)
                class_weight = {
                    0: total / (2 * class_counts[0]),
                    1: total / (2 * class_counts[1])
                }
                print(f"✓ Class weights: {class_weight}")

            # Build model
            self.build_model((X_train.shape[1], X_train.shape[2]))

            # Callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=8,
                    restore_best_weights=True,
                    verbose=verbose
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=4,
                    min_lr=1e-6,
                    verbose=verbose
                )
            ]

            # Train
            print("\nTraining model...")
            self.training_history = self.model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=50,
                batch_size=32,
                verbose=verbose,
                callbacks=callbacks,
                class_weight=class_weight
            )

            # Evaluate
            test_probs = self.model.predict(X_test, verbose=0).flatten()
            test_preds = (test_probs > 0.5).astype(int)

            # Calculate metrics
            acc = accuracy_score(y_test, test_preds) * 100

            # Get final prediction
            X_full_scaled = self.scaler.transform(X)
            last_seq = X_full_scaled[-self.lookback_medium:].reshape(
                1, self.lookback_medium, X_full_scaled.shape[1]
            )
            next_prob = self.model.predict(last_seq, verbose=0)[0][0]

            # Detailed diagnostics
            diagnostics = {
                'class_balance_train': class_counts,
                'class_balance_test': np.bincount(y_test),
                'confusion_matrix': confusion_matrix(y_test, test_preds),
                'test_probs': test_probs,
                'test_preds': test_preds,
                'test_actuals': y_test,
                'classification_report': classification_report(y_test, test_preds, output_dict=True),
                'training_time_seconds': time.time() - start_time,
                'training_history': self.training_history.history
            }

            print(f"\n✓ Training completed in {diagnostics['training_time_seconds']:.2f}s")
            print(f"✓ Test accuracy: {acc:.2f}%")

            return next_prob, acc, diagnostics

        except Exception as e:
            print(f"✗ Training failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return None


class ConfidenceFilter:
    """Analyze predictions by confidence levels"""

    @staticmethod
    def analyze_by_confidence(diagnostics, confidence_threshold=0.6):
        """Analyze performance by confidence levels"""
        probs = diagnostics['test_probs']
        preds = diagnostics['test_preds']
        actuals = diagnostics['test_actuals']

        # Calculate confidence (distance from 0.5)
        confidence = np.abs(probs - 0.5) * 2

        # High confidence predictions
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
                'range': f'{min_conf * 100:.0f}-{max_conf * 100:.0f}%',
                'count': mask.sum(),
                'pct': (mask.sum() / len(probs)) * 100,
                'accuracy': tier_acc
            })

        return results


if __name__ == "__main__":
    # Test with synthetic data
    import pandas as pd

    # Create synthetic data
    n = 500
    dates = pd.date_range('2020-01-01', periods=n, freq='D')

    df = pd.DataFrame({
        'Open': np.random.randn(n) * 10 + 100,
        'High': np.random.randn(n) * 10 + 105,
        'Low': np.random.randn(n) * 10 + 95,
        'Close': np.random.randn(n) * 10 + 100,
        'AdjClose': np.random.randn(n) * 10 + 100,
        'Volume': np.random.randint(1000, 10000, n)
    }, index=dates)

    # Add some features for testing
    df['Feature1'] = df['Close'].pct_change()
    df['Feature2'] = df['Close'].rolling(20).mean()
    df['Target'] = (df['Close'].pct_change().shift(-1).abs() > 0.01).astype(int)
    df = df.dropna()

    print("Testing LSTM Model...")
    lstm = MultiTimeframeLSTM()
    result = lstm.train_and_predict(df, verbose=1)

    if result:
        next_prob, acc, diagnostics = result
        print(f"\n{'=' * 80}")
        print("TEST RESULTS")
        print('=' * 80)
        print(f"Test Accuracy: {acc:.2f}%")
        print(f"Next Prediction Probability: {next_prob * 100:.2f}%")
        print(f"Confidence: {abs(next_prob - 0.5) * 200:.2f}%")
        print('=' * 80)