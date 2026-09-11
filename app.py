"""
AI Typing Biometrics — Keystroke Dynamics for Continuous Authentication
=======================================================================

Identifies users by their typing patterns using keystroke dynamics:
  - Dwell time (key press duration)
  - Flight time (time between key presses)
  - Typing rhythm and cadence
  - Error patterns and backspace behavior

Uses a simulated LSTM + statistical profiling ensemble for biometric
authentication. All neural network components implemented from scratch in NumPy.

Author: Varshini487
"""

import numpy as np
import re
import json
import hashlib
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class Config:
    num_users: int = 20
    sessions_per_user: int = 15
    keystrokes_per_session: int = 150
    common_phrase: str = "the quick brown fox jumps over the lazy dog"
    lstm_hidden: int = 64
    lstm_layers: int = 2
    feature_dim: int = 32
    learning_rate: float = 0.01
    epochs: int = 100
    auth_threshold: float = 0.65
    continuous_window: int = 50  # keystrokes for continuous auth window
    verbose: bool = True

# ============================================================================
# 1. Keystroke Data Simulator
# ============================================================================

class KeystrokeSimulator:
    """Simulates realistic keystroke timing data for multiple users."""

    # Common English bigrams for realistic typing patterns
    COMMON_BIGRAMS = ['th', 'he', 'in', 'er', 'an', 're', 'on', 'at', 'en', 'nd',
                      'ti', 'es', 'or', 'te', 'of', 'ed', 'is', 'it', 'al', 'ar']

    def __init__(self, config: Config):
        self.config = config
        self.rng = np.random.default_rng(42)
        # Each user has unique typing profile parameters
        self.user_profiles = self._generate_user_profiles()

    def _generate_user_profiles(self) -> Dict[int, dict]:
        """Generate unique typing biometric profiles for each user."""
        profiles = {}
        for uid in range(self.config.num_users):
            profiles[uid] = {
                'mean_dwell': float(self.rng.uniform(80, 150)),       # ms
                'std_dwell': float(self.rng.uniform(15, 35)),         # ms
                'mean_flight': float(self.rng.uniform(60, 130)),      # ms
                'std_flight': float(self.rng.uniform(20, 50)),        # ms
                'typing_speed': float(self.rng.uniform(2.5, 5.5)),    # chars/sec
                'rhythm_consistency': float(self.rng.uniform(0.55, 0.92)),  # 0-1
                'error_rate': float(self.rng.uniform(0.01, 0.08)),    # backspace freq
                'digraph_dwell_offset': {},  # per-bigram offsets
                'digraph_flight_offset': {},
                'burst_tendency': float(self.rng.uniform(0.1, 0.6)),  # burst typing
                'pause_frequency': float(self.rng.uniform(0.02, 0.12)), # thinking pauses
                'pause_duration': float(self.rng.uniform(300, 800)),   # ms
            }
            # Per-bigram timing offsets (user-specific digraph patterns)
            for bg in self.COMMON_BIGRAMS:
                profiles[uid]['digraph_dwell_offset'][bg] = float(self.rng.normal(0, 10))
                profiles[uid]['digraph_flight_offset'][bg] = float(self.rng.normal(0, 15))
        return profiles

    def generate_session(self, user_id: int, text: Optional[str] = None) -> List[dict]:
        """Generate a single typing session for a user."""
        profile = self.user_profiles[user_id]
        if text is None:
            text = self.config.common_phrase
        # Repeat text to reach desired keystroke count
        full_text = (text + ' ') * (self.config.keystrokes_per_session // len(text) + 2)
        full_text = full_text[:self.config.keystrokes_per_session]

        keystrokes = []
        timestamp = 0.0
        for i, char in enumerate(full_text):
            # Determine bigram for digraph-specific timing
            prev_char = full_text[i-1] if i > 0 else ''
            bigram = (prev_char + char).lower() if prev_char else ''

            # Dwell time (key press duration)
            dwell_offset = profile['digraph_dwell_offset'].get(bigram, 0)
            dwell = max(30, self.rng.normal(profile['mean_dwell'] + dwell_offset, profile['std_dwell']))

            # Flight time (time from previous key release to current key press)
            flight_offset = profile['digraph_flight_offset'].get(bigram, 0)
            flight = max(10, self.rng.normal(profile['mean_flight'] + flight_offset, profile['std_flight']))

            # Rhythm consistency — reduce variance for consistent typists
            if self.rng.random() < profile['rhythm_consistency']:
                dwell = profile['mean_dwell'] + dwell_offset + self.rng.normal(0, profile['std_dwell'] * 0.5)
                flight = profile['mean_flight'] + flight_offset + self.rng.normal(0, profile['std_flight'] * 0.5)

            # Burst typing — sometimes very fast consecutive keys
            if self.rng.random() < profile['burst_tendency']:
                flight *= 0.4

            # Thinking pauses
            if self.rng.random() < profile['pause_frequency']:
                flight += self.rng.normal(profile['pause_duration'], 100)

            # Errors (backspace)
            is_error = self.rng.random() < profile['error_rate']

            key = char if not is_error else '\b'
            keystrokes.append({
                'user_id': user_id,
                'key': key,
                'key_char': char,
                'press_time': timestamp,
                'release_time': timestamp + dwell,
                'dwell_time': dwell,
                'flight_time': flight if i > 0 else 0,
                'is_error': is_error,
                'position': i,
            })
            timestamp += dwell + flight

        return keystrokes

    def generate_all_sessions(self) -> Tuple[List[dict], List[dict]]:
        """Generate training and testing sessions for all users."""
        train_sessions, test_sessions = [], []
        for uid in range(self.config.num_users):
            for s in range(self.config.sessions_per_user):
                session = self.generate_session(uid)
                if s < int(self.config.sessions_per_user * 0.7):
                    train_sessions.append({'user_id': uid, 'session_id': s, 'keystrokes': session})
                else:
                    test_sessions.append({'user_id': uid, 'session_id': s, 'keystrokes': session})
        return train_sessions, test_sessions

# ============================================================================
# 2. Feature Extraction
# ============================================================================

class FeatureExtractor:
    """Extracts biometric features from keystroke timing data."""

    def __init__(self):
        self.common_keys = list("etaoinsrhldcumfpgwybvkxjqz ")

    def extract_session_features(self, keystrokes: List[dict]) -> np.ndarray:
        """Extract 32-dimensional feature vector from a typing session."""
        valid = [k for k in keystrokes if not k['is_error'] and k['flight_time'] > 0]
        if len(valid) < 5:
            return np.zeros(32)

        dwells = [k['dwell_time'] for k in valid]
        flights = [k['flight_time'] for k in valid]

        features = []

        # 1-5: Dwell statistics
        features.extend([
            np.mean(dwells),
            np.std(dwells),
            np.median(dwells),
            np.min(dwells),
            np.max(dwells),
        ])

        # 6-10: Flight statistics
        features.extend([
            np.mean(flights),
            np.std(flights),
            np.median(flights),
            np.min(flights),
            np.max(flights),
        ])

        # 11-13: Dwell percentiles
        features.extend([
            np.percentile(dwells, 25),
            np.percentile(dwells, 75),
            np.percentile(dwells, 95),
        ])

        # 14-16: Flight percentiles
        features.extend([
            np.percentile(flights, 25),
            np.percentile(flights, 75),
            np.percentile(flights, 95),
        ])

        # 17-18: Coefficient of variation (rhythm consistency proxy)
        features.extend([
            np.std(dwells) / (np.mean(dwells) + 1e-6),
            np.std(flights) / (np.mean(flights) + 1e-6),
        ])

        # 19-20: Typing speed metrics
        total_time = valid[-1]['release_time'] - valid[0]['press_time']
        features.extend([
            len(valid) / (total_time / 1000.0 + 1e-6),  # keys per second
            total_time / len(valid),  # avg time per key
        ])

        # 21-22: Per-key dwell patterns (top common keys)
        key_dwells = defaultdict(list)
        for k in valid:
            key_dwells[k['key_char'].lower()].append(k['dwell_time'])
        for ck in self.common_keys[:2]:
            vals = key_dwells.get(ck, [0])
            features.append(np.mean(vals))

        # 23-26: Digraph (bigram) flight times
        digraph_flights = defaultdict(list)
        for i in range(1, len(valid)):
            bg = (valid[i-1]['key_char'] + valid[i]['key_char']).lower()
            digraph_flights[bg].append(valid[i]['flight_time'])

        # Top 4 common digraphs
        for bg in ['th', 'he', 'in', 'er']:
            vals = digraph_flights.get(bg, [0])
            features.append(np.mean(vals))

        # 27-28: Error metrics
        errors = [k for k in keystrokes if k['is_error']]
        features.extend([
            len(errors) / len(keystrokes),  # error rate
            len(errors),  # total errors
        ])

        # 29-30: Burst patterns (consecutive fast keys)
        fast_keys = sum(1 for f in flights if f < np.percentile(flights, 25))
        features.extend([
            fast_keys / len(flights),  # burst ratio
            np.mean(np.diff(flights)) if len(flights) > 1 else 0,  # flight trend
        ])

        # 31: Dwell-flight correlation
        if len(dwells) == len(flights):
            features.append(np.corrcoef(dwells, flights)[0, 1] if np.std(dwells) > 0 and np.std(flights) > 0 else 0)
        else:
            features.append(0)

        # 32: Rhythm regularity (autocorrelation lag-2 of flight times)
        if len(flights) > 3:
            f_arr = np.array(flights) - np.mean(flights)
            denom = np.sum(f_arr ** 2) + 1e-6
            features.append(np.sum(f_arr[:-2] * f_arr[2:]) / denom)
        else:
            features.append(0)

        return np.array(features, dtype=np.float64)

    def extract_sequence(self, keystrokes: List[dict], max_len: int = 100) -> np.ndarray:
        """Extract a [dwell, flight] time sequence for LSTM processing."""
        valid = [k for k in keystrokes if not k['is_error']]
        seq = []
        for k in valid[:max_len]:
            seq.append([k['dwell_time'], k['flight_time']])
        # Pad to max_len
        while len(seq) < max_len:
            seq.append([0.0, 0.0])
        # Normalize
        arr = np.array(seq)
        arr = (arr - arr.mean(axis=0)) / (arr.std(axis=0) + 1e-6)
        return arr

# ============================================================================
# 3. Simulated LSTM (from scratch in NumPy)
# ============================================================================

class LSTMCell:
    """Single LSTM cell implemented from scratch."""

    def __init__(self, input_dim: int, hidden_dim: int):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        scale = np.sqrt(2.0 / (input_dim + hidden_dim))

        # Combined weights for [input, forget, output, gate] gates
        self.Wf = np.random.randn(hidden_dim, input_dim + hidden_dim) * scale
        self.Wi = np.random.randn(hidden_dim, input_dim + hidden_dim) * scale
        self.Wo = np.random.randn(hidden_dim, input_dim + hidden_dim) * scale
        self.Wg = np.random.randn(hidden_dim, input_dim + hidden_dim) * scale
        self.bf = np.zeros(hidden_dim)
        self.bi = np.zeros(hidden_dim)
        self.bo = np.zeros(hidden_dim)
        self.bg = np.zeros(hidden_dim)

        # Gradients
        self.dWf = np.zeros_like(self.Wf)
        self.dWi = np.zeros_like(self.Wi)
        self.dWo = np.zeros_like(self.Wo)
        self.dWg = np.zeros_like(self.Wg)
        self.dbf = np.zeros_like(self.bf)
        self.dbi = np.zeros_like(self.bi)
        self.dbo = np.zeros_like(self.bo)
        self.dbg = np.zeros_like(self.bg)

    def forward(self, x_t: np.ndarray, h_prev: np.ndarray, c_prev: np.ndarray):
        """Forward pass for one time step."""
        concat = np.concatenate([x_t, h_prev])

        f = self._sigmoid(self.Wf @ concat + self.bf)  # forget gate
        i = self._sigmoid(self.Wi @ concat + self.bi)  # input gate
        o = self._sigmoid(self.Wo @ concat + self.bo)  # output gate
        g = np.tanh(self.Wg @ concat + self.bg)        # candidate

        c_t = f * c_prev + i * g
        h_t = o * np.tanh(c_t)

        cache = (concat, f, i, o, g, c_prev, c_t, h_t)
        return h_t, c_t, cache

    def backward(self, dh_t: np.ndarray, dc_t: np.ndarray, cache: tuple):
        """Backward pass for one time step."""
        concat, f, i, o, g, c_prev, c_t, h_t = cache

        tanh_c = np.tanh(c_t)
        do = dh_t * tanh_c * o * (1 - o)
        dc = dc_t + dh_t * o * (1 - tanh_c ** 2)

        df = dc * c_prev * f * (1 - f)
        di = dc * g * i * (1 - i)
        dg = dc * i * (1 - g ** 2)
        dc_prev = dc * f

        self.dWf += np.outer(df, concat)
        self.dWi += np.outer(di, concat)
        self.dWo += np.outer(do, concat)
        self.dWg += np.outer(dg, concat)
        self.dbf += df
        self.dbi += di
        self.dbo += do
        self.dbg += dg

        d_concat = self.Wf.T @ df + self.Wi.T @ di + self.Wo.T @ do + self.Wg.T @ dg
        dh_prev = d_concat[self.input_dim:]
        dx_t = d_concat[:self.input_dim]

        return dh_prev, dc_prev, dx_t

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def update_params(self, lr: float):
        self.Wf -= lr * self.dWf
        self.Wi -= lr * self.dWi
        self.Wo -= lr * self.dWo
        self.Wg -= lr * self.dWg
        self.bf -= lr * self.dbf
        self.bi -= lr * self.dbi
        self.bo -= lr * self.dbo
        self.bg -= lr * self.dbg
        self.zero_grad()

    def zero_grad(self):
        self.dWf.fill(0); self.dWi.fill(0); self.dWo.fill(0); self.dWg.fill(0)
        self.dbf.fill(0); self.dbi.fill(0); self.dbo.fill(0); self.dbg.fill(0)


class LSTMNetwork:
    """Multi-layer LSTM for sequence classification."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, num_classes: int):
        self.cells = [LSTMCell(input_dim if l == 0 else hidden_dim, hidden_dim) for l in range(num_layers)]
        # Classification head
        self.W_out = np.random.randn(hidden_dim, num_classes) * np.sqrt(2.0 / hidden_dim)
        self.b_out = np.zeros(num_classes)
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_layers = num_layers

    def forward(self, x_seq: np.ndarray):
        """Forward pass over sequence. x_seq: [seq_len, input_dim]"""
        seq_len = x_seq.shape[0]
        caches = [[] for _ in range(self.num_layers)]
        h = [np.zeros(self.hidden_dim) for _ in range(self.num_layers)]
        c = [np.zeros(self.hidden_dim) for _ in range(self.num_layers)]

        for t in range(seq_len):
            x_t = x_seq[t]
            for l in range(self.num_layers):
                inp = x_t if l == 0 else h[l-1]
                h[l], c[l], cache = self.cells[l].forward(inp, h[l], c[l])
                caches[l].append(cache)

        # Use last hidden state of last layer for classification
        final_h = h[-1]
        logits = self.W_out.T @ final_h + self.b_out
        probs = self._softmax(logits)
        return probs, caches, final_h

    def backward(self, caches, final_h, probs, y_true):
        """Backward pass through time (BPTT)."""
        # Gradient from classification loss
        dlogits = probs.copy()
        dlogits[y_true] -= 1
        dlogits /= 1  # batch size = 1

        dW_out = np.outer(final_h, dlogits)
        db_out = dlogits.copy()

        # Initial gradient into last LSTM layer
        dh = self.W_out @ dlogits  # [hidden_dim]
        dc = np.zeros(self.hidden_dim)

        # Backprop through time
        for t in range(len(caches[0]) - 1, -1, -1):
            for l in range(self.num_layers - 1, -1, -1):
                if l == self.num_layers - 1:
                    dh_l, dc_l, _ = self.cells[l].backward(dh, dc, caches[l][t])
                    dh, dc = dh_l, dc_l
                else:
                    dh_l, dc_l, _ = self.cells[l].backward(dh, dc, caches[l][t])
                    dh, dc = dh_l, dc_l

        # Update classification head
        self.W_out -= self.lr * dW_out
        self.b_out -= self.lr * db_out

        # Update LSTM cells
        for cell in self.cells:
            cell.update_params(self.lr)

    def train(self, X_train, y_train, lr=0.01, epochs=50):
        self.lr = lr
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            for i in range(len(X_train)):
                probs, caches, final_h = self.forward(X_train[i])
                # Cross-entropy loss
                loss = -np.log(probs[y_train[i]] + 1e-10)
                total_loss += loss
                if np.argmax(probs) == y_train[i]:
                    correct += 1
                self.backward(caches, final_h, probs, y_train[i])
            if (epoch + 1) % 10 == 0:
                acc = correct / len(X_train)
                print(f"  LSTM Epoch {epoch+1}/{epochs} — Loss: {total_loss/len(X_train):.4f} — Acc: {acc:.2%}")

    def predict(self, x_seq):
        probs, _, _ = self.forward(x_seq)
        return probs

    def _softmax(self, x):
        e = np.exp(x - np.max(x))
        return e / e.sum()

# ============================================================================
# 4. Statistical Profiler (Template Matching)
# ============================================================================

class StatisticalProfiler:
    """Statistical template-matching classifier for keystroke biometrics."""

    def __init__(self):
        self.templates = {}  # user_id -> mean feature vector
        self.template_stds = {}  # user_id -> std feature vector

    def train(self, features: np.ndarray, labels: np.ndarray):
        for uid in np.unique(labels):
            mask = labels == uid
            user_features = features[mask]
            self.templates[uid] = np.mean(user_features, axis=0)
            self.template_stds[uid] = np.std(user_features, axis=0) + 1e-6

    def predict_proba(self, feature: np.ndarray) -> np.ndarray:
        """Return probability distribution over users."""
        scores = []
        for uid in self.templates:
            # Mahalanobis-like distance
            diff = feature - self.templates[uid]
            dist = np.sqrt(np.sum((diff / self.template_stds[uid]) ** 2))
            scores.append((uid, dist))
        # Convert distances to probabilities (closer = higher prob)
        distances = np.array([s[1] for s in scores])
        probs = np.exp(-distances / (distances.min() + 1e-6))
        probs /= probs.sum()
        user_ids = [s[0] for s in scores]
        result = np.zeros(max(user_ids) + 1)
        for uid, p in zip(user_ids, probs):
            result[uid] = p
        return result


class TypingBiometricsSystem:
    """Main system: ensemble of LSTM + statistical profiler."""

    def __init__(self, config: Config):
        self.config = config
        self.simulator = KeystrokeSimulator(config)
        self.extractor = FeatureExtractor()
        self.statistical_model = StatisticalProfiler()
        self.lstm = None
        self.is_trained = False

    def _prepare_data(self, sessions):
        """Extract features and sequences from sessions."""
        features, sequences, labels = [], [], []
        for session in sessions:
            feat = self.extractor.extract_session_features(session['keystrokes'])
            seq = self.extractor.extract_sequence(session['keystrokes'])
            features.append(feat)
            sequences.append(seq)
            labels.append(session['user_id'])
        return np.array(features), np.array(sequences), np.array(labels)

    def train(self):
        print("=" * 70)
        print("AI Typing Biometrics — Training")
        print("=" * 70)

        # Generate data
        print("\n[1] Generating keystroke data for", self.config.num_users, "users...")
        train_sessions, test_sessions = self.simulator.generate_all_sessions()
        print(f"    Training sessions: {len(train_sessions)}  Test sessions: {len(test_sessions)}")

        # Extract features
        print("\n[2] Extracting biometric features (32-dim per session)...")
        X_train_feat, X_train_seq, y_train = self._prepare_data(train_sessions)
        X_test_feat, X_test_seq, y_test = self._prepare_data(test_sessions)
        print(f"    Feature matrix: {X_train_feat.shape}  Sequence matrix: {X_train_seq.shape}")

        # Normalize features
        feat_mean = X_train_feat.mean(axis=0)
        feat_std = X_train_feat.std(axis=0) + 1e-6
        X_train_feat = (X_train_feat - feat_mean) / feat_std
        X_test_feat = (X_test_feat - feat_std) / feat_std
        self.feat_mean = feat_mean
        self.feat_std = feat_std

        # Train statistical model
        print("\n[3] Training Statistical Profiler (template matching)...")
        self.statistical_model.train(X_train_feat, y_train)

        # Train LSTM
        print(f"\n[4] Training LSTM ({self.config.lstm_layers} layers, {self.config.lstm_hidden} hidden)...")
        self.lstm = LSTMNetwork(
            input_dim=2, hidden_dim=self.config.lstm_hidden,
            num_layers=self.config.lstm_layers, num_classes=self.config.num_users
        )
        self.lstm.train(X_train_seq, y_train, lr=self.config.learning_rate, epochs=self.config.epochs)

        # Evaluate
        print("\n[5] Evaluating ensemble (60% LSTM + 40% Statistical)...")
        predictions, confidences = [], []
        for i in range(len(X_test_feat)):
            lstm_probs = self.lstm.predict(X_test_seq[i])
            stat_probs = self.statistical_model.predict_proba(X_test_feat[i])
            # Normalize stat_probs to match lstm_probs size
            stat_padded = np.zeros_like(lstm_probs)
            stat_padded[:len(stat_probs)] = stat_probs
            ensemble = 0.60 * lstm_probs + 0.40 * stat_padded
            pred = np.argmax(ensemble)
            conf = ensemble[pred]
            predictions.append(pred)
            confidences.append(conf)

        y_test = y_test[:len(predictions)]
        accuracy = np.mean(np.array(predictions) == y_test)
        avg_conf = np.mean(confidences)

        print(f"\n{'='*70}")
        print(f"RESULTS")
        print(f"{'='*70}")
        print(f"  Ensemble Accuracy:     {accuracy:.2%}")
        print(f"  Avg Confidence:        {avg_conf:.2%}")
        print(f"  Authentication threshold: {self.config.auth_threshold:.2f}")

        # False Accept / False Reject analysis
        far, frr = self._compute_far_frr(X_test_feat, X_test_seq, y_test)
        print(f"  False Accept Rate:     {far:.2%}")
        print(f"  False Reject Rate:     {frr:.2%}")
        print(f"  EER (Equal Error Rate): ~{(far + frr) / 2:.2%}")

        self.is_trained = True
        return accuracy, far, frr

    def _compute_far_frr(self, X_feat, X_seq, y_true):
        """Compute False Accept Rate and False Reject Rate."""
        genuine_accepts = 0
        genuine_rejects = 0
        impostor_accepts = 0
        impostor_rejects = 0

        for i in range(len(X_feat)):
            lstm_probs = self.lstm.predict(X_seq[i])
            stat_probs = self.statistical_model.predict_proba(X_feat[i])
            stat_padded = np.zeros_like(lstm_probs)
            stat_padded[:len(stat_probs)] = stat_probs
            ensemble = 0.60 * lstm_probs + 0.40 * stat_padded

            claimed_user = y_true[i]
            score = ensemble[claimed_user]

            # Genuine attempt
            if score >= self.config.auth_threshold:
                genuine_accepts += 1
            else:
                genuine_rejects += 1

            # Impostor attempt (use another user's data)
            impostor_id = (y_true[i] + 1) % self.config.num_users
            impostor_score = ensemble[impostor_id] if impostor_id < len(ensemble) else 0
            if impostor_score >= self.config.auth_threshold:
                impostor_accepts += 1
            else:
                impostor_rejects += 1

        far = impostor_accepts / (impostor_accepts + impostor_rejects + 1e-6)
        frr = genuine_rejects / (genuine_accepts + genuine_rejects + 1e-6)
        return far, frr

    def continuous_auth_demo(self, user_id: int = 0):
        """Demonstrate continuous authentication over a typing session."""
        if not self.is_trained:
            print("System not trained. Run train() first.")
            return

        print(f"\n{'='*70}")
        print(f"CONTINUOUS AUTHENTICATION DEMO — User {user_id}")
        print(f"{'='*70}")

        session = self.simulator.generate_session(user_id)
        window_size = self.config.continuous_window

        print(f"Monitoring {len(session)} keystrokes in windows of {window_size}...\n")
        print(f"{'Window':<10} {'Keystroke Range':<20} {'Top Match':<12} {'Confidence':<12} {'Status'}")
        print("-" * 70)

        for w in range(0, len(session) - window_size, window_size // 2):
            window = session[w:w+window_size]
            feat = self.extractor.extract_session_features(window)
            feat = (feat - self.feat_mean) / self.feat_std
            seq = self.extractor.extract_sequence(window)

            lstm_probs = self.lstm.predict(seq)
            stat_probs = self.statistical_model.predict_proba(feat)
            stat_padded = np.zeros_like(lstm_probs)
            stat_padded[:len(stat_probs)] = stat_probs
            ensemble = 0.60 * lstm_probs + 0.40 * stat_padded

            top_user = np.argmax(ensemble)
            top_conf = ensemble[top_user]
            status = "✓ AUTHENTIC" if (top_user == user_id and top_conf >= self.config.auth_threshold) else "✗ ALERT"

            print(f"{w// (window_size//2):<10} {w}-{w+window_size:<20} User {top_user:<12} {top_conf:<12.2%} {status}")

        print("\nContinuous authentication monitors typing in sliding windows,")
        print("detecting if the current typist matches the authorized user.")

    def feature_importance_report(self):
        """Report on which features are most discriminative."""
        print(f"\n{'='*70}")
        print("FEATURE IMPORTANCE ANALYSIS")
        print(f"{'='*70}")

        feature_names = [
            "Dwell Mean", "Dwell Std", "Dwell Median", "Dwell Min", "Dwell Max",
            "Flight Mean", "Flight Std", "Flight Median", "Flight Min", "Flight Max",
            "Dwell 25th%", "Dwell 75th%", "Dwell 95th%",
            "Flight 25th%", "Flight 75th%", "Flight 95th%",
            "Dwell CV", "Flight CV",
            "Typing Speed (keys/s)", "Time per Key",
            "Key 'e' Dwell", "Key 't' Dwell",
            "Digraph 'th' Flight", "Digraph 'he' Flight", "Digraph 'in' Flight", "Digraph 'er' Flight",
            "Error Rate", "Error Count",
            "Burst Ratio", "Flight Trend",
            "Dwell-Flight Corr", "Rhythm Autocorr (lag-2)",
        ]

        # Compute inter-user variance as importance proxy
        print("\nTop 10 Most Discriminative Features (by inter-user variance):")
        print("-" * 50)

        # Use templates to compute importance
        if self.statistical_model.templates:
            template_matrix = np.array([self.statistical_model.templates[uid] for uid in self.statistical_model.templates])
            importance = np.std(template_matrix, axis=0)
            top_idx = np.argsort(importance)[::-1][:10]
            for rank, idx in enumerate(top_idx, 1):
                print(f"  {rank:>2}. {feature_names[idx]:<30} Importance: {importance[idx]:.4f}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    config = Config()
    system = TypingBiometricsSystem(config)

    # Train and evaluate
    accuracy, far, frr = system.train()

    # Continuous authentication demonstration
    system.continuous_auth_demo(user_id=0)

    # Feature importance
    system.feature_importance_report()

    print(f"\n{'='*70}")
    print("AI Typing Biometrics — Complete")
    print(f"{'='*70}")
    print("All neural network components (LSTM with forget/input/output/candidate gates,")
    print("full BPTT backprop, softmax classifier) implemented from scratch in NumPy.")
    print("Statistical template matching uses Mahalanobis-like distance.")
    print("Ensemble: 60% LSTM + 40% Statistical")
    print(f"\nFinal — Accuracy: {accuracy:.2%}  FAR: {far:.2%}  FRR: {frr:.2%}")
