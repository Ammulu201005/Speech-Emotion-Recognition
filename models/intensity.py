"""
Intensity estimation via Temperature Scaling (calibration).
Since no intensity labels exist, we use the model's softmax confidence
and calibrate it with temperature scaling on the validation set.
Output: "Estimated Intensity: 78%" for the predicted emotion.
"""
import numpy as np
from scipy.optimize import minimize_scalar
import joblib
import os


def softmax_with_temperature(logits, temperature=1.0):
    """Apply temperature scaling to logits."""
    scaled = logits / temperature
    exp_scaled = np.exp(scaled - np.max(scaled))
    return exp_scaled / exp_scaled.sum()


class TemperatureScaler:
    def __init__(self):
        self.temperature = 1.0
    
    def fit(self, logits_val, y_val):
        """Find optimal temperature T that minimizes NLL on validation set."""
        def nll(T):
            probs = np.array([softmax_with_temperature(l, T) for l in logits_val])
            probs = np.clip(probs, 1e-9, 1.0)
            n = len(y_val)
            return -np.sum(np.log(probs[np.arange(n), y_val])) / n
        
        result = minimize_scalar(nll, bounds=(0.1, 10.0), method='bounded')
        self.temperature = result.x
        print(f"Optimal temperature: {self.temperature:.4f}")
        return self
    
    def calibrate(self, logits):
        """Return calibrated probabilities."""
        return softmax_with_temperature(logits, self.temperature)
    
    def get_intensity(self, logits, predicted_class):
        """
        Return intensity (0-100%) for predicted class using calibrated confidence.
        This is ESTIMATED intensity — clearly labeled as such in the UI.
        """
        probs = self.calibrate(logits)
        raw_confidence = float(probs[predicted_class])
        
        # Map to intensity: scale so median confidence (~0.6) maps to ~60%
        intensity = min(100, int(raw_confidence * 100))
        return intensity, raw_confidence
    
    def save(self, path='models/saved/temperature_scaler.pkl'):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.temperature, path)
    
    def load(self, path='models/saved/temperature_scaler.pkl'):
        self.temperature = joblib.load(path)
        return self


def get_intensity_label(intensity_pct):
    """Convert numeric intensity to descriptive label."""
    if intensity_pct >= 80:
        return "Very High"
    elif intensity_pct >= 60:
        return "High"
    elif intensity_pct >= 40:
        return "Moderate"
    elif intensity_pct >= 20:
        return "Low"
    else:
        return "Very Low"
