import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Define parameters
fs = 1e6
T = 1
t = np.arange(0, T, 1/fs)

# Drone and Bird Doppler shifts
propeller_speed = 3000
propeller_freq = propeller_speed / 60
wing_flap_rate = 5

doppler_shift_drone = np.cos(2 * np.pi * propeller_freq * t)
doppler_shift_bird = np.cos(2 * np.pi * wing_flap_rate * t)

# Perform time-frequency analysis
f_drone, t_drone, Sxx_drone = spectrogram(doppler_shift_drone, fs, nperseg=256, noverlap=250, nfft=256)
f_bird, t_bird, Sxx_bird = spectrogram(doppler_shift_bird, fs, nperseg=256, noverlap=250, nfft=256)

# Extract features and labels
mean_freq_drone = np.mean(np.abs(Sxx_drone), axis=1)
mean_freq_bird = np.mean(np.abs(Sxx_bird), axis=1)
max_length = max(len(mean_freq_drone), len(mean_freq_bird))
mean_freq_drone = np.pad(mean_freq_drone, (0, max_length - len(mean_freq_drone)), 'constant')
mean_freq_bird = np.pad(mean_freq_bird, (0, max_length - len(mean_freq_bird)), 'constant')
features = np.vstack([mean_freq_drone, mean_freq_bird]).T
labels = np.hstack([np.ones(len(mean_freq_drone)), np.zeros(len(mean_freq_bird))])

# Train SVM classifier
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
model = SVC(kernel='linear')
model.fit(X_train, y_train)
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f'Model Accuracy: {accuracy * 100:.2f}%')
print(classification_report(y_test, predictions, target_names=['Bird', 'Drone']))
