import joblib
import numpy as np
import string
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataset import combine_valid_invald, generate_random_ids, encode_id

CHARS = string.ascii_lowercase + string.digits
CHAR2IDX = {c: i for i, c in enumerate(CHARS)}
all_prediction_ids = generate_random_ids(1_000_000)


# Path to your saved logistic regression model
model_path = 'data/id_classifier_model.joblib'

# Load the model
logistic_regression_model = joblib.load(model_path)

# Example: print model parameters
print(logistic_regression_model)

# Prepare features for all ids using one-hot encoding
X = np.array([encode_id(id_str) for id_str in tqdm(all_prediction_ids, desc="Encoding IDs")])

# Prepare features for all ids
# X = []
# # for id_str in tqdm(all_ids, desc="Preparing features"):
# #     # Convert id string to indices
# #     X.append([char_to_idx[c] for c in id_str])
# X = np.array(all_ids)

# Predict probabilities
probs = logistic_regression_model.predict_proba(X)[:, 1]

# Plot
# print(all_ids)




X_, y, ground_truth_ids, y_ordered = combine_valid_invald()
X_ordered_encoded = np.array([encode_id(id_str) for id_str in ground_truth_ids])


# test_1 = encode_id("4qdqn")
# test_2 = encode_id("q4qnd")

# print(test_1)
# print(test_2)
plt.figure(figsize=(16, 6))

num_ticks = 10
plt.subplot(2, 1, 1)
plt.plot(ground_truth_ids, y_ordered, '.', color='red', markersize=1)
plt.title('Ground Truth Probability for Valid IDs')
plt.xlabel('ID Index')
plt.ylabel('Probability')
tick_indices_gt = np.linspace(0, len(ground_truth_ids) - 1, num_ticks, dtype=int)
plt.xticks(tick_indices_gt, [ground_truth_ids[i] for i in tick_indices_gt], rotation=45)



plt.subplot(2, 1, 2)
plt.plot(all_prediction_ids, probs, '.', markersize=1, label='Predicted Probability')

# Compute moving average
window_size = 1000
moving_avg = np.convolve(probs, np.ones(window_size)/window_size, mode='valid')

# For x-axis, align moving average with the center of each window
ma_x = all_prediction_ids[window_size//2 : len(all_prediction_ids) - window_size//2 + 1]

plt.plot(ma_x, moving_avg, color='blue', linewidth=2, label=f'Moving Avg (window={window_size})')

plt.title('Predicted Probability for All IDs')
plt.ylabel('Probability')
tick_indices = np.linspace(0, len(all_prediction_ids) - 1, num_ticks, dtype=int)
plt.xticks(tick_indices, [all_prediction_ids[i] for i in tick_indices], rotation=45)
plt.legend()

# print(X_ordered)
# plt.legend()



plt.title('Probability of Valid ID for Each Possible ID')
plt.xlabel('ID Index')
plt.ylabel('Probability of Being Valid')
# Show only 10 evenly spaced IDs on the x-axis
num_ticks = 10
# tick_indices = np.linspace(0, len(all_prediction_ids) - 1, num_ticks, dtype=int)
# plt.xticks(tick_indices, [all_prediction_ids[i] for i in tick_indices], rotation=45)
plt.xticks([])
plt.show()
