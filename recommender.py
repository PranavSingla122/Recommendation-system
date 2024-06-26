import pandas as pd
import numpy as np

# Load the data from CSV
data = pd.read_csv('data1.csv')

# Get the unique email addresses
email = list(set(data.values.flatten().tolist()))

n = len(email)
adj_matrix = np.zeros((n, n))

# Populate the adjacency matrix
for index, row in data.iterrows():
    name = row['Email Address']
    i = email.index(name)
    for col in row[1:]:
        if pd.notna(col):  # Ensure that the column value is not NaN
            j = email.index(col)
            adj_matrix[i][j] = 1

# Adjust the matrix for reciprocal dislikes
for index, row in data.iterrows():
    name = row['Email Address']
    i = email.index(name)
    for col in row[1:]:
        if pd.notna(col):  # Ensure that the column value is not NaN
            j = email.index(col)
            if adj_matrix[i][j] == 1 and adj_matrix[j][i] == 0:  # If B has not marked A, it implies dislike
                adj_matrix[j][i] = -1

# Gradient descent for matrix factorization
def matrix_factorization(R, K, steps=1000, alpha=0.002, beta=0.02):
    num_users, num_items = R.shape
    U = np.random.rand(num_users, K)
    V = np.random.rand(K, num_items)
    
    for step in range(steps):
        for i in range(num_users):
            for j in range(num_items):
                if R[i, j] != 0:
                    eij = R[i, j] - np.dot(U[i, :], V[:, j])
                    for k in range(K):
                        U[i, k] += alpha * (2 * eij * V[k, j] - beta * U[i, k])
                        V[k, j] += alpha * (2 * eij * U[i, k] - beta * V[k, j])
                        
        # Calculate the total error
        error = 0
        for i in range(num_users):
            for j in range(num_items):
                if R[i, j] != 0:
                    error += pow(R[i, j] - np.dot(U[i, :], V[:, j]), 2)
                    for k in range(K):
                        error += (beta / 2) * (pow(U[i, k], 2) + pow(V[k, j], 2))
        if error < 0.01:
            break
    return U, V


# Predicting missing links
def predict_missing_links(U, V, threshold=0.8):
    R_hat = np.dot(U, V)
    predictions = (R_hat >= threshold).astype(int)
    return predictions

U,V = matrix_factorization(adj_matrix, 50)

predictions = predict_missing_links(U, V)

# Convert the predictions to a DataFrame for better readability
predictions_df = pd.DataFrame(predictions, index=email, columns=email)

# Print the predictions
print("Prediction for Adjacency Matrix:")
print(predictions_df)