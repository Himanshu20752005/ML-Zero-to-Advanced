import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# ---------------------------------------------------
# Parameters
# ---------------------------------------------------

np.random.seed(42)

B = 100          # Number of training sets
N = 30           # Training points in each set
degrees = [1, 4, 15]

# Fixed evaluation grid
X_grid = np.linspace(0, 1, 100).reshape(-1, 1)

# True function
y_true = np.sin(2 * np.pi * X_grid).ravel()

# Store final results
bias_squared_results = []
variance_results = []

# ---------------------------------------------------
# (a) Fit 100 models for each degree
# ---------------------------------------------------

for degree in degrees:

    # Store predictions from all 100 models
    predictions = np.zeros((B, len(X_grid)))

    poly = PolynomialFeatures(degree=degree)

    for b in range(B):

        # Generate a fresh training set
        X_train = np.random.uniform(0, 1, N).reshape(-1, 1)

        noise = np.random.normal(0, 0.2, N)

        y_train = np.sin(2 * np.pi * X_train).ravel() + noise

        # Polynomial transformation
        X_train_poly = poly.fit_transform(X_train)
        X_grid_poly = poly.transform(X_grid)

        # Fit model
        model = LinearRegression()
        model.fit(X_train_poly, y_train)

        # Predict on fixed evaluation grid
        predictions[b] = model.predict(X_grid_poly)

    # ---------------------------------------------------
    # (b) Bias² and Variance
    # ---------------------------------------------------

    # Average prediction at each grid point
    mean_prediction = np.mean(predictions, axis=0)

    # Squared bias at each grid point
    squared_bias = (mean_prediction - y_true) ** 2

    # Variance at each grid point
    variance = np.var(predictions, axis=0)

    # Average over all grid points
    bias_squared = np.mean(squared_bias)
    variance_value = np.mean(variance)

    bias_squared_results.append(bias_squared)
    variance_results.append(variance_value)


# ---------------------------------------------------
# Display results
# ---------------------------------------------------

print("Degree\tBias²\t\tVariance\tBias² + Variance")

for i, degree in enumerate(degrees):

    total = bias_squared_results[i] + variance_results[i]

    print(
        f"{degree}\t"
        f"{bias_squared_results[i]:.4f}\t\t"
        f"{variance_results[i]:.4f}\t\t"
        f"{total:.4f}"
    )


# ---------------------------------------------------
# (c) Plot Bias², Variance and their Sum
# ---------------------------------------------------

total_results = np.array(bias_squared_results) + np.array(variance_results)

plt.figure(figsize=(8, 5))

plt.plot(degrees, bias_squared_results, marker='o', label='Bias²')
plt.plot(degrees, variance_results, marker='o', label='Variance')
plt.plot(degrees, total_results, marker='o', label='Bias² + Variance')

plt.xlabel("Polynomial Degree")
plt.ylabel("Error")
plt.title("Bias²-Variance Tradeoff")
plt.xticks(degrees)
plt.legend()
plt.grid(True)

plt.show()
