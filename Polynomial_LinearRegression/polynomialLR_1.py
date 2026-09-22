import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# ---------------------------------------------------
# (a) Generate Training and Test Data
# ---------------------------------------------------

np.random.seed(42)

# Training data: 25 points
X_train = np.random.uniform(0, 1, 25)
y_train = np.sin(2 * np.pi * X_train) + np.random.normal(0, 0.2, 25)

# Test data: 300 points
X_test = np.random.uniform(0, 1, 300)
y_test = np.sin(2 * np.pi * X_test) + np.random.normal(0, 0.2, 300)

# Sort x values for smooth plotting
X_plot = np.linspace(0, 1, 500)
y_true = np.sin(2 * np.pi * X_plot)

degrees = [1, 4, 15]

results = []

# ---------------------------------------------------
# Fit models and plot
# ---------------------------------------------------

plt.figure(figsize=(15, 4))

for i, degree in enumerate(degrees):

    # Create polynomial features
    poly = PolynomialFeatures(degree=degree)

    X_train_poly = poly.fit_transform(X_train.reshape(-1, 1))
    X_test_poly = poly.transform(X_test.reshape(-1, 1))
    X_plot_poly = poly.transform(X_plot.reshape(-1, 1))

    # Unregularized linear regression
    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    # Predictions
    y_train_pred = model.predict(X_train_poly)
    y_test_pred = model.predict(X_test_poly)
    y_plot_pred = model.predict(X_plot_poly)

    # MSE
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)

    results.append([degree, train_mse, test_mse])

    # Plot
    plt.subplot(1, 3, i + 1)

    plt.scatter(X_train, y_train, label="Training data")
    plt.plot(X_plot, y_true, label="True function")
    plt.plot(X_plot, y_plot_pred, label=f"Degree {degree} fit")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Polynomial Degree {degree}")
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()

# ---------------------------------------------------
# (b) Display MSE results
# ---------------------------------------------------

print("Degree\tTraining MSE\tTest MSE")

for degree, train_mse, test_mse in results:
    print(f"{degree}\t{train_mse:.4f}\t\t{test_mse:.4f}")



# Degree	 Behavior	       Reason
# ===============================================================================================================================================
# 1	         Underfitting	   A straight line is too simple to capture the curved sine relationship. Both training and test errors are relatively high.
# 4	         Good fit	       Degree 4 is flexible enough to capture the main sinusoidal pattern without excessively fitting the random noise.
# 15	     Overfitting	   The high-degree polynomial has enough flexibility to closely fit the 25 training points, including noise. Training MSE becomes 
#                               very small, while test MSE generally increases.
