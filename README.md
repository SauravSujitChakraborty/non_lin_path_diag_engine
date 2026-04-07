# non_lin_path_diag_engine

NOTE : This project was made by me during Aug'25, preserved and finally published on Apr7,'26

STOCHASTIC SIMULATION WITH DETERMINISTIC LOGIC

             CANCER DETECTION

THEORY :-

A. Feature Scaling (Standardization)

​==> SVMs(Support Vector Machines) are distance based learners. They calculate the Euclidean distance between data points to find a boundary.

==> ​The Problem: $Perimeter$ values are ~100, while Smoothness values are ~0.1. Without scaling, the model would think $Perimeter$ is much more important.

==> ​The Solution: We use StandardScaler to transform all features so they have a Mean $(\mu)$ of 0 and a Standard Deviation $(\sigma)$ of 1

• $$ z = \frac{x - \mu}{\sigma} $$

==> $x$ (The Raw Input): This is the actual measurement from the biopsy. For example, if a cell has a Radius of 18.5, then    $$ x = 18.5 $$ Raw x values are difficult for models because they have different units (e.g., Radius is 10–20, but Perimeter is 80–150).

==> $z$ (The Z-Score / Standardized Value): This is the output after scaling. It tells the model how many standard deviations the cell is away from the average.

• If $$ z = 2.0 $$, that specific cell's radius is much larger than normal (2 standard deviations above mean).

• If $$ z = -1.0 $$, the radius is smaller than average.

B. SVM & Maximum Margin

==> ​Unlike other models that just split the data, an SVM seeks the Optimal Hyperplane that maximizes the Margin (the distance to the nearest data points, called Support Vectors).

​==> This makes the model more robust to noise because it creates a wide buffer zone between classes.

​C. The RBF Kernel & The Kernel Trick

==> ​Biological data is rarely Linearly Separable (that is, you can't draw a single straight line to separate sick from healthy). We use the Radial Basis Function (RBF) Kernel.

==> ​Theory: 

• The RBF kernel mathematically projects the 4D cellular data into an Infinite-Dimensional Space.

• ​In this higher dimension, the data that looked tangled in 2D becomes easily separable by a flat plane. This is known as the Kernel Trick.

​D. Probabilistic Inference (Platt Scaling)

==> ​Standard SVMs are Hard Classifiers which only tell us in 0 or 1. In medicine, doctors need a Risk Score.
​We enabled $$ probability=True $$, which uses Platt Scaling.

==> ​This fits a Logistic Regression model on top of the SVM's distance scores to produce a Calibrated Probability (e.g., "96.0% Malignancy").

​4. Data Simulation: Stochastic vs. Deterministic

==> ​The project uses Stochastic Simulation to generate the dataset:

==> ​Stochastic Features: Measurements are drawn from Gaussian (Normal) Distributions to mimic natural biological variance.

==> ​Deterministic Labels: The "Ground Truth" (Cancer vs. Healthy) is assigned based on a hidden logic gate: $(Radius > 16) | (Texture > 22)$.

==> ​Reproducibility: We use $np.random.seed(42)$ to ensure the "random" data is identical every time the code runs, allowing for reliable Backtesting
