# House Price Predictor

## Project Overview
This is a simple machine learning project using linear regression to predict house prices based on features from the Boston Housing dataset. It demonstrates basic ML concepts like data preprocessing, model training, evaluation, and visualization. Focused on supervised learning, multiple linear regression, feature scaling, and cost functions. The project uses Python with libraries like Scikit-Learn, NumPy, Pandas, and Matplotlib.

Note: The Boston Housing dataset has known ethical issues (e.g., potential racial bias in features like neighborhood demographics). This project uses it for educational purposes only—real-world applications should use unbiased datasets.

## Dataset
- **Source**: Boston Housing dataset from Scikit-Learn (506 samples, 13 features like number of rooms (RM), lower status population (LSTAT), etc., target: median house price in $1000s).
- **Features Used**: For simplicity, RM (average rooms per dwelling) and LSTAT (% lower status population).
- **Why This Dataset?**: Classic for regression practice; small size for quick training.

## Installation and Setup
1. **Environment**: Use Google Colab (browser-based, no install needed) or local Python (via Anaconda).
2. **Libraries**: Run in a notebook:
   ```python
   !pip install scikit-learn matplotlib pandas numpy
3. **Run the Notebook**: Open house_price_predictor.ipynb in Colab or Jupyter.

## Usage
1. Load and preprocess data (scaling, train/test split).
2. Train linear regression model.
3. Evaluate with MSE/R2.
4. Visualize predictions.

## Results
- Model Accuracy: R2 score ~0.6-0.7 (varies with features; full dataset better).
- Example Prediction: For a house with 6 rooms and 10% LSTAT, predicts ~$25k (median value).
- Visualization: Scatter plot shows fit (actual vs. predicted prices).

## What I Practiced
- **Linear Regression**: Fit model with multiple features (y^=w1x1+w2x2+b\hat{y} = w_1 x_1 + w_2 x_2 + b\hat{y} = w_1 x_1 + w_2 x_2 + b
).
- **Feature Scalin**g: Use StandardScaler to normalize (mean 0, std 1) for efficient GD.
- **Evaluation**: MSE measures error; R2 shows fit quality (1 = perfect).
- **Ethics**: Datasets like Boston have bias—always check for fairness.
- **Python Tools**: Scikit-Learn for models, Pandas for data handling.

## Improvements and Future Work
- Add more features (e.g., all 13 from Boston) for better accuracy.
- Handle overfitting with regularization (e.g., Ridge from Scikit-Learn).
- Use modern datasets (e.g., California Housing on Kaggle) to avoid bias.
- Integrate React frontend: Input features via form, predict via API (Flask backend).
- Extend to polynomial regression for non-linear patterns.

## References
- **Dataset**: Scikit-Learn Boston Housing (via fetch_openml).
- **Libraries**: Scikit-Learn, NumPy, Pandas, Matplotlib.
