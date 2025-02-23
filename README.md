# Linear Regression and Stochastic Gradient Descent

This repository contains implementations of linear regression using both the **normal equation** and **stochastic gradient descent (SGD)**, as well as an extended **polynomial regression** with L2 regularization.

## 📌 Project Overview

The goal of this project is to explore different optimization techniques for linear regression and extend it to polynomial regression using **SGD with L2 regularization (Ridge Regression)**.

### 📂 Files Included

- **`Linear_Regression.py`** - Implements linear regression using the **ormal equation**.
- **`SGD.py`** - Implements linear regression using **stochastic gradient descent (SGD)**.
- **`poly_SGD.py`** - Implements **polynomial regression** (up to degree 4) with **L2 regularization** using SGD.

## 📊 Dataset

The scripts requires the presence of a dataset file named **`hw2data1.txt`** with the following structure:

Each row corresponds to:

- **X**: Population of a city
- **Y**: Profit of a food truck in that city

Ensure this dataset file is in the same directory before running the scripts.

---

## 🔧 Installation & Dependencies

The scripts require Python and the following libraries:

- `numpy`
- `matplotlib`

You can install dependencies using:

```bash
pip install numpy matplotlib
```

## 📝 Results

* The **normal equation** provides an exact solution but may be inefficient for large datasets.
* **SGD** is more scalable but requires careful tuning of hyperparameters.
* **Polynomial regression** captures more complex relationships but requires regularization to prevent overfitting.

## 🔗 License

This project is open-source under the  **MIT License** .
