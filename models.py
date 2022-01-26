
import pandas as pd
from sklearn.model_selection import train_test_split
import prepare_HW3
my_id = 3
or_id = 0
from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np
import matplotlib.pyplot as plt

from LinearRegressor import LinearRegressor





def numerical_subgradient(w, b, X, y, delta=1e-4):
    w_ = w.copy()
    g_w = np.zeros_like(w_)
    orig_objective = LinearRegressor.loss(w_, b, X, y)
    for i in range(g_w.shape[0]):
        w_[i] += delta
        perturbed_objective = LinearRegressor.loss(w_, b, X, y)
        w_[i] -= delta
        g_w[i] = (perturbed_objective - orig_objective) / delta

    g_b = (LinearRegressor.loss(w_, b + delta, X, y) - orig_objective) / delta
    return g_w, g_b


def compare_gradients(X, y, deltas, C=1, REPEATS=10, figsize=(10, 6)):
    residual_means_w = []
    residual_means_b = []

    for delta in deltas:
        residuals_w = []
        residuals_b = []

        for _ in range(REPEATS):
            # Randomize vectors in which the gradient is computed
            w = np.random.randn(X.shape[1])
            b = np.random.randn(1)

            # Compute the two types of gradients
            analytic_grad_w, analytic_grad_b = LinearRegressor.gradient(w, b, X, y)

            numeric_grad_w, numeric_grad_b = numerical_subgradient(w, b, X, y, delta=delta)

            residual_w = np.linalg.norm(numeric_grad_w - analytic_grad_w)
            residuals_w.append(residual_w)

            residual_b = np.linalg.norm(numeric_grad_b - analytic_grad_b)
            residuals_b.append(residual_b)

        residual_means_w.append(np.mean(residuals_w))
        residual_means_b.append(np.mean(residuals_b))

    fig, axs = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(16, 8))
    plt.suptitle('Residuals of analytical and numerical gradients',
                 fontsize=22, fontweight="bold")
    axs[0].set_title(r'Residuals of $\nabla_{w}L\left(w,b\right)$')
    axs[1].set_title(r'Residuals of $\frac{\partial}{\partial{b}}L\left(w,b\right)$')
    axs[0].plot(deltas, residual_means_w, linewidth=3)
    axs[1].plot(deltas, residual_means_b, linewidth=3)
    axs[0].set_yscale('log')
    axs[0].set_xscale('log')
    axs[1].set_yscale('log')
    axs[1].set_xscale('log')
    axs[0].set_xlabel('$\delta_w$', fontsize=18)
    axs[1].set_xlabel('$\delta_b$', fontsize=18)
    axs[0].set_ylabel(r'$\left\Vert \nabla_{w}L\left(w,b\right) - u_{\delta_w} \left(w,b\right)\right\Vert$',
                      fontsize=18)
    axs[1].set_ylabel(
        r'$\left\Vert \frac{\partial}{\partial{b}}L\left(w,b\right) - u_{\delta_b} \left(w,b\right)\right\Vert$',
        fontsize=18)

    axs[0].grid(alpha=0.5)
    axs[1].grid(alpha=0.5)
    plt.show()


def test_lr(X_train, y_train, X_val, y_val, max_iter=1500):
    lr_list = np.logspace(-9, -1, 9)

    fig, axs = plt.subplots(3, 3, sharey=True, figsize=(20, 12))
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.5)

    axs = np.ravel(axs)
    for i, lr in enumerate(lr_list):
        cur_linear_reggressor = LinearRegressor(lr)
        train_losses, val_losses = cur_linear_reggressor.fit_with_logs(X_train, y_train, keep_losses=True, X_val=X_val,
                                                                       y_val=y_val, max_iter=max_iter)
        print('lr size = ' + str(lr) + ', Best train loss = ' + str(
            min(train_losses)) + ', Best validation loss = ' + str(min(val_losses)))

        iterations = np.arange(max_iter + 1)
        axs[i].semilogy(iterations, train_losses, label="Train")
        axs[i].semilogy(iterations, val_losses, label="Validation")
        axs[i].grid(alpha=0.5)
        axs[i].legend()
        axs[i].set_title('lr = ' + str(lr))
        axs[i].set_xlabel('iteration')
        axs[i].set_ylabel('MSE')



if __name__ == '__main__':
   # read the data set
    df = pd.read_csv('virus_labeled.csv')
    train, test = prepare_HW3.prepare_data(df)
    train['new_fecutre'] = np.where((train['blood_type_A-'] == 1) | (train['blood_type_A+'] == 1), 1,0)
    train.drop(columns=['blood_type_A-', 'blood_type_A+', 'blood_type_B-', 'blood_type_B+'
                   , 'blood_type_AB-', 'blood_type_O-','blood_type_O+'],inplace = True)
    train.pop('blood_type')
    train_subset , train_subset_test = train_test_split(train,train_size=0.8 , random_state= my_id+or_id)
    X_train = train_subset.copy()
    X_train.pop('VirusScore')
    y_train = train_subset['VirusScore'].values
    X_train = X_train
    X_val = train_subset_test.copy()
    X_val.pop('VirusScore')
    X_val = X_val.values
    y_val = train_subset_test['VirusScore'].values
    compare_gradients(X_train.values, y_train, deltas=np.logspace(-7, -2, 9))
    test_lr(X_train.values,y_train,X_val,y_val)


