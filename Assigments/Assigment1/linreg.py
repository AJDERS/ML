import inspect
import numpy as np
import matplotlib.pyplot as plt


class LinReg:

    def __init__(self, xs, ys):
        self.xs = xs
        self.ys = ys

    def _make_homogeneous(self):
        self.X = np.array(self.xs)
        self.X = np.column_stack(
            (
            self.xs,
            np.ones(len(self.xs))
            )
        )

    def linear_regression(self, transform=None):
        # Calculate pseudo-inverse.
        if transform:
            self.xs = list(map(transform, self.xs))
        self._make_homogeneous()
        xs_transpose = self.X.transpose()
        xtx = xs_transpose.dot(self.X)
        pseudo_inv = np.linalg.inv(xtx).dot(xs_transpose).dot(self.ys)
        w, b = pseudo_inv
        return w, b

    def plot(self, transform=None):
        w, b = self.linear_regression(transform)
        x = np.linspace(min(self.xs), max(self.xs), 50)
        y = w*x + b
        _, ax = plt.subplots()
        ax.plot(self.xs, self.ys, 'ro', label='Data')
        ax.plot(x, y, label='Regression Line')
        ax.set_ylabel('Radiated Energy')
        if transform:
            transform_str = inspect.getsource(transform)
            x_label = f'Absolute Temperature, {transform_str}'
        else:
            x_label = 'Absolute Temperature'
        ax.set_xlabel(x_label)
        ax.set_title('Linear Regression, Absolute Temperature/Radiated Energy')
        plt.show()

    def mse(self, transform=None):
        w, b = self.linear_regression(transform)
        mse = np.mean(self.ys - (w * np.array(self.xs) + b))
        return mse