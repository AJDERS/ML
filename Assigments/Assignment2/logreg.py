import math
import numpy as np

class LogReg:

    def __init__(self, xs, labels, learning_rate=0.1):
        self.N = len(xs)
        self.learning_rate = learning_rate
        self.weights = [np.random.normal(0, 0.001, 3)]
        self.labels_dict = {k:v for k,v in zip(set(labels),[-1,1])}
        self.labels = list(map(lambda x: self.labels_dict.get(x), labels))
        self.parameters = list(zip(xs, self.labels))
        self.errors = [float('inf')]

    def _compute_gradient(self, weights):
        def _compute_term(x, y, w):
            numerator = y*x
            denominator = 1 + math.exp(y*np.transpose(w).dot(x))
            return numerator/denominator

        gradient = -(1/self.N)*sum(
            map(
                lambda par: _compute_term(par[0], par[1], weights),
                self.parameters
            )
        )
        return gradient

    def _update_weight(self, gradient):
        v_t = -gradient
        self.weights.append(self.weights[-1] + self.learning_rate * v_t)

    def _calculate_error(self, weights):
        def _compute_term(x, y, w):
            return math.log(
                1 + math.exp(-y * np.transpose(self.weights[-1]).dot(x))
            )

        error = (1/self.N)*sum(
            map(
                lambda par: _compute_term(par[0], par[1], weights),
                self.parameters
            )
        )
        self.errors.append(error)

    def _change_of_error(self):
        if not len(self.errors) > 3:
            return True
        else:
            return abs(self.errors[-2]-self.errors[-1]) > 0.000001

    def _calculate_weights(self):
        step_count = 0
        while all(
            [
                (step_count < 100000),
                (self.errors[-1] > 0.01),
                (self._change_of_error())
            ]
        ):
            gradient = self._compute_gradient(self.weights[-1])
            self._update_weight(gradient)
            self._calculate_error(self.weights[-1])
            step_count += 1
            if step_count % 10000 == 0:
               print(f'Reached {step_count} iterations.') 

    def _theta(self, s):
        return math.exp(s)/(1+math.exp(s))

    def predict(self, x):
        signal = self.weights[-1].dot(x)
        return round(self._theta(signal))

    def validate(self, xs, labels):
        if len(self.weights) == 1:
            self._calculate_weights()
        self.predictions = np.array(
            list(map(lambda x: self.predict(x), xs))
        )
        return 1-np.sum(self.predictions == labels)/len(labels)
