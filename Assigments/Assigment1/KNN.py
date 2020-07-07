import numpy as np
import random
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm


class Preprocess:

    def __init__(self, picture_matrix, labels):
        self.picture_matrix = picture_matrix
        self.labels = labels

    def pick_digits(self, dig1, dig2):
        new_picture_matrix = []
        new_labels = []
        for i, label in enumerate(self.labels):
            if label in [dig1, dig2]:
                new_labels.append(label)
                new_picture_matrix.append(self.picture_matrix[i,:])
            else:
                pass
        new_picture_matrix = np.array(new_picture_matrix)
        return new_picture_matrix, new_labels

    def make_train_val(self, picture_matrix, labels):
        data_train, data_val, labels_train, labels_val = train_test_split(
            picture_matrix,
            labels,
            test_size=0.20,
            random_state=42
        )
        return data_train, data_val, labels_train, labels_val

class KNearestNeighbor:

    def __init__(self, points, labels):
        self.points = points
        assert len(set(labels)) == 2
        self.labels = labels
        self.labels_dict = {k:v for k,v in zip(set(labels),[-1,1])}
        self.labels_dict_reverse = {k:v for v,k in zip(set(labels),[-1,1])}

    def _distance(self, x):
        assert len(x) == np.shape(self.points)[1]
        distances = np.linalg.norm(self.points-x, axis=1)
        sorted_pos = np.argsort(distances)
        return sorted_pos

    def predict(self, x, K):
        sorted_pos = self._distance(x)
        if np.signbit(sum([self.labels_dict.get(self.labels[i]) for i in sorted_pos[:K]])):
            return self.labels_dict_reverse.get(-1)
        else:
            return self.labels_dict_reverse.get(1)
    
    def predict_matrix(self, matrix, K):
        predictions = []
        for picture in matrix:
            predictions.append(self.predict(picture, K))
        return np.array(predictions)
            
class Validation:

    def __init__(self, data_train, data_val, labels_train, labels_val):
        self.knn = KNearestNeighbor(data_train, labels_train)
        self.data_val = data_val
        self.labels_val = labels_val

    def validation_error(self, K):
        predictions = self.knn.predict_matrix(self.data_val, K)
        self.predictions = predictions
        return np.sum(predictions == self.labels_val)/len(self.labels_val)

    def validation_error_plot(self):
        self.val_error = []
        Ks = range(34)[1::2]
        for K in tqdm(Ks): 
            self.val_error.append(round(self.validation_error(K), 5))
        self.best = Ks[self.val_error.index(max(self.val_error))]
        # Make plot
        labels = [str(x) for x in Ks]

        x = np.arange(len(labels))  # the label locations
        width = 0.35  # the width of the bars

        fig, ax = plt.subplots()
        ax.step(x - width/2, self.val_error, width, label='Validation Error')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Validation error')
        ax.set_title('Validation error as a function of K')
        ax.set_xticks(x-width/2)
        ax.set_xticklabels(labels)
        ax.set_ylim([0.999*min(self.val_error), 1.001*max(self.val_error)])
        ax.grid(True)
        fig.tight_layout()
        
        plt.show()


