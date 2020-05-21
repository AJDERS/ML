import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


class SelfAssessment:

    def __init__(self, file):
        self.data = self.load_data_file(file)
        self.groups = self.data.groupby('class')
        self.data_matrix = self.make_matrix(file)

    def load_data_file(self, file):
        data = pd.read_csv(
            file,
            names=[
                "sepal_length",
                "sepal_width",
                "petal_length",
                "petal_width",
                'class'
            ]
        )
        return data

    def make_matrix(self, file):
        data = pd.read_csv(
            file,
            header=None,
            usecols=[0,1,2,3]
        )
        return data.values

    def calculate_lengths(self):
        matrix = self.data_matrix
        first = matrix[0]
        remaining = matrix[1:]
        distances = np.linalg.norm(remaining-first, axis=1)
        average_dist = np.average(distances)
        variance_dist = np.var(distances)
        return (average_dist, variance_dist)

    def calculate_group_average(self, col):
        if col in self.data.columns:
            averages = {}
            for name, group in self.groups:
                nx = len(group)
                averages[name] = (group[col].sum())/nx
            return averages
        else:
            raise ValueError(f'{col} is not in dataframe.')

    def _square_diff(self, value, average):
        return (value-average)**2

    def calculate_group_variance(self, col):
        try:
            averages = self.calculate_group_average(col)
        except ValueError:
            print(f'{col} is not in dataframe.')
        variances = {}
        for name, group in self.groups:
            nx = len(group)
            average = averages.get(name)
            square_diff = group[col].apply(
                lambda x: self._square_diff(x, average)
            )
            ssd = square_diff.sum()
            variances[name] = ssd/nx
        return variances
        

    def make_plot(self):
        data = self.data
        groups = self.groups
        sepal_length_list = []
        sepal_width_list = []
        patches = []
        colors = ['blue', 'red', 'green']
        for idx, class_ in enumerate(data['class'].unique()):
            group = groups.get_group(class_)
            sepal_length_list.append(
                group.sepal_length.values
            )
            sepal_width_list.append(
                group.sepal_width.values
            )
            patch = mpatches.Patch(
                color=colors[idx],
                label=f'{class_}')
            patches.append(patch)
        for index in range(len(sepal_length_list)):
            plt.scatter(
                sepal_length_list[index],
                sepal_width_list[index],
                c=colors[index])
        plt.legend(handles=patches)
        plt.savefig('iris.png')
        plt.show()

