import matplotlib.pyplot as plt
import seaborn as sns

class grafico:
    def __init__(self,title,X,labels):
        self.title = title
        self.X = X
        self.labels = labels
    def plotar(self):
        plt.set_title(self.title)
        plt.scatter(self.X[:, 0], self.X[:, 1], c=self.labels, s=40, cmap='viridis');
        pltshow()