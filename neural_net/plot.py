import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


class Plotter:
    def __init__(self, x_test, y_test, networks, legend_based_on_names: bool = True):
        self.x_test = x_test
        self.y_test = y_test
        self.networks = networks
        self.predictions = []
        self.legend_based_on_names = legend_based_on_names
        self.palette = ['g', 'r', 'b', 'y', 'c', 'k', 'm', 'orange', 'w']

    def set_params(self, x_test=None, y_test=None, networks=None):
        if x_test is not None:
            self.x_test = x_test
        if y_test is not None:
            self.y_test = y_test
        if networks is not None:
            self.networks = networks

    def get_color(self, index):
        return self.palette[index]

    def set_labels(self, title=None, ylabel=None, xlabel=None):
        if title is not None:
            plt.title(title)
        if xlabel is not None:
            plt.xlabel(xlabel)
        if ylabel is not None:
            plt.ylabel(ylabel)
        return self

    # visualization of 2d data
    @FutureWarning
    def plot_data_2d(self, x, y, colors, show=True):
        pass
        # uniques = np.unique(colors)
        # labels = []
        # for index, unique in enumerate(uniques):
        #     indexes = colors == unique
        #     plt.plot(x[indexes], y[indexes], 'o', self.get_color(index))
        #     labels.append('class ' + str(int(unique)))
        # plt.legend(labels=labels)
        # plt.title('Points on the plane')
        # plt.xlabel('y')
        # plt.ylabel('x')
        # if show:
        #     plt.show()
        # return self

    # plot which shows mse or accuracy
    def plot_measure_results_data(self, measure_function, measure_name, labels=None, from_error=0, show=True, y_log=False):
        errors = []
        for network in self.networks:
            errors.append(measure_function(network))
        for index in range(len(errors)):
            plt.plot([i + 1 for i in range(from_error, len(errors[index]))], errors[index][from_error:], color=self.get_color(index))
        labels = self._get_labels(labels, default_label_name='network')
        plt.legend(labels=labels)
        plt.title(f'{measure_name} through epochs')
        plt.xlabel(f'epochs')
        plt.ylabel(f'{measure_name}')
        if y_log:
            plt.yscale("log")
        if show:
            plt.show()
        return self

    # visualization of 1d data with both predictions and true values
    def plot_data_1d(self, labels=None, show=True):
        self._predict_networks()
        for index, prediction in enumerate([self.y_test] + self.predictions):
            plt.plot(self.x_test[:, 0], prediction[:, 0], 'o', color=self.get_color(index))
        labels = self._get_labels(labels, label0="true data", default_label_name='predictions')
        plt.legend(labels=labels)
        plt.title('Fictional data with values')
        plt.xlabel('observed values')
        plt.ylabel('result values')
        if show:
            plt.show()
        return self

    def _get_labels(self, labels, label0=None, default_label_name='label'):
        if labels is None:
            if label0:
                labels = [label0]
            else:
                labels = []
            if self.legend_based_on_names:
                for network in self.networks:
                    labels.append(network.name)
            elif len(self.networks) == 1:
                labels.append(default_label_name)
            else:
                for i in range(len(self.networks)):
                    labels.append(f'{default_label_name} {i+1}')
        return labels

    def _predict_networks(self):
        self.predictions = []
        for network in self.networks:
            self.predictions.append(network.predict(self.x_test))
