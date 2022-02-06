<<<<<<< HEAD
import csv
import os
from datetime import date, datetime
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


class Draw:
    MARKERS_LIST = ['o', 'v', 'p', 'D', '+', '*', '^', 'x', '.', 's']

    @classmethod
    def points(cls, data):
        return plt.scatter(data['x'], data['y'], s=data['space'], marker=data['marker'], facecolors=data['color'],
                           edgecolors='none', alpha=data['alpha'], zorder=data['zorder'])

    @classmethod
    def circles(cls, data):
        for x, y, r in zip(data['x'], data['y'], data['radius']):
            plt.gcf().gca().add_artist(plt.Circle(
                (x, y),
                r,
                alpha=data['alpha'],
                color=data['color'],
                zorder=data['zorder'],
                fill=False)
            )

    @classmethod
    def lines(cls, data):
        for key, marker in zip(data, Draw.MARKERS_LIST):
            plt.plot(data[key]['x'], data[key]['y'], marker=marker, label=key)

    @classmethod
    def set_plot_feature(cls, x_ticks, y_ticks, title=None, x_label=None, y_label=None, handles=[], labels=[],
                         loc="lower right", font_size=10, line_style=':', x_ticks_labels=[], y_ticks_labels=[]):
        plt.legend(handles=handles, labels=labels, loc=loc, fontsize=font_size)
        plt.grid(True, linestyle=line_style)
        plt.xticks(x_ticks, x_ticks_labels)
        plt.yticks(y_ticks, y_ticks_labels)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)


def new_name_if_exist(path, file_name, ext):
    files_names = os.listdir(path)
    if file_name in files_names:
        today_time = str(datetime.now().hour) + '-' + str(datetime.now().minute) + '-' + str(datetime.now().second)
        today_date = str(date.today())
        additional = '_' + today_date + '_' + today_time + ext
        file_name = file_name[:-4]
        file_name += additional
    return file_name


def to_csv(path, file_name, data):
    file_name = new_name_if_exist(path, file_name, ext='.csv')

    with open(path + '/' + file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)


def from_csv(path, file_name):
    with open(path + '/' + file_name) as file:
        csv_reader = csv.reader(file, delimiter=',')
        data = [[float(elem) for elem in row] for row in csv_reader]
    return data


def create_arithmetic_scale_line_diagram(path, file_name, data, title=None, x_label=None, y_label=None, y_ticks=None,
                                  x_ticks=None, x_ticks_labels=None, y_ticks_labels=None):
    file_name = new_name_if_exist(path, file_name, ext='.svg')

    fig = plt.figure(figsize=(10, 8))
    Draw.lines(data)

    Draw.set_plot_feature(x_ticks, y_ticks, title=title, x_label=x_label, y_label=y_label, labels=[key for key in data],
                          loc="lower left", font_size=10, line_style=':', x_ticks_labels=x_ticks_labels,
                          y_ticks_labels=y_ticks_labels)

    fig.savefig(path+'/'+file_name, transparent=True)


def create_scenario_diagram(path, file_name, data, title=None, x_label=None, y_label=None, x_ticks=None, y_ticks=None,
                     x_ticks_labels=None, y_ticks_labels=None):
    file_name = new_name_if_exist(path, file_name, ext='.svg')

    fig = plt.figure()
    for key in data:
        if data[key]['shape'] == 'points':
            Draw.points(data[key])
        elif data[key]['shape'] == 'circles':
            Draw.circles(data[key])

    patches = [mlines.Line2D([], [], color=data[key]['color'], marker=data[key]['marker'], linestyle='None') for key in data]
    Draw.set_plot_feature(x_ticks, y_ticks, title=title, x_label=x_label, y_label=y_label, handles=patches, labels=[key for key in data],
                          loc="lower right", font_size=10, line_style=':', x_ticks_labels=x_ticks_labels,
                          y_ticks_labels=y_ticks_labels)

    fig.savefig(path + '/' + file_name, transparent=True)
=======
import csv
import os
from datetime import date, datetime


def to_csv(path, file_name, data):
    files_names = os.listdir(path)
    if file_name in files_names:
        today_time = str(datetime.now().hour) + '-' + str(datetime.now().minute) + '-' + str(datetime.now().second)
        today_date = str(date.today())
        additional = '_' + today_date + '_' + today_time + '.csv'
        file_name = file_name[:-4]
        file_name += additional

    with open(path + '/' + file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)


def from_csv(path, file_name):
    with open(path + '/' + file_name) as file:
        csv_reader = csv.reader(file, delimiter=',')
        data = [[float(elem) for elem in row] for row in csv_reader]
    return data
>>>>>>> be4a292f30b149cf5d706beafa268b472b77d334
