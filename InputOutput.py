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
