from datetime import datetime
import pandas as pd
import numpy as np
import math

from PySide2.QtWidgets import QDialog
import sys
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtGui import QImage
from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

def read_stock_data(zone, sitting_people, standing_people):
    now = datetime.today()
    dt_str = now.strftime("%d/%m/%Y")
    dt_str1 = now.strftime("%H:%M:%S")

    filename = '/home/sisifo/PycharmProjects/ML/venv/Utils_4/Utils_dataStock/data_storage.csv'
    df = pd.read_csv(filename)
    relevant_cols = ['date', 'time', 'zone', 'sitting_people', 'standing_people']
    df = df[relevant_cols]
    list1 = [dt_str, dt_str1, zone, sitting_people, standing_people]
    df.loc[len(df)] = list1
    df.to_csv('/home/sisifo/PycharmProjects/ML/venv/Utils_4/Utils_dataStock/data_storage.csv')

'''
zone = 1
sitting_people = 9
standing_people = 2
read_stock_data(zone, sitting_people, standing_people)
'''


def basic_search(self):
    filename = '/home/sisifo/PycharmProjects/ML/venv/Utils_4/Utils_dataStock/sintetic_data_v3.csv'
    df = pd.read_csv(filename)
    relevant_cols = ['date', 'time', 'zone', 'sitting_people', 'standing_people']
    df = df[relevant_cols]

    date_t = self.ui.lineEdit.text()
    time_t = self.ui.lineEdit_2.text()

    filtered_values = np.where((df['date'] == date_t) & (df['time'] == time_t))
    df2 = df.loc[filtered_values]
    if df2.empty:
        a = 'la fecha no existe'
        #print(a)
        self.ui.label_12.setText('{}'.format(a))
    else:
        print("else")
        sitting_people = df2['sitting_people']
        standing_people = df2['standing_people']
        total_col = sitting_people + standing_people
        total_chairs = 50
        free_chairs = total_chairs - sitting_people
        self.ui.label_4.setText('{}'.format(total_col.to_numpy()[0]))
        self.ui.label_3.setText('{}'.format(sitting_people.to_numpy()[0]))
        self.ui.label.setText('{}'.format(free_chairs.to_numpy()[0]))

'''
date_t = "01/05/2020"
time_t = "15:00:00"
basic_search(date_t, time_t)
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_sitting_people(self):
    date_t = self.ui.lineEdit_13.text()
    time_tl = self.ui.lineEdit_14.text()
    time_tu = self.ui.lineEdit_15.text()

    #date_t = "01/01/2017"
    #time_tl = "07:00:00"
    #time_tu = "17:00:00"

    filename = '/home/sisifo/PycharmProjects/ML/venv/Utils_4/Utils_dataStock/sintetic_data_v3.csv'
    df = pd.read_csv(filename)

    relevant_cols = ['date', 'time', 'zone', 'sitting_people', 'standing_people']
    df = df[relevant_cols]

    x = df[(df['date'] == date_t) & ((df['time'] <= time_tu) & (df['time'] >= time_tl))]
    print(x)

    timeS = x.time.tolist()
    dateS = x.date.tolist()

    list_str = []
    for i in range(len(timeS)): list_str.append(dateS[i] + ":" + timeS[i])

    if self.ui.checkBox_4.isChecked():
        plt.plot(list_str, x.sitting_people, color='r', label='Personas Sentadas')
        if self.ui.checkBox_2.isChecked():
            plt.plot(list_str, x.standing_people, color='b', label='Personas Paradas')
        if self.ui.checkBox_3.isChecked():
            total_people = x.sitting_people + x.standing_people
            plt.plot(list_str, total_people, color='g', label='total de Personas ')
        plt.xticks(rotation=90)
        plt.xlabel("fecha - hora")
        plt.ylabel("Personas")
        plt.title("Personas sentadas")
        plt.legend()
        #plt.show()
    #################bar plot###########33
    if self.ui.checkBox_5.isChecked():
        plt.figure()
        plt.bar(list_str, x.sitting_people, color='r', label='Personas Sentadas')
        if self.ui.checkBox_2.isChecked():
            plt.bar(list_str, x.standing_people, color='b', label='Personas Paradas')
        if self.ui.checkBox_3.isChecked():
            total_people = x.sitting_people + x.standing_people
            plt.bar(list_str, total_people, color='g', label='total de Personas ')
        plt.xticks(rotation=90)
        plt.xlabel("fecha - hora")
        plt.ylabel("Personas")
        plt.title("Personas sentadas")
        plt.legend()
        #plt.show()

    ####### Pie plot #############3
    if self.ui.checkBox_6.isChecked():

        if self.ui.checkBox.isChecked():
            plt.figure()
            plt.pie(x.sitting_people, labels=list_str)
            plt.title("Personas sentadas")
        if self.ui.checkBox_2.isChecked():
            plt.figure()
            plt.pie(x.standing_people, labels=list_str)
            plt.title("Personas paradas")

        if self.ui.checkBox_3.isChecked():
            total_people = x.sitting_people + x.standing_people
            plt.figure()
            plt.pie(total_people, labels=list_str)
            plt.title("Total de Personas")

        #plt.xticks(rotation=90)
        #plt.xlabel("fecha - hora")
        #plt.ylabel("Personas")
        #plt.title("Personas sentadas")
        #plt.legend()
        #plt.show()

    plt.show()
#plot_sitting_people()




def plot_sitting_people_month(self):
    date_t = self.ui.lineEdit_7.text()
    time_t = self.ui.lineEdit_8.text()

    filename = '/home/sisifo/PycharmProjects/ML/venv/Utils_4/Utils_dataStock/sintetic_data_v3.csv'
    df = pd.read_csv(filename)

    relevant_cols = ['date', 'time', 'zone', 'sitting_people', 'standing_people']
    df = df[relevant_cols]

    #date_t = "01/12/2017"
    #time_t = "17:00:00"

    x = str(date_t).split("/")
    int_x = []

    for i in range(len(x)): int_x.append(int(x[i]))

    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    x = df[(df['date'].dt.year == int_x[2])
           & (df['date'].dt.month == int_x[1])
           & (df['time'] == time_t)
           ]

    print(x)
    x.date = x.date.dt.strftime('%Y-%m-%d')
    timeS = x.time.tolist()
    dateS = x.date.tolist()

    list_str = []
    for i in range(len(timeS)): list_str.append(dateS[i] + ":" + timeS[i])

    if self.ui.checkBox_7.isChecked():
        if self.ui.checkBox_10.isChecked():
            plt.plot(list_str, x.sitting_people, color='r', label='Personas Sentadas')
        if self.ui.checkBox_11.isChecked():
            plt.plot(list_str, x.standing_people, color='b', label='Personas Paradas')
        if self.ui.checkBox_12.isChecked():
            total_people = x.sitting_people + x.standing_people
            plt.plot(list_str, total_people, color='g', label='total de Personas ')
        plt.xticks(rotation=90)
        plt.xlabel("fecha - hora")
        plt.ylabel("Personas")
        plt.title("Personas sentadas")
        plt.legend()
        #plt.show()
    #################bar plot###########33
    if self.ui.checkBox_8.isChecked():
        if self.ui.checkBox_10.isChecked():
            plt.figure()
            plt.bar(list_str, x.sitting_people, color='r', label='Personas Sentadas')
        if self.ui.checkBox_11.isChecked():
            plt.bar(list_str, x.standing_people, color='b', label='Personas Paradas')
        if self.ui.checkBox_12.isChecked():
            total_people = x.sitting_people + x.standing_people
            plt.bar(list_str, total_people, color='g', label='total de Personas ')
        plt.xticks(rotation=90)
        plt.xlabel("fecha - hora")
        plt.ylabel("Personas")
        plt.title("Personas sentadas")
        plt.legend()
        #plt.show()

    ####### Pie plot #############3
    if self.ui.checkBox_9.isChecked():

        if self.ui.checkBox_10.isChecked():
            plt.figure()
            plt.pie(x.sitting_people, labels=list_str)
            plt.title("Personas sentadas")
        if self.ui.checkBox_11.isChecked():
            plt.figure()
            plt.pie(x.standing_people, labels=list_str)
            plt.title("Personas paradas")

        if self.ui.checkBox_12.isChecked():
            total_people = x.sitting_people + x.standing_people
            plt.figure()
            plt.pie(total_people, labels=list_str)
            plt.title("Total de Personas")

        #plt.xticks(rotation=90)
        #plt.xlabel("fecha - hora")
        #plt.ylabel("Personas")
        #plt.title("Personas sentadas")
        #plt.legend()
        #plt.show()

    plt.show()


def Analitic_day(self):
    date_t = self.ui.lineEdit_16.text()
    time_tl = self.ui.lineEdit_17.text()
    time_tu = self.ui.lineEdit_18.text()

    # date_t = "01/01/2017"
    # time_tl = "07:00:00"
    # time_tu = "17:00:00"

    total_seat = 50

    filename = '/home/sisifo/PycharmProjects/ML/venv/Utils_4/Utils_dataStock/sintetic_data_v3.csv'
    df = pd.read_csv(filename)

    relevant_cols = ['date', 'time', 'zone', 'sitting_people', 'standing_people']
    df = df[relevant_cols]

    x = df[(df['date'] == date_t) & ((df['time'] <= time_tu) & (df['time'] >= time_tl))]
    print(x)

    # Hora con mayor cantidad de personas sentadas
    max_ind = x['sitting_people'].idxmax()
    h1 = x['time'][max_ind]
    h1_1 = x['sitting_people'][max_ind]
    h1_2 = x['standing_people'][max_ind]
    h1_3 = total_seat - h1_1
    h1_4 = h1_1 + h1_2
    # end


    ##### Hora con mayor cantidad de personas paradas ################3
    max_ind2 = x['standing_people'].idxmax()
    h2 = x['time'][max_ind2]
    h2_1 = x['sitting_people'][max_ind2]
    h2_2 = x['standing_people'][max_ind2]
    h2_3 = total_seat - h2_1
    h2_4 = h2_1 + h2_2
    ################ end ##########################

    ##### Hora con mayor cantidad de personas : ################3
    x2 = x.copy()
    x2['total_people'] = x.apply(lambda y: y['standing_people'] + y['sitting_people'], axis=1)

    max_ind3 = x2['total_people'].idxmax()
    h3 = x2['time'][max_ind3]
    h3_1 = x2['sitting_people'][max_ind3]
    h3_2 = x2['standing_people'][max_ind3]
    h3_3 = total_seat - h3_1
    h3_4 = h3_1 + h3_2
    # end

    # Hora con menor cantidad de personas sentadas
    min_ind = x['sitting_people'].idxmin()
    h4 = x['time'][min_ind]
    h4_1 = x['sitting_people'][min_ind]
    h4_2 = x['standing_people'][min_ind]
    h4_3 = total_seat - h4_1
    h4_4 = h4_1 + h4_2
    # end

    # Hora con mayor cantidad de personas paradas
    min_ind2 = x['standing_people'].idxmin()
    h5 = x['time'][min_ind2]
    h5_1 = x['sitting_people'][min_ind2]
    h5_2 = x['standing_people'][min_ind2]
    h5_3 = total_seat - h5_1
    h5_4 = h5_1 + h5_2
    # end

    # Hora con mayor cantidad de personas :
    # x2 = x.copy()
    # x2['total_people'] = x.apply(lambda y: y['standing_people'] + y['sitting_people'], axis=1)

    min_ind3 = x2['total_people'].idxmin()
    h6 = x2['time'][min_ind3]
    h6_1 = x2['sitting_people'][min_ind3]
    h6_2 = x2['standing_people'][min_ind3]
    h6_3 = total_seat - h6_1
    h6_4 = h6_1 + h6_2
    # end

    # get the mean of the columns
    mean_sitting_people = x['sitting_people'].mean()
    mean_standing_people = x['standing_people'].mean()
    mean_total_people = x2['total_people'].mean()

    self.ui.label_29.setText('{}'.format(h1))
    self.ui.label_31.setText('{}'.format(h1_1))
    self.ui.label_32.setText('{}'.format(h1_2))
    self.ui.label_33.setText('{}'.format(h1_3))
    self.ui.label_159.setText('{}'.format(h1_4))

    self.ui.label_30.setText('{}'.format(h2))
    self.ui.label_140.setText('{}'.format(h2_1))
    self.ui.label_143.setText('{}'.format(h2_2))
    self.ui.label_148.setText('{}'.format(h2_3))
    self.ui.label_158.setText('{}'.format(h2_4))

    self.ui.label_134.setText('{}'.format(h3))
    self.ui.label_139.setText('{}'.format(h3_1))
    self.ui.label_144.setText('{}'.format(h3_2))
    self.ui.label_149.setText('{}'.format(h3_3))
    self.ui.label_156.setText('{}'.format(h3_4))

    self.ui.label_135.setText('{}'.format(h4))
    self.ui.label_140.setText('{}'.format(h4_1))
    self.ui.label_145.setText('{}'.format(h4_2))
    self.ui.label_150.setText('{}'.format(h4_3))
    self.ui.label_160.setText('{}'.format(h4_4))

    self.ui.label_136.setText('{}'.format(h5))
    self.ui.label_141.setText('{}'.format(h5_1))
    self.ui.label_146.setText('{}'.format(h5_2))
    self.ui.label_151.setText('{}'.format(h5_3))
    self.ui.label_157.setText('{}'.format(h5_4))

    self.ui.label_137.setText('{}'.format(h6))
    self.ui.label_142.setText('{}'.format(h6_1))
    self.ui.label_147.setText('{}'.format(h6_2))
    self.ui.label_153.setText('{}'.format(h6_3))
    self.ui.label_161.setText('{}'.format(h6_4))

    self.ui.label_112.setText('{}'.format(mean_sitting_people))
    self.ui.label_114.setText('{}'.format(mean_standing_people))
    self.ui.label_115.setText('{}'.format(mean_total_people))


def Analitic_period_time(self):
    date_tl = self.ui.lineEdit_19.text()
    date_tu = self.ui.lineEdit_26.text()
    time_tl = self.ui.lineEdit_20.text()
    time_tu = self.ui.lineEdit_27.text()

    # date_tl = "15/01/2017"
    # date_tu = "22/01/2017"
    # time_tl = "16:00:00"
    # time_tu = "17:00:00"

    total_seat = 50

    filename = '/home/sisifo/PycharmProjects/ML/venv/Utils_4/Utils_dataStock/sintetic_data_v3.csv'
    df = pd.read_csv(filename)

    relevant_cols = ['date', 'time', 'zone', 'sitting_people', 'standing_people']
    df = df[relevant_cols]

    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')

    x =df.loc[((df['date'] >= date_tl) & (df['date'] <= date_tu) ) & ((df['time']<= time_tu) & (df['time'] >= time_tl))]
    print(x)

    # día con mayor cantidad de personas sentadas
    max_ind = x['sitting_people'].idxmax()
    d1 = x['date'][max_ind]
    d1_1 = x['sitting_people'][max_ind]
    d1_2 = x['standing_people'][max_ind]
    d1_3 = total_seat - d1_1
    d1_4 = d1_1 + d1_2
    # end

    # día con mayor cantidad de personas paradas
    max_ind2 = x['standing_people'].idxmax()
    d2 = x['date'][max_ind2]
    d2_1 = x['sitting_people'][max_ind2]
    d2_2 = x['standing_people'][max_ind2]
    d2_3 = total_seat - d2_1
    d2_4 = d2_1 + d2_2
    # end

    # día con mayor cantidad de personas:
    x2 = x.copy()
    x2['total_people'] = x.apply(lambda y: y['standing_people'] + y['sitting_people'], axis=1)

    max_ind3 = x2['total_people'].idxmax()
    d3 = x2['date'][max_ind3]
    d3_1 = x2['sitting_people'][max_ind3]
    d3_2 = x2['standing_people'][max_ind3]
    d3_3 = total_seat - d3_1
    d3_4 = d3_1 + d3_2
    # end

    # día con menor cantidad de personas sentadas
    min_ind = x['sitting_people'].idxmin()
    d4 = x['date'][min_ind]
    d4_1 = x['sitting_people'][min_ind]
    d4_2 = x['standing_people'][min_ind]
    d4_3 = total_seat - d4_1
    d4_4 = d4_1 + d4_2
    # end

    # día con mayor cantidad de personas paradas
    min_ind2 = x['standing_people'].idxmin()
    d5 = x['date'][min_ind2]
    d5_1 = x['sitting_people'][min_ind2]
    d5_2 = x['standing_people'][min_ind2]
    d5_3 = total_seat - d5_1
    d5_4 = d5_1 + d5_2
    # end

    # día con mayor cantidad de personas :
    # x2 = x.copy()
    # x2['total_people'] = x.apply(lambda y: y['standing_people'] + y['sitting_people'], axis=1)

    min_ind3 = x2['total_people'].idxmin()
    d6 = x2['date'][min_ind3]
    d6_1 = x2['sitting_people'][min_ind3]
    d6_2 = x2['standing_people'][min_ind3]
    d6_3 = total_seat - d6_1
    d6_4 = d6_1 + d6_2
    # end

    # get the mean of the columns
    mean_sitting_people = math.floor(x['sitting_people'].mean())
    mean_standing_people = math.floor(x['standing_people'].mean())
    mean_total_people = math.floor(x2['total_people'].mean())

    self.ui.label_116.setText('{}'.format(d1))
    self.ui.label_179.setText('{}'.format(d1_1))
    self.ui.label_180.setText('{}'.format(d1_2))
    self.ui.label_190.setText('{}'.format(d1_3))
    self.ui.label_197.setText('{}'.format(d1_4))

    self.ui.label_119.setText('{}'.format(d2))
    self.ui.label_177.setText('{}'.format(d2_1))
    self.ui.label_183.setText('{}'.format(d2_2))
    self.ui.label_188.setText('{}'.format(d2_3))
    self.ui.label_192.setText('{}'.format(d2_4))

    self.ui.label_117.setText('{}'.format(d3))
    self.ui.label_175.setText('{}'.format(d3_1))
    self.ui.label_185.setText('{}'.format(d3_2))
    self.ui.label_186.setText('{}'.format(d3_3))
    self.ui.label_194.setText('{}'.format(d3_4))

    self.ui.label_121.setText('{}'.format(d4))
    self.ui.label_176.setText('{}'.format(d4_1))
    self.ui.label_182.setText('{}'.format(d4_2))
    self.ui.label_187.setText('{}'.format(d4_3))
    self.ui.label_195.setText('{}'.format(d4_4))

    self.ui.label_120.setText('{}'.format(d5))
    self.ui.label_174.setText('{}'.format(d5_1))
    self.ui.label_181.setText('{}'.format(d5_2))
    self.ui.label_189.setText('{}'.format(d5_3))
    self.ui.label_193.setText('{}'.format(d5_4))

    self.ui.label_118.setText('{}'.format(d6))
    self.ui.label_178.setText('{}'.format(d6_1))
    self.ui.label_184.setText('{}'.format(d6_2))
    self.ui.label_191.setText('{}'.format(d6_3))
    self.ui.label_196.setText('{}'.format(d6_4))

    self.ui.label_122.setText('{}'.format(mean_sitting_people))
    self.ui.label_124.setText('{}'.format(mean_standing_people))
    self.ui.label_123.setText('{}'.format(mean_total_people))
