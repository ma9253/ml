import os
import pandas as pd
import hashlib
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit

def get_last_byte_of_hash(id,ratio,hash):
    return hash(np.int64(id)).digest()[-1]<265*ratio

class DataReader:
    """
    Data reader class
    """

    def __init__(self, path='', file_name=''):
        if path != '' and file_name != '':
            self.file_path = os.path.join(path, file_name)
        else:
            self.file_path = '../Data/LR/'+'housing.csv'

    def load_data(self):
        """load csv file and return Pandas DataFrame Object"""
        return pd.read_csv(self.file_path)

    def analyze_data(self):
        """
        load data, show first five rows and provides basic info about the data e.g. column names, type etc.
        """
        data = self.load_data()
        print('---------Top Five Rows--------')
        print(data.head())
        print('--------Basic Info------')
        print(data.info())

    def analyze_data_for_categorical_features(self):
        """
        load data, show statistics of categorical attributes
        e.g. total categories and number of instances for each category
        """
        data = self.load_data()
        print('-------Statistics of Categorical Columns--------')
        # ls_columns = data.columns.values.tolist()
        ls_columns = data.select_dtypes(include=['object']).columns.values.tolist()
        for c in ls_columns:
            print(data[c].value_counts())

    def analyze_data_for_numerical_features(self):
        """
        load data, show summary of all numerical attributes e.g. count, std, mean, min, max, etc.
        """
        data = self.load_data()
        print(data.describe())

    def plot_histogram_of_data_columns(self,lsColumns=[]):
        """
        load data, and display histogram of each feature/column
        :param lsColumns:
        """
        data = self.load_data()
        ls_columns = data.columns.values.tolist()
        if lsColumns:
            ls_columns=lsColumns
        data.hist(column=ls_columns,bins=50, figsize=(20,15))
        plt.show()

    def add_id_column_to_data(self,column1=None,column2=None):
        """
        add id column to data set for splitting data into train and test
        :param column1:
        :param column2:
        :return:
        """
        data= self.load_data()
        column_name = "id"

        if column1 and column2:
            # adding an index column based on values of two columns
            data['id']= data[column1]*1000+data[column2]
        elif column1:
            # adding an index column based on values of a column
            data['id'] = data[column1]
        else:
            # adding an index column based on row number
            data = data.reset_index()
            column_name = "index"

        return data, column_name

    def split_train_test(self,testSetPercentage,idColumn1=None,idColumn2=None,hash=hashlib.md5):
        """
        split train and test set by id
        :param testSetPercentage:
        :param idColumn1:
        :param idColumn2:
        :return: train and test set
        """
        data_with_id, column_name = self.add_id_column_to_data(idColumn1,idColumn2)
        ids = data_with_id[column_name]
        test_data_indexs = ids.apply(lambda id:get_last_byte_of_hash(id,testSetPercentage,hash))
        return data_with_id.loc[~test_data_indexs], data_with_id.loc[test_data_indexs]

    def stratified_train_test_split(self,columnName,testSetPercentage=0.2):
        data= self.load_data()
        data["income_cat"]=np.ceil(data[columnName]/1.5)
        data["income_cat"].where(data["income_cat"]<5,5,inplace=True)
        print(data["income_cat"].value_counts()/len(data))

        split= StratifiedShuffleSplit(n_splits=1,test_size=testSetPercentage,random_state=92)
        # print(data.head())
        for train_index,test_index in split.split(data,data["income_cat"]):
            train_set= data.loc[train_index]
            test_set = data.loc[test_index]

data_reader = DataReader()
# data_reader.analyze_data()
# data_reader.analyze_data_for_categorical_features()
# data_reader.analyze_data_for_numerical_features()
# data_reader.plot_histogram_of_data_columns()
# train,test=data_reader.split_train_test(0.2,'longitude','latitude')
data_reader.stratified_train_test_split("median_income",0.2)