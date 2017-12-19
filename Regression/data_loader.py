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

    def stratified_train_test_split(self,normalizedColumnName,normalizedValue,mergingValue, testSetPercentage=0.2,labelColumn=''):
        """
        make stratified train and test split
        :param normalizedColumnName:
        :param normalizedValue:
        :param mergingValue:
        :param testSetPercentage:
        :param labelColumn:
        :return: train and test set
        """
        data= self.load_data()
        # make a new column for the standardized values of normalized column and do stratified sampling based on that column
        helper_column_name = "income_cat"
        data[helper_column_name]=np.ceil(data[normalizedColumnName]/normalizedValue)
        data[helper_column_name].where(data[helper_column_name]<mergingValue,mergingValue,inplace=True)
        # print(data[helper_column_name].value_counts()/len(data))

        split= StratifiedShuffleSplit(n_splits=1,test_size=testSetPercentage,random_state=92)
        for train_index,test_index in split.split(data,data[helper_column_name]):
            train_set= data.loc[train_index]
            test_set = data.loc[test_index]
        # removing helper column
        for set in (train_set,test_set):
            set.drop([helper_column_name],axis=1,inplace=True)
        if labelColumn:
            train_data = train_set.drop(labelColumn,axis=1)
            train_labels = train_set[labelColumn].copy()

            test_data = test_set.drop(labelColumn, axis=1)
            test_labels = test_set[labelColumn].copy()
            return train_data,train_labels, test_data, test_labels

        return train_set,test_set

# data_reader = DataReader()
# data_reader.analyze_data()
# data_reader.analyze_data_for_categorical_features()
# data_reader.analyze_data_for_numerical_features()
# data_reader.plot_histogram_of_data_columns()
# train,test=data_reader.split_train_test(0.2,'longitude','latitude')
# train,test = data_reader.stratified_train_test_split("median_income",1.5,5, 0.2,'median_house_value')
# d= train.copy()
# corr_matrix = d.corr()
# print(corr_matrix['median_house_value'].sort_values(ascending=False))

# d.plot(kind="scatter",x='longitude',y='latitude', alpha=0.4, s=d['population']/100,label='population',c=d['median_house_value'],cmap=plt.get_cmap('jet'),colorbar=True,)
# plt.legend()
# plt.show()