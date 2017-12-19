import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelBinarizer

class AttributesCombiner(BaseEstimator,TransformerMixin):
    """
    class to combine different attributes to generates new attributes
    """

    def __init__(self,rooms_column_index=3,bedrooms_column_index=4,population_column_index=5,household_column_index=6):
        self.rooms_column_index = rooms_column_index
        self.bedrooms_column_index = bedrooms_column_index
        self.population_column_index = population_column_index
        self.household_column_index = household_column_index

    def fit(self,X,y=None):
        return self

    def transform(self,X,y=None):
        # selecting population column values by column id
        rooms_per_household=X[:,self.rooms_column_index]/X[:,self.household_column_index]
        population_per_household = X[:,self.population_column_index]/X[:,self.household_column_index]
        bedrooms_per_rooms = X[:,self.bedrooms_column_index]/X[:,self.rooms_column_index]
        # concatinating and return the results
        return np.c_[X,rooms_per_household,population_per_household,bedrooms_per_rooms]


class DataFrameToNumpyArrayConverter(BaseEstimator,TransformerMixin):
    """
    class to convert pandas datafame into numpy array
    """
    def __init__(self, columnNames):
        self.column_names = columnNames

    def fit(self,X,y=None):
        return self

    def transform(self,X):
        return X[self.column_names].values

class CustomLabelBinarizer(LabelBinarizer):
    """
    custom label binarizer class to avoid the error in pipeline
    """
    def fit_transform(self, X, y=None):
        return super(CustomLabelBinarizer, self).fit_transform(X)