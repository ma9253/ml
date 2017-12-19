from Regression.data_loader import DataReader
from Regression.data_transformation import AttributesCombiner, DataFrameToNumpyArrayConverter, CustomLabelBinarizer
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from matplotlib import pyplot as plt


class Main():
    """
    main class to run every functionality
    """

    def __init__(self):
        dataReader = DataReader(path='../Data/LR/', file_name='housing.csv')
        self.train_X, self.train_Y, self.test_X, self.test_Y = dataReader.stratified_train_test_split(
            normalizedColumnName="median_income", normalizedValue=1.5, mergingValue=5, testSetPercentage=0.2,
            labelColumn="median_house_value")

    def get_numerical_attributes_data(self, categoricalColumn):
        """
        get numerical attributes
        :param data:
        :param categoricalColumn:
        :return: numerical columns
        """
        # ls_columns = data.select_dtypes(include=['object']).columns.values.tolist()
        numerical_X = self.train_X.drop(categoricalColumn, axis=1)
        return list(numerical_X)

    def prepare_full_pipeline(self, categoricalColumn):
        """
        prepare full pipeline
        :param categoricalColumn:
        :return: pipeline
        """
        try:
            # categorical_X, numerical_X = self.split_numeric_and_categorical_data(data,categoricalColumn)
            numerical_columns = self.get_numerical_attributes_data(categoricalColumn)
            categorical_columns = [categoricalColumn]
            # print(numerical_columns)

            num_pipeline = Pipeline([
                ('numpy_transformer', DataFrameToNumpyArrayConverter(numerical_columns)),
                ('imputer', Imputer(strategy='median')),  # fill the missing values with median of the column
                ('attribute_combiner', AttributesCombiner()),
                ('standard_scaler', StandardScaler()),
            ])

            cat_pipeline = Pipeline([
                ('numpy_transformer', DataFrameToNumpyArrayConverter(categorical_columns)),
                ('label_encoder', CustomLabelBinarizer()),  # binarizer for categorical data

            ])

            combined_pipeline = FeatureUnion(transformer_list=[
                ('num_pipeline', num_pipeline),
                ('cat_pipeline', cat_pipeline),
            ])

            return combined_pipeline
        except Exception as ex:
            print(str(ex))
            return None

    def fit_model(self, model):
        try:
            pipeline = self.prepare_full_pipeline(categoricalColumn="ocean_proximity")
            X = pipeline.fit_transform(self.train_X)
            model.fit(X, self.train_Y)
            predictions_train = model.predict(X)
            X_test = pipeline.transform(self.test_X)
            predictions_test = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(self.test_Y, predictions_test))
            print('Error (RMSE): ', rmse)

            # ------------ setting plot style
            # plt.style.use('fivethirtyeight')
            plt.style.use('ggplot')

            # -----------plotting residual errors in training data
            # plt.scatter(predictions_train, predictions_train - self.train_Y,
            #             color="green", s=5, label='Train data')

            # ----------plotting residual errors in test data
            # on x axis we have predictions of the model whereas on y axis residual error is displayed
            plt.scatter(predictions_test, predictions_test - self.test_Y,
                        color="blue", s=10, label='Test data')

            # ------------ plotting line for zero residual error
            plt.hlines(y=0, xmin=0, xmax=800000, linewidth=2)
            plt.legend(loc='upper right')
            plt.title("Residual errors")
            plt.show()

        except Exception as ex:
            print(str(ex))


if __name__ == "__main__":
    main = Main()
    # model = LinearRegression()
    # model = Ridge() # linear regression with l2 regularization
    # model = Lasso() # linear regression with l1 regularization
    # model = SVR(kernel='rbf') # svm regressor
    # model = DecisionTreeRegressor()
    model = RandomForestRegressor()
    main.fit_model(model)
