import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        
        '''
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            numerical_pipeline = Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")), #because of outliers in the data
                ("scaler",StandardScaler())
                ]
            )

            categorical_pipeline = Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder()),
                ("scaler",StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline",numerical_pipeline,numerical_columns),
                    ("cat_pipelines",categorical_pipeline,categorical_columns)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):

        try:

            # --------------------------------------------------------------------------------------------
            # Reading Data

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            # --------------------------------------------------------------------------------------------


            # --------------------------------------------------------------------------------------------
            # Preprocessing Data Object

            logging.info("Obtaining preprocessing object")            
            preprocessing_obj = self.get_data_transformer_object()
            # --------------------------------------------------------------------------------------------


            # --------------------------------------------------------------------------------------------
            # Separating features and target variable

            target_column_name = "math_score"
            numerical_columns = ["writing_score", "reading_score"]

            X_train = train_df.drop(columns=[target_column_name],axis=1)
            y_train = train_df[target_column_name]

            X_test = test_df.drop(columns=[target_column_name],axis=1)
            y_test = test_df[target_column_name]
            # --------------------------------------------------------------------------------------------


            # --------------------------------------------------------------------------------------------
            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe.")

            X_train_arr = preprocessing_obj.fit_transform(X_train)
            X_test_arr = preprocessing_obj.transform(X_test)
            # --------------------------------------------------------------------------------------------


            # --------------------------------------------------------------------------------------------
            # making a complete matrix of features and target variable

            train_arr = np.c_[X_train_arr, np.array(y_train)]
            test_arr = np.c_[X_test_arr, np.array(y_test)]
            # --------------------------------------------------------------------------------------------


            # --------------------------------------------------------------------------------------------
            logging.info(f"Saved preprocessing object.")

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        
        except Exception as e:
            raise CustomException(e,sys)