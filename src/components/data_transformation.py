import os
import sys
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
class DataTransformationConfig:
    def __init__(self):
        self.preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            categorical_columns = [
                'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
                'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
                'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
                'stalk-surface-below-ring', 'stalk-color-above-ring',
                'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',
                'ring-type', 'spore-print-color', 'population', 'habitat'
            ]

            cat_pipeline = Pipeline([
                ("one_hot_encoder", OneHotEncoder())
            ])

            preprocessor = ColumnTransformer([
                ("cat_pipelines", cat_pipeline, categorical_columns)
            ])

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "class"

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            # Encode the target features
            label_encoder = LabelEncoder()
            target_feature_train_arr = label_encoder.fit_transform(target_feature_train_df).reshape(-1, 1)
            target_feature_test_arr = label_encoder.transform(target_feature_test_df).reshape(-1, 1)
            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Convert to sparse matrix if necessary
            if sparse.issparse(input_feature_train_arr):
                target_feature_train_arr = sparse.csr_matrix(target_feature_train_arr)
                target_feature_test_arr = sparse.csr_matrix(target_feature_test_arr)

            train_arr = sparse.hstack([input_feature_train_arr, target_feature_train_arr]) if sparse.issparse(input_feature_train_arr) else np.concatenate([input_feature_train_arr, target_feature_train_arr], axis=1)
            test_arr = sparse.hstack([input_feature_test_arr, target_feature_test_arr]) if sparse.issparse(input_feature_test_arr) else np.concatenate([input_feature_test_arr, target_feature_test_arr], axis=1)

            # Save the preprocessor object and label encoder
            logging.info(f"Saved preprocessing object.")
            save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path, obj=preprocessing_obj)
            # Example: save_object(file_path='path_to_save_label_encoder.pkl', obj=label_encoder)

            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path

        except Exception as e:
            raise CustomException(e, sys)