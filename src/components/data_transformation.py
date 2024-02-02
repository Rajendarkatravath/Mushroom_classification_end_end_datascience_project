import os
import sys
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

class DataTransformationConfig:
    def __init__(self):
        self.preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")
        self.label_encoder_file_path = os.path.join('artifacts', "label_encoder.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            preprocessor = OrdinalEncoder()  # Use OrdinalEncoder for all feature columns
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data completed")

            target_column_name = "class"

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            label_encoder = LabelEncoder()
            target_feature_train_arr = label_encoder.fit_transform(target_feature_train_df).reshape(-1, 1)
            target_feature_test_arr = label_encoder.transform(target_feature_test_df).reshape(-1, 1)
            logging.info("Label encoding of target variable completed")

            preprocessing_obj = self.get_data_transformer_object()
            logging.info("Obtaining preprocessing object")

            # Apply transformations
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            logging.info("Preprocessing of input features completed")

            # Handling sparse matrices
            if sparse.issparse(input_feature_train_arr):
                input_feature_train_arr = input_feature_train_arr.toarray()
                input_feature_test_arr = input_feature_test_arr.toarray()

            # Concatenate features and target
            train_arr = np.hstack([input_feature_train_arr, target_feature_train_arr])
            test_arr = np.hstack([input_feature_test_arr, target_feature_test_arr])

            # Save the preprocessor object and label encoder
            logging.info("Saving preprocessing object and label encoder")
            save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path, obj=preprocessing_obj)
            save_object(file_path=self.data_transformation_config.label_encoder_file_path, obj=label_encoder)

            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path, self.data_transformation_config.label_encoder_file_path

        except Exception as e:
            raise CustomException(e, sys)


