import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


current_dir = os.path.dirname(os.path.realpath(__file__))
src_dir = os.path.join(current_dir, '..', 'src')
sys.path.append(src_dir)
 

if __name__=='__main__':
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()
    tranform=DataTransformation()
    train_arr,test_arr,m=tranform.initaite_data_transformation(train_data,test_data)
    trainer=ModelTrainer()
    trainer.initate_model_training(train_arr,test_arr)
    
