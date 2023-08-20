import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import pandas as pd


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            model_path=os.path.join('artifacts','model.pkl')

            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)

            data_scaled=preprocessor.transform(features)

            pred=model.predict(data_scaled)
            return pred
            

        except Exception as e:
            logging.info("Exception occured in prediction")
            raise CustomException(e,sys)
        
        
class CustomData:
    def __init__(self,
                 Gender:str,
                 family_history_with_overweight:str,
                 FAVC:str,
                 CAEC:str,
                 SMOKE:str,
                 SCC:str,
                 CALC:str,
                 MTRANS:str,
                 Age:float,
                 Height:float,
                 Weight:float,
                 FCVC:float,
                 NCP:float,
                 CH2O:float,
                 FAF:float,
                 TUE:float):
        
        self.Gender=Gender
        self.family_history_with_overweight=family_history_with_overweight
        self.FAVC=FAVC
        self.CAEC=CAEC
        self.SMOKE=SMOKE
        self.SCC=SCC
        self.CALC=CALC
        self.MTRANS = MTRANS
        self.Age = Age
        self.Height = Height
        self.Weight = Weight
        self.FCVC = FCVC
        self.NCP = NCP
        self.CH2O = CH2O
        self.FAF = FAF
        self.TUE = TUE


    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'Gender':[self.Gender],
                'family_history_with_overweight':[self.family_history_with_overweight],
                'FAVC':[self.FAVC],
                'CAEC':[self.CAEC],
                'SMOKE':[self.SMOKE],
                'SCC':[self.SCC],
                'CALC':[self.CALC],
                'MTRANS':[self.MTRANS],
                'Age':[self.Age],
                'Height':[self.Height],
                'Weight':[self.Weight],
                'FCVC':[self.FCVC],
                'NCP':[self.NCP],
                'CH2O':[self.CH2O],
                'FAF':[self.FAF],
                'TUE':[self.TUE]

            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
        except Exception as e:
            logging.info('Exception Occured in prediction pipeline')
            raise CustomException(e,sys)
