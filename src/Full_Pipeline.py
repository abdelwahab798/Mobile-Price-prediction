import logging
import os 
import pandas as pd
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error , mean_squared_error , root_mean_squared_error, r2_score
import joblib
import json

log_dir="logs"
os.makedirs(log_dir,exist_ok=True)

logger=logging.getLogger("pipline")
logger.setLevel("DEBUG")

consle_handler=logging.StreamHandler()
consle_handler.setLevel("DEBUG")

log_file_path=os.path.join(log_dir,"pipline.log")
file_handler=logging.FileHandler(log_file_path)
file_handler.setLevel("DEBUG")

formater=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
consle_handler.setFormatter(formater)
file_handler.setFormatter(formater)

logger.addHandler(consle_handler)
logger.addHandler(file_handler)

def load_Data(data_path)->pd.DataFrame:
    try:
        df=pd.read_csv(data_path)
        logger.info("Data is loaded")
        df=df.drop(columns=["thickness_mm","build_material","wifi_version","camera_setup"],axis=1)
        logger.info("Delet a low important features")
        return df
    except Exception as e:
        logger.error("error while load data",e)
        raise

def create_pipeline():
    try:
        cat=["brand","chipset","display_type"]
        processesor=ColumnTransformer(transformers=[("cat_process",OrdinalEncoder(handle_unknown="use_encoded_value",unknown_value=-1),cat)],remainder="passthrough")
        logger.info("processesor is created")
        full_pipeline=Pipeline(steps=[
            ('preprocessor', processesor),
            ('regressor', XGBRegressor(early_stopping_rounds=10))])
        logger.info("full pipline is created")
        return full_pipeline
    except Exception as e:
        logger.error("error while creating full pipeline,", e)
        raise

def spilt_data(df:pd.DataFrame):
    try:
        x=df.drop(columns=["price_inr"])
        y=df["price_inr"]
        x_train,x_temp, y_train, y_temp = train_test_split(x,y, test_size=0.3, random_state=42)
        x_val, x_test, y_val, y_test = train_test_split(x_temp,y_temp, test_size=0.5, random_state=42)
        return x_train, x_val,y_train,y_val,x_test,y_test
    except Exception as e:
        logger.error("error while split data")

def train_model(full_pipeline,x_train,y_train,x_val,y_val,x_test):
    try:
        full_pipeline.named_steps['preprocessor'].fit(x_train)
        x_val_transformed=full_pipeline.named_steps['preprocessor'].transform(x_val)
        full_pipeline.fit(x_train,y_train,regressor__eval_set=[(x_val_transformed, y_val)],regressor__verbose=False)
        logger.info("train model is done")
        return full_pipeline
    except Exception as e:
        logger.error("error while train model",e)

def eval_model(full_pipeline,y_test,x_test):
    try:
        y_pred = full_pipeline.predict(x_test)
        MAE=mean_absolute_error(y_test,y_pred)
        MSE=mean_squared_error(y_test,y_pred)
        RSME=root_mean_squared_error(y_test,y_pred)
        r2=r2_score(y_test,y_pred)
        metrics={
            "MAE":MAE,
            "MSE":MSE,
            "RSME":RSME,
            "r2":r2}
        logger.info("eval is done")
        return metrics
    except Exception as e:
        logger.error("error while eval model ",e)


def save_metrics(metrics: dict, file_path: str) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logger.debug('Metrics saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the metrics: %s', e)
        raise
    


def save_model(full_pipeline):
    try:
        joblib.dump(full_pipeline,r"C:\Users\nice\Desktop\Big Data project\Mobile-Price-prediction\model\full_pipline.pkl")
        logger.info("model is saved")
    except Exception as e:
        logger.error("error while save model")
        

if __name__ == "__main__":
    try:
        data_path = r"C:\Users\nice\Desktop\Big Data project\Mobile-Price-prediction\Data\proccessed\processed_data.csv" 
        df = load_Data(data_path)
        x_train, x_val, y_train, y_val, x_test, y_test=spilt_data(df)
        pipeline=create_pipeline()
        pipeline=train_model(pipeline, x_train, y_train, x_val, y_val, x_test)
        metric=eval_model(pipeline, y_test, x_test)
        save_model(pipeline)
        save_metrics(metric,"reports/metric.json")
        logger.info("All steps completed successfully")
    except Exception as main_e:
        logger.critical("Pipeline failed at some point: %s", main_e)
    











