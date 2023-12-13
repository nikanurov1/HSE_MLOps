from pydantic import BaseModel


class AddModelRequest(BaseModel):
    task: str
    model_name: str
    hyperparameters: dict


class TrainingRequest(BaseModel):
    task: str
    model_name: str
    data: dict
    data_name: str


class PredictionRequest(BaseModel):
    task: str
    model_name: str
    data: dict


class DropModelRequest(BaseModel):
    task: str
    model_name: str
