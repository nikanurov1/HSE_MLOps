from fastapi import FastAPI, HTTPException
from models import Models
from request import AddModelRequest, TrainingRequest, DropModelRequest, PredictionRequest
import uvicorn

app = FastAPI()
models = Models()


@app.post("/add_model")
def add_model(request: AddModelRequest):
    """
    Добавляет новую модель с заданными гиперпараметрами.

    Args:
        request (AddModelRequest): Объект запроса содержащий информацию о модели и ее гиперпараметрах.

    Returns:
        dict: Сообщение о успешном добавлении модели.
    """
    try:
        models.add_model(request.task, request.model_name, request.hyperparameters)
        return {"message": f"Модель {request.model_name} успешно добавлена"}
    except KeyError:
        raise HTTPException(
            status_code=400,
            detail="Передана некорректная задача или некорректные гиперпараметры",
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/train")
def train_model(request: TrainingRequest):
    """
    Обучает указанную модель на предоставленных данных.

    Args:
        request (TrainingRequest): Объект запроса содержащий информацию о модели и данных для обучения.

    Returns:
        dict: Сообщение о успешном обучении модели.
    """
    try:
        models.train(request.task, request.model_name, request.data)
        return {"message": f"Модель {request.model_name} успешно обучена"}
    except KeyError:
        raise HTTPException(
            status_code=400,
            detail="Модель не существует или передана некорректная задача",
        )


@app.post("/predict")
@app.get("/predict")
def predict_model(request: PredictionRequest):
    """
    Возвращает предсказания для предоставленных данных на основе обученной модели.

    Args:
        request (PredictionRequest): Объект запроса содержащий информацию о модели и данных для предсказания.

    Returns:
        dict: Содержит предсказанные значения.
    """
    try:
        prediction = models.predict(request.task, request.model_name, request.data)
        return {"prediction": prediction}
    except KeyError:
        raise HTTPException(
            status_code=400,
            detail="Модель не существует или передана некорректная задача",
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/drop_model")
def drop_model(request: DropModelRequest):
    """
    Удаляет указанную модель.

    Args:
        request (DropModelRequest): Объект запроса содержащий информацию о модели для удаления.

    Returns:
        dict: Сообщение о успешном удалении модели.
    """
    try:
        models.drop_model(request.task, request.model_name)
        return {"message": f"Модель {request.model_name} успешно удалена"}
    except KeyError:
        raise HTTPException(
            status_code=400,
            detail="Модель не cуществует или передана некорректная задача",
        )


@app.get("/models")
@app.post("/models")
def return_models():
    """
    Возвращает список и количество доступных моделей
    """
    return models.available_models()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7755)
