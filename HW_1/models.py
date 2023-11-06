import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from typing import Literal, Tuple, Union


class Models:
    def __init__(self):
        self.models = {
            "regression": {"models": {}, "cnt": 0},
            "classification": {"models": {}, "cnt": 0},
        }

    def available_models(self) -> dict:
        """
        Данный метод возвращает количество моделей и их количество для каждого класса

        Returns:
            dict: словарь с доступными
        """
        return {
            "regression": {
                "Количество": self.models["regression"]["cnt"],
                "Модели": [*self.models["regression"]["models"].keys()],
            },
            "classification": {
                "Количество": self.models["classification"]["cnt"],
                "Модели": [*self.models["classification"]["models"].keys()],
            },
        }

    def add_model(
        self,
        task: Literal["classification", "regression"],
        model_name: str,
        hypeparams: dict = {},
    ) -> None:
        """
        Данный метод инициализирует модель и добавляет ее в список моделей

        Args:
            task (Literal['classification', 'regression']): Тип задачи, который может быть 'classification' или 'regression'
            model_name (str): Название модели
            hyperparams (dict, optional): Гиперпараметры

        Raises:
            ValueError: Если неверно передана решаемая моделью задача

        Returns:
            None
        """
        if model_name in self.models[task]["models"].keys():
            raise ValueError(
                "Модель с таким именем существует - добавьте модель с другим именем или удалите уже существующую"
            )
        if task == "regression":
            try:
                self.models[task]["models"][model_name] = {
                    "model": LinearRegression(**hypeparams),
                    "is trained": False,
                }
            except TypeError:
                raise ValueError('Введите корректные гипарпараметры для модели регрессии')
            self.models[task]["cnt"] += 1
        else:
            try:
                self.models[task]["models"][model_name] = {
                    "model": LogisticRegression(**hypeparams),
                    "is trained": False
                }
            except TypeError:
                raise ValueError('Введите корректные гипарпараметры для модели классификации')
            self.models[task]["cnt"] += 1

    def prepare_data(
        self, data: dict, mod: Literal["train", "predict"] = "train"
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Подготавливает входные данные в соответствии с указанным режимом.

        Args:
            data (dict): Словарь с данными для подготовки.
            mod (Literal['train', 'predict'], optional): Режим подготовки данных. Доступны режимы:
                - 'train': Данные будут разделены на признаки (X) и целевую переменную (y). По умолчанию.
                - 'predict': Вернет данные в виде DataFrame.

        Returns:
            pd.DataFrame: Если mod установлен в 'predict'.
            tuple: Если mod установлен в 'train'. Возвращает кортеж из признаков (X) и целевой переменной (y).
        """
        df = pd.DataFrame(data)
        if mod == "predict":
            return df
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        return X, y

    def train(
        self, task: Literal["classification", "regression"], model_name: str, data: dict
    ) -> None:
        """
        Обучает указанную модель на предоставленных данных.

        Args:
            task (Literal['classification', 'regression']): Тип задачи, который может быть 'classification' или 'regression'.
            model_name (str): Имя модели из словаря self.models для обучения.
            data (dict): Словарь с данными для обучения. Формат данных должен соответствовать ожиданиям функции self.prepare_data.

        Raises:
            KeyError: Если указанное имя модели или тип задачи отсутствует в словаре self.models.

        Returns:
            None
        """
        X, y = self.prepare_data(
            data,
        )
        model_data = self.models[task]["models"].get(model_name)
        if not model_data:
            raise KeyError(f"Модель с именем '{model_name}' не найдена для задачи '{task.value}'")
        model_data["model"].fit(X, y)
        model_data["is trained"] = True

    def predict(
        self, task: Literal["classification", "regression"], model_name: str, data: dict
    ) -> list:
        """
        Предсказывает результаты на основе обученной модели.

        Args:
            task (Literal['classification', 'regression']): Тип задачи, который может быть 'classification' или 'regression'.
            model_name (str): Имя модели из словаря self.models для выполнения предсказаний.
            data (dict): Словарь с данными для предсказания. Формат данных должен соответствовать ожиданиям функции self.prepare_data.

        Raises:
            KeyError: Если указанное имя модели или тип задачи отсутствует в словаре self.models.
            Exception: Если модель еще не была обучена.

        Returns:
            list: Список предсказанных значений.
        """
        X = self.prepare_data(data, mod="predict")
        model_data = self.models[task]["models"].get(model_name)
        if not model_data:
            raise KeyError(f"Модель с именем '{model_name}' не найдена для задачи '{task.value}'")
        if not model_data["is trained"]:
            raise Exception("Модель еще не обучена :(")
        return model_data["model"].predict(X).tolist()

    def drop_model(
        self, task: Literal["classification", "regression"], model_name: str
    ) -> None:
        """
        Удаляет указанную модель из словаря моделей и уменьшает счетчик моделей для заданной задачи.

        Args:
            task (Literal['classification', 'regression']): Тип задачи, который может быть 'classification' или 'regression'.
            model_name (str): Имя модели из словаря self.models, которую необходимо удалить.

        Raises:
            KeyError: Если указанное имя модели или тип задачи отсутствует в словаре self.models.


        Returns:
            None
        """
        if model_name in self.models[task]["models"]:
            del self.models[task]["models"][model_name]
            self.models[task]["cnt"] -= 1
        else:
            raise KeyError(f"Модель с именем '{model_name}' не найдена для задачи '{task.value}'")
