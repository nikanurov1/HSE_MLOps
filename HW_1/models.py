import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from typing import Literal, Tuple, Union


class Models:
    def __init__(self):
        self.models = {'regression': {'models': {},
                                     'cnt': 0},
                       'classification': {'models': {},
                                          'cnt': 0}}
        
    def available_models(self) -> dict:
        """
        Данный метод возвращает количество моделей и их количество для каждого классса

        Returns:
        """
        return {'regression': {"Количество": self.models['regression']['cnt'], "Модели": [*self.models['regression']['models'].keys()]},
                                 'classification': {"Количество": self.models['classification']['cnt'], "Модели": [*self.models['classification']['models'].keys()]}}
        
    def add_model(self, task: Literal['classification', 'regression'], model_name: str, hypeparams: dict={}) -> None:
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
        if model_name in self.models[task]['models'].keys():
            raise Exception('Модель с таким именем существует - добавьте модель с другим именем или удалите уже существующую')
        if task == 'regression':
            self.models[task]['models'][model_name] = {'model': LinearRegression(**hypeparams),  
                                        'is trained': False}
            self.models[task]['cnt'] += 1
        else: 
            self.models[task]['models'][model_name] = {'model': LogisticRegression(**hypeparams), 
                                       'is trained': False}
            self.models[task]['cnt'] += 1
        
    def prepare_data(self, data:dict, mod: Literal['train', 'predict']='train') -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Подготавливает входные данные в соответствии с указанным режимом.

        Args:
            data (dict): Словарь с данными для подготовки.
            mod (Literal['train', 'predict'], optional): Режим подготовки данных. Доступны режимы:
                - 'train': Данные будут разделены на признаки (X) и целевую переменную (y). По умолчанию.
                - 'predict': Вернет данные в виде DataFrame.

        Returns:
            - pd.DataFrame: Если mod установлен в 'predict'.
            - tuple: Если mod установлен в 'train'. Возвращает кортеж из признаков (X) и целевой переменной (y).
        """
        if mod == 'predict':
            return pd.DataFrame(data)
        else:
            self.data = pd.DataFrame(data)
            X = self.data.iloc[:, : -1]  
            y = self.data.iloc[:, -1] 
            return X, y  

    
    def train(self, task: Literal['classification', 'regression'], model_name: str, data: dict) -> None:
        """
        Обучает указанную модель на предоставленных данных.

        Args:
            task (Literal['classification', 'regression']): Тип задачи, который может быть 'classification' или 'regression'.
            model_name (str): Имя модели из словаря self.models для обучения.
            data (dict): Словарь с данными для обучения. Формат данных должен соответствовать ожиданиям функции self.prepare_data.

        Raises:
            KeyError: Если указанное имя модели или тип задачи отсутствует в словаре self.models.

        Side Effects:
            - Обновляет состояние модели в словаре self.models.
            - Выводит информацию о словаре self.models после обучения.

        Returns:
            None
        """
        _X, _y = self.prepare_data(data, )
        train_model = self.models[task]['models'][model_name]
        train_model['model'].fit(_X, _y)
        train_model['is trained'] = True

    def predict(self, task: Literal['classification', 'regression'], model_name: str, data: dict) -> list:
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
        _X = self.prepare_data(data, mod='predict')
        pred_model = self.models[task]['models'][model_name]
        if pred_model['is trained']:
            return pred_model['model'].predict(_X).tolist()
        else:
            raise Exception("Model has not been trained yet :(")
        
    def drop_model(self, task: Literal['classification', 'regression'], model_name: str) -> None:
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
        del self.models[task]['models'][model_name]
        self.models[task]['cnt'] -= 1


        
