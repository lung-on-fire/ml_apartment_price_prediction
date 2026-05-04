# Housing Price Prediction using Linear Regression and Gradient Boosting

This notebook demonstrates a machine learning pipeline for predicting housing prices, primarily focusing on **custom implementations of linear regression (including regularized versions) and various gradient boosting models**. The dataset used is from the Two Sigma Connect: Rental Listing Inquiries competition on Kaggle.

## Project Goal
To build and evaluate various regression models for housing price prediction, emphasizing both hands-on implementation of core machine learning algorithms and the application of established libraries.

## Dataset
The dataset is originally from the Two Sigma Connect: Rental Listing Inquiries competition on Kaggle.

## Notebook Contents:

*   **Imports**: Necessary libraries for data manipulation, visualization, and machine learning are imported.
*   **Data Loading and Initial Exploration**: The `train.json` and `test.json` datasets are downloaded and extracted. Initial data inspection using `info()` and `head()` is performed.
*   **Data Preprocessing**:
    *   Identification of target variable (`price`) and feature types.
    *   Dropping irrelevant columns such as `display_address`, `street_address`, `building_id`, `created`, `listing_id`, `manager_id`, `photos`, and `description`.
    *   Handling outliers in `price`, `bathrooms`, and `bedrooms`.
    *   Encoding `interest_level` as numerical categories.
*   **Target Analysis**: Distribution of the `price` variable is visualized (histograms, box plots), and logarithmic transformation is applied to achieve a more normal distribution for modeling.
*   **Feature Engineering**:
    *   Creation of new features based on `bathrooms` and `bedrooms` (e.g., `bathrooms_bedrooms`, `bathrooms_more_2`, `bedrooms_more_2`).
    *   Calculation of distance to Manhattan's center (`dist_to_center`) and its logarithmic transformation (`dist_log`).
    *   Extraction and one-hot encoding of the top 20 most frequent features from the `features` column.
    *   Clustering geographical coordinates (`latitude`, `longitude`) using K-Means to create a `cluster_label` feature.
*   **Model Training and Evaluation**: The dataset is split into training and validation sets. Different regression models are trained and evaluated:
    *   **Custom Linear Regression**: Implementation of a linear regression model with `fit` and `predict` methods, utilizing different optimization modes: analytical solution, non-stochastic gradient descent, and stochastic gradient descent.
    *   **Regularized Models (Custom)**: Extension of the custom linear regression to include L1 (Lasso), L2 (Ridge), and ElasticNet regularization.
    *   **CatBoostRegressor**: A gradient boosting model, initially trained on all engineered features, then on a selected subset of features.
    *   **Scikit-learn Linear Regression**: Trained with polynomial features and one-hot encoded categorical features (for comparison with custom LR).
    *   **Decision Tree Regressor**: Used to demonstrate overfitting in a simple tree model.
    *   **Random Forest Regressor**: Evaluated with default and regularized parameters to mitigate overfitting.
    *   **Naive Models**: Baseline models predicting mean or median price for comparison.
    *   **Hyperparameter Optimization**: Attempts to optimize CatBoostRegressor using Optuna and RandomizedSearchCV are demonstrated.
*   **Normalization Techniques**: Implementation and comparison of custom `MinMaxScaler` and `StandardScaler` against scikit-learn's versions are included within the model pipelines.

## Key Findings:

*   The **CatBoostRegressor** consistently performed best among the models, achieving the lowest MAE and RMSE on both training and validation sets without significant overfitting.
*   The **custom implementations of linear regression**, especially with the analytical solution, closely match the performance of scikit-learn's `LinearRegression`.
*   Gradient descent-based methods often require careful tuning of hyperparameters (learning rate, iterations) for stability and optimal convergence, especially after feature scaling.
*   Linear Regression performed reasonably but was outperformed by tree-based models.
*   Decision Trees and Random Forests showed signs of overfitting with default parameters, which could be improved through regularization and hyperparameter tuning.
*   Regularization proves effective in controlling overfitting when dealing with complex feature sets, such as high-degree polynomial features.
*   Normalization (MinMaxScaler, StandardScaler) significantly impacts the stability and performance of gradient descent-based models.
*   All trained models significantly outperformed naive baseline models.
*   Among the compared linear models, `Lasso` and `scikit-learn Linear Regression` with appropriate scaling often showed strong performance.

## Project Structure

This project is organized into several key components to facilitate the development and evaluation of linear regression models:

*   **`help_functions.py`**: This module contains auxiliary functions used throughout the notebook for various tasks such as data loading, preprocessing, and potentially custom evaluation metrics or utility operations that are not part of the core model implementations.

*   **`my_linreg.py`**: This module is central to the project, housing the custom implementation of the `MyLinearRegression` class. This class provides different modes for solving linear regression (analytical, non-stochastic gradient descent, and stochastic gradient descent) and also incorporates various regularization techniques (L1, L2, ElasticNet).

*   **Jupyter Notebook (`.ipynb`)**: The main notebook orchestrates the entire machine learning pipeline, from data loading and preprocessing to model training, evaluation, and comparison. It integrates the custom modules (`help_functions.py`, `my_linreg.py`) with standard scikit-learn functionalities.

---

# Предсказание цен на жилье с использованием линейной регрессии и градиентного бустинга

Этот ноутбук демонстрирует пайплайн машинного обучения для предсказания цен на жилье, в основном фокусируясь на **собственных реализациях линейной регрессии (включая регуляризованные версии) и различных моделях градиентного бустинга**. Набор данных взят из соревнования Two Sigma Connect: Rental Listing Inquiries на Kaggle.

## Цель проекта
Создать и оценить различные регрессионные модели для предсказания цен на жилье, акцентируя внимание как на практической реализации основных алгоритмов машинного обучения, так и на применении стандартных библиотек.

## Набор данных
Набор данных взят из соревнования Two Sigma Connect: Rental Listing Inquiries на Kaggle.

## Содержание ноутбука:

*   **Импорты**: Импортируются необходимые библиотеки для манипуляции данными, визуализации и машинного обучения.
*   **Загрузка данных и первоначальное исследование**: Наборы данных `train.json` и `test.json` загружаются и извлекаются. Выполняется первоначальный осмотр данных с использованием `info()` и `head()`.
*   **Предварительная обработка данных**:
    *   Идентификация целевой переменной (`price`) и типов признаков.
    *   Удаление нерелевантных столбцов, таких как `display_address`, `street_address`, `building_id`, `created`, `listing_id`, `manager_id`, `photos` и `description`.
    *   Обработка выбросов в `price`, `bathrooms` и `bedrooms`.
    *   Кодирование `interest_level` как числовых категорий.
*   **Анализ целевой переменной**: Визуализируется распределение переменной `price` (гистограммы, бокс-плоты), и применяется логарифмическое преобразование для достижения более нормального распределения для моделирования.
*   **Разработка признаков**:
    *   Создание новых признаков на основе `bathrooms` и `bedrooms` (например, `bathrooms_bedrooms`, `bathrooms_more_2`, `bedrooms_more_2`).
    *   Вычисление расстояния до центра Манхэттена (`dist_to_center`) и его логарифмическое преобразование (`dist_log`).
    *   Извлечение и one-hot кодирование 20 наиболее часто встречающихся признаков из столбца `features`.
    *   Кластеризация географических координат (`latitude`, `longitude`) с использованием K-Means для создания признака `cluster_label`.
*   **Обучение и оценка моделей**: Набор данных разделяется на обучающую и валидационную выборки. Обучаются и оцениваются различные регрессионные модели:
    *   **Пользовательская линейная регрессия**: Реализация модели линейной регрессии с методами `fit` и `predict`, использующая различные режимы оптимизации: аналитическое решение, нестохастический градиентный спуск и стохастический градиентный спуск.
    *   **Регуляризованные модели (Пользовательские)**: Расширение пользовательской линейной регрессии для включения регуляризации L1 (Lasso), L2 (Ridge) и ElasticNet.
    *   **CatBoostRegressor**: Модель градиентного бустинга, первоначально обученная на всех разработанных признаках, затем на выбранном подмножестве признаков.
    *   **Линейная регрессия Scikit-learn**: Обучена с полиномиальными признаками и one-hot закодированными категориальными признаками (для сравнения с пользовательской LR).
    *   **Регрессор дерева решений (Decision Tree Regressor)**: Используется для демонстрации переобучения в простой модели дерева.
    *   **Регрессор случайного леса (Random Forest Regressor)**: Оценивается с параметрами по умолчанию и регуляризованными параметрами для уменьшения переобучения.
    *   **Наивные модели**: Базовые модели, предсказывающие среднюю или медианную цену для сравнения.
    *   **Оптимизация гиперпараметров**: Демонстрируются попытки оптимизации CatBoostRegressor с использованием Optuna и RandomizedSearchCV.
*   **Методы нормализации**: Реализация и сравнение пользовательских `MinMaxScaler` и `StandardScaler` с версиями из scikit-learn включены в пайплайны моделей.

## Структура проекта

Этот проект организован в несколько ключевых компонентов для облегчения разработки и оценки моделей линейной регрессии:

*   **`help_functions.py`**: Этот модуль содержит вспомогательные функции, используемые на протяжении всего ноутбука для различных задач, таких как загрузка данных, предобработка и, возможно, пользовательские метрики оценки или служебные операции, которые не являются частью основных реализаций модели.

*   **`my_linreg.py`**: Этот модуль является центральным в проекте, содержащим пользовательскую реализацию класса `MyLinearRegression`. Этот класс предоставляет различные режимы для решения линейной регрессии (аналитическое решение, нестохастический градиентный спуск и стохастический градиентный спуск), а также включает различные методы регуляризации (L1, L2, ElasticNet).

*   **Jupyter Notebook (`.ipynb`)**: Основной ноутбук организует весь пайплайн машинного обучения, от загрузки и предобработки данных до обучения, оценки и сравнения моделей. Он интегрирует пользовательские модули (`help_functions.py`, `my_linreg.py`) со стандартными функциями scikit-learn.


