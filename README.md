# Housing Price Prediction using Linear Regression and Gradient Boosting

This notebook demonstrates a machine learning pipeline for predicting housing prices, primarily focusing on linear regression and gradient boosting models. The dataset used is from the Two Sigma Connect: Rental Listing Inquiries competition on Kaggle.

## Notebook Contents:

1.  **Imports**: Necessary libraries for data manipulation, visualization, and machine learning are imported.
2.  **Data Loading and Initial Exploration**: The `train.json` and `test.json` datasets are downloaded and extracted. Initial data inspection using `info()` and `head()` is performed.
3.  **Data Preprocessing**: 
    *   Identification of target variable (`price`) and feature types.
    *   Dropping irrelevant columns such as `display_address`, `street_address`, `building_id`, `created`, `listing_id`, `manager_id`, `photos`, and `description`.
    *   Handling outliers in `price`, `bathrooms`, and `bedrooms`.
    *   Encoding `interest_level` as numerical categories.
4.  **Target Analysis**: Distribution of the `price` variable is visualized (histograms, box plots), and logarithmic transformation is applied to achieve a more normal distribution for modeling.
5.  **Feature Engineering**: 
    *   Creation of new features based on `bathrooms` and `bedrooms` (e.g., `bathrooms_bedrooms`, `bathrooms_more_2`, `bedrooms_more_2`).
    *   Calculation of distance to Manhattan's center (`dist_to_center`) and its logarithmic transformation (`dist_log`).
    *   Extraction and one-hot encoding of the top 20 most frequent features from the `features` column.
    *   Clustering geographical coordinates (latitude, longitude) using K-Means to create a `cluster_label` feature.
6.  **Model Training and Evaluation**: The dataset is split into training and validation sets. Different regression models are trained and evaluated:
    *   **CatBoostRegressor**: A gradient boosting model, initially trained on all engineered features, then on a selected subset of features.
    *   **Linear Regression**: Trained with polynomial features and one-hot encoded categorical features.
    *   **Decision Tree Regressor**: Used to demonstrate overfitting in a simple tree model.
    *   **Random Forest Regressor**: Evaluated with default and regularized parameters to mitigate overfitting.
    *   **Naive Models**: Baseline models predicting mean or median price for comparison.
7.  **Hyperparameter Optimization**: Attempts to optimize CatBoostRegressor using Optuna and RandomizedSearchCV are demonstrated.

## Key Findings:

*   The CatBoostRegressor consistently performed best among the models, achieving the lowest MAE and RMSE on both training and validation sets without significant overfitting.
*   Linear Regression performed reasonably but was outperformed by tree-based models.
*   Decision Trees and Random Forests showed signs of overfitting with default parameters, which could be improved through regularization and hyperparameter tuning.
*   All trained models significantly outperformed naive baseline models.

## Dataset:

The dataset is originally from the [Two Sigma Connect: Rental Listing Inquiries competition on Kaggle](https://www.kaggle.com/competitions/two-sigma-connect-rental-listing-inquiries/overview).



```markdown
# Housing Price Prediction using Linear Regression and Gradient Boosting

This notebook demonstrates a machine learning pipeline for predicting housing prices, primarily focusing on linear regression and gradient boosting models. The dataset used is from the Two Sigma Connect: Rental Listing Inquiries competition on Kaggle.

## Notebook Contents:

1.  **Imports**: Necessary libraries for data manipulation, visualization, and machine learning are imported.
2.  **Data Loading and Initial Exploration**: The `train.json` and `test.json` datasets are downloaded and extracted. Initial data inspection using `info()` and `head()` is performed.
3.  **Data Preprocessing**: 
    *   Identification of target variable (`price`) and feature types.
    *   Dropping irrelevant columns such as `display_address`, `street_address`, `building_id`, `created`, `listing_id`, `manager_id`, `photos`, and `description`.
    *   Handling outliers in `price`, `bathrooms`, and `bedrooms`.
    *   Encoding `interest_level` as numerical categories.
4.  **Target Analysis**: Distribution of the `price` variable is visualized (histograms, box plots), and logarithmic transformation is applied to achieve a more normal distribution for modeling.
5.  **Feature Engineering**: 
    *   Creation of new features based on `bathrooms` and `bedrooms` (e.g., `bathrooms_bedrooms`, `bathrooms_more_2`, `bedrooms_more_2`).
    *   Calculation of distance to Manhattan's center (`dist_to_center`) and its logarithmic transformation (`dist_log`).
    *   Extraction and one-hot encoding of the top 20 most frequent features from the `features` column.
    *   Clustering geographical coordinates (latitude, longitude) using K-Means to create a `cluster_label` feature.
6.  **Model Training and Evaluation**: The dataset is split into training and validation sets. Different regression models are trained and evaluated:
    *   **CatBoostRegressor**: A gradient boosting model, initially trained on all engineered features, then on a selected subset of features.
    *   **Linear Regression**: Trained with polynomial features and one-hot encoded categorical features.
    *   **Decision Tree Regressor**: Used to demonstrate overfitting in a simple tree model.
    *   **Random Forest Regressor**: Evaluated with default and regularized parameters to mitigate overfitting.
    *   **Naive Models**: Baseline models predicting mean or median price for comparison.
7.  **Hyperparameter Optimization**: Attempts to optimize CatBoostRegressor using Optuna and RandomizedSearchCV are demonstrated.

## Key Findings:

*   The CatBoostRegressor consistently performed best among the models, achieving the lowest MAE and RMSE on both training and validation sets without significant overfitting.
*   Linear Regression performed reasonably but was outperformed by tree-based models.
*   Decision Trees and Random Forests showed signs of overfitting with default parameters, which could be improved through regularization and hyperparameter tuning.
*   All trained models significantly outperformed naive baseline models.

## Dataset:

The dataset is originally from the [Two Sigma Connect: Rental Listing Inquiries competition on Kaggle](https://www.kaggle.com/competitions/two-sigma-connect-rental-listing-inquiries/overview).

---

# Прогнозирование цен на жилье с использованием линейной регрессии и градиентного бустинга

Этот ноутбук демонстрирует пайплайн машинного обучения для прогнозирования цен на жилье, в основном ориентированный на модели линейной регрессии и градиентного бустинга. Используемый набор данных взят из соревнования Two Sigma Connect: Rental Listing Inquiries на Kaggle.

## Содержание ноутбука:

1.  **Импорты**: Импортируются необходимые библиотеки для манипуляции данными, визуализации и машинного обучения.
2.  **Загрузка данных и первоначальное исследование**: Загружаются и распаковываются наборы данных `train.json` и `test.json`. Проводится первоначальный анализ данных с использованием `info()` и `head()`.
3.  **Предварительная обработка данных**: 
    *   Идентификация целевой переменной (`price`) и типов признаков.
    *   Удаление нерелевантных столбцов, таких как `display_address`, `street_address`, `building_id`, `created`, `listing_id`, `manager_id`, `photos` и `description`.
    *   Обработка выбросов в `price`, `bathrooms` и `bedrooms`.
    *   Кодирование `interest_level` как числовых категорий.
4.  **Анализ целевой переменной**: Визуализируется распределение переменной `price` (гистограммы, боксплоты) и применяется логарифмическое преобразование для достижения более нормального распределения для моделирования.
5.  **Создание признаков**: 
    *   Создание новых признаков на основе `bathrooms` и `bedrooms` (например, `bathrooms_bedrooms`, `bathrooms_more_2`, `bedrooms_more_2`).
    *   Расчет расстояния до центра Манхэттена (`dist_to_center`) и его логарифмическое преобразование (`dist_log`).
    *   Извлечение и one-hot кодирование 20 наиболее частых признаков из столбца `features`.
    *   Кластеризация географических координат (широты, долготы) с использованием K-Means для создания признака `cluster_label`.
6.  **Обучение и оценка модели**: Набор данных делится на обучающую и валидационную выборки. Обучаются и оцениваются различные регрессионные модели:
    *   **CatBoostRegressor**: Модель градиентного бустинга, первоначально обученная на всех разработанных признаках, затем на выбранном подмножестве признаков.
    *   **Линейная регрессия**: Обучена с полиномиальными признаками и one-hot кодированными категориальными признаками.
    *   **Дерево решений (Decision Tree Regressor)**: Используется для демонстрации переобучения в простой модели дерева.
    *   **Случайный лес (Random Forest Regressor)**: Оценивается с параметрами по умолчанию и регуляризованными параметрами для снижения переобучения.
    *   **Наивные модели**: Базовые модели, предсказывающие среднюю или медианную цену для сравнения.
7.  **Оптимизация гиперпараметров**: Демонстрируются попытки оптимизировать CatBoostRegressor с использованием Optuna и RandomizedSearchCV.

## Ключевые выводы:

*   CatBoostRegressor неизменно показывал лучшие результаты среди моделей, достигая наименьших MAE и RMSE на обучающей и валидационной выборках без значительного переобучения.
*   Линейная регрессия показала разумные результаты, но была превзойдена моделями на основе деревьев.
*   Деревья решений и случайные леса демонстрировали признаки переобучения с параметрами по умолчанию, что можно улучшить с помощью регуляризации и настройки гиперпараметров.
*   Все обученные модели значительно превзошли наивные базовые модели.

## Набор данных:

Набор данных изначально взят из соревнования [Two Sigma Connect: Rental Listing Inquiries на Kaggle](https://www.kaggle.com/competitions/two-sigma-connect-rental-listing-inquiries/overview).
