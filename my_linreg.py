import numpy as np


class MyLinearRegression:

  """Реализует модель линейной регрессии с возможностью использования различных режимов обучения
  (аналитическое решение, нестохастический и стохастический градиентный спуск - см. класс выше)
  и типов регуляризации (Lasso, Ridge, ElasticNet)."""

  def __init__(self, mode=None, penalty=None, alpha=0.1, lr=0.005, iters=1000, l1_ratio=0.5, clip_value=None):
    """Инициализирует параметры модели.
    `mode`: Режим обучения ("non-stochastic" для нестохастического градиентного спуска,
    "stochastic" для стохастического градиентного спуска, `None` для аналитического решения).
    `penalty`: Тип регуляризации ("lasso", "ridge", "elasticnet" или `None` для отсутствия регуляризации).
    `alpha`: Коэффициент регуляризации (гиперпараметр).
    `lr`: Скорость обучения (learning rate) для градиентных методов.
    `iters`: Количество итераций (для градиентных методов).
    `l1_ratio`: Соотношение L1 к L2 для ElasticNet регуляризации (0.0 для Ridge, 1.0 для Lasso).
    `clip_value`: Значение для отсечения градиентов (если `None`, отсечение не применяется).
    Инициализирует веса (`self.weights`) как `None` и свободный член (`self.b`) как 0.
    Устанавливает `np.random.seed` для воспроизводимости."""

    self.weights = None
    self.b = 0
    np.random.seed(21)
    self.mode = mode
    self.penalty = penalty
    self.alpha = alpha
    self.lr = lr
    self.iters = iters
    self.l1_ratio = l1_ratio
    self.clip_value = clip_value


  def fit(self, X, Y, intercept=True):
    """Обучает модель, подбирая оптимальные веса на основе входных данных `X` и целевых значений `Y`.
    `X`: Матрица признаков.
    `Y`: Вектор целевых значений.
    `intercept`: Флаг, указывающий, следует ли добавлять свободный член (перехват) в модель.
    Включает инициализацию весов, добавление единичного вектора для свободного члена (если `intercept=True`),
    и итеративный процесс обучения с использованием градиентного спуска
    (нестохастического или стохастического) или прямое аналитическое решение."""

    self.X = np.array(X)
    self.Y = np.array(Y)
    self.obs, self.feat = self.X.shape

    if intercept:
      self.X = np.c_[np.ones(self.obs), self.X] #добавляем единичный вектор
      self.weights = np.zeros(self.feat + 1) # матрица заполненнми нулями
      self.weights[0] = np.mean(self.Y) # b - нач инициализация
      self.weights[1:] = np.random.randn(self.feat) * 0.01 # Small random values for features
    else:
      self.weights = np.random.randn(self.feat) * 0.01

    if self.mode == "analytical" or self.mode is None:
        self.weights = np.linalg.pinv(self.X.T @ self.X) @ self.X.T @ self.Y
        self.penalty = None

        if intercept:
            self.intercept_ = self.weights[0]
            self.coef_ = self.weights[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = self.weights
        return


    for i in range(self.iters):
        if self.mode=="non-stochastic":
            self.gradient_descent(self.X, self.Y)

        elif self.mode=="stochastic":
            self.stochastic_gradient_descent(self.X, self.Y)


    if intercept:
        self.intercept_ = self.weights[0]
        self.coef_ = self.weights[1:]
    else:
        self.intercept_ = 0.0
        self.coef_ = self.weights


  def gradient_descent(self, X, Y):
    """Выполняет один шаг нестохастического градиентного спуска.
    Рассчитывает градиент функции потерь (MSE плюс члены регуляризации, если применимо) по всему набору данных `X` и `Y` и обновляет веса.
    Включает L1 (Lasso), L2 (Ridge) и ElasticNet регуляризацию, если она указана."""

    self.weights = np.clip(self.weights, -1e10, 1e10)
    grad = None
    preds_array = X @ self.weights
    error_array = preds_array - Y
    grad_mse = (2/len(Y)) * (X.T @ error_array)
    has_intercept = X.shape[1] > self.feat

    if self.penalty == "lasso":
      grad_reg = self.alpha * np.sign(self.weights)
      if has_intercept:
        grad_reg[0] = 0
      grad = grad_reg + grad_mse

    elif self.penalty == 'ridge':
      grad_reg = 2 * self.alpha * self.weights
      if has_intercept:
        grad_reg[0] = 0
      grad = grad_reg + grad_mse

    elif self.penalty == "elasticnet":
      l1_grad = self.alpha * self.l1_ratio * np.sign(self.weights)
      l2_grad = self.alpha * (1 - self.l1_ratio) * self.weights
      grad_reg = l1_grad + l2_grad
      if has_intercept:
        grad_reg[0] = 0
      grad = grad_mse + grad_reg

    elif self.penalty == None:
      grad = grad_mse

    else:
      raise Exception("Допустимые значения для параметра penalty: None, lasso, ridge, elasticnet")

    if self.clip_value is not None:
        grad = np.clip(grad, -self.clip_value, self.clip_value)

    self.weights -= self.lr * grad

  def stochastic_gradient_descent(self, X, Y):
    """Выполняет один шаг стохастического градиентного спуска (с использованием мини-батчей).
    Случайным образом выбирает мини-батч данных из `X` и `Y`, рассчитывает градиент функции потерь
    для этого батча и обновляет веса.
    Включает L1 (Lasso), L2 (Ridge) и ElasticNet регуляризацию, если она указана."""

    self.weights = np.clip(self.weights, -1e10, 1e10)

    batch_size = 32
    row_indices = np.random.choice(len(Y), batch_size, replace=False)
    X_batch = X[row_indices]
    Y_batch = Y[row_indices]

    preds_array = X_batch @ self.weights
    error_array = preds_array - Y_batch

    grad_mse = (2 / batch_size) * (X_batch.T @ error_array)
    has_intercept = X.shape[1] > self.feat

    if self.penalty == "lasso":
      grad_reg = self.alpha * np.sign(self.weights)
      if has_intercept:
        grad_reg[0] = 0
      grad = grad_reg + grad_mse

    elif self.penalty == 'ridge':
      grad_reg = 2 * self.alpha * self.weights
      if has_intercept:
        grad_reg[0] = 0
      grad = grad_reg + grad_mse

    elif self.penalty == "elasticnet":
      l1_grad = self.alpha * self.l1_ratio * np.sign(self.weights)
      l2_grad = 2 * self.alpha * (1 - self.l1_ratio) * self.weights
      grad_reg = l1_grad + l2_grad
      if has_intercept:
        grad_reg[0] = 0
      grad = grad_mse + grad_reg

    elif self.penalty == None:
      grad = grad_mse

    else:
      raise Exception("Допустимые значения для параметра penalty: None, lasso, ridge, elasticnet")

    if self.clip_value is not None:
        grad = np.clip(grad, -self.clip_value, self.clip_value)

    self.weights -= self.lr * grad



  def loss_function(self, X, Y):
    """Вычисляет значение функции потерь для текущих весов модели.
    Рассчитывает среднеквадратическую ошибку (MSE) и добавляет соответствующий член регуляризации
    (L1, L2 или ElasticNet), если он был указан при инициализации модели."""

    if self.weights is None:
      raise Exception("Модель ещё обучается")
    preds_array = X @ self.weights

    reg_weights = self.weights[1:] if self.weights.shape[0] > 1 else np.array([])
    loss = np.mean((preds_array - Y)** 2)

    if self.penalty=='elasticnet':
      loss += self.alpha * self.l1_ratio * np.sum(np.abs(reg_weights)) + 0.5 * self.alpha *(1-self.l1_ratio) * np.sum(reg_weights ** 2)
    elif self.penalty == 'lasso':
      loss += self.alpha * np.sum(np.abs(reg_weights))
    elif self.penalty == 'ridge':
      loss += 0.5 * self.alpha * np.sum(reg_weights ** 2)
    elif self.penalty == None:
      pass
    else:
      print("Допустимые значения для параметра penalty: None, lasso, ridge, elasticnet")
    return loss

  def predict(self, X, intercept=True):
    """ Делает предсказания для новых входных данных `X`.
    `X`: Матрица признаков для которых нужно сделать предсказания.
    `intercept`: Флаг, указывающий, следует ли учитывать свободный член при предсказании.
    Возвращает предсказанные значения."""

    if self.weights is None:
      raise Exception("Модель ещё обучается")

    X = np.array(X)
    if intercept is True:
      X = np.c_[np.ones(len(X)), X]
    return X @ self.weights