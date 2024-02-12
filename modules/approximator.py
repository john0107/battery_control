import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.extmath import safe_sparse_dot


class customLinearRegression(LinearRegression):
    """customized Linear regression model from sklearn 
    predict function is modified to allow for prediction of targets specified by indices.
    """

    def predict(self, X, target_index=None):
        """
        Predict using the linear model.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            Samples.
        indices: array-like, shape (n_targets, )
            indices of targets to predict

        Returns
        -------
        C : array, shape (n_samples,)
            Returns predicted values.
        """
        check_is_fitted(self)

        X = self._validate_data(
            X, accept_sparse=["csr", "csc", "coo"], reset=False)
        if isinstance(target_index, int):
            assert self.coef_.ndim == 2, self.coef.shape
            return safe_sparse_dot(X, self.coef_[[target_index], :].squeeze(axis=0).T, dense_output=True) + self.intercept_[target_index]
        elif isinstance(target_index, list):
            assert self.coef_.ndim == 2, self.coef.shape
            return safe_sparse_dot(X, self.coef_[target_index, :].T, dense_output=True) + self.intercept_[target_index]
        else:
            return safe_sparse_dot(X, self.coef_.T, dense_output=True) + self.intercept_


class BaseApproximator():
    def __init__(self, environment):
        self.environment = environment
        # save information from environment for easier access
        # get all available control states
        self.control_set = self.environment.control_set
        self.n_controls = len(self.control_set)
        self.n_steps = self.environment.n_steps  # get number of steps
        self.y_dim = self.environment.get_dim()

    def fit(self, step, y, X, target):
        raise NotImplementedError

    def predict(self, step, y, X):
        raise NotImplementedError


class RegressionModel(BaseApproximator):
    def __init__(self, environment, transformer, score=False):
        super(RegressionModel, self).__init__(environment)
        self.transformer = transformer
        self.function = {}
        if score:
            self.scores = {}

    def fit(self, step, features, targets):
        '''
        Parameters:
        -----------
        step: int
        X: np.ndarray with dimensions (n_samples, x_dim, n_steps)
        targets: np.ndarray with dimensions (n_samples, n_targets)
        '''
        model = customLinearRegression(
            fit_intercept=False, copy_X=False)  # create model
        # save fitted model in self.function
        self.function[step] = model.fit(X=features, y=targets)
        self.function[step].coef_ = self.function[step].coef_.copy()  # wtf
        if hasattr(self, 'scores'):
            self.scores[step] = model.score(X=features, y=targets)

    def predict(self, step, control, X, features, y_dim_axis=1):
        '''
        Parameters:
        -----------
        step: int
        control: np.ndarray (n_samples, y_dim, ...)
        X: np.ndarray (n_samples, x_dim, n_steps)
        features: np.ndarray (n_samples, n_features)

        Returns:
        --------
        prediction: np.ndarray (n_samples, n_actions) or (n_samples, n_controls, n_actions)
        '''
        # check if parameters are valid
        # check if step is valid
        assert isinstance(
            step, (int, np.integer)) and 0 <= step and step < self.n_steps, f'step {step} is invalid'
        # check if control axis has right dimensions
        assert control.shape[y_dim_axis] == self.environment.y_dim
        # check if y_dim_axis is valid
        assert y_dim_axis < control.ndim, (y_dim_axis, control.ndim)

        # get interpolation controls for control
        interpolation_controls, interpolation_coeffs = self.environment.get_interpolation_controls(
            control)
        assert interpolation_controls.shape == control.shape + (4, )

        # compute prediction for all controls in control set
        prediction_values = self.function[step].predict(features)

        # convert controls to index
        nan_value = -1  # ensure that this value is valid index for convenience, BUT not contained in the output of convert_control_to_index
        interpolation_controls_idx = self.environment.convert_control_to_index(interpolation_controls,
                                                                               y_dim_axis=y_dim_axis,
                                                                               nan_value=nan_value
                                                                               ).squeeze(axis=y_dim_axis)  # remove axis y_dim_axis
        assert interpolation_controls_idx.shape == control.shape[:y_dim_axis] + \
            control.shape[y_dim_axis+1:] + \
            (4, ), interpolation_controls_idx.shape

        # create final prediction
        final_prediction = 0
        for i in range(4):  # loop over all interpolation controls instead of computing all predictions at once, saves memory by factor 4
            prediction = prediction_values[
                np.expand_dims(np.arange(X.shape[0]), axis=tuple(
                    range(1, interpolation_controls_idx.ndim-1))),
                interpolation_controls_idx[..., i]]
            prediction[np.broadcast_to(interpolation_controls_idx[..., i] == nan_value,
                                       shape=prediction.shape)] = np.NaN
            final_prediction += prediction * \
                interpolation_coeffs.squeeze(axis=y_dim_axis)[..., i]
        return final_prediction
