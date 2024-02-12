import numpy as np
import warnings


######## Battery Model ########


class BatteryModel():
    '''Class for the battery model.

    Parameters
    ----------
    K_SoC : int
        Discretization of SoC component of control set.
    K_Cav : int, optional
        Discretization of (scaled) average C-rate component of control set, default is 1.
    K_Tac : int, optional
        Discretization of (scaled) active time component of control set, default is 1.
    Q_max : float, optional
        Maximum capacity of the battery in Ah or kWh, default is 1.
    C_min : float, optional
        Minimum C-rate of the battery, default is 0.
    C_max : float, optional
        Maximum C-rate of the battery, default is 1.
    running_cost : float, optional
        Parameter for running cost, default is 0.
    ageing_cost : float, optional
        Parameter for ageing cost, default is 0.
    battery_value : float, optional
        Value of the battery in € with full max capacity, default is 0.
    degradation_model : str, optional
        Name of the degradation model, default is 'no'. See degradation_models for all options.
    ignore_degradation_state: bool, optional
        If True, the degradation state is not considered in the state space, default is False.
    '''

    def __init__(self, K_SoC, K_Cav=1, K_Tac=1, Q_max=1, C_min=0, C_max=1, running_cost=0, ageing_cost=0, battery_value=0, degradation_model=None, ignore_degradation_state=False):
        # battery specifications
        # value of the battery in € with full max capacity
        self.battery_value = battery_value
        # maximum capacity of the battery in Ah or kWh; assume constant voltage U=1V
        self.Q_max = Q_max
        self.I_C = self.Q_max  # current to fully (dis)charge the battery in 1h
        # minimum and maximum C-rate of the battery
        self.C_min = C_min
        self.C_max = C_max

        self.running_cost = running_cost  # parameter for running cost
        self.ageing_cost = ageing_cost  # parameter for ageing cost

        degradation_models = {
            'old': OldDegradationModel,
            'old continuous': OldContinuousDegradationModel,
            'new': NewDegradationModel,
            'quad': QuadraticDegradationModel,
            'no': NoDegradationModel,
        }
        if degradation_model in degradation_models.keys():
            self.degradation_model = degradation_models[degradation_model](
                battery_value=self.battery_value,
                Q_max=self.Q_max,
                I_C=self.I_C,
                C_max=self.C_max,
                C_min=self.C_min,
                running_cost=self.running_cost,
                ageing_cost=self.ageing_cost,
            )
        else:
            raise KeyError(f'{degradation_model} not found')

        if degradation_model == 'no':  # ignore degradation state for no degradation model
            self.ignore_degradation_state = True
        else:  # do not ignore degradation state by default for all other degradation models
            self.ignore_degradation_state = ignore_degradation_state

        # control and action set
        self.round = 4  # rounding parameter for control and action set
        self.K_SoC = K_SoC  # discretization of control set
        self.K_Cav = K_Cav  # discretization of control set
        self.K_Tac = K_Tac  # discretization of control set
        # check if discretization is fine enough
        assert self.K_SoC >= 1 and self.K_Cav >= 1 and self.K_Tac >= 1

    def set_time_discretization(self, time_discretization):
        '''Set time discretization for degradation model.

        Parameters
        ----------
        time_discretization : TimeDiscretization
            Time discretization of the problem.
        '''
        # save time discretization and all its attributes
        self.time_discretization = time_discretization
        self.__dict__.update(self.time_discretization.get_attributes())
        # set time discretization for degradation_model
        self.degradation_model.set_time_discretization(
            self.time_discretization)

        self.y_SoC_grid = np.linspace(
            start=0, stop=1, num=self.K_SoC+1).round(self.round)
        self.C_av_grid = np.linspace(
            start=0, stop=1, num=self.K_Cav+1).round(self.round)
        self.T_ac_grid = np.linspace(
            start=0, stop=1, num=self.K_Tac+1).round(self.round)
        self.control_set = [
            np.array([y, C_av, T_ac])
            for y in self.y_SoC_grid
            for C_av in self.C_av_grid
            for T_ac in self.T_ac_grid
        ]  # the order is important, if changed, convert_control_to_index has to be changed as well
        # battery cannot bet discharged more than demand
        self.action_set = [
            a for a in np.linspace(start=-1/self.delta_t,
                                   stop=1/self.delta_t,
                                   num=2*self.K_SoC+1
                                   )  # no rounding here
            if np.abs(a) <= self.C_max  # C-rate cannot be larger than C_max
        ]

    def convert_control_to_index(self, control, y_dim_axis=1, nan_value=None):
        '''convert array of controls to corresponding index in self.control_set

        Parameters
        ----------
        control: np.ndarray (..., y_dim, ...)
        y_dim_axis: int, default = 1
            axis of control
        nan_value: int, default = n_controls
            fill value for np.NaN controls

        Returns
        -------
        indices: np.ndarray (..., 1, ...), dtype = int
            array with corresponding indices and similar shape to control
        '''
        y_SoC = control.take(indices=[0], axis=y_dim_axis)
        C_av = control.take(indices=[1], axis=y_dim_axis)
        T_ac = control.take(indices=[2], axis=y_dim_axis)
        indices = y_SoC * self.K_SoC * (self.K_Cav+1) * (self.K_Tac+1)
        indices += C_av * self.K_Cav * (self.K_Tac+1)
        indices += T_ac * self.K_Tac
        assert indices.shape == control.shape[:y_dim_axis] + \
            (1, ) + control.shape[y_dim_axis+1:], indices.shape
        assert ((0 <= indices[np.isfinite(indices)]) & (indices[np.isfinite(
            indices)] < len(self.control_set))).all(), indices[np.isfinite(indices)]
        nan_idx = np.isnan(indices)
        if nan_value is None:
            nan_value = len(self.control_set)
            if nan_idx.any():
                warnings.warn(
                    f'No nan_value set, automatically set to {nan_value}, proceed with caution!')
        indices[nan_idx] = nan_value
        indices = np.rint(indices).astype(int)
        return indices

    def compute_degradation_cost(self, step, action, control, **kwargs):
        '''compute degradation cost for given step, action and control

        Parameters
        ----------
        step: int
            current step
        action: np.ndarray (..., ..., n_actions)
            action for which the degradation cost is computed
        control: np.ndarray (..., y_dim, ...)
            control for which the degradation cost is computed
        **kwargs: dict
            additional keyword arguments for degradation model
        '''
        return self.degradation_model.compute_degradation_cost(step, action, control, **kwargs)

    def get_degradation_state(self, step, control, y_dim_axis=1):
        ''' compute degradation state for given step and control

        Parameters:
        ----------
        step: int
            current step
        control: nd.array with dimensions (..., y_dim, ...)
            control for which the degradation state is computed
        y_dim_axis: int, default = 1
            axis of control

        Returns:
        --------
        degradation_state: nd.array with dimensions (..., 1, ...)
        '''
        if self.ignore_degradation_state:  # if degradation state is ignored
            # return ones
            return np.ones(control.shape[:y_dim_axis] + (1, ) + control.shape[y_dim_axis+1:])
        else:  # otherwise compute degradation state from degradation model
            return self.degradation_model.get_degradation_state(step, control, y_dim_axis=1)

    def get_interpolation_controls(self, control, y_dim_axis=1):
        '''compute interpolation controls and coefficients for interpolation using bilinear interpolation

        Parameters
        ----------
        control: np.ndarray (..., y_dim, ...)
            control for which the interpolation controls and coefficients are computed
        y_dim_axis: int, default = 1
            axis of control

        Returns
        -------
        interpolation_controls: np.ndarray (.., y_dim, ..., 4)
            computed interpolation controls
        interpolation_coeffs: np.ndarray (..., 1, ..., 4)
            computed interpolation coefficients
        '''
        y_SoC = control.take(indices=[0], axis=y_dim_axis)
        C_av = control.take(indices=[1], axis=y_dim_axis)
        T_ac = control.take(indices=[2], axis=y_dim_axis)

        # assert y in np.linspace(start=0, stop=1, num=self.K+1).round(self.battery.round), f'{y}'
        for comp in [C_av, T_ac]:  # ensure that the values of C_av and T_ac are between 0 and 1
            assert (-0.01 <= comp[np.isfinite(comp)]
                    ).all(), comp[comp < 0.0].item(0)
            assert (comp[np.isfinite(comp)] <=
                    1.001).all(), comp[1.0 < comp].item(0)
            if not ((0.0 <= comp[np.isfinite(comp)]) & (comp[np.isfinite(comp)] <= 1.0)).all():
                # correct small rounding errors
                comp[comp > 1.0], comp[comp < 0.0] = 1.0, 0.0

        # compute supporting grid points for C_av and T_ac
        grid_points = {}
        for comp_label, comp, grid, K in zip(['C_av', 'T_ac'], [C_av, T_ac], [self.C_av_grid, self.T_ac_grid], [self.K_Cav, self.K_Tac]):
            # use searchsorted to obtain index of supporting grid points
            comp_idx = np.searchsorted(a=grid, v=comp, side='left')
            # index where component is NaN
            comp_idx_nan = (comp_idx == K+1)
            # set index where component should be NaN to some value for convenience, this will be corrected below
            comp_idx[comp_idx_nan] = K - 1
            assert ((0 <= comp_idx) & (comp_idx <= K)
                    ).all(), (grid, comp, comp_idx)
            # if component == 0.0, then comp_idx is 0 -> change to 1 so that comp_idx-1 is a valid index
            comp_idx[comp_idx == 0] += 1
            # get upper gridpoint for component
            comp_upper_gridpoint = grid[comp_idx]
            # get lower gridpoint for component
            comp_lower_gridpoint = grid[comp_idx-1]
            assert ((0 <= comp_upper_gridpoint) & (
                comp_upper_gridpoint <= 1)).all(), comp_upper_gridpoint
            assert ((0 <= comp_lower_gridpoint) & (
                comp_lower_gridpoint <= 1)).all(), comp_lower_gridpoint
            # correct where component should be NaN to NaN
            comp_upper_gridpoint[comp_idx_nan] = np.NaN
            # correct where component should be NaN to NaN
            comp_lower_gridpoint[comp_idx_nan] = np.NaN
            grid_points[comp_label] = {
                'upper': comp_upper_gridpoint,  # upper gridpoint for component
                'lower': comp_lower_gridpoint,  # lower gridpoint for component
            }

        # create interpolation controls and compute coefficients
        interpolation_controls, interpolation_coeffs = [], []
        normalization = (grid_points['C_av']['upper'] - grid_points['C_av']['lower']) * (
            grid_points['T_ac']['upper'] - grid_points['T_ac']['lower'])
        lower_upper = ['lower', 'upper']
        # loop over upper and lower gridpoint for C_av and T_ac
        for C_av_pos, T_ac_pos in [(1, 1), (1, 0), (0, 1), (0, 0)]:
            # create interpolation control for corresponding gridpoints
            interpolation_control = np.concatenate(
                [y_SoC, grid_points['C_av'][lower_upper[C_av_pos]], grid_points['T_ac'][lower_upper[T_ac_pos]]], axis=y_dim_axis)
            interpolation_controls += [
                np.expand_dims(interpolation_control, axis=-1)]
            # compute interpolation coefficient, see https://en.wikipedia.org/wiki/Bilinear_interpolation
            interpolation_coeff = (C_av - grid_points['C_av'][lower_upper[1-C_av_pos]]) * (
                T_ac - grid_points['T_ac'][lower_upper[1-T_ac_pos]]) / normalization
            if C_av_pos != T_ac_pos:  # if C_av_pos != T_ac_pos, then the coefficient is negative
                interpolation_coeff = -interpolation_coeff
            interpolation_coeffs += [
                np.expand_dims(interpolation_coeff, axis=-1)]

        interpolation_controls = np.concatenate(
            interpolation_controls, axis=-1)
        interpolation_coeffs = np.concatenate(interpolation_coeffs, axis=-1)
        assert interpolation_controls.shape == control.shape + \
            (4, ), (interpolation_controls.shape, control.shape)
        assert interpolation_coeffs.shape == control.shape[:y_dim_axis] + (
            1, ) + control.shape[y_dim_axis+1:] + (4, ), (interpolation_coeffs.shape, control.shape)

        return interpolation_controls, interpolation_coeffs


######## Degradaton Models ########


class BaseDegradationModel():
    def __init__(self, battery_value=0, Q_max=1, I_C=1, C_max=1, C_min=0, running_cost=0, ageing_cost=0):
        # battery specifications
        # value of the battery in € with full max capacity
        self.battery_value = battery_value
        self.Q_max = Q_max  # maximum capacity of the battery in Ah
        self.I_C = I_C  # current to fully (dis)charge the battery in 1h
        self.C_max = C_max  # maximum power of the battery
        self.C_min = C_min  # minimum power of the battery

        self.running_cost = running_cost  # parameter for running cost
        self.ageing_cost = ageing_cost  # parameter for ageing cost

    def set_time_discretization(self, time_discretization):
        self.time_discretization = time_discretization
        self.__dict__.update(self.time_discretization.get_attributes())

    def compute_degradation_cost(self, step, action, control):
        raise NotImplementedError

    def get_degradation_state(self, step, control, y_dim_axis=1):
        '''
        Parameters:
        ----------
        step: int
        control: np.ndarray (..., y_dim, ...)
        y_dim_axis: int, default=1

        Returns:
        --------
        degradation_state: np.ndarray (..., 1, ...)
        '''
        # get components from current control
        C_av = control.take(indices=[1], axis=y_dim_axis) / self.delta_t
        T_ac = control.take(indices=[2], axis=y_dim_axis) * step * self.delta_t
        y_tot = C_av * T_ac
        # compute degradation state
        input = y_tot / 2 * C_av
        degradation_state = 1 - self.ageing_cost * input
        assert np.all(degradation_state >=
                      0.0), f'input {input}, step {step}, control {control}'
        degradation_state = np.maximum(
            degradation_state, 0.0)  # set negative values to 0
        assert degradation_state.shape == control.shape[:y_dim_axis] + (
            1, ) + control.shape[y_dim_axis+1:], (degradation_state.shape, control.shape)
        return degradation_state


class OldDegradationModel(BaseDegradationModel):
    def compute_degradation_cost(self, step, action, control, y_dim_axis=1, action_dim_axis=-1, **kwargs):
        '''
        Parameters
        ----------
        step: int
        action: np.ndarray (..., ..., n_actions)
        control: np.ndarray (..., y_dim, ...)
        y_dim_axis: int, default=1
        action_dim_axis: int, default=-1

        Returns
        -------
        degradation_cost: np.ndarray (..., ..., n_actions)
        '''
        assert hasattr(self, 'delta_t'), 'time discretization not set'
        assert np.isfinite(control).all(), f'control {control} is invalid'
        # move action dimension axis to y dimension axis
        action = np.moveaxis(action, source=action_dim_axis,
                             destination=y_dim_axis)
        # get y_tot and T_ac from control and preprocess
        C_av = control.take(indices=[1], axis=y_dim_axis) / self.delta_t
        T_ac = control.take(indices=[2], axis=y_dim_axis) * step * self.delta_t
        y_tot = C_av * T_ac

        # compute input for degradation cost
        current_input = y_tot / 2 * C_av
        updated_y_tot = (y_tot + np.abs(action) * self.delta_t)
        if self.C_min > 0:
            updated_T_ac = T_ac + np.abs(np.sign(action)) * np.where(np.abs(
                action) < self.battery.C_min, np.abs(action) / self.battery.C_min, 1) * self.delta_t
        else:
            updated_T_ac = T_ac + np.abs(np.sign(action)) * self.delta_t
        with np.errstate(divide='ignore', invalid='ignore'):
            updated_C_av = np.where(
                updated_T_ac > 0, updated_y_tot / updated_T_ac, 0)
        # TODO: check if next_input can and should be computed using update_control
        next_input = updated_y_tot / 2 * updated_C_av
        # compute degradation cost
        degradation_cost = self.ageing_cost * (next_input - current_input)
        degradation_cost = np.where(np.abs(action) > 0, degradation_cost, 0)
        degradation_cost = degradation_cost * self.battery_value  # * self.Q_max
        assert np.isfinite(degradation_cost).any(
            axis=1).all(), degradation_cost
        # shape BEFORE moving action dimension axis back
        assert degradation_cost.shape == action.shape, (
            degradation_cost.shape, action.shape)
        # move back action dimension axis
        degradation_cost = np.moveaxis(
            degradation_cost, source=y_dim_axis, destination=action_dim_axis)
        return degradation_cost


class OldContinuousDegradationModel(BaseDegradationModel):
    def compute_degradation_cost(self, step, action, control, y_dim_axis=1, action_dim_axis=-1, **kwargs):
        '''
        Parameters
        ----------
        step: int
        action: np.ndarray (..., ..., n_actions)
        control: np.ndarray (..., y_dim, ...)
        y_dim_axis: int, default=1
        action_dim_axis: int, default=-1

        Returns
        -------
        degradation_cost: np.ndarray (..., ..., n_actions)
        '''
        assert hasattr(self, 'delta_t'), 'time discretization not set'
        assert np.isfinite(control).all(), f'control {control} is invalid'
        # move action dimension axis to y dimension axis
        action = np.moveaxis(action, source=action_dim_axis,
                             destination=y_dim_axis)
        # get y_tot and T_ac from control and preprocess
        C_av = control.take(indices=[1], axis=y_dim_axis) / self.delta_t
        T_ac = control.take(indices=[2], axis=y_dim_axis) * step * self.delta_t
        y_tot = C_av * T_ac

        # compute input for degradation cost
        current_input = y_tot / 2 * C_av
        updated_y_tot = (y_tot + np.abs(action) * self.delta_t)
        if self.C_min > 0:
            # updated_T_ac = T_ac + np.abs(np.sign(action)) * np.where(np.abs(
            #     action) < self.battery.C_min, np.abs(action) / self.battery.C_min, 1) * self.delta_t
            updated_T_ac = T_ac + np.where(np.abs(action) < self.battery.C_min, np.abs(
                action) / self.battery.C_min, 1) * self.delta_t
        else:
            # updated_T_ac = T_ac + np.abs(np.sign(action)) * self.delta_t
            updated_T_ac = T_ac + self.delta_t
        with np.errstate(divide='ignore', invalid='ignore'):
            updated_C_av = np.where(
                updated_T_ac > 0, updated_y_tot / updated_T_ac, 0)
        # TODO: check if next_input can and should be computed using update_control
        next_input = updated_y_tot / 2 * updated_C_av
        # compute degradation cost
        degradation_cost = self.ageing_cost * (next_input - current_input)
        # the next line is omitted so that the degradation cost is continuous in 0
        # degradation_cost = np.where(np.abs(action) > 0, degradation_cost, 0)
        degradation_cost = degradation_cost * self.battery_value  # * self.Q_max
        assert np.isfinite(degradation_cost).any(
            axis=1).all(), degradation_cost
        # shape BEFORE moving action dimension axis back
        assert degradation_cost.shape == action.shape, (
            degradation_cost.shape, action.shape)
        # move back action dimension axis
        degradation_cost = np.moveaxis(
            degradation_cost, source=y_dim_axis, destination=action_dim_axis)
        return degradation_cost


class NewDegradationModel(BaseDegradationModel):
    def compute_degradation_cost(self, step, action, control, y_dim_axis=1, action_dim_axis=-1, **kwargs):
        '''
        Parameters
        ----------
        step: int
        action: np.ndarray (..., ..., n_actions)
        control: np.ndarray (..., y_dim, ...)
        y_dim_axis: int, default=1
        action_dim_axis: int, default=-1

        Returns
        -------
        degradation_cost: np.ndarray (..., ..., n_actions)
        '''
        # compute cost of battery ageing
        assert hasattr(self, 'delta_t'), 'time discretization not set'
        assert np.isfinite(control).all(), f'control {control} is invalid'
        # move action dimension axis to y dimension axis
        action = np.moveaxis(action, source=action_dim_axis,
                             destination=y_dim_axis)
        # get y_tot and T_ac from control and preprocess
        C_av = control.take(indices=[1], axis=y_dim_axis) / self.delta_t
        T_ac = control.take(indices=[2], axis=y_dim_axis) * step * self.delta_t
        y_tot = C_av * T_ac

        # compute input for degradation cost
        current_input = y_tot / 2 * C_av
        updated_y_tot = (y_tot + np.abs(action) * self.delta_t)
        if self.C_min > 0:
            updated_T_ac = T_ac + np.abs(np.sign(action)) * np.where(np.abs(
                action) < self.battery.C_min, np.abs(action) / self.battery.C_min, 1) * self.delta_t
        else:
            updated_T_ac = T_ac + np.abs(np.sign(action)) * self.delta_t
        with np.errstate(divide='ignore', invalid='ignore'):
            updated_C_av = np.where(
                updated_T_ac > 0, updated_y_tot / updated_T_ac, 0)
        # TODO: check if next_input can and should be computed using update_control
        next_input = updated_y_tot / 2 * updated_C_av
        # compute degradation_cost
        degradation_cost = self.ageing_cost * \
            np.where(np.abs(action) < C_av, np.square(action) / 2
                     * self.delta_t, next_input - current_input)
        degradation_cost = degradation_cost * self.battery_value  # * self.Q_max
        assert np.isfinite(degradation_cost).any(
            axis=y_dim_axis).all(), degradation_cost[np.isnan(degradation_cost)]
        # shape BEFORE moving action dimension axis back
        assert degradation_cost.shape == action.shape, (
            degradation_cost.shape, action.shape)
        # move back action dimension axis
        degradation_cost = np.moveaxis(
            degradation_cost, source=y_dim_axis, destination=action_dim_axis)
        return degradation_cost


class QuadraticDegradationModel(BaseDegradationModel):
    def compute_degradation_cost(self, step, action, control, **kwargs):
        '''
        Parameters
        ----------
        step: int
        action: np.ndarray (..., ..., n_actions)

        Returns
        -------
        degradation_cost: np.ndarray (..., ..., n_actions)
        '''
        # compute cost of battery ageing
        assert hasattr(self, 'delta_t'), 'time discretization not set'
        action = np.where(np.abs(action) >= self.C_min, action,
                          self.C_min)  # C-rate is at least C_min
        degradation_cost = self.ageing_cost * \
            np.square(action) / 2 * self.delta_t
        degradation_cost = degradation_cost * self.battery_value  # * self.Q_max
        assert np.isfinite(degradation_cost).any(
            axis=1).all(), degradation_cost
        return degradation_cost


class NoDegradationModel(BaseDegradationModel):
    def compute_degradation_cost(self, step, action, control, **kwargs):
        '''
        Parameters
        ----------
        step: int
        action: np.ndarray (..., ..., n_actions)

        Returns
        -------
        degradation_cost: np.ndarray (..., ..., n_actions)
        '''
        # compute cost of battery ageing
        degradation_cost = np.zeros_like(action)
        return degradation_cost
