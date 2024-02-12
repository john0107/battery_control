import numpy as np

from modules.battery_model import BatteryModel


class BaseEnvironment():
    def __init__(self, time_discretization, y_dim=1):
        # save time discretization and all its attributes
        self.time_discretization = time_discretization
        self.__dict__.update(self.time_discretization.get_attributes())

        self.y_dim = y_dim

    def get_T(self):
        return self.T

    def get_steps(self):
        return self.n_steps

    def get_dim(self):
        return self.y_dim

    def get_admissible_actions(self, y):
        raise NotImplementedError


class BatteryProblem(BaseEnvironment):
    """ Battery control problem with a given battery and demand model.

    Parameters
    ----------
    time_discretization: TimeDiscretization
        specifies the time discretization of the problem
    battery: BatteryModel
        specifies the battery model
    demand: DemandModel
        specifies the demand model
    sell_to_market: bool, default=True
        specifies whether selling to the market is allowed or not
    """

    def __init__(self, time_discretization, battery, demand, sell_to_market=True):
        # initialize time discretization
        super(BatteryProblem, self).__init__(time_discretization,
                                             y_dim=3)  # create attributes T, n_steps, y_dim

        # battery specifications
        self.battery = battery
        self.battery.set_time_discretization(self.time_discretization)
        # control and action set
        self.control_set = self.battery.control_set
        self.n_controls = len(self.control_set)
        self.action_set = self.battery.action_set
        self.n_actions = len(self.action_set)
        # consumption model
        self.demand = demand
        self.demand.set_time_discretization(self.time_discretization)
        # other parameters
        self.sell_to_market = sell_to_market

    def convert_control_to_index(self, control, **kwargs):
        '''convert array of controls to corresponding index in self.control_set

        Parameters
        ----------
        control: np.ndarray (..., y_dim, ...)
            array of controls
        y_dim_axis: int, default=1
            dimension axis of y dimension
        **kwargs: dict
            additional keyword arguments for self.battery.convert_control_to_index

        Returns
        -------
        indices: np.ndarray(..., ...)
            array with corresponding indices, similar shape to control, BUT y_dim_axis is removed
        '''
        return self.battery.convert_control_to_index(control=control, **kwargs)

    def get_admissible_actions(self, step, control, y_dim_axis=1, action_dim_axis=-1):
        ''' get admissible actions for each given control
        Parameters
        ----------
        step: int
            current step
        control: np.ndarray (..., y_dim, ...)
            given controls
        y_dim_axis: int, default=1
            dimension axis of y dimension
        action_dim_axis: int, default=-1
            dimension axis of action dimension

        Returns
        -------
        admissible_actions: np.ndarray (..., ..., n_actions)
            admissible actions for each given control
        '''
        # admissible actions only depend on (SoC) component at index 0
        y_SoC = control.take(indices=[0], axis=y_dim_axis)
        assert y_SoC.shape == control.shape[:y_dim_axis] + \
            (1, ) + control.shape[y_dim_axis+1:], (y_SoC.shape, control.shape)

        if self.sell_to_market:  # selling is allowed, so allow all actions
            filtered_action_set = self.action_set  
        else:  # if selling is not allowed, only allow actions that do not lead to selling
            C_demand = self.demand.get_demand_per_hour(step) / self.battery.Q_max
            filtered_action_set = [C_rate for C_rate in self.action_set if -C_demand <= C_rate]
        assert len(filtered_action_set) > 0, f'no admissible actions for step {step}, control is {control}'

        # create array of all actions with appropriate dimensions
        a = np.expand_dims(np.array(filtered_action_set), axis=tuple(
            i for i in range(y_SoC.ndim) if i != y_dim_axis))

        admissible_actions = np.kron(a, np.ones(y_SoC.shape))
        assert admissible_actions.shape == control.shape[:y_dim_axis] + (
            len(filtered_action_set), ) + control.shape[y_dim_axis+1:], (admissible_actions.shape, control.shape)

        # TODO: make this function depend on self.update_control
        updated_y_SoC = np.around(y_SoC + admissible_actions *
                                  self.delta_t, self.battery.round)
        assert updated_y_SoC.shape == admissible_actions.shape, (updated_y_SoC.shape, admissible_actions.shape)
        # set invalid actions to np.NaN
        degradation_state = self.battery.get_degradation_state(step, control)
        assert degradation_state.shape == control.shape[:y_dim_axis] + (
            1, ) + control.shape[y_dim_axis+1:], (degradation_state.shape, control.shape)
        idx = (0 <= updated_y_SoC) & (updated_y_SoC <= degradation_state)
        admissible_actions[idx == False] = np.NaN
        assert np.isfinite(admissible_actions).any(axis=1).all(
        ), f'no admissible actions {a}, updated control is {updated_y_SoC}, control is {control}'
        admissible_actions = np.moveaxis(
            admissible_actions, source=y_dim_axis, destination=action_dim_axis)
        return admissible_actions

    def update_control(self, step, action, control, y_dim_axis=1, action_dim_axis=-1):
        ''' update control with given action
        Parameters:
        ----------
        step: int
            current step
        action: np.ndarray (..., ..., n_actions)
            action to be taken for each given control
        control: np.ndarray (..., y_dim, ...)
            control to be updated
        y_dim_axis: int, default=1
            dimension axis of y dimension
        action_dim_axis: int, default=-1
            dimension axis of action dimension

        Returns:
        --------
        updated_control: np.ndarray (..., y_dim, ..., n_actions)
            updated control
        '''
        # move action dimension axis to y dimension axis
        action = np.moveaxis(action, source=action_dim_axis,
                             destination=y_dim_axis)
        # check if dimensions match
        assert control.shape[:y_dim_axis] == action.shape[:
                                                          y_dim_axis], (control.shape, action.shape)
        assert control.shape[y_dim_axis+1:] == action.shape[y_dim_axis +
                                                            1:], (control.shape, action.shape)
        # get components of current control
        y_SoC = control.take(indices=[0], axis=y_dim_axis)
        C_av = control.take(indices=[1], axis=y_dim_axis) / self.delta_t
        T_ac = control.take(indices=[2], axis=y_dim_axis) * step * self.delta_t
        y_tot = C_av * T_ac

        # compute updated control
        updated_y_SoC = np.around(
            y_SoC + action * self.delta_t, self.battery.round)
        updated_y_tot = y_tot + np.abs(action) * self.delta_t
        # if np.abs(action) < self.battery.C_min, then T_ac is updated as if action = self.battery.C_min,
        # i.e. with time step np.abs(action) / self.battery.C_min * self.delta_t
        if self.battery.C_min > 0:
            updated_T_ac = T_ac + np.abs(np.sign(action)) * np.where(np.abs(
                action) < self.battery.C_min, np.abs(action) / self.battery.C_min, 1) * self.delta_t
        else:
            updated_T_ac = T_ac + np.abs(np.sign(action)) * self.delta_t
        with np.errstate(divide='ignore', invalid='ignore'):
            updated_C_av = np.where(
                updated_T_ac > 0, updated_y_tot / updated_T_ac, 0)

        # concatenate components together
        updated_control = np.stack([
            updated_y_SoC,
            updated_C_av * self.delta_t,
            updated_T_ac / ((step + 1) * self.delta_t),
        ], axis=action_dim_axis).swapaxes(y_dim_axis, action_dim_axis)
        assert updated_control.shape == control.shape[:y_dim_axis] + (
            self.y_dim, ) + control.shape[y_dim_axis+1:] + (action.shape[y_dim_axis], ), updated_control.shape
        return updated_control

    def cash_flow(self, step, action, control, X, detailed_breakdown=False, **kwargs):
        ''' Compute cash flows for given actions and controls.

        Parameters
        ----------
        step: int
            current time step
        action: np.ndarray (:, ..., n_actions)
            actions for which cash flow is computed
        control: nd.array with dimensions (n_samples, y_dim, ...)
            controls for which cash flow is computed
        X: nd.array with dimensions (n_samples, x_dim)
            sample paths of electricity prices
        detailed_breakdown: bool, default=False
            if True, return detailed breakdown of cash flow
        **kwargs: dict, optional
            additional keyword arguments

        Returns
        -------
        cost: nd.array with dimensions (n_samples, ..., n_controls, n_actions)
            cost of actions for given controls
        '''
        # compute cost of buying electricity
        assert X.ndim == 2  # check if X has appropriate dimensions
        # discount_factor = np.exp(- self.interest_rate * step / self.n_steps * self.T)  # compute discount factor
        # X = discount_factor * X.flatten()  # discount prices
        X = X[:, 0]  # current electricity price is always given at 0th index
        X = X / 10**3  # convert price from €/MWh to €/kWh
        if isinstance(action, np.ndarray):
            # check if dimensions match
            assert X.shape[0] == action.shape[0] or action.shape[
                0] == 1, f'{X.shape[0]} =/= {action.shape[0]}'
            # add axis to X for broadcasting
            X = np.expand_dims(X, axis=tuple(range(1, action.ndim)))
        C_demand = self.demand.get_demand_per_hour(step) / self.battery.Q_max
        C_buy = action + C_demand
        electricity_cost = C_buy * self.battery.I_C * X * self.delta_t
        assert electricity_cost.shape[1:] == action.shape[1:], (
            electricity_cost.shape, action.shape)

        # compute cost of running the battery
        running_cost = self.battery.running_cost * \
            np.abs(action) * self.battery.I_C * self.delta_t

        # compute degredation cost of battery
        ageing_cost = self.battery.compute_degradation_cost(
            step, action, control, **kwargs)

        # compute total cost
        total_cost = electricity_cost + running_cost + ageing_cost
        if detailed_breakdown:
            return total_cost, (electricity_cost, running_cost, ageing_cost)
        else:
            return total_cost

    def get_interpolation_controls(self, control, y_dim_axis=1):
        ''' Get interpolation controls for given control(s).

        Parameters
        ----------
        control: nd.array with dimensions (..., y_dim, ...)
            control for which interpolation controls are computed
        y_dim_axis: int, default=1
            axis of y dimension

        Returns:
        --------
        interpolation_controls: List[Tuple(interpolation_control, coeff)], where
            interpolation_control: nd.array(shape=(n_samples, y_dim)) or nd.array(shape=(n_samples, y_dim, n_actions))
            coeff: nd.array(shape=n_samples, 1) or nd.array(shape=(n_samples, 1, n_actions))
        '''
        return self.battery.get_interpolation_controls(control, y_dim_axis=y_dim_axis)


######## old code - DEPRACATED ########


class GasStorageProblem(BaseEnvironment):
    def __init__(self, T, n_steps, N, r=0.1):
        super(GasStorageProblem, self).__init__(T, n_steps)
        self.delta = 1 / N  # discretization of control set
        self.control_set = [round(self.delta * i, 3) for i in range(N+1)]
        self.action_set = [-1, 0, 1]
        self.admissible_actions = {y: [a for a in self.action_set if (self.update_control(a, y) in self.control_set)]
                                   for y in self.control_set}
        self.r = r  # interest rate for discounting cash flows

    def update_control(self, a, y):
        if isinstance(a, np.ndarray) and isinstance(y, np.ndarray):
            assert len(y) == a.shape[0]
            if a.ndim == 1:
                return (y + a * self.delta).round(decimals=3)
            else:
                return (y[:, np.newaxis] + a * self.delta).round(decimals=3)
        return round(y + a * self.delta, 3)

    def cash_flow(self, step, a, y, X):
        # check if X has appropriate dimensions
        assert X.ndim == 2 and X.shape[1] == 2
        # compute discount factor
        discount_factor = np.exp(- self.r * step / self.n_steps * self.T)
        if isinstance(a, np.ndarray):
            assert X.shape[0] == a.shape[0]
            if a.ndim == 1:
                return - a * self.delta * X[:, 1] * discount_factor
            else:
                return - a * self.delta * X[:, 1][:, np.newaxis] * discount_factor
        else:
            return - a * self.delta * X[:, 1] * discount_factor


class BermudanMaxCallOption(BaseEnvironment):
    def __init__(self, T, n_steps, y_max=1, strike=100, r=0.05):
        super(BermudanMaxCallOption, self).__init__(T, n_steps)
        self.y_max = y_max
        self.control_set = list(range(self.y_max+1))
        self.action_set = [0, 1]
        self.admissible_actions = {
            y: [0, 1] if y > 0 else [0] for y in self.control_set}
        self.strike = strike
        self.r = r  # interest rate for discounting cash flows

    def update_control(self, a, y):
        if isinstance(a, np.ndarray) and isinstance(y, np.ndarray):
            assert len(y) == a.shape[0]
            if a.ndim == 1:
                return y - a
            else:
                return y[:, np.newaxis] - a
        return y - a

    def cash_flow(self, step, a, y, X):
        # and X.shape[1] == 2  # check if X has appropriate dimensions
        assert X.ndim == 2
        # compute discount factor
        discount_factor = np.exp(- self.r * step / self.n_steps * self.T)
        payoff = np.nanmax(np.maximum((X - self.strike), 0), axis=1)
        if isinstance(a, np.ndarray):
            assert X.shape[0] == a.shape[0]
            if a.ndim == 1:
                return a * payoff * discount_factor
            else:
                return a * payoff[:, np.newaxis] * discount_factor
        else:
            return a * payoff * discount_factor


class SimpleBatteryProblem(BaseEnvironment):
    def __init__(self, T, n_steps, N, c, d=0, r=0.1):
        super(SimpleBatteryProblem, self).__init__(T, n_steps)
        self.delta = 1 / N  # discretization of control set
        self.total_consumption = c  # total consumption in time T
        self.consumption = self.total_consumption / \
            self.n_steps  # constant consumption per time step
        self.battery_degradation = d
        self.control_set = [round(self.delta * i, 3) for i in range(N+1)]
        self.action_set = list(range(0, N+1))
        self.admissible_actions = {
            y: [a for a in self.action_set
                if (self.update_control(0, a, y) in self.control_set)]
            for y in self.control_set}
        self.r = r  # interest rate for discounting cash flows

    def update_control(self, step, a, y):
        if isinstance(a, np.ndarray) and isinstance(y, np.ndarray):
            assert len(y) == a.shape[0]
            if a.ndim == 1:
                return (y - self.consumption + a * self.delta).round(decimals=3)
            else:
                return (y[:, np.newaxis] - self.consumption + a * self.delta).round(decimals=3)
        # subtract consumption in each time step
        return round(y - self.consumption + a * self.delta, 3)

    def cash_flow(self, step, a, y, X):
        # assert X.ndim == 1  # check if X has appropriate dimensions
        # compute discount factor
        discount_factor = np.exp(- self.r * step / self.n_steps * self.T)
        penalty = np.abs(a * self.delta - self.consumption) * \
            self.battery_degradation
        if isinstance(a, np.ndarray):
            assert X.shape[0] == a.shape[0]
            if a.ndim == 1:
                return - a * self.delta * X.flatten() * discount_factor - penalty
            else:
                return - a * self.delta * X * discount_factor - penalty
        else:
            return - a * self.delta * X.flatten() * discount_factor - penalty
