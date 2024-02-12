import numpy as np
import pandas as pd
import scipy.stats as stats
import sys
import os
import gc
import pickle
import blosc2
from tqdm.autonotebook import tqdm, trange

# import modules required for `create_parameters` function
from modules.time_discretization import TimeDiscretization
from modules.demand import ConstantDemandModel
from modules.simulator import ElectricityPriceMultiFactorSimulator, GasPriceSimulator
from modules.battery_model import BatteryModel
from modules.environment import BatteryProblem
from modules.transformer import PolynomialTransformer
from modules.approximator import RegressionModel
from modules.utils import deep_update


class BatteryControlSolver():
    """Class for solving the battery control problem.

    Parameters
    ----------
    environment: BaseEnvironment
        environment of the problem.
    simulator: BaseSimulator
        state simulator of the problem.
    conti_approximator: BaseApproximator
        approximator for the continuation function.
    value_approximator: BaseApproximator, optional
        approximator for the value function, by default None. 
        If not given, the value function is not approximated.
    """

    def __init__(self, environment, simulator, conti_approximator, value_approximator=None):
        self.environment = environment
        self.simulator = simulator
        self.conti_func = conti_approximator
        self.value_func = value_approximator

        self.T = self.environment.get_T()
        self.n_steps = self.environment.get_steps()
        self.x_dim = self.simulator.get_dim()
        self.y_dim = self.environment.get_dim()

    def compute_action_values(self, step, control, X, features):
        '''compute the value of each admissible action for each given control
        Parameters
        ----------
        step: int
        control: np.ndarray (n_samples, y_dim, ) or (1, y_dim, n_controls)
        X: np.ndarray (n_samples, x_dim, n_steps)
        features: np.ndarray (n_samples, n_features)

        Returns
        -------
        action_values: np.ndarray (n_samples, n_actions) or (n_samples, n_controls, n_actions)
        '''
        assert control.ndim >= 2, control.ndim  # it is assumed that control is at least 2-dimensional
        # compute all possible next controls; updated_control has dim (n_samples, y_dim, n_actions)
        admissible_actions = self.environment.get_admissible_actions(
            step, control)  # get all admissible actions for current control
        updated_control = self.environment.update_control(
            step, admissible_actions, control)  # compute next possible controls
        # check that updated_control has correct shape
        assert updated_control.shape[:-
                                     1] == control.shape, updated_control.shape
        assert updated_control.shape[-1] > (
            self.environment.n_actions - 1) // 2, updated_control.shape
        # compute the value of the continuation function for each ((future) y, x) pair
        continuation_values = self.conti_func.predict(
            step, updated_control, X, features)
        assert np.isfinite(continuation_values).any(
            axis=1).all(), updated_control
        cash_flow = self.environment.cash_flow(
            step, admissible_actions, control, X[:, :, step])
        assert cash_flow.shape == continuation_values.shape, (
            cash_flow.shape, continuation_values.shape)
        action_values = cash_flow + continuation_values
        return action_values

    def solve(self, n_samples, X=None):
        """solve the battery control problem via the partial least squares Monte Carlo algorithm.

        Parameters
        ----------
        n_samples: int
            number of samples used for the regression.
        X: np.ndarray (n_samples, x_dim, n_steps), optional
            sample paths used for the regression, by default None.
            If not given, sample paths are simulated.

        Returns
        -------
        self: BatteryControlSolver
            returns itself.
        """
        # simulate sample paths for regression
        if X is None:
            X = self.simulator.simulate(time_discretization=self.environment.time_discretization,
                                        n_samples=n_samples)

        if self.value_func:  # algorithm with additional value function approximation
            # TODO: THIS CASE IS DEPRECATED
            # set value function at terminal time
            # value function at terminal time is zero
            zero_target = np.zeros(shape=(n_samples, 1))
            for y in self.environment.control_set:
                self.value_func.fit(self.n_steps, y, X, target=zero_target)

            # backward iteration
            # approximate value functions backwards
            for step in trange(self.n_steps-1, -1, -1):
                # approximate continuation function using regression
                for y in self.environment.control_set:  # regression for each control
                    target = self.value_func.predict(
                        step+1, y, X)  # compute target
                    # approximate continuation function
                    self.conti_func.fit(step, y, X, target=target)

                # approximate value function with second regression
                # simulate a new set of sample paths
                _X = self.simulator.simulate(
                    time_discretization=self.environment.time_discretization, n_samples=n_samples)
                for y in self.environment.control_set:  # compute value function for each control
                    action_values = self.compute_action_values(
                        step, y, _X)  # compute action values
                    # approx. value function is action value with optimal action
                    target = np.nanmin(action_values, axis=1)
                    # approximate value function
                    self.value_func.fit(step, y, _X, target=target)
        else:  # algorithm without additional value function approximation
            # set value function at terminal time
            # value function at terminal time is zero
            zero_targets = np.zeros(
                shape=(n_samples, self.environment.n_controls))
            features = self.conti_func.transformer.transform(
                X[:, :, self.n_steps-1])  # create features
            self.conti_func.fit(
                self.n_steps-1, features=features, targets=zero_targets)

            # backward iteration
            # approximate value functions backwards
            for step in trange(self.n_steps-2, -1, -1, leave=False):
                next_features = features  # create features for step+1
                features = self.conti_func.transformer.transform(
                    X[:, :, step])  # create features for step

                # approximate continuation function using regression
                controls = np.stack(self.environment.control_set, axis=1)[
                    np.newaxis, ...]
                assert controls.shape == (
                    1, self.y_dim, self.environment.n_controls)
                action_values = self.compute_action_values(
                    step+1, controls, X, features=next_features)  # compute action values
                assert action_values.shape[:-1] == (
                    n_samples, self.environment.n_controls), action_values.shape
                assert action_values.shape[-1] > (
                    self.environment.n_actions - 1) // 2, action_values.shape
                # approximated value function is action value with optimal action
                targets = np.nanmin(action_values, axis=-1)
                assert targets.shape == (
                    n_samples, self.environment.n_controls)
                assert np.isfinite(targets).all(
                ), f'target {targets} is not finite'
                # approximate continuation function
                self.conti_func.fit(step, features=features, targets=targets)
                gc.collect()  # collect garbage, otherwise memory will explode
        return self

    def lower_bound(self, initial_control, n_samples, return_cash_flows=False, X=None, return_X=False):
        """compute the lower bound of the battery control problem.

        Parameters
        ----------
        initial_control: np.ndarray (y_dim, 1)
            initial control.
        n_samples: int
            number of samples used for the estimation of the lower bound.
        return_cash_flows: bool, optional
            if True, return the computed final cash flow for each sample, by default False.
        X: np.ndarray (n_samples, x_dim, n_steps), optional
            sample paths used for the estimation of the lower bound, by default None.
            If not given, sample paths are simulated.
        return_X: bool, optional
            if True, return the sample paths used for the estimation of the lower bound, by default False.

        Returns
        -------
        result: dict
            dictionary containing the lower bound, confidence intervals and optionally the final cash flow.
        X: np.ndarray (n_samples, x_dim, n_steps), optional
            sample paths used for the estimation of the lower bound, by default None.
        """
        if not X:
            X = self.simulator.simulate(
                time_discretization=self.environment.time_discretization, n_samples=n_samples)  # simulate samples

        initial_control = np.repeat(
            np.array(initial_control, ndmin=2), repeats=n_samples, axis=0)
        control_history = [initial_control]
        cash_flow_history = []
        for step in trange(self.n_steps, leave=False):
            # get current control
            current_control = control_history[-1]
            # compute action values for all samples
            features = self.conti_func.transformer.transform(X[:, :, step])
            action_values = self.compute_action_values(
                step, current_control, X, features)

            # compute optimal actions for all samples
            optimal_actions_idx = np.nanargmin(action_values, axis=-1)
            optimal_actions = np.array(self.environment.action_set)[
                optimal_actions_idx][:, np.newaxis]

            # update cash flow and control history
            cash_flow_history += [self.environment.cash_flow(
                step, optimal_actions, current_control, X[:, :, step])]
            control_history += [self.environment.update_control(
                step, optimal_actions, current_control)[:, :, 0]]

        total_cash_flows = np.stack(cash_flow_history).sum(axis=0)
        lower_bound = np.mean(total_cash_flows)
        std = np.std(total_cash_flows, ddof=1)
        if self.value_func is not None:
            v_0 = self.value_func.predict(0, initial_control, X).mean()
        else:
            v_0 = 'value function not approximated'
        result = {'lower bound estimate': lower_bound,
                  '95% confidence interval': [lower_bound-std*2/n_samples**0.5,
                                              lower_bound+std*2/n_samples**0.5],
                  'v_0': v_0}
        if return_cash_flows and return_X:
            return result, total_cash_flows, X
        elif return_cash_flows:
            return result, total_cash_flows
        elif return_X:
            return result, X
        else:
            return result


####### solve with parameters #######

def str_to_class(classname):
    '''Convert string to class

    Parameters
    ----------
    classname : str
        Name of class

    Returns
    -------
    class: class
        class with given name
    '''
    if isinstance(classname, str):
        return getattr(sys.modules[__name__], classname)
    else:
        return classname


def create_parameters(base_parameter, parameter_changes=None, label=None):
    '''Create parameters from base parameter and parameter changes

    Parameters
    ----------
    base_parameter : dict
        Base parameter
    parameter_changes : dict, optional
        Parameter changes, by default None
    label : str, optional
        Label of parameter if no parameter changes given, by default None

    Returns
    -------
    parameters : dict
        Parameters
    '''

    parameters = {}
    if parameter_changes is None:  # no parameter changes given
        if label is not None:  # return base parameter with given label
            parameter_changes = {label: {}}
        else:
            raise KeyError(f'Parameter label is missing.')

    # create parameters with parmeter changes
    for label, parameter_change in parameter_changes.items():
        parameters[label] = deep_update(base_parameter, parameter_change)

    # update environment parameter by initializing classes
    for parameter in parameters.values():
        updated_env_params = {}
        for key, env_param in parameter['environment'].items():
            if key in parameter['meta'].keys():
                key_class = str_to_class(parameter['meta'][key])
                updated_env_params[key] = key_class(**env_param)
            else:
                updated_env_params[key] = env_param
        parameter['environment'] = updated_env_params
    return parameters


def solve_with_param(param, save=False, path=None, label=None):
    '''
    Parameters
    ----------
    param: dict
    save: bool, default=False
    path: str, default=None
    label: str, default=None

    Returns
    -------
    solver: Solver
    '''
    print(f'----- {label} -----')
    sim = str_to_class(param['meta']['simulator'])(**param['simulator'])
    tra = str_to_class(param['meta']['transformer'])(
        sim=sim, **param['transformer'])
    env = str_to_class(param['meta']['environment'])(**param['environment'])
    conti_app = str_to_class(param['meta']['approximator'])(env, tra)
    # value_app = str_to_class(param['meta']['approximator'])(env, tra)
    sol = str_to_class(param['meta']['solver'])(env, sim, conti_app)
    sol.solve(n_samples=param['meta']['n_samples'])

    if save:  # save solver
        assert path is not None and label is not None, (path, label)
        save_solver(solver=sol, label=label, path=path)

    return sol


def solve_with_parameters(parameters, save=False, path=None):
    '''
    Parameters
    ----------
    param: dict
    save: bool, default=False
    path: str, default=None
    label: str, default=None

    Returns
    -------
    solvers: Dict[str, Solver]
    '''
    solvers = {}
    for label, param in tqdm(parameters.items()):
        solvers[label] = solve_with_param(
            param, save=save, path=path, label=label)
    return solvers


####### compute lower bound #######

def compute_lower_bound(solvers, initial_control, n_samples=10**3, return_cash_flows=True, benchmark=True):
    '''estimates lower bounds for solvers

    Parameters
    ----------
    solvers: Dict[str, Solver]
        dictionary of solvers with labels as keys
    initial_control: np.ndarray
        initial control for lower bound
    n_samples: int
        number of samples for lower bound estimation
    benchmark: bool
        whether to compute benchmark

    Returns
    -------
    lower_bounds: Dict[str, float]
        dictionary of estimated lower_bound
    cash_flows: Dict[str, np.ndarray]
        dictionary of computed cash flows
    '''
    lower_bounds, cash_flows = {}, {}
    for label, solver in solvers.items():
        print(f'----- {label} -----')
        lower_bound, cash_flow = solver.lower_bound(initial_control=initial_control,
                                                    n_samples=n_samples,
                                                    return_cash_flows=return_cash_flows)
        lower_bounds[label] = lower_bound
        cash_flows[label] = cash_flow
        print(lower_bound)
    if benchmark:  # benchmark computation
        print(f'----- benchmark no battery -----')
        label = 'benchmark no battery'
        # get environment and simulator to compute benchmark
        if isinstance(benchmark, str):
            sol = solvers[benchmark]
        else:
            sol = next(iter(solvers.values()))
        env = sol.environment

        # compute cash flow for benchmark
        cash_flow = (sol.simulator.simulate(
            time_discretization=env.time_discretization, n_samples=n_samples,
        )[:, [0], :-1]
            / 10**3  # convert prices from €/MWh to €/kWh
            * env.time_discretization.delta_t  # delta t, time step in hours
            * env.demand.demand_per_hour  # demand per time step
        ).sum(axis=2)
        cash_flows[label] = cash_flow
        lower_bound = {'lower bound estimate': cash_flow.mean(),
                       '95% confidence interval': [cash_flow.mean()-cash_flow.std()*2/n_samples**0.5,
                                                   cash_flow.mean()+cash_flow.std()*2/n_samples**0.5]}
        lower_bounds[label] = lower_bound
        print(lower_bound)

    return lower_bounds, cash_flows


def compute_lower_bound_files(solvers, initial_control, path,
                              n_samples=10**3, n_batches=1, confidence=[0.95, 0.997],
                              benchmark=True, compute_cash_flows=True, append=False):
    '''computes cash flows in batches from saved solvers, saves them to csv and computes lower bounds

    Parameters
    ----------
    solvers: Dict[str, Solver]
        dictionary of solvers with labels as keys
    initial_control: np.ndarray
        initial control for lower bound
    path: str
        path to save cash flows
    n_samples: int, default=10**3
        total number of samples for lower bound estimation
    n_batches: int, default=1
        number of batches
    confidence: float
        confidence interval for lower bound estimation
    benchmark: bool, default=True
        whether to compute benchmark
    compute_cash_flows: bool, default=True
        whether to compute cash flows
    append: bool, default=False
        whether to append to already computed cash flows

    Returns
    -------
    lower_bounds: pd.DataFrame
        dataframe containing estimated lower_bound and Monte-Carlo confidence intervals
    '''
    assert n_samples % n_batches == 0, f'n_samples ({n_samples}) must be divisible by n_batches ({n_batches})'
    batch_size = n_samples // n_batches

    if append and compute_cash_flows:
        print('----- loading already computed cash flows -----')
        df = pd.read_csv(f'{path}/cash_flows.csv', index_col=0)
        assert len(
            df) == n_samples, f'length of found cash_flows ({len(df)}) does not match n_samples ({n_samples})'
        solvers_to_compute = list(solvers.keys())
        for label in df.columns:
            if label in solvers_to_compute:
                print(f'----- cash flow for {label} already computed -----')
                solvers_to_compute.remove(label)
    else:
        solvers_to_compute = solvers.keys()

    if compute_cash_flows:
        print('----- computing cash flows -----')
        cash_flows = {}
        for label in tqdm(solvers_to_compute, unit='solver'):
            cash_flows[label] = []
            for _ in trange(n_batches, leave=False, unit='batch'):
                _, cash_flow = solvers[label].lower_bound(initial_control=initial_control,
                                                          n_samples=batch_size,
                                                          return_cash_flows=True)
                cash_flows[label] += [cash_flow]
            cash_flows[label] = np.concatenate(
                cash_flows[label], axis=0).squeeze()
            # save cash flows in case of interruption
            if append:
                pd.concat([df, pd.DataFrame(cash_flows)], axis=1).to_csv(
                    path_or_buf=f'{path}/cash_flows.csv')
            else:
                pd.DataFrame(cash_flows).to_csv(
                    path_or_buf=f'{path}/cash_flows.csv')
            print(f'----- cash flow for {label} saved -----')

        if benchmark:  # compute benchmark
            print(f'----- computing benchmark -----')
            # get environment and simulator to compute benchmark
            if isinstance(benchmark, str):
                sol = solvers[benchmark]
            else:
                sol = next(iter(solvers.values()))
            env = sol.environment

            # compute cash flow for benchmark
            cash_flows['benchmark'] = []
            for _ in trange(n_batches, leave=False, unit='batch'):
                cash_flow = (sol.simulator.simulate(
                    time_discretization=env.time_discretization, n_samples=batch_size,
                )[:, [0], :-1]
                    / 10**3  # convert prices from €/MWh to €/kWh
                    * env.time_discretization.delta_t  # delta t, time step in hours
                    * env.demand.demand_per_hour  # demand per time step
                ).sum(axis=2).squeeze()
                cash_flows['benchmark'] += [cash_flow]
            cash_flows['benchmark'] = np.concatenate(
                cash_flows['benchmark'], axis=0).squeeze()

        # save cash flows
        if append:  # append cash flows to existing file
            df = pd.concat([df, pd.DataFrame(cash_flows)],
                           axis=1)
        else:  # create new file
            df = pd.DataFrame(cash_flows)
        df.to_csv(path_or_buf=f'{path}/cash_flows.csv')
        print('----- all cash flows saved -----')
    else:
        print('----- loading cash flows -----')
        df = pd.read_csv(f'{path}/cash_flows.csv', index_col=0)

    # compute lower bounds
    print('----- computing lower bounds -----')
    lower_bounds = {}
    for label, cash_flow in df.items():
        confidence_intervals = {}
        if isinstance(confidence, float):  # convert confidence to list
            confidence = [confidence]
        for c in confidence:  # compute confidence intervals for
            confidence_interval = stats.norm.interval(
                confidence=c, loc=np.mean(cash_flow), scale=stats.sem(cash_flow))
            confidence_intervals[f'{100*c:g}% confidence interval'] = confidence_interval
        lower_bound = {
            'value estimate': cash_flow.mean(),
            'std': cash_flow.std(),
        }
        # add confidence intervals
        for key, confidence_interval in confidence_intervals.items():
            lower_bound[key] = confidence_interval

            lower_bound[f'\u00B1{key.split("%")[0]}% Monte-Carlo error'] = (
                confidence_interval[1] - confidence_interval[0]) / 2
        lower_bounds[label] = lower_bound
    lower_bounds = pd.DataFrame(lower_bounds)
    lower_bounds.to_csv(path_or_buf=f'{path}/lower_bounds.csv')
    print('----- lower bounds saved -----')
    return lower_bounds


####### load and save #######

def load_solvers(path):
    """load solvers from path

    Parameters
    ----------
    path : str
        path to solvers

    Returns
    -------
    solvers : Dict[str, Solver]
        dictionary of solvers with labels as keys
    """
    solvers = {}
    for root, _, files in os.walk(path):
        for file_name in tqdm(files):
            if file_name.endswith('.pkl'):
                label = file_name.split('.')[0]
                print(f'----- found {label}; loading -----')
                with open(os.path.join(root, file_name), "rb") as f:
                    compressed_pickle = f.read()
                    depressed_pickle = blosc2.decompress(compressed_pickle)
                    solvers[label] = pickle.loads(depressed_pickle)
        print('solvers loaded')
    return solvers


def save_solver(solver, label, path):
    """save solver to path

    Parameters
    ----------
    solver : Solver
        solver to be saved
    label : str
        label of the solver
    path : str
        path to save the solver
    """
    if path[-1] == '/':
        file_name = f'{path}{label}.pkl'
    else:
        file_name = f'{path}/{label}.pkl'

    with open(file_name, "wb") as f:
        pickled_data = pickle.dumps(solver, protocol=-1)
        compressed_pickle = blosc2.compress(pickled_data)
        f.write(compressed_pickle)
    print(f'"{label}" solver saved in {path}')


def save_solvers(solvers, path):
    """save multiple solvers to path

    Parameters
    ----------
    solvers : Dict[str, Solver]
        dictionary of solvers with labels as keys
    path : str
        path to save the solvers
    """
    for label, solver in solvers.items():
        save_solver(solver, label, path)
