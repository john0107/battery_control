import numpy as np
import pandas as pd
from scipy import stats, signal
from tqdm.autonotebook import tqdm
import functools

from modules.utils import gamma_kernel, convert_ghyd_to_nig_params, compute_CARMA_parameters
from modules.time_discretization import TimeDiscretization, scale_time


class BaseSimulator():
    def __init__(self, n_dim=1):
        self.n_dim = n_dim
        # if self.n_dim == 1:
        #     self.state_labels = ('price', )

    def get_dim(self):
        """Return dimension of the process."""
        return self.n_dim

    def simulate(self, time_discretization, n_samples):
        """Simulate sample paths of the process.

        Parameters
        ----------
        time_discretization : TimeDiscretization
            Time discretization for simulating sample paths.
        n_samples : int
            Number of samples to simulate.

        Returns
        -------
        sample_paths : array_like
            Sample paths of the process.
        """
        raise NotImplementedError

    def get_state_labels(self, ):
        """Return labels of the state variables."""
        raise NotImplementedError

    def simulate_in_batches(self, time_discretization, n_samples, batch_size=10, file=None):
        """simulate sample paths in batches instead of all at once

        Parameters
        ----------
        time_discretization: TimeDiscretization
            time discretization for simulating sample paths
        n_samples: int
            number of samples to simulate in total
        batch_size: int, default is 10
            size of batches
        file: str, default is None
            If given the simulated sample paths are saved in the file location as .npy file

        Returns
        -------
        sample_paths: array_like
            Sample paths of the process
        """
        # compute batch sizes
        n_batches = [batch_size] * (n_samples // batch_size)
        if n_samples % batch_size != 0:
            n_batches += [n_samples % batch_size]

        sample_paths = []
        for n_batch in tqdm(n_batches):
            sample_paths.append(self.simulate(
                time_discretization=time_discretization, n_samples=n_batch))
        # concatenate along sample dimension
        sample_paths = np.concatenate(sample_paths, axis=0)
        # save as npy file in compressed format
        if file is not None:  # if path is specified save the sample paths
            np.save(file, sample_paths)
        return sample_paths

    def load(self, file):
        """Load sample paths from npz file.

        Parameters
        ----------
        path : str
            Path to npz file.

        Returns
        -------
        sample_paths : array_like
            Sample paths of the process."""
        return np.load(f'{file}.npy')


class ConstantProcess(BaseSimulator):
    def __init__(self, initial_value=0, n_dim=1):
        super().__init__(n_dim=n_dim)
        self.initial_value = initial_value

    def simulate(self, time_discretization, n_samples):
        n_steps = time_discretization.n_steps
        return np.full(shape=(n_samples, self.n_dim, n_steps+1), fill_value=self.initial_value)


class ModifiedProcess(BaseSimulator):
    def __init__(self, process, func=None, time_change=None):
        self.process = process
        self.func = func
        self.time_change = time_change
        assert self.func is None or self.time_change is None

    def simulate(self, time_discretization, n_samples):
        if self.func:
            sample_paths = self.process.simulate(
                time_discretization, n_samples)
            return self.func(sample_paths)
        elif self.time_change:
            changed_time_discretization = self.time_change(time_discretization)
            return self.process.simulate(changed_time_discretization, n_samples)
        else:
            return self.process.simulate(time_discretization, n_samples)


class SeasonSimulator(BaseSimulator):
    def __init__(self, start_date='01/01/2011') -> None:
        self.start_date = pd.to_datetime(start_date)
        self.season_param = {
            'level': 50.575,
            'trend': -0.0134,
            'seasonal_factor': -3.027,
            'seasonal_shift': 8328,
        }
        self.weekday_hourly_variations = {
            "Monday": {
                "0": -6.65,
                "1": -13.881,
                "2": -17.95,
                "3": -20.872,
                "4": -23.412,
                "5": -21.61,
                "6": -14.757,
                "7": -3.92,
                "8": 12.131,
                "9": 15.472,
                "10": 17.202,
                "11": 19.294,
                "12": 25.392,
                "13": 19.633,
                "14": 18.355,
                "15": 14.873,
                "16": 11.743,
                "17": 10.276,
                "18": 14.361,
                "19": 17.836,
                "20": 14.66,
                "21": 10.245,
                "22": 3.596,
                "23": 0.75,
            },
            "Tuesday": {
                "0": -6.108,
                "1": -9.135,
                "2": -12.804,
                "3": -15.548,
                "4": -18.165,
                "5": -17.359,
                "6": -10.587,
                "7": -1.554,
                "8": 14.877,
                "9": 18.461,
                "10": 20.279,
                "11": 22.109,
                "12": 26.746,
                "13": 19.95,
                "14": 18.552,
                "15": 15.929,
                "16": 12.603,
                "17": 11.266,
                "18": 15.075,
                "19": 17.724,
                "20": 14.595,
                "21": 11.024,
                "22": 4.104,
                "23": 1.397,
            },
            "Wednesday": {
                "0": -5.776,
                "1": -8.495,
                "2": -11.912,
                "3": -15.136,
                "4": -17.606,
                "5": -16.752,
                "6": -10.395,
                "7": -1.869,
                "8": 13.287,
                "9": 17.129,
                "10": 19.796,
                "11": 21.953,
                "12": 26.407,
                "13": 20.461,
                "14": 19.24,
                "15": 16.732,
                "16": 13.102,
                "17": 11.514,
                "18": 14.591,
                "19": 17.306,
                "20": 14.543,
                "21": 10.847,
                "22": 4.514,
                "23": 1.502,
            },
            "Thursday": {
                "0": -6.085,
                "1": -7.194,
                "2": -11.175,
                "3": -14.212,
                "4": -16.979,
                "5": -15.986,
                "6": -10.031,
                "7": -1.617,
                "8": 13.423,
                "9": 16.474,
                "10": 18.087,
                "11": 19.954,
                "12": 24.08,
                "13": 18.432,
                "14": 16.619,
                "15": 13.97,
                "16": 11.117,
                "17": 9.643,
                "18": 13.089,
                "19": 17.044,
                "20": 13.668,
                "21": 10.438,
                "22": 3.206,
                "23": 1.027,
            },
            "Friday": {
                "0": -6.203,
                "1": -8.119,
                "2": -12.032,
                "3": -15.0,
                "4": -17.689,
                "5": -16.817,
                "6": -10.731,
                "7": -2.262,
                "8": 12.286,
                "9": 15.522,
                "10": 17.626,
                "11": 19.428,
                "12": 22.065,
                "13": 16.614,
                "14": 11.743,
                "15": 7.73,
                "16": 4.344,
                "17": 4.032,
                "18": 8.186,
                "19": 10.586,
                "20": 8.449,
                "21": 5.743,
                "22": 0.437,
                "23": 0.212,
            },
            "Saturday": {
                "0": -9.89,
                "1": -5.567,
                "2": -10.079,
                "3": -13.012,
                "4": -15.715,
                "5": -17.409,
                "6": -16.93,
                "7": -18.33,
                "8": -11.393,
                "9": -5.189,
                "10": 1.373,
                "11": 4.176,
                "12": 5.82,
                "13": 3.753,
                "14": -1.224,
                "15": -4.762,
                "16": -6.636,
                "17": -6.093,
                "18": -0.582,
                "19": 4.266,
                "20": 3.485,
                "21": -1.369,
                "22": -4.96,
                "23": -3.281,
            },
            "Sunday": {
                "0": -8.829,
                "1": -12.44,
                "2": -18.147,
                "3": -21.677,
                "4": -24.611,
                "5": -25.749,
                "6": -26.193,
                "7": -31.463,
                "8": -27.006,
                "9": -18.894,
                "10": -12.531,
                "11": -7.998,
                "12": -3.325,
                "13": -4.799,
                "14": -10.263,
                "15": -14.092,
                "16": -16.557,
                "17": -15.969,
                "18": -8.657,
                "19": -1.527,
                "20": 0.817,
                "21": -1.445,
                "22": -3.417,
                "23": -1.12,
            },
        }

    def simulate(self, time_discretization, n_samples, factor=1.0):
        T = time_discretization.get_T_as_timedelta()
        end = self.start_date + T
        freq = T / time_discretization.n_steps
        # generate dates for each hour (instead of given frequency -> later resample to given frequency)
        dates = pd.date_range(start=self.start_date,
                              end=end, freq='1h', inclusive='both')

        df = pd.DataFrame(dates, columns=['date'])
        df['weekday'] = df['date'].dt.day_name()
        df['hour'] = df['date'].dt.hour
        df.loc[df['hour'] == 0, ['hour']] = 24  # replace 0 with 24
        df = pd.get_dummies(df, columns=['weekday', 'hour'])
        df['time_since_start'] = (
            df['date'] - self.start_date) / pd.Timedelta('1d')

        # add level, trend and seasonality to the price
        df['price'] = self.season_param['level'] + self.season_param['trend'] * df['time_since_start'] + \
            self.season_param['seasonal_factor'] * np.cos(
                (self.season_param['seasonal_shift'] + 2 * np.pi * df['time_since_start']) / 365)
        # add weekday and hour effects
        for weekday, hourly_variations in self.weekday_hourly_variations.items():
            for hour, coeff in hourly_variations.items():
                # check if hour and weekday are in the dataframe
                if f'hour_{hour}' in df.columns and f'weekday_{weekday}' in df.columns:
                    df['price'] += factor * df[f'hour_{hour}'] * \
                        df[f'weekday_{weekday}'] * coeff

        # convert to numpy array
        # resample to the desired frequency
        sample_paths = df.resample(freq, on='date')['price'].mean().values
        # add n_samples and n_dim dimensions
        sample_paths = np.expand_dims(sample_paths, axis=[0, 1])
        # repeat the sample paths n_samples times
        sample_paths = sample_paths.repeat(repeats=n_samples, axis=0)
        return sample_paths


class BrownianMotionSimulator(BaseSimulator):
    """Simulator for multi-dimensional Brownian motion.

    Parameters
    ----------
    mean : array_like
        Mean of the Brownian motion increments per unit time.
        If `mean` is a scalar, it is interpreted as the mean of a Brownian motion with `n_dim` dimensions.
    cov : array_like
        Covariance matrix of the Brownian motion increments per unit time.
        If `cov` is a scalar, it is assumed that the Brownian motion increments are uncorrelated and have variance `cov`.
    initial_value : array_like, optional
        Initial value of the Brownian motion. The default is 0.
        If `initial_value` is a scalar, it is interpreted as the initial value of a Brownian motion with `n_dim` dimensions.
    unit : str, optional
        Time unit. The default is 'd'.
    n_dim : int, optional
        Number of dimensions of the Brownian motion.
        If not specified, it is inferred from the shape of `mean` if `mean` is an array. 
        If `mean` is a scalar, `n_dim` is set to 1.
    """

    def __init__(self, mean=0, cov=1, unit='d', initial_value=0, n_dim=1) -> None:
        if np.isscalar(mean):
            mean = np.full(n_dim, mean)
        else:
            n_dim = len(mean)  # infer n_dim from mean
        assert mean.shape == (n_dim,), "mean must have shape (n_dim,)"
        if np.isscalar(cov):
            cov = np.eye(n_dim) * cov
        assert cov.shape == (
            n_dim, n_dim), "cov must have shape (n_dim, n_dim)"
        assert np.all(np.linalg.eigvals(cov) >=
                      0), "cov must be positive semi-definite"
        if np.isscalar(initial_value):
            initial_value = np.full(n_dim, initial_value)
        assert initial_value.shape == (
            n_dim,), "initial_value must have shape (n_dim,)"
        super().__init__(n_dim=n_dim)  # initialize base class

        self.mean, self.cov = mean, cov
        self.unit = unit
        self.initial_value = initial_value

    def simulate(self, time_discretization, n_samples):
        # get time discretization
        T, n_steps = time_discretization.get_T(
            unit=self.unit), time_discretization.n_steps
        dt = T / n_steps
        # simulate Browninan increments
        rv = stats.multivariate_normal(
            mean=self.mean*dt, cov=self.cov*dt, allow_singular=True)
        increments = rv.rvs(size=(n_samples, n_steps)).reshape(
            n_samples, n_steps, self.n_dim)
        # swap axes to get shape (n_samples, n_dim, n_steps)
        increments = np.swapaxes(increments, 1, 2)
        # add initial value to increments
        initial_value = np.expand_dims(
            self.initial_value, axis=(0, -1)).repeat(n_samples, axis=0)
        increments = np.append(initial_value, increments,
                               axis=2)  # add initial value
        # add Brownian motion increments to get Brownian motion sample paths
        sample_paths = np.cumsum(increments, axis=2)
        return sample_paths


class PoissonProcessSimulator(BaseSimulator):
    def __init__(self, intensity=1, unit='d', initial_value=0, n_dim=1) -> None:
        super().__init__(n_dim=n_dim)
        self.intensity = intensity
        self.unit = unit
        self.initial_value = initial_value

    def simulate(self, time_discretization, n_samples):
        T, n_steps = time_discretization.get_T(
            unit=self.unit), time_discretization.n_steps
        dt = T / n_steps  # compute time step dt
        self.rv = stats.poisson(mu=self.intensity*dt)

        # simulate Poisson process increments
        increments = self.rv.rvs(size=(n_samples, 1, n_steps))
        initial_value = self.initial_value * np.ones((n_samples, 1, 1))
        # append initial value to simulated increments
        increments = np.append(initial_value, increments, axis=2)
        sample_paths = increments.cumsum(axis=2)
        return sample_paths


class CompoundPoissonProcess(BaseSimulator):
    def __init__(self, jumper=stats.chi2(df=1), intensity=1, unit='d', initial_value=0, n_dim=1) -> None:
        super().__init__(n_dim=n_dim)
        self.jumper = jumper
        self.intensity = intensity
        self.poisson_process = PoissonProcessSimulator(intensity=self.intensity,
                                                       unit=unit,
                                                       initial_value=initial_value,
                                                       n_dim=self.n_dim)
        self.unit = unit
        self.initial_value = initial_value

    def simulate(self, time_discretization, n_samples):
        T, n_steps = time_discretization.get_T(
            unit=self.unit), time_discretization.n_steps
        dt = T / n_steps  # compute time step dt
        poisson_process = self.poisson_process.simulate(
            time_discretization=time_discretization, n_samples=n_samples).astype(int)
        max_jumps = poisson_process[:, :, -1].max()

        # simulate Poisson process increments
        jumps = self.jumper.rvs(size=(n_samples, self.n_dim, max_jumps))
        jumps = np.append(
            np.zeros(shape=(n_samples, self.n_dim, 1)), jumps, axis=2)
        jumps = jumps.cumsum(axis=2)
        # generate jump process with with Poisson process
        sample_paths = np.take_along_axis(
            arr=jumps, indices=poisson_process, axis=2)
        return sample_paths


class NIGLevySimulator(BaseSimulator):
    def __init__(self, a=1, b=0, loc=0, scale=1, unit='d', initial_value=0, n_dim=1) -> None:
        super().__init__(n_dim=n_dim)
        self.a, self.b = a, b
        self.loc, self.scale = loc, scale
        self.unit = unit

        self.initial_value = initial_value

    def simulate(self, time_discretization, n_samples):
        T, n_steps = time_discretization.get_T(
            unit=self.unit), time_discretization.n_steps
        dt = T / n_steps  # compute time step dt
        self.rv = stats.norminvgauss(
            a=self.a*dt, b=self.b*dt, loc=self.loc*dt, scale=self.scale*dt)

        # simulate Levy process increments
        increments = self.rv.rvs(size=(n_samples, 1, n_steps))
        initial_value = self.initial_value * np.ones((n_samples, 1, 1))
        # append initial value to simulated increments
        increments = np.append(initial_value, increments, axis=2)
        sample_paths = increments.cumsum(axis=2)
        return sample_paths


class IGLevySimulator(BaseSimulator):
    def __init__(self, mu=1, lambd=1, unit='d', initial_value=0, n_dim=1) -> None:
        super().__init__(n_dim=n_dim)
        self.mu = mu / lambd
        self.loc, self.scale = 0, lambd
        self.unit = unit

        self.initial_value = initial_value

    def simulate(self, time_discretization, n_samples):
        T, n_steps = time_discretization.get_T(
            unit=self.unit), time_discretization.n_steps
        dt = T / n_steps  # compute time step dt
        self.rv = stats.invgauss(mu=self.mu/dt, scale=self.scale*dt**2)

        # simulate Levy process increments
        increments = self.rv.rvs(size=(n_samples, 1, n_steps))
        initial_value = self.initial_value * np.ones((n_samples, 1, 1))
        # append initial value to simulated increments
        increments = np.append(initial_value, increments, axis=2)
        sample_paths = increments.cumsum(axis=2)
        return sample_paths


class OrnsteinUhlenbeckSimulator(BaseSimulator):
    def __init__(self, driver, stationary=True, resample_factor=10, reversion_rate=1, mean=0, unit='d', initial_value=0, n_dim=1) -> None:
        super().__init__(n_dim=n_dim)
        self.driver = driver
        self.stationary = stationary
        self.resample_factor = resample_factor
        self.reversion_rate = reversion_rate  # reversion parameter
        self.mean = mean  # mean parameter
        self.unit = unit
        self.initial_value = initial_value

    def simulate(self, time_discretization, n_samples, increments=None, return_increments=False):
        # compute parameters
        T, n_steps = time_discretization.get_T(
            unit=self.unit), time_discretization.n_steps

        # create new time discretization if stationary
        if self.stationary:  # if stationary, double time horizon and number of steps
            refined_td = TimeDiscretization(
                T=2*T, n_steps=2*n_steps*self.resample_factor, unit=self.unit)
        else:  # else use given time discretization
            refined_td = TimeDiscretization(
                T=T, n_steps=n_steps*self.resample_factor, unit=self.unit)
        refined_dt = refined_td.get_T(
            unit=self.unit) / refined_td.n_steps  # compute time step dt

        # compute increments
        if increments is not None:  # check increments
            if self.stationary:
                assert increments.shape == (
                    n_samples, 1, 2*n_steps*self.resample_factor), increments.shape
            else:
                assert increments.shape == (
                    n_samples, 1, n_steps*self.resample_factor), increments.shape
        else:
            # compute increments
            increments = np.diff(self.driver.simulate(
                time_discretization=refined_td, n_samples=n_samples), n=1, axis=2)

        # Euler-Maruyama scheme
        X = {0: np.full((n_samples, self.n_dim, 1),
                        self.initial_value, dtype=float)}
        for step in range(1, refined_td.n_steps+1):
            X[step] = X[step-1] + self.reversion_rate * \
                (self.mean - X[step-1]) * \
                refined_dt + increments[:, :, [step-1]]

        # concatenate sample paths and subsample
        if self.stationary:  # if stationary, only return last n_steps
            sample_paths = np.concatenate(
                [X[step] for step in range(n_steps*self.resample_factor, refined_td.n_steps+1, self.resample_factor)], axis=2)
        else:
            sample_paths = np.concatenate(
                [X[step] for step in range(0, refined_td.n_steps+1, self.resample_factor)], axis=2)

        if return_increments:
            return sample_paths, increments
        else:
            return sample_paths


class CARMA21Simulator(BaseSimulator):
    def __init__(self, driver, alpha1, alpha2, lambd1, lambd2, unit='d', initial_value=0, n_dim=1) -> None:
        super().__init__(n_dim=n_dim)
        self.driver = driver
        self.alpha1, self.alpha2 = alpha1, alpha2
        self.lambd1, self.lambd2 = lambd1, lambd2
        self.unit = unit
        self.initial_value = initial_value

        self.ou_sim1 = OrnsteinUhlenbeckSimulator(driver=self.driver,
                                                  stationary=True,
                                                  reversion_rate=self.lambd1,
                                                  unit=self.unit,
                                                  initial_value=self.initial_value,
                                                  n_dim=self.n_dim)
        self.ou_sim2 = OrnsteinUhlenbeckSimulator(driver=self.driver,
                                                  stationary=True,
                                                  reversion_rate=self.lambd2,
                                                  unit=self.unit,
                                                  initial_value=self.initial_value,
                                                  n_dim=self.n_dim)

    def simulate(self, time_discretization, n_samples):
        sample_paths1, increments = self.ou_sim1.simulate(
            time_discretization, n_samples, return_increments=True)
        sample_paths2 = self.ou_sim1.simulate(
            time_discretization, n_samples, increments=increments)
        return self.alpha1 * sample_paths1 + self.alpha2 * sample_paths2


class MultiFactorSimulator(BaseSimulator):
    def __init__(self, simulators, coeffs=[], arithmetic=True, n_dim=1):
        super().__init__(n_dim=n_dim)
        self.simulators = simulators
        self.coeffs = coeffs
        self.arithmetic = arithmetic

    def simulate(self, time_discretization, n_samples):
        sample_paths = 0
        for i, sim in enumerate(self.simulators):
            coeff = self.coeffs[i] if self.coeffs else 1
            sample_paths += coeff * \
                sim.simulate(
                    time_discretization=time_discretization, n_samples=n_samples)
        return sample_paths


class LevySemistationarySimulator(BaseSimulator):
    def __init__(self, kernel, driver=BrownianMotionSimulator(), vol=ConstantProcess(1),
                 mu=0, c=1, gamma=0, drift_kernel=None,
                 truncation_factor=2, unit='d', n_dim=1):
        super().__init__(n_dim=n_dim)

        self.driver = driver
        self.kernel = kernel
        self.vol = vol

        self.mu = mu
        self.c = c
        self.gamma = gamma
        self.drift_kernel = drift_kernel

        self.truncation_factor = truncation_factor

    def simulate(self, time_discretization, n_samples):
        # create extended time discretization
        T, n_steps, unit = time_discretization.get_T(
        ), time_discretization.n_steps, time_discretization.unit
        dt = T / n_steps
        M = self.truncation_factor*n_steps

        # add constant mu
        sample_paths = np.full(
            shape=(n_samples, self.n_dim, n_steps+1), fill_value=self.mu, dtype=np.float64)

        # add vol term
        extended_td = TimeDiscretization(
            T=dt*(M+n_steps), n_steps=M+n_steps, unit=unit)
        vol = self.vol.simulate(
            time_discretization=extended_td, n_samples=n_samples)
        extended_td = TimeDiscretization(
            T=dt*(M+n_steps+1), n_steps=M+n_steps+1, unit=unit)  # to have matching increments
        increments = np.diff(self.driver.simulate(
            time_discretization=extended_td, n_samples=n_samples), axis=2)
        Sigma = vol * increments
        G = self.kernel(np.arange(start=0, stop=M+n_steps+1)
                        * dt)[np.newaxis, np.newaxis, :]
        Y = signal.fftconvolve(G, Sigma, axes=[2])
        sample_paths += self.c * Y[:, :, M:M+n_steps+1]

        # add drift term
        if self.gamma != 0:
            Sigma = np.square(vol) * dt
            assert self.drift_kernel is not None, 'No drift_kernel given'
            G = self.drift_kernel(
                np.arange(start=0, stop=M+n_steps+1) * dt)[np.newaxis, np.newaxis, :]
            Y = signal.fftconvolve(G, Sigma, axes=[2])
            sample_paths += self.gamma * Y[:, :, M:M+n_steps+1]

        return sample_paths


class WindPenetrationIndexBSS(LevySemistationarySimulator):
    def __init__(self, lambd=-0.5, chi=1.71, phi=1.71, mu=-0.07, Sigma=0.22**2, gamma=0.07,
                 ny_bar=0.88, lambd_bar=0.39,
                 unit='d', n_dim=1):
        # initialize subordinator
        simulators = [IGLevySimulator(mu=np.sqrt(phi/chi)/2, lambd=phi/4),
                      CompoundPoissonProcess(intensity=2/np.sqrt(phi*chi))]
        coeffs = [1, 1 / chi]

        time_change = functools.partial(scale_time, scale=lambd_bar)
        self.subordinator = ModifiedProcess(MultiFactorSimulator(simulators=simulators, coeffs=coeffs),
                                            time_change=time_change)

        # initialize stochastic volatility process
        self.vol_kernel = functools.partial(
            gamma_kernel, shape=2-2*ny_bar, rate=lambd_bar, div=lambd_bar)
        self.vol2 = LevySemistationarySimulator(
            kernel=self.vol_kernel, driver=self.subordinator)
        self.vol = ModifiedProcess(process=self.vol2, func=np.sqrt)

        # initialize wind model as Brownian semi-stationary process
        self.drift_kernel = functools.partial(
            gamma_kernel, shape=2*ny_bar-1, rate=lambd_bar)
        self.bss_kernel = functools.partial(
            gamma_kernel, shape=2*ny_bar-1, rate=lambd_bar, func=np.sqrt)
        driver = BrownianMotionSimulator()
        super().__init__(kernel=self.bss_kernel, driver=driver, vol=self.vol,
                         mu=mu, c=np.sqrt(Sigma), gamma=gamma, drift_kernel=self.drift_kernel,
                         truncation_factor=2,
                         unit=unit, n_dim=n_dim)


class WindPenetrationIndexNIG(OrnsteinUhlenbeckSimulator):
    """Simulator for wind penetration index, based on Ornstein-Uhlenbeck process with normal inverse Gaussian Lévy process as driving process"""

    def __init__(self,
                 reversion_rate=0.013509848425914805,
                 a=0.831151507599864,
                 b=-0.050690482837973345,
                 loc=0.0021776979127245263,
                 scale=0.142817561629721574,
                 unit='h',):
        self.unit = unit
        self.ghyd_params = {'lambd': -0.5,
                            'alpha_bar': 0.82960430539817,
                            'mu': 0.0021776979127245263,
                            'Sigma': 0.024586246451637945,
                            'gamma': -0.008726438748745399}
        self.nig_params = {
            'a': a,
            'b': b,
            'loc': loc,
            'scale': scale
        }
        self.driver = NIGLevySimulator(**self.nig_params)
        self.reversion_rate = reversion_rate

        super().__init__(driver=self.driver, stationary=True,
                         resample_factor=10, reversion_rate=self.reversion_rate, unit=self.unit, n_dim=1)


class ElectricityPriceMultiFactorSimulator(BaseSimulator):
    """Simulator for electricity prices, based on the model of Rowinska et al. (2017)

    Parameters
    ----------
    wind : LevySemistationarySimulator, optional
        Simulator for wind penetration index, by default WindPenetrationIndex()
    seasonal_impact_factor : float, optional
        Impact factor of seasonal component, by default 1
    long_term_impact_factor : float, optional
        Impact factor of long-term component, by default 1
    short_term_impact_factor : float, optional
        Impact factor of short-term component, by default 1
    wind_impact_factor : float, optional
        Impact factor of wind component, by default 0
    """

    def __init__(self,
                 wind=WindPenetrationIndexNIG(),
                 seasonal_impact_factor=1,
                 long_term_impact_factor=1,
                 short_term_impact_factor=1,
                 wind_impact_factor=0,
                 K_f=0,
                 ):
        # impact factors, to impact of seasonal and stochastic components
        self.seasonal_impact_factor = seasonal_impact_factor
        self.long_term_impact_factor = long_term_impact_factor
        self.short_term_impact_factor = short_term_impact_factor
        self.wind_impact_factor = wind_impact_factor
        self.K_f = K_f

        # initialize simulator and state labels
        if self.wind_impact_factor == 0:  # no wind component
            super().__init__(n_dim=1)
            self.state_labels = ('price', )
        else:
            super().__init__(n_dim=2+self.K_f)
            self.state_labels = ('price', 'wpi', ) + \
                tuple(f'wpi forecast {i}' for i in range(1, K_f+1))

        # seasonal and trend component
        self.seasonal_component = SeasonSimulator()

        # long-term component
        lambd, alpha_bar, mu, Sigma, gamma = -0.5, 0.236, - \
            0.005, 0.385, 0  # see Rowinska et al
        long_term_nig_params = convert_ghyd_to_nig_params(lambd=lambd, alpha_bar=alpha_bar, mu=mu, Sigma=Sigma, gamma=gamma,
                                                          parameterization='alpha_bar')
        self.long_term_component = NIGLevySimulator(
            **long_term_nig_params, unit='d')

        # short-term component
        lambd, alpha_bar, mu, Sigma, gamma = - \
            0.5, 0.962, 1.31, 7.606, -1.342  # see Rowinska et al
        short_term_driver_params = convert_ghyd_to_nig_params(lambd=lambd, alpha_bar=alpha_bar, mu=mu, Sigma=Sigma, gamma=gamma,
                                                              parameterization='alpha_bar')
        driver = NIGLevySimulator(**short_term_driver_params)

        # short_term_CARMA_params = compute_CARMA_parameters(1.413, -0.446, -0.826)
        short_term_CARMA_params = {
            'alpha1': 0.08074445067694089,
            'alpha2': -0.9192555493230591,
            'lambd1': 0.06504421815502415,
            'lambd2': 0.7423921088070488
        }
        self.short_term_component = CARMA21Simulator(
            driver=driver, **short_term_CARMA_params)

        # wind component (model 5)
        self.wind_component = wind

    def simulate(self, time_discretization, n_samples):

        sample_paths = 0
        sample_paths += self.seasonal_component.simulate(
            time_discretization, n_samples, factor=self.seasonal_impact_factor)
        sample_paths += self.long_term_impact_factor * self.long_term_component.simulate(
            time_discretization, n_samples)
        sample_paths += self.short_term_impact_factor * self.short_term_component.simulate(
            time_discretization, n_samples)

        if self.wind_impact_factor == 0:
            return sample_paths
        else:
            wpi = self.wind_component.simulate(time_discretization, n_samples)
            wpi_forecasts = [np.roll(wpi, shift=-i, axis=2)
                             for i in range(1, self.K_f+1)]

            sample_paths += self.wind_impact_factor * \
                wpi * (-74.7)  # add wind component to price
            # add wind and wind forecast to sample paths
            sample_paths = np.concatenate(
                [sample_paths, wpi] + wpi_forecasts, axis=1)
            return sample_paths

    def get_state_labels(self, ):
        return self.state_labels


class GasPriceSimulator(BaseSimulator):
    def __init__(self, initial_value=(100, 100), beta=45, alpha_1=0.25, alpha_2=0.5, sigma_1=0.2, sigma_2=0.2, rho_W=0.6,
                 lambd=2, mu_1=100, mu_2=100, eta_1=30, eta_2=30, rho_J=0.6):
        super().__init__(n_dim=2)
        self.state_labels = ('oil', 'gas')

        self.rng = np.random.default_rng()
        self.initial_value = initial_value
        self.beta = beta
        self.alpha_1, self.alpha_2 = alpha_1, alpha_2
        self.sigma_1, self.sigma_2 = sigma_1, sigma_2
        self.rho_W = rho_W
        self.lambd = lambd
        self.mu_1, self.mu_2 = mu_1, mu_2
        self.eta_1, self.eta_2 = eta_1, eta_2
        self.rho_J = rho_J
        self.n_dim = 2

    def simulate(self, time_discretization, n_samples, drift=True, vol=True, jump=True):
        T, n_steps = time_discretization.get_T(
            unit='y'), time_discretization.n_steps
        dt = T / n_steps  # compute time step dt
        # initialize process X with initial value and Poisson process N
        X = {0: np.repeat(np.array(self.initial_value, dtype=float)[
                          np.newaxis, :, np.newaxis], n_samples, axis=0)}
        N = {0: np.zeros(shape=(n_samples), dtype=int)}

        # compute covariance matrix for Brownian and jump increments
        cov_W = [[self.sigma_1**2 * dt, self.sigma_1 * self.sigma_2 * self.rho_W * dt],
                 [self.sigma_1 * self.sigma_2 * self.rho_W * dt, self.sigma_2**2 * dt]]
        cov_J = [[self.eta_1**2, self.rho_J * self.eta_1 * self.eta_2],
                 [self.rho_J * self.eta_1 * self.eta_2, self.eta_2**2]]

        # pre-computations for jump terms
        # compute Poisson process using Poisson bridge
        N[n_steps] = self.rng.poisson(lam=self.lambd*T, size=n_samples)
        for step in range(1, n_steps):
            # compute N_t conditional on N_t-1 and N_T
            N[step] = N[step-1] + \
                self.rng.binomial(n=N[n_steps]-N[step-1], p=1/(n_steps-step+1))
        indices = np.sort(np.vstack(list(N.values())), axis=0)[
            :, :, np.newaxis]
        max_jumps = N[n_steps].max()  # get maximum amount of jumps
        jumps = self.rng.multivariate_normal(mean=[self.mu_1, self.mu_2], cov=cov_J, size=(
            max_jumps, n_samples))  # generate jumps
        jumps = np.append(arr=np.zeros(shape=(1, n_samples, self.n_dim)),
                          values=jumps, axis=0)  # append initial value zero to jumps
        jumps = jumps.cumsum(axis=0)  # compute cumulative jumps
        # generate jump process with with Poisson process
        jumps = np.take_along_axis(arr=jumps, indices=indices, axis=0)

        # Euler-Maruyama scheme
        for step in range(1, n_steps+1):
            X[step] = X[step-1]  # get previous X value
            # mean reversion term
            if drift:
                X1, X2 = X[step-1][:, 0, :], X[step -
                                               1][:, 1, :]  # separate X1 and X2
                drift_1, drift_2 = self.alpha_1 * \
                    (self.beta - X1) * dt, self.alpha_2 * \
                    (X1 - X2) * dt  # compute drift term
                # add drift increment to X
                X[step] = X[step] + \
                    np.concatenate([drift_1, drift_2], axis=1)[
                    :, :, np.newaxis]
            # volatility term
            if vol:
                sigma_dWt = self.rng.multivariate_normal(mean=[0, 0], cov=cov_W, size=n_samples)[
                    :, :, np.newaxis]  # compute vol term
                # add vol increment term to X
                X[step] = X[step] + X[step-1] * sigma_dWt
            # jump term
            if jump:
                JdNt = (jumps[step, :, :] -
                        jumps[step-1, :, :])[:, :, np.newaxis]
                XdNt = X[step-1] * (N[step] - N[step-1])[:,
                                                         np.newaxis, np.newaxis]
                X[step] = X[step] + (JdNt - XdNt)  # add jump increment to X
        return np.concatenate(list(X.values()), axis=2)

    def get_state_labels(self, ):
        return self.state_labels
