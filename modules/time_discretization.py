import pandas as pd


class TimeDiscretization():
    """specifies time discretization, can be used to pass to price simulators and environments.

    Parameters
    ----------
    T : float
        Terminal time T in time unit.
    n_steps : int
        Number of time steps.
        If freq is given, n_steps is inferred from T and freq, and n_steps is overwritten.
    unit : str, optional
        Time unit of terminal time T, by default 'h'.
    freq : str, optional
        Frequency of time steps, by default None.
        If given, n_steps is inferred from T and freq, and overwrites given n_steps.
    """

    def __init__(self, T, n_steps, unit='h', freq=None) -> None:
        self.T = T  # terminal time T in time unit
        self.unit = unit  # time unit
        if freq is not None:  # if frequency is given, infer n_steps
            self.n_steps = int(self.convert_time(
                T, from_unit=self.unit, to_unit=freq))
        else:
            self.n_steps = n_steps  # number of time steps
        # time step size in hours
        self.delta_t = self.get_T(unit='h') / self.n_steps

    def convert_time(self, T, from_unit, to_unit):
        """Convert time from one unit to another.

        Parameters
        ----------
        T : float
            Time to be converted.
        from_unit : str
            Unit of time to be converted from.
        to_unit : str
            Unit of time to be converted to.

        Returns
        -------
        float
            Converted time.
        """
        if from_unit == 'y':  # convert from_unit from 'y' to 'd'
            T, from_unit = 365 * T, 'd'
        if to_unit == 'y':  # convert to_unit from 'y' to 'd'
            T, to_unit = T / 365, 'd'
        # convert to_unit to pandas Timedelta object
        if to_unit[0].isdigit():  # check if to_unit has a number
            to_unit = pd.Timedelta(to_unit)
        else:  # else assume to_unit is 'h', 'd', 'w', 'm', 'y'
            to_unit = pd.Timedelta(1, unit=to_unit)
        return pd.Timedelta(T, unit=from_unit) / to_unit

    def get_T(self, unit=None):
        """Get terminal time T in given time unit.

        Parameters
        ----------
        unit : str, optional
            Time unit, by default time unit of terminal time.

        Returns
        -------
        float
            Terminal time T in given time unit.
        """
        if unit is None:
            unit = self.unit
        return self.convert_time(T=self.T, from_unit=self.unit, to_unit=unit)

    def get_T_as_timedelta(self):
        """Get terminal time T as pandas Timedelta object.

        Returns
        -------
        pd.Timedelta
            Terminal time T as pandas Timedelta object.
        """
        if self.unit == 'y':
            T, unit = 365 * T, 'd'
        else:
            unit = self.unit
        return pd.Timedelta(value=self.T, unit=unit)

    def get_attributes(self):
        """Get all attributes of the time discretization.

        Returns
        -------
        dict
            Dictionary of all attributes of the time discretization.
        """
        return self.__dict__


# ---- util function for time discretization ---- #

def scale_time(td, scale):
    """Scale the time discretization by a factor `scale`.
    Parameters
    ----------
    td : TimeDiscretization
        Time discretization to be scaled.
    scale : float
        Factor to scale the time discretization.

    Returns
    -------
    TimeDiscretization 
        Scaled time discretization.
    """
    return TimeDiscretization(T=td.T*scale, n_steps=td.n_steps, unit=td.unit)
