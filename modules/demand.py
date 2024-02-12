import pandas as pd


class ConstantDemandModel():
    '''Constant demand model
    Parameters
    ----------
    demand : float
        Demand in kWh per time unit
    unit : str
        Time unit of demand. Default is 'h' for hours.
    '''

    def __init__(self, demand, unit='h') -> None:
        self.demand = demand  # demand in kWh per time unit
        self.unit = unit  # time unit
        self.demand_per_hour = self.demand * \
            (pd.Timedelta('1h') / pd.Timedelta(1, unit=self.unit))

    def set_time_discretization(self, time_discretization):
        '''Set time discretization

        Parameters
        ----------
        time_discretization : TimeDiscretization
            Time discretization
        '''
        self.time_discretization = time_discretization
        self.__dict__.update(self.time_discretization.get_attributes())

    def get_demand_per_hour(self, step):
        return self.demand_per_hour
