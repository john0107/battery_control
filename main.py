import numpy as np
import pandas as pd
import yaml

from modules import solver, visualize


def main():
    """Main function for the battery control problem."""

    # specify the experiment name, which is the name of the folder in experiments
    experiment = '_demand_comparison'
    # specify the specific paths of the experiment folder
    experiment_folder = f'/experiments/{experiment}'
    saved_solvers_path = f'{experiment_folder}/saved_solvers/'
    cash_flows_path = f'{experiment_folder}/cash_flows/'
    config_file = f'{experiment_folder}/parameters.yaml'

    # Read the config file of the experiment
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)

    base_parameter = config['base_parameter']  # base parameter for the problem
    # parameter changes for the problem
    parameter_changes = config['parameter_changes']
    # create the parameters for the problem
    parameters = solver.create_parameters(base_parameter=base_parameter,
                                          parameter_changes=parameter_changes)

    # Solve the battery control problems
    load = False  # if True, load the solvers from the saved solvers folder
    if load:
        solvers = solver.load_solvers(path=saved_solvers_path)
    else:  # else solve the battery control problems for the given parameters
        save = True  # if True, save the solvers in the saved solvers folder
        solvers = solver.solve_with_parameters(parameters=parameters,
                                               save=save,
                                               path=saved_solvers_path)

    # Compute lower bounds
    initial_control = np.array([0.0, 0.0, 0.0])  # specify the initial control
    n_samples_test = 10**3  # specify the number of samples for the lower bound estimation
    n_batches = 1  # specify the number of batches for the lower bound estimation
    # compute the lower bounds for the given solvers
    lower_bounds = solver.compute_lower_bound_files(solvers,
                                                    initial_control=initial_control,
                                                    path=cash_flows_path,
                                                    n_samples=n_samples_test,
                                                    n_batches=n_batches,
                                                    benchmark=True)

    # Visualize confidence intervals of estimated lower bounds
    # and save the figure in the images folder
    cash_flows = pd.read_csv(f'{cash_flows_path}/cash_flows.csv')
    visualize.plot_confidence_intervals(cash_flows=cash_flows,
                                        path=f'experiments/{experiment}/images/',
                                        filename='confidence_intervals')
    # visualize one sample of the control for each solver
    # and save the figure in the images folder
    visualize.visualize_sample(solvers=solvers,
                               initial_control=initial_control,
                               path=f'experiments/{experiment}/images/')


if __name__ == "__main__":
    main()
