import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from tqdm.autonotebook import tqdm, trange

from modules.time_discretization import TimeDiscretization


def save_or_show_fig(fig, path=None, filename=None, dpi=600):
    """Save figure or show it.
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save or show
    path : str, optional
        Path to the folder where the figure will be saved, by default None
        If None, the figure will be shown
    filename : str, optional
        Name of the figure file, by default None
        '.png' will be added to the filename
        If None, the figure will be shown
    dpi : int, optional
        Dots per inch of the figure, by default 600
    """
    if path:
        if path[-1] != '/':
            path += '/'
        if filename is None:
            raise ValueError('filename must be specified if path is specified')
        # save figure
        fig.get_figure().savefig(f'{path}{filename}.png', dpi=dpi)
    else:  # show figure
        plt.show()


def plot_confidence_intervals(cash_flows, path=None, filename='confidence_intervals', dpi=600, **kwargs):
    """Plot the confidence intervals of the cash flows.
    Parameters
    ----------
    cash_flows : dict
        Dictionary with the cash flows of the solvers.
    path : str
        Path to the folder where the image will be saved
    filename : str, optional
        Name of the image file, by default 'confidence_intervals'
        '.png' will be added to the filename
    dpi : int, optional
        Dots per inch of the image, by default 600
    **kwargs : dict
        Keyword arguments passed to sns.pointplot
    """
    data = pd.DataFrame(np.concatenate(list(cash_flows.values()), axis=1),
                        columns=cash_flows.keys())
    confidence_intervals_plot = sns.pointplot(data=data,
                                              errorbar=('se', 2),
                                              orient='h',
                                              capsize=.4,
                                              join=False,
                                              **kwargs)
    save_or_show_fig(confidence_intervals_plot, path=path,
                     filename=filename, dpi=dpi)


def simulate_sample(solver, initial_control, X=None, progress_bar=True):
    if X is None:
        X = solver.simulator.simulate(
            time_discretization=solver.environment.time_discretization, n_samples=1)
    control_history = [np.array(initial_control, ndmin=2)]
    optimal_actions_history = []
    cash_flow_history, breakdown_cash_flow_history = [], []
    for step in trange(solver.n_steps, disable=(not progress_bar), leave=False):
        control = control_history[-1]  # get current control
        features = solver.conti_func.transformer.transform(X[:, :, step])
        action_values = solver.compute_action_values(
            step, control, X, features)  # compute action values
        optimal_actions_idx = np.nanargmin(action_values, axis=1)
        admissible_actions = solver.environment.get_admissible_actions(
            step=step, control=control)
        optimal_actions = solver.environment.get_admissible_actions(
            step=step, control=control)[0, [optimal_actions_idx]]
        optimal_actions_history += [optimal_actions]
        total_cash_flow, breakdown_cash_flow = solver.environment.cash_flow(
            step, optimal_actions, control, X[:, :, step], detailed_breakdown=True)
        # add cash flow for optimal action
        cash_flow_history += [total_cash_flow]
        # add breakdown cashflow as tuple
        breakdown_cash_flow_history += [breakdown_cash_flow]
        # update control with optimal action
        control_history += [solver.environment.update_control(
            step, optimal_actions, control)[:, :, 0]]
    return X, control_history, optimal_actions_history, cash_flow_history, breakdown_cash_flow_history


def visualize_sample(solvers, initial_control, X=None, combined=False, path=None, dpi=600, only_SoC=False, title_map=None, reference_solver=None, max_state_index=1):
    history, time_grid = {}, {}
    if X is None:
        tds = {label: solver.environment.time_discretization for label,
               solver in solvers.items()}
        if reference_solver is not None:
            T, unit = tds[reference_solver].get_T(), tds[reference_solver].unit
        else:
            T, unit = next(iter(tds.values())).get_T(), next(
                iter(tds.values())).unit
        n_steps = {}
        for label, td in tds.items():
            if td.get_T(unit=unit) != T:
                raise ValueError()
            n_steps[label] = td.n_steps
        finest_n_steps = math.lcm(*n_steps.values())
        finest_td = TimeDiscretization(T=T, n_steps=finest_n_steps, unit=unit)
        if reference_solver is not None:
            finest_X = solvers[reference_solver].simulator.simulate(
                time_discretization=finest_td, n_samples=1)
        else:
            finest_X = next(iter(solvers.values())).simulator.simulate(
                time_discretization=finest_td, n_samples=1)
        X = {}
        for label, td in tds.items():
            X[label] = finest_X[:, :, ::finest_n_steps//td.n_steps]
    for label, solver in tqdm(solvers.items(), leave=True):
        _, control_history, optimal_actions_history, cash_flow_history, breakdown_cash_flow_history = simulate_sample(
            solver, initial_control=initial_control, X=X[label])
        history[label] = (control_history, cash_flow_history,
                          breakdown_cash_flow_history)
        time_grid[label] = [T/n_steps[label] *
                            k for k in range(n_steps[label]+1)]
        assert len(control_history) == len(time_grid[label])

    # prepare subplots
    if combined:
        fig, axes = plt.subplots(len(solvers.keys())+1, 1,
                                 figsize=(20, max(12, 3*(len(solvers.keys())+1))))
    else:
        fig, axes = plt.subplots(len(solvers.keys()), 1,
                                 figsize=(20, max(12, 3*len(solvers.keys()))))
    # plt.subplots_adjust(right=0.7)
    plt.tight_layout()

    # prepare combined plot
    if reference_solver is not None:
        price_df = pd.DataFrame(finest_X[0, 0:max_state_index, :].T,
                                columns=solvers[reference_solver].simulator.get_state_labels()[
            0:max_state_index],
            index=[T/finest_n_steps * k for k in range(finest_n_steps+1)])
    else:
        price_df = pd.DataFrame(finest_X[0, 0:max_state_index, :].T,
                                columns=next(iter(solvers.values())).simulator.get_state_labels()[
            0:max_state_index],
            index=[T/finest_n_steps * k for k in range(finest_n_steps+1)])
    if combined:
        axes[0].title.set_text('combined')
        prices_axis = axes[0].twinx()
        prices_axis.set_ylabel('Electricity Price', color='r')
        sns.lineplot(data=price_df, palette=sns.color_palette(
            "husl", len(price_df.columns)), ax=prices_axis)

    # plot control history
    SoC, C_av, Y_tot_normalized = {}, {}, {}
    dfs = {}
    flag = 1 if combined else 0
    for i, (label, (control_history, cash_flow_history, breakdown_cash_flow_history)) in enumerate(history.items()):
        if title_map is not None:
            axes[i+flag].title.set_text(title_map(label))
        else:
            axes[i+flag].title.set_text(label)
        axes[i+flag].set_ylabel('State of Charge', color='b')
        # axes[i+flag].set_xlabel('Days')
        # plot price
        prices_axis = axes[i+flag].twinx()
        prices_axis.set_ylabel('Electricity Price', color='r')
        sns.lineplot(data=price_df, palette=sns.color_palette(
            "husl", len(price_df.columns)), ax=prices_axis)

        SoC[label] = [control[0][0] for control in control_history]
        C_av[label] = [control[0][1] for control in control_history]
        Y_tot_normalized[label] = [control[0][1] * control[0][2]
                                   for i, control in enumerate(control_history)]
        if only_SoC:
            dfs[label] = pd.DataFrame(np.array([SoC[label]]).T,
                                      columns=[f'SoC'],
                                      index=time_grid[label])
        else:
            dfs[label] = pd.DataFrame(np.array([SoC[label], C_av[label], Y_tot_normalized[label]]).T,
                                      columns=[
                f'SoC {label}', f'C_av {label}', f'Y_tot {label}'],
                index=time_grid[label])
        if combined:
            sns.lineplot(dfs[label], ax=axes[0])
        sns.lineplot(dfs[label], ax=axes[i+flag])
        axes[i+flag].get_legend().remove()
        # axes[i+flag].legend(loc='center left', bbox_to_anchor=(1.04, 0.5))
        # added these three lines
        lines, labels = prices_axis.get_legend_handles_labels()
        lines2, labels2 = axes[i+flag].get_legend_handles_labels()
        prices_axis.legend(lines + lines2, labels + labels2, loc=0)

    if combined:
        axes[0].legend(loc='center left', bbox_to_anchor=(1.04, 0.5))
    save_or_show_fig(fig, path=path, filename='visualized_sample', dpi=dpi)

    # plot of unnormalized y_tot
    Y_tot = {}
    for i, (label, (control_history, _, _)) in enumerate(history.items()):
        Y_tot[label] = [control[0][1] * control[0][2] *
                        i for i, control in enumerate(control_history)]
        fig = sns.lineplot(x=time_grid[label],
                           y=Y_tot[label], label=f'Y_tot {label}')
    plt.show()

    # plot of C_av
    for label in history.keys():
        sns.lineplot(x=time_grid[label], y=C_av[label], label=f'C_av {label}')
    plt.show()

    # plot cash flow
    # convert prices from €/MWh to €/kWh
    baseline = finest_X[0, 0, :].cumsum()[:-1] / 10**3
    # get environment of first solver (only time discretization and demand is needed)
    env0 = next(iter(solvers.values())).environment  #
    baseline *= finest_td.delta_t  # time step in hours
    baseline *= env0.demand.demand_per_hour  # demand per hour
    sns.lineplot(x=[T/finest_n_steps * k for k in range(finest_n_steps)],
                 y=baseline, label='base line')

    for label, (_, cash_flow_history, _) in history.items():
        cum_cash = np.concatenate(cash_flow_history).cumsum()
        sns.lineplot(x=time_grid[label][:-1],
                     y=cum_cash, label=f'cashflow {label}')
    plt.show()

    # plot total cash flow difference to baseline
    for label, (_, cash_flow_history, _) in history.items():
        cum_cash = np.concatenate(cash_flow_history).cumsum()
        diff = cum_cash - baseline[::finest_n_steps //
                                   solvers[label].environment.time_discretization.n_steps]
        sns.lineplot(x=time_grid[label][:-1], y=diff,
                     label=f'combined cashflow {label} compared to base line')
    plt.show()

    # plot electricity cost difference compared to baseline
    for label, (_, _, breakdown_cash_flow_history) in history.items():
        breakdown_cash_flow_history = np.array(
            breakdown_cash_flow_history).squeeze().cumsum(axis=0).T
        for cash_flow, cost in zip(breakdown_cash_flow_history, ['elec', 'running', 'ageing']):
            if cost == 'elec':
                diff = cash_flow - \
                    baseline[::finest_n_steps //
                             solvers[label].environment.time_discretization.n_steps]
                sns.lineplot(x=time_grid[label][:-1], y=diff,
                             label=f'{cost} cost {label} compared to baseline')
    plt.show()

    # plot ageing cost
    for label, (_, _, breakdown_cash_flow_history) in history.items():
        breakdown_cash_flow_history = np.array(
            breakdown_cash_flow_history).squeeze().cumsum(axis=0).T
        for cash_flow, cost in zip(breakdown_cash_flow_history, ['elec', 'running', 'ageing']):
            if cost == 'ageing':
                sns.lineplot(x=time_grid[label][:-1],
                             y=cash_flow, label=f'{cost} cost {label}')
    plt.show()

    # TODO: add plot for specific degradation functions independent of the model

    # plot degradation
    for label, (control_history, _, _) in history.items():
        degradation_state = np.concatenate([
            solvers[label].environment.battery.get_degradation_state(step=i, control=control) for i, control in enumerate(control_history)
        ]).flatten()
        sns.lineplot(x=time_grid[label], y=degradation_state,
                     label=f'degradation {label}')
    plt.show()
