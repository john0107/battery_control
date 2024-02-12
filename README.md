# Stochastic Optimal Control of a Battery with Degradation in Stochastic Electricity Markets

This GitHub repository contains the implementation of a stochastic optimal control framework for battery energy storage systems, developed as part of my master thesis.

## Abstract

As the world strives for a more sustainable future, integrating renewable energy sources into power systems presents complex challenges.
In this context, battery energy storage systems emerge as a key component in ensuring a successful transition.
This thesis presents a stochastic optimal control framework for battery energy storage systems from the perspective of a consumer who seeks to minimize her cost of electricity over a fixed time horizon.
The framework includes a realistic battery degradation model and allows for the incorporation of stochastic electricity prices, weather forecasts and uncertain electricity demand.
We formulate this optimal battery control problem as a Markov decision problem and present a least squares Monte Carlo algorithm as a method to solve the problem numerically.
Our numerical results demonstrate the feasibility of our approach and highlight the importance of incorporating battery degradation, stochastic electricity prices, and weather forecasts into the optimal control strategy.

## Features

- Realistic battery degradation model based on average C-rate and Ah-throughput.
- Incorporation of stochastic electricity price models, including a versatile multifactor approach.
- Integration of wind forecasts to improve the performance of the battery storage system.
- Consideration of daily seasonality in influencing optimal battery control strategies.
- Efficient numerical solution using the least squares Monte Carlo algorithm.
