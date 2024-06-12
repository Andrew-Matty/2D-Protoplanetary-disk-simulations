# 2D-Protoplanetary-disk-simulations
Astrophysics degree dissertation project.

## Overview
This repository contains the code and data for two-dimensional simulations of dust and gas dynamics in the mid-plane of a protoplanetary disc model. The simulations were conducted using the Dedalus framework in Python, utilizing a spectral method to solve the differential equations governing the system. The project explores various instabilities that arise under different initial conditions, with a focus on their role in the early stages of planetary formation.

## Features
- **Spectral Method Simulations**: Numerical simulations using the Dedalus framework.
- **Instability Analysis**: Identification and analysis of instabilities, including Kelvin-Helmholtz instabilities.
- **Linear Stability Analysis**: Fourier transform analysis to investigate the presence and nature of instabilities.
- **Parameter Variation**: Ten simulations with different initial parameters.

## Getting Started

### Prerequisites
- Python 3.11.0
- Dedalus3

## Project Structure

- `shearv7.py`: Script to execute the simulations.
- `Fourier.py`: Script to perform fourier analysis on the simulations.
- `Plotting Programmes/`: Directory with the code to visualise the data. 
- `plot_snapshots.py`: Script creates snapshots of the simualtion at each timestep.
- `Plot_Single2.py`: Combines the snapshots into a single video file.
- `Previous Sims/`: Directory containing older versions of the simulations.



## Results

The project observed various instabilities in the simulations, primarily Kelvin-Helmholtz and combinations of other instabilities. The streaming instability, first discovered by Youdin & Goodman (2005), was not observed independently in these simulations. The findings contribute to understanding the early stages of planetary formation and the diversity of planetary systems.

## References

- Youdin, A. N., & Goodman, J. (2005). Streaming Instabilities in Protoplanetary Disks.



