# harnessing-mine-thermal-energy

<!-- [![DOI](https://zenodo.org/badge/206265040.svg)](https://zenodo.org/badge/latestdoi/206265040) -->

Repository with  code used for the execution of optimisation model and post-processing results for long-distance energy network; further detail is found in XXXXX.

---

## Features


---

## Requirements
### Overall 
 - Python 3.10 (code has not been tested with other Python versions).
 - Python package dependencies:
 -   Numpy
 -   Pandas
 -   Scipy
 -   Matplotlib
 -   Optional: [TSAM](https://github.com/FZJ-IEK3-VSA/tsam) for time series aggregation.
 -   Optional: [Scienceplots](https://github.com/garrettj403/SciencePlots) for figure formatting.
 
### Model Execution
 - Python package dependencies:
   - [COMANDO](https://jugit.fz-juelich.de/iek-10/public/optimization/comando) and its dependencies (Optimisation framework).
   - [Psweep](https://github.com/elcorto/psweep/tree/0.9.0) (Parametric analysis and housekeeping).
 - [GUROBI](https://www.gurobi.com/) 9.5 or higher

## Setup
 - Check for Python version and package dependencies (virtual environment or separated Conda environment is recommended)
 - SETUP GUROBI solver installation, further details can be found in the COMANDO [readthedocs](https://comando.readthedocs.io/en/latest/interfaces.html#interfaces)
  
---
## Required Inputs

- Thermal load time series. The original study considered time series for single-family houses generated using [RC_BuildingSimulator](https://github.com/architecture-building-systems/RC_BuildingSimulator)
- Ambient temperature time series. The original study used TMY data from [PV-GIS](https://re.jrc.ec.europa.eu/pvg_tools/en/)
- Ground temperature time series (for pipe losses calculation). The original study derived the ground temperature from the ambient temperature.
- Horizon of analysis
- Cost of components and commodities. Source of included data detailed in XXXXX

## Usage

The code includes sample cases: a single-family house (SFH) and a decentralised thermal system.

Further component examples are found in the [COMANDO](https://jugit.fz-juelich.de/iek-10/public/optimization/comando) list of examples, 

## Considerations
- COMANDO is solver agnostic; minor modifications to the code should be needed to run the code with other solvers, although this hasn't been tested so far.
- The individual heat pumps/RCAC components are reversible. If time aggregation is used, be careful with simultaneous heating and cooling demand, which will result in an unfeasible model.
- The study file is set up to run using Python multiprocessing capabilities. If this isn't desired, the code must be changed accordingly.
