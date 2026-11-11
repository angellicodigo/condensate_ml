"""Helper modules for the Rubisco-EPYC1 differentiable Brownian Dynamics simulation.

This package holds the pure, reusable building blocks that the main sweep script
imports and assembles. Each module covers one area of concern:

- geometry.py : coordinate wrapping, minimum image displacement
- initial_snapshot.py : initial grid layout and full initial coordinate snapshot
- topology.py   : particle labels, type ids, mobilities, bond topologies, steric matrices
- forces.py     : steric, attraction, bond, Brownian integration force computations
- post_simulation_analysis.py : soft/hard contacts and cluster graph metrics
- io.py         : XYZ, CSV, summary JSON, and diagnostic plot export

The main script (torch_kf_bd_kf_sweep_copy.py) imports from here and wires
everything together using NEFI's TimeStepper base class in functional style.
"""
