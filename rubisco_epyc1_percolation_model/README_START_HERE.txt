START HERE — Percolation-theory Rubisco–EPYC1 model

This is a complementary network model, not a spatial MD simulation.
Rubisco nodes have valency 8; EPYC1 nodes have valency 6. Finite binding
sites are randomly occupied with probability p. Connected components are
computed for a bipartite Rubisco–EPYC1 graph.

Run:
  python percolation_epyc1_rubisco.py --params percolation_params.json --reps 500

Outputs:
  data/*.csv
  plots/*.png

Use this model to ask whether connectivity/percolation alone can explain the
emergence of a system-spanning cluster. Do not interpret p as a directly
measured biochemical on-rate without calibration.
