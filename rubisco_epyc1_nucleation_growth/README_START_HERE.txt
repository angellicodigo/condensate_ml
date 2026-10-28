START HERE — Rubisco–EPYC1 nucleation/growth package

This package captures the latest model state from the conversation:
- HOOMD-blue GPU Brownian dynamics
- 24 Rubisco scaffolds
- 8 tethered patches per Rubisco
- 48 EPYC1 chains
- 12 beads/chain, 6 stickers
- 2:1 EPYC1:Rubisco molecular ratio
- 1.5:1 sticker:patch site ratio
- force-based T-P attraction
- bounded soft sterics
- harmonic EPYC1 bonds
- harmonic patch tethers
- GSD + XYZ (OVITO) + CSV + plots
- dual-GPU sweep by running independent parameter sets on GPU 0 and GPU 1

IMPORTANT:
The validated production Python file lives on the lab server from the interactive
debugging session. The placeholder script in this archive does NOT pretend to
be that validated file. Copy the validated server script into this folder before
running. This avoids losing or silently changing the exact working HOOMD API
version-specific implementation.

Recommended parameter ladder is in params.json.
