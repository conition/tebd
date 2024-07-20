# Time evolving block decimation (TEBD)
Python implementation of the TEBD algorithm for simulation of quantum spin chains.

## Requirements
```bash
pip install numpy
pip install scipy
pip install tensornetwork
pip install matplotlib
pip install seaborn
```

## File structure
`report/`: Details on TEBD theory, implementation considerations, and results

`mps.py`: Functions for creating and operating on matrix product states (MPS)

`tebd.py`: Functions for running the TEBD algorithm

`runtimes.ipynb`: Tests runtime scaling of TEBD and exact diagonlization with chain length

`ground_state.ipynb`: Finds ground state of Heisenberg XXZ model using TEBD and compares to analytical solution

`spin_wave.ipynb`: Uses TEBD to simulate propagation of a spin wave in the Heisenberg XXX model and generates animations in `videos/`