# Monte Carlo Electronic & Ionic Transport Device Simulation

This repository contains a work-in-progress MATLAB-based Monte Carlo simulation for electronic and ionic transport in device structures. The simulation currently focuses on modeling electronic transport, including potential and field computations, particle injection, and observables tracking. Proper scattering interactions and ionic hopping mechanisms are still under development.

## Features

- **Monte Carlo Particle Simulation:** Simulates carrier (electrons and holes) transport using random sampling techniques.
- **Potential and Field Calculation:** Solves the Poisson equation to obtain potential and computes the electric field.
- **GPU Acceleration:** Uses MATLAB’s GPU arrayfun for parallel computation in updating particle positions.
- **Live Plot Updates:** Observables such as current, carrier densities, and potential profiles are updated every 100 steps.
- **Work in Progress:** Note that scattering interactions and ionic hopping (for ions) have not yet been fully implemented.

## Repository Structure

- **Core_refactored.m:** Main simulation script containing the core loop, observables updates, and inline helper functions.
- **Helper Functions:** Included within the main script to handle tasks such as parameter initialization, particle injection, potential/field calculation, and plotting.
- **Live Plot Updates:** The simulation live-updates plots of key observables every 100 time steps to help monitor the simulation progress.

## Requirements

- **MATLAB:** The code is developed and tested using MATLAB.
- **Parallel Computing Toolbox:** Required for GPU acceleration (using `gpuArray.arrayfun`).

## Usage

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/monte-carlo-transport-simulation.git
   cd monte-carlo-transport-simulation
2. **Open MATLAB and Navigate to the Repository Directory.**

3. **Run the Main Script:**

```matlab
Core.m
```
Run code sections in the Core.m script
The simulation will run for the specified number of time steps (default is 3000) and update plots every 100 steps.

## Current Limitations
- **Scattering Interactions:** The current implementation includes placeholder logic for scattering events. Detailed scattering mechanisms are not fully implemented.
- **Ionic Hopping:** While the framework is in place, ionic hopping and associated interactions are not yet incorporated.
- **Performance Tuning:** As this is a work in progress, further code optimizations and validations are planned.
## Future Work
- **Implement Detailed Scattering Models:** Incorporate proper scattering interactions based on physical models.
- **Develop Ionic Hopping Mechanisms:** Add models for ionic transport and hopping to extend the simulation’s applicability.
- **Enhanced Visualization and Analysis:** Improve plotting functions and observables tracking to provide more detailed insight into the simulation dynamics.
- **Code Optimization:** Further optimize code for performance and readability.
## Contributing
Contributions are welcome! If you have suggestions, improvements, or bug fixes, please open an issue or submit a pull request.

## License
This project is open source and available under the MIT License.

## Contact
For questions or further information, please contact HaroldsonPhysics@gmail.com.
