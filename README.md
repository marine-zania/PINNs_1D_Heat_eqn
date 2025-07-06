# 1D Heat Equation Solver using Physics-Informed Neural Networks (PINNs)

This project uses a Physics-Informed Neural Network (PINN) to solve the one-dimensional heat equation:

\[
\frac{\partial u}{\partial t} = \frac{1}{16} \frac{\partial^2 u}{\partial x^2}, \quad \text{for } 0 \leq x \leq 4,\ t > 0
\]

### Conditions:
- **Initial condition**:  
  \[
  u(x, 0) = \frac{1}{2}x(8 - x)
  \]
- **Boundary conditions**:  
  \[
  u(0, t) = 0,\quad u(4, t) = 8
  \]

---

## What's in the code?

- A fully-connected neural network (4 layers) is used to approximate the solution \( u(x, t) \).
- The PDE is enforced using automatic differentiation (PyTorch).
- Loss is computed from three parts: physics (PDE residual), initial condition, and boundary conditions.
- An analytical solution (via Fourier series) is used to compare and verify the model’s performance.
- The results are plotted and saved in the `plots/` folder.

---

## Files
- **`pinns_heat1d.py`**: Main script that contains everything—network definition, training, comparison with exact solution, and plotting.
- **`plots/`**: Folder where plots are saved (it’s created automatically if not present).

---

## How to run

1. Make sure you have Python 3.7+ and install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the script:
   ```bash
   python pinns_heat1d.py
   ```

The script will:
- Train the network for 5000 epochs,
- Compare the predicted and exact solutions at different time steps,
- Plot both the solution curves and the training loss.

---

## Output

- A side-by-side comparison table is printed in the terminal.
- A plot showing predicted vs exact solution at multiple time points is saved as `plots/pinn_vs_exact_comparison.png`.
- A convergence plot (Loss vs Epochs) is also displayed.

---

## Notes

- Code runs on GPU if available, otherwise defaults to CPU.
- You can tweak number of training epochs, network architecture, or domain size if needed.
