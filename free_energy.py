#%%%%%%
from qiskit.quantum_info import SparsePauliOp
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm



hamiltonian = SparsePauliOp.from_list([("XYXI", 1.0), ("IYXX", 1.0)]).to_matrix()
beta0 = 2

def entropy(rho:np.array):
    #compute entropy of rho
    eigvals = np.linalg.eigvals(rho)
    return -np.sum([np.log(eigval)*eigval for eigval in eigvals])
def energy(rho:np.array,hamiltonian:np.array):
    return np.trace(rho@hamiltonian)
    

betas = np.linspace(0.8*beta0, beta0*1.2, 200)
free_energy = []
for beta in betas:
    rho = expm(-beta * hamiltonian)
    part_fun = np.trace(rho)
    rho = rho / part_fun
    free_energy.append(energy(rho,hamiltonian) - (1/beta0) * entropy(rho)  )
    

plt.plot(betas, free_energy)
print(f"The minimum free energy is obtained for ")

# %%
