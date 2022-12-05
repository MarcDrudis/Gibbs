import numpy as np
from qiskit.algorithms.time_evolvers.variational.var_qite import VarQITE
from qiskit.algorithms.time_evolvers.time_evolution_problem import TimeEvolutionProblem
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit.circuit.library import EfficientSU2
from qiskit.utils import algorithm_globals
from qiskit.primitives import Estimator
from qiskit.algorithms.time_evolvers.variational import (
    ImaginaryMcLachlanPrinciple,
)
from qiskit.primitives import Sampler

from qiskit.algorithms.gradients import LinCombQFI, LinCombEstimatorGradient


from qiskit.algorithms.gibbs_state_preparation.default_ansatz_builder import build_ansatz, build_init_ansatz_params_vals

def create_hamiltonian_lattice(num_sites: int) -> SparsePauliOp:
    """Creates an Ising Hamiltonian on a lattice."""
    j_const = 0.1
    g_const = -1.0

    zz_op = ["I" * i + "ZZ" + "I" * (num_sites - i - 2) for i in range(num_sites - 1)]
    x_op = ["I" * i + "X" + "I" * (num_sites - i - 1) for i in range(num_sites)]
    return SparsePauliOp(zz_op) * j_const + SparsePauliOp(x_op) * g_const


hamiltonian = -SparsePauliOp.from_list([("XX",1.0)])

problem = TimeEvolutionProblem(hamiltonian=hamiltonian,time=5)


# ansatz = build_ansatz(2*hamiltonian.num_qubits,3)
# init_param_values = build_init_ansatz_params_vals(2*hamiltonian.num_qubits,1)

ansatz = EfficientSU2(2, reps=1)
init_param_values = np.random.rand(ansatz.num_parameters)

print(ansatz.decompose())
print(init_param_values)

algorithm_globals.random_seed = 123
estimator = Estimator()
qfi = LinCombQFI(estimator)
gradient = LinCombEstimatorGradient(estimator)
var_principle = ImaginaryMcLachlanPrinciple(qfi, gradient)

var_qite = VarQITE(
    ansatz=ansatz,
    initial_parameters = init_param_values,
    variational_principle = var_principle,
    estimator = estimator,
    num_timesteps=None,
)

evolution_result = var_qite.evolve(problem)




print(evolution_result.evolved_state.data[0][0].params)



final_circuit = evolution_result.evolved_state.copy()
final_circuit.measure_all()

sampler =Sampler()
probs = sampler.run(final_circuit,shots=1000)

print(probs.probabilities_dict())