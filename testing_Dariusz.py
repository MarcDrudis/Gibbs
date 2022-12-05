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

from qiskit.algorithms.gibbs_state_preparation.gibbs_state_sampler import GibbsStateSampler

from qiskit.algorithms.gibbs_state_preparation.default_ansatz_builder import (
    build_ansatz,
    build_init_ansatz_params_vals,
)

from qiskit.algorithms.gradients import LinCombQFI, LinCombEstimatorGradient
from qiskit.primitives import Sampler
from qiskit.quantum_info import Pauli

from qiskit.algorithms.time_evolvers.variational import ImaginaryMcLachlanPrinciple

from qiskit.algorithms.gibbs_state_preparation.varqite_gibbs_state_builder import (
    VarQiteGibbsStateBuilder,
)

from qiskit import QuantumCircuit

def create_hamiltonian_lattice(num_sites: int) -> SparsePauliOp:
    """Creates an Ising Hamiltonian on a lattice."""
    j_const = 0.1
    g_const = -1.0

    zz_op = ["I" * i + "ZZ" + "I" * (num_sites - i - 2) for i in range(num_sites - 1)]
    x_op = ["I" * i + "X" + "I" * (num_sites - i - 1) for i in range(num_sites)]
    return SparsePauliOp(zz_op) * j_const + SparsePauliOp(x_op) * g_const


# hamiltonian = -SparsePauliOp.from_list([("XX",1.0)])
# ansatz = EfficientSU2(hamiltonian.num_qubits, reps=1)

 
# algorithm_globals.random_seed = 123
# estimator = Estimator()
# qfi = LinCombQFI(estimator)
# gradient = LinCombEstimatorGradient(estimator)
# var_principle = ImaginaryMcLachlanPrinciple(qfi, gradient)
# params = np.ones(ansatz.num_parameters)
# var_qite = VarQITE(
#     ansatz,
#     params,
#     var_principle,
#     estimator,
#     num_timesteps=None,
# )

# gibbs_builder = VarQiteGibbsStateBuilder(sampler=Sampler(), qite_algorithm=var_qite)
# gibbs_sampler = gibbs_builder.build(hamiltonian,temperature=1.0)
# final_circ = gibbs_sampler.gibbs_state_function
# print(final_circ.draw())
# print(final_circ.parameters)



hamiltonian = -SparsePauliOp.from_list([("XX",1.0)])
num_qubits = hamiltonian.num_qubits
hamiltonian_identity = hamiltonian.tensor(SparsePauliOp.from_list([("I"*num_qubits, 1.0)]))
temperature = 1000

seed = 170
sampler = Sampler(options={"seed": seed, "shots": 1024})

depth = 1

aux_registers = set(range(num_qubits, 2*num_qubits))

ansatz = build_ansatz(2*num_qubits, depth)
# param_values_init = build_init_ansatz_params_vals(2*num_qubits, depth)
param_values_init = np.ones(2*num_qubits)

params_dict = None#dict(zip(ansatz.ordered_parameters, param_values_init))
# gibbs_state = GibbsStateSampler(
#     sampler,
#     gibbs_state_function,
#     hamiltonian,
#     temperature,
#     ansatz,
#     params_dict,
#     aux_registers=aux_registers,
# )

###################################################
estimator = Estimator()
qfi = LinCombQFI(estimator)
gradient = LinCombEstimatorGradient(estimator)
var_principle = ImaginaryMcLachlanPrinciple(qfi, gradient)
var_qite = VarQITE(
    ansatz,
    param_values_init,
    var_principle,
    estimator,
    num_timesteps=None,
)

gibbs_builder = VarQiteGibbsStateBuilder(sampler=sampler, qite_algorithm=var_qite)
####################################################



gibbs_state = gibbs_builder.build(hamiltonian,temperature=temperature,problem_hamiltonian_param_dict=None)
print(gibbs_state.gibbs_state_function)
# probs = gibbs_state.sample()

# print(probs)