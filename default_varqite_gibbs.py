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
from scipy.sparse.linalg import expm_multiply,expm
from qiskit.primitives import Sampler

from qiskit.algorithms.gradients import LinCombQFI, LinCombEstimatorGradient

from qiskit.quantum_info import partial_trace

from qiskit.algorithms.gibbs_state_preparation.default_ansatz_builder import build_ansatz, build_init_ansatz_params_vals

from qiskit.algorithms.gibbs_state_preparation.varqite_gibbs_state_builder import VarQiteGibbsStateBuilder
from qiskit.visualization import plot_histogram


def qite_gibbs_state_builder(hamiltonian: SparsePauliOp,beta: float,steps: int = None, depth:int = 1, seed: int = 0):
    ansatz = build_ansatz(2*hamiltonian.num_qubits,depth)
    init_param_values = build_init_ansatz_params_vals(2*hamiltonian.num_qubits,depth)
    algorithm_globals.random_seed = seed
    estimator = Estimator()
    qfi = LinCombQFI(estimator)
    gradient = LinCombEstimatorGradient(estimator)
    var_principle = ImaginaryMcLachlanPrinciple(qfi, gradient)

    var_qite = VarQITE(
        ansatz=ansatz,
        initial_parameters = init_param_values,
        variational_principle = var_principle,
        estimator = estimator,
        num_timesteps=steps,
    )

    BOLTZMANN_CONSTANT = 1.38064852e-2
    temperature = 1/(BOLTZMANN_CONSTANT*beta)

    param_dict = dict(zip(ansatz.parameters, init_param_values))
    sampler  = Sampler()
    gibbs_state_builder = VarQiteGibbsStateBuilder(sampler, var_qite,ansatz,param_dict)
    gibbs_state = gibbs_state_builder.build(hamiltonian, temperature=temperature, problem_hamiltonian_param_dict = param_dict)


    final_circuit = gibbs_state.gibbs_state_function.copy()
    qr = final_circuit.qregs[0]
    non_aux_registers = int(len(qr) / 2)
    new_creg = final_circuit._create_creg(non_aux_registers, "meas")
    final_circuit.add_register(new_creg)
    final_circuit.barrier()
    for i in range(non_aux_registers):
        final_circuit.measure(qr[i], new_creg[i])
        
    qite_mixed_state = partial_trace(Statevector(gibbs_state.gibbs_state_function),list(range(0,hamiltonian.num_qubits)))
    qite_state_probs = qite_mixed_state.probabilities()
    
    
    sampler = Sampler()
    sampled_probs = np.array(list(sampler.run(final_circuit,shots=1000).result().quasi_dists[0].values()))
        
    return {"qite_state_probs": qite_state_probs , "sampled": sampled_probs}


def classical_gibbs_distr(hamiltonian: SparsePauliOp,beta: float):
    theoretical_final_mixed_state = expm(-hamiltonian.to_matrix(sparse=True)*beta)
    probs = np.array(theoretical_final_mixed_state.diagonal())
    probs = probs/np.sum(probs)
    return probs
