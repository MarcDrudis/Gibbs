"""Tests QiteGibbsStateBuilder class."""
import unittest

import numpy as np

from qiskit.algorithms.time_evolvers.variational import ImaginaryMcLachlanPrinciple
from qiskit.algorithms.gibbs_state_preparation.varqite_gibbs_state_builder import (
    VarQiteGibbsStateBuilder,
)
from qiskit.algorithms.time_evolvers.variational.var_qite import VarQITE
from qiskit.primitives import Sampler
from qiskit.algorithms.gibbs_state_preparation.default_ansatz_builder import build_ansatz, build_init_ansatz_params_vals
from qiskit.quantum_info import Pauli
from qiskit.circuit.library import EfficientSU2
from qiskit.quantum_info import Statevector, partial_trace



seed = 11
np.random.seed(seed)

hamiltonian = Pauli("ZX")
temperature = 120

ansatz = build_ansatz(2*hamiltonian.num_qubits,1)
init_param_values = build_init_ansatz_params_vals(2*hamiltonian.num_qubits,1)


print(ansatz)


param_dict = dict(zip(ansatz.parameters, init_param_values))



var_princip = ImaginaryMcLachlanPrinciple()

# no sampler given so matrix multiplication will be used
qite_algorithm = VarQITE(ansatz, var_princip, init_param_values, num_timesteps=None)

expected_parameter_values = [
    2.48553698605043e-17,
    -0.392553646634161,
    4.80865003568788e-17,
    6.61879494105776e-17,
    4.23399339414617e-17,
    -0.392553646634161,
    2.05198219009111e-17,
    8.40731618835306e-17,
    -2.73540840610989e-17,
    -0.392553646634161,
    3.02199171321979e-17,
    1.26270155363491e-16,
    3.6188537047339e-17,
    -0.392553646634161,
    3.57368105999006e-17,
    1.94753553277173e-16,
]


sampler_shots = Sampler(options={"seed": seed, "shots": 40})

samplers_dict = {"sampler": Sampler(), "sampler_shots": sampler_shots}

sampler_names = ["sampler", "sampler_shots"]

sampler = samplers_dict["sampler_shots"]

gibbs_state_builder = VarQiteGibbsStateBuilder(sampler, qite_algorithm,ansatz,param_dict)
gibbs_state = gibbs_state_builder.build(hamiltonian, temperature, param_dict)
parameter_values = gibbs_state.gibbs_state_function.data[0][0].params

expected_aux_registers = {1}
expected_hamiltonian = hamiltonian
expected_temperature = temperature


params_dict = dict(zip(ansatz.ordered_parameters, parameter_values))


print(partial_trace(Statevector(gibbs_state.gibbs_state_function), hamiltonian.num_qubits))

# print(gibbs_state.sample())
# print(gibbs_state.gibbs_state_function.draw())

# sampler = Sampler(options={"seed": 123, "shots": 400})
# print(parameter_values)
# result = sampler.run(gibbs_state.gibbs_state_function).result()
# print(result)
