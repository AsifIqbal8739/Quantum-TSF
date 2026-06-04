import torch
import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import real_amplitudes, efficient_su2, pauli_two_design, n_local

from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms.gradients import ReverseEstimatorGradient, ParamShiftEstimatorGradient, SPSAEstimatorGradient

from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector

from qiskit_torch_module import QuantumModule

from Results_Gen.Helper_Utilities.circuit import generate_circuit, generate_circuit_2, _encoding_layer_with_scaling
from Results_Gen.Helper_Utilities.cctTTN import generate_TTN


def ansatz(num_qubits, depth=1, type='RealAmp', entang='full'):
    vqc = QuantumCircuit(num_qubits)

    num_params_encoding = num_qubits
    num_params_scaling = 2 * num_qubits * depth
    params_encoding = ParameterVector('e', length=num_params_encoding)
    params_scaling = ParameterVector('s', length=num_params_scaling)
    params_var = []
    for nd in range(depth):
        # Input Encoding Layer
        vqc.barrier()
        _encoding_layer_with_scaling(vqc, params_encoding, params_scaling, num_qubits=num_qubits, depth=nd)
        vqc.barrier()
        if type == 'RealAmp':
            cct = real_amplitudes(num_qubits=num_qubits, parameter_prefix=f'v{nd}', reps=1,
                                  entanglement=entang, insert_barriers=True)
        elif type == 'EFSU2':
            cct = efficient_su2(num_qubits, reps=1, entanglement=entang, insert_barriers=True,
                                parameter_prefix=f'v{nd}')
        elif type == 'Pauli2D':
            cct = pauli_two_design(num_qubits, reps=1, insert_barriers=True, parameter_prefix=f'v{nd}')
        elif type == 'nLoc':
            cct = n_local(num_qubits, rotation_blocks=["ry", "rz"], entanglement_blocks="cz",
                          entanglement=entang, insert_barriers=True,
                          reps=1, parameter_prefix=f'v{nd}')

        params_var.append(list(cct.parameters))
        vqc = vqc.compose(cct, vqc.qubits)

    params_variational = [item for sublist in params_var for item in sublist]
    return vqc, params_encoding, (params_variational, params_scaling)


class ModelFactory(torch.nn.Module):
    def __init__(self, config):
        super(ModelFactory, self).__init__()

        self.config = config
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.num_threads = config.num_threads
        self.depth = config.cct_depth

        # for QuantumReUP
        self.QuReUp_type = config.QuReUp_type
        self.QuReUp_Ent = config.QuReUp_Ent

        # for QuantumTTN
        self.ctype = config.ctype
        self.rottype = config.rottype
        self.input_bias = config.input_bias
        self.enttype = config.enttype

        # for BuiltIn models
        self.entang = config.entang

        self.Name = config.model_type

        self._build_model()

# model_names = ['Linear', 'QuReUp', 'QuTNN', 'RealAmp', 'EFSU2', 'Pauli2D', 'nLoc']
    def _build_model(self):
        if self.Name == 'Linear':
            self.activation = getattr(torch.nn, self.config.activation)()

            self.layers = torch.nn.ModuleList()
            self.layers.append(torch.nn.Linear(self.seq_len, self.config.num_nodes[0]))
            for ii in range(self.depth-1):
                self.layers.append(torch.nn.Linear(self.config.num_nodes[ii], self.config.num_nodes[ii+1]))
            self.output_layer = torch.nn.Linear(self.config.num_nodes[self.depth-1], self.pred_len)

            weights = [param.data.view(-1) for param in self.parameters()]
            self.weights = torch.cat(weights).tolist()

        elif self.Name == 'QuReUp':
            if self.QuReUp_type == 1:
                vqc, encoding, (variational, scaling) = generate_circuit(num_qubits=self.seq_len, depth=self.depth,
                                                                         input_scaling=True,
                                                                         entanglement_structure=self.QuReUp_Ent)
            elif self.QuReUp_type == 2:
                vqc, encoding, (variational, scaling) = generate_circuit_2(num_qubits=self.seq_len, depth=self.depth,
                                                                           input_scaling=True,
                                                                           entanglement_structure=self.QuReUp_Ent)
            self.weights = variational.params + scaling.params

        elif self.Name == 'QuTNN':
            if not self.input_bias:
                vqc, encoding, (variational, scaling) = generate_TTN(num_qubits=self.seq_len, ctype=self.ctype,
                                                                     input_scaling=True,
                                                                     rottype=self.rottype, enttype=self.enttype)
                self.weights = variational.params + scaling.params

            else:
                vqc, encoding, (variational, scaling, bias) = generate_TTN(num_qubits=self.seq_len, ctype=self.ctype,
                                                                           input_scaling=True, rottype=self.rottype,
                                                                           input_bias=self.input_bias,
                                                                           enttype=self.enttype)
                self.weights = variational.params + scaling.params + bias.params

        else:
            vqc, encoding, (variational, scaling) = ansatz(self.seq_len, depth=self.depth, type=self.Name,
                                                           entang=self.entang)
            self.weights = variational + scaling.params

        # display the number of parameters
        print(f'Model: {self.Name} has {len(self.weights)} Variational Parameters')

        # Here we setup a quantum module to train the vqc
        if self.Name != 'Linear':
            if self.config.useQTM:
                self.model = QuantumModule(circuit=vqc, encoding_params=encoding, variational_params=self.weights,
                                           variational_params_initial=('uniform', {'a': -1.0, 'b': 1.0}),
                                           observables='individualZ'
                                           if self.seq_len == self.pred_len else [
                                               SparsePauliOp(i * 'I' + 'Z' + (self.seq_len - i - 1) * 'I')
                                               for i in range(self.pred_len)],
                                           # [SparsePauliOp(self.seq_len * 'Z') for _ in range(self.pred_len)],
                                           num_threads_forward=self.num_threads,
                                           num_threads_backward=self.num_threads
                                           )
            else:
                qnn = EstimatorQNN(circuit=vqc, input_params=encoding.params, weight_params=self.weights,
                                   observables=[SparsePauliOp(i * 'I' + 'Z' + (self.seq_len - i - 1) * 'I') for i in range(self.seq_len)]
                                   if self.seq_len == self.pred_len else [SparsePauliOp((i * 'I' + 'Z' + (self.seq_len - i - 1) * 'I'))
                                                                          for i in range(self.pred_len)],
                                   gradient=ReverseEstimatorGradient())
                p = np.random.uniform(-1.0, 1.0, size=qnn.num_weights)
                self.model = TorchConnector(qnn, initial_weights=p)

    def forward(self, x):
        if self.Name == 'Linear':
            for layer in self.layers:
                x = self.activation(layer(x))
            x = self.output_layer(x)
        else:
            x = self.model(x)
        return x