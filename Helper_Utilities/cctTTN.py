# An attempt at creating a TTN based Quantum Circuit
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import RGate
import matplotlib.pyplot as plt
plt.ion()


def _unitary_2q(vqc, qbits, params, ctype='top'):
    # A two qubit unitary which applies single bit unitaries and a cx gate
    vqc.rz(params[0], qbits[0])
    vqc.ry(params[1], qbits[1])
    if ctype == 'bot':
        vqc.cx(qbits[0], qbits[1])
    else:
        vqc.cx(qbits[1], qbits[0])


def _unitary_2q_R(vqc, qbits, params, ctype='top'):
    # A two qubit unitary which applies single bit R roations and a cx gate
    vqc.append(RGate(params[0], params[1]), [qbits[0]])
    vqc.append(RGate(params[2], params[3]), [qbits[1]])
    if ctype == 'bot':
        vqc.cx(qbits[0], qbits[1])
    else:
        vqc.cx(qbits[1], qbits[0])


def _encoding_layer_with_scaling(vqc, parameters_encoding, parameters_scaling, num_qubits):
    """ Feature map with multiplicative scaling parameters
    """
    for q in range(num_qubits):
        idx = (2 * q + 1) - 1
        vqc.ry(parameters_scaling[idx] * parameters_encoding[q], q)
        vqc.rz(parameters_scaling[idx+1] * parameters_encoding[q], q)


def _encoding_layer_with_scaling_bias(vqc, parameters_encoding, parameters_scaling, parameters_bias, num_qubits):
    """ Feature map with multiplicative scaling parameters with bias
    """
    for q in range(num_qubits):
        idx = (2 * q + 1) - 1
        vqc.ry(parameters_scaling[idx] * parameters_encoding[q] + parameters_bias[idx], q)
        vqc.rz(parameters_scaling[idx+1] * parameters_encoding[q]+ parameters_bias[idx+1], q)


def _entanglement_layer_full_cz(vqc, qubits):
    """ CZ Entangling layer (all-to-all)
    """
    for i, q in enumerate(qubits):
        for qq in range(i+1, len(qubits)):
            vqc.cz(q, qubits[qq])


def ttn_gate_indices(num_qubits, ctype='top'):
    """
    Computes the indices of qubits that participate in 2-qubit gates at each layer
    of a Tree Tensor Network (TTN) until a single qubit remains.

    :param num_qubits: Total number of input qubits
        ctype = 'bot' or 'top': Bottom or Top Qubit to be measured
    :return: A list where each element is a list of (qubit1, qubit2) pairs for a layer
    """
    layers = []  # Store layers of 2-qubit gate pairs
    qubits = list(range(num_qubits))  # Initial qubit indices

    while len(qubits) > 1:
        layer = []
        next_qubits = []

        for i in range(0, len(qubits) - 1, 2):
            layer.append((qubits[i], qubits[i + 1]))  # Form a 2-qubit gate
            if ctype == 'bot':
                next_qubits.append(qubits[i+1])  # Keep one qubit from each pair ( first or the second one)
            elif ctype == 'top':
                next_qubits.append(qubits[i])

        # If odd number of qubits, carry forward the last unpaired qubit
        if len(qubits) % 2 == 1:
            next_qubits.append(qubits[-1])

        layers.append(layer)  # Save current layer
        qubits = next_qubits  # Move to next layer

    return layers


def generate_TTN(num_qubits: int = 4, ctype='top', input_scaling: bool = True, rottype='yz', input_bias=False, enttype='nn'):
    vqc = QuantumCircuit(num_qubits)

    num_params_encoding = num_qubits
    num_params_scaling = 2 * num_qubits
    if rottype == 'yz':
        num_params_variational = 2 * (num_qubits - 1)
    elif rottype == 'rr':
        num_params_variational = 4 * (num_qubits - 1)

    params_encoding = ParameterVector('e', length=num_params_encoding)
    params_scaling = ParameterVector('s', length=num_params_scaling)
    params_variational = ParameterVector('v', length=num_params_variational)
    if input_bias:
        params_bias = ParameterVector('b', length=num_params_scaling)

    # Apply input encoding with scaling
    if not input_bias:
        _encoding_layer_with_scaling(vqc, params_encoding, params_scaling, num_qubits)
    else:
        _encoding_layer_with_scaling_bias(vqc, params_encoding, params_scaling, params_bias, num_qubits)

    vqc.barrier()
    # Adding 2 qubit unitaries
    ttn_layers = ttn_gate_indices(num_qubits, ctype=ctype)
    var_param_indx = 0
    for ii in range(len(ttn_layers)):
        layer = ttn_layers[ii]
        # implement full-CZ entanglement here
        if enttype == 'full' and ii > 0:
            flattened_list = [item for tup in layer for item in tup]
            _entanglement_layer_full_cz(vqc, flattened_list)
            vqc.barrier()
        for jj in range(len(layer)):
            idx = layer[jj]
            if rottype == 'yz':
                _unitary_2q(vqc, [idx[0], idx[1]], [params_variational[var_param_indx],
                                                params_variational[var_param_indx+1]], ctype=ctype)
                var_param_indx += 2
            elif rottype == 'rr':
                _unitary_2q_R(vqc, list(idx[0:2]), list(params_variational[var_param_indx:var_param_indx+4]),
                              ctype=ctype)
                var_param_indx += 4

        vqc.barrier()

    if not input_bias:
        return vqc, params_encoding, (params_variational, params_scaling)
    else:
        return vqc, params_encoding, (params_variational, params_scaling, params_bias)

if __name__ == '__main__':
    num_qubits = 12
    # ll = ttn_gate_indices(12)
    vqc, num_pe, (num_pv, num_ps) = generate_TTN(num_qubits=num_qubits)
    vqc.draw(output="mpl", style="clifford")
