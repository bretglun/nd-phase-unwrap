import numpy as np
import thinqpbo as tq


def QPBO(D, V, cyclic):
    shape = V.shape[2:]
    numNodes = np.prod(shape)
    strides = np.zeros(shape, dtype=bool).strides

    graph = tq.QPBOFloat()
    graph.add_node(numNodes)
    
    # Add unary terms:
    if D is not None:
        D = D.reshape(2, numNodes)
        for i in range(numNodes):
            graph.add_unary_term(i, D[0, i], D[1, i])
    
    # Add binary terms:
    V = V.reshape(4, len(shape), numNodes)
    for dim in range(len(shape)):
        for i in range(numNodes):
            j = i + strides[dim]
            if (j//strides[dim])%shape[dim]==0: # at edge
                if not cyclic[dim]:
                    continue
                j -= shape[dim] * strides[dim] # cycle over edge
            graph.add_pairwise_term(i, j, V[0, dim, i], V[1, dim, i], V[2, dim, i], V[3, dim, i])
    
    graph.solve()
    
    label = np.zeros(numNodes)
    for i in range(numNodes):
        label[i] = graph.get_label(i)

    return label.reshape(shape)


def beta_jump_move(p, beta, wD, wV, cyclic):
    pb = p + beta
    D = np.array((wD * p**2, wD * pb**2)) if wD is not None else None
    V = np.zeros((4, p.ndim, *p.shape))
    for dim in range(p.ndim):
        V[:, dim, ...] = wV[dim] * [
            (p  - np.roll(p , -1, axis=dim))**2,
            (p  - np.roll(pb, -1, axis=dim))**2,
            (pb - np.roll(p , -1, axis=dim))**2,
            (pb - np.roll(pb, -1, axis=dim))**2
        ]
    label = QPBO(D, V, cyclic)
    p[label==1] += beta
    return p


def get_weights(magn, pixel_spacing, data_cost=False):
    M2p = magn**2
    wD = M2p if data_cost else None
    wV = np.zeros((magn.ndim, *magn.shape))
    for dim in range(magn.ndim):
        M2q = np.roll(M2p, -1, axis=dim)
        with np.errstate(divide='ignore', invalid='ignore'): # suppress divide-by-zero warning
            wV[dim] = M2p * M2q / (M2p + M2q) / pixel_spacing[dim]
    wV[np.isnan(wV)] = 0.
    return wD, wV


def get_energy_with_weights(phase, wD, wV, cyclic):
    energy = np.sum(wD * phase**2) if wD is not None else 0.
    for dim in range(phase.ndim):
        energy += np.sum(wV[dim] * (phase  - np.roll(phase , -1, axis=dim))**2)
        if not cyclic[dim]:
            energy -= np.sum(wV[dim].take(-1, axis=dim) * (phase.take(-1, axis=dim) - phase.take(0, axis=dim))**2)
    return energy


def get_energy(phase, magn, pixel_spacing, data_cost, cyclic):
    wD, wV = get_weights(magn, pixel_spacing, data_cost)
    return get_energy_with_weights(phase, wD, wV, cyclic)


def remove_drift(phase, magn, period):
    mean_phase = np.mean(magn * phase) / np.mean(magn)
    mean_period = int(round(mean_phase/period))
    return phase - mean_period * period


def unwrap(arr, config):
    magn, phase = np.abs(arr), np.angle(arr)

    wD, wV = get_weights(magn, config.pixel_spacing, config.data_cost)
    
    min_energy = get_energy_with_weights(phase, wD, wV, config.cyclic)
    improved = True
    while improved:
        improved = False
        for beta in [2 * np.pi, -2 * np.pi]: # 2pi jump moves with alternating sign
            phase_updated = beta_jump_move(phase, beta, wD, wV, config.cyclic)
            energy = get_energy_with_weights(phase_updated, wD, wV, config.cyclic)
            if energy < min_energy:
                improved = True
                phase = phase_updated
                min_energy = energy
    if not config.data_cost:
        phase = remove_drift(phase, magn, period=2*np.pi)
    return phase