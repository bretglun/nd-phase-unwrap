import numpy as np
import thinqpbo as tq


def QPBO(D, V, cyclic):
    shape = D.shape[1:]
    numNodes = np.prod(shape)
    strides = np.zeros(shape, dtype=bool).strides
    D = D.reshape(2, numNodes)
    V = V.reshape(4, len(shape), numNodes)

    graph = tq.QPBOFloat()
    graph.add_node(numNodes)
    
    # Add unary terms:
    for i in range(numNodes):
        graph.add_unary_term(i, D[0, i], D[1, i])
    
    # Add binary terms:
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


def betaJumpMove(p, beta, wD, D, wV, V, cyclic):
    pb = p + beta
    D[0,...] = wD * p**2
    D[1,...] = wD * pb**2
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


def get_weights(magn, pixel_spacing):
    M2p = magn**2
    wD = M2p # Data cost weights (to avoid global jumps)
    wV = np.zeros((magn.ndim, *magn.shape))
    for dim in range(magn.ndim):
        M2q = np.roll(M2p, -1, axis=dim)
        with np.errstate(divide='ignore', invalid='ignore'): # suppress divide-by-zero warning
            wV[dim] = M2p * M2q / (M2p + M2q) / pixel_spacing[dim]
    wV[np.isnan(wV)] = 0.
    return wD, wV


def getEnergyWithWeights(p, wD, wV, cyclic):
    e = np.sum(wD * p**2)
    for dim in range(p.ndim):
        e += np.sum(wV[dim] * (p  - np.roll(p , -1, axis=dim))**2)
        if not cyclic[dim]:
            e -= np.sum(wV[dim].take(-1, axis=dim) * (p.take(-1, axis=dim) - p.take(0, axis=dim))**2)
    return e


def getEnergy(p, m, mu):
    wD, wV = get_weights(m, mu)
    cyclic = [False, False, False, True] # which dims are cyclic?
    return getEnergyWithWeights(p, wD, wV, cyclic)


def removeDrift(p, m, period):
    meanPhase = np.mean(m * p) / np.mean(m)
    nDriftedPeriods = int(round(meanPhase/period))
    return p - nDriftedPeriods * period


def unwrap(arr, config):
    magn, phase = np.abs(arr), np.angle(arr)

    wD, wV = get_weights(magn, config.pixel_spacing)
    if not config.data_cost:
        wD *= 0
    
    D = np.zeros((2, *arr.shape))
    V = np.zeros((4, arr.ndim, *arr.shape))

    minEnergy = getEnergyWithWeights(phase, wD, wV, config.cyclic)
    improved = True
    while improved:
        improved = False
        for beta in [2 * np.pi, -2 * np.pi]: # 2pi jump moves with alternating sign
            phase_updated = betaJumpMove(phase, beta, wD, D, wV, V, config.cyclic)
            energy = getEnergyWithWeights(phase_updated, wD, wV, config.cyclic)
            if energy < minEnergy:
                improved = True
                phase = phase_updated
                minEnergy = energy
    if not config.data_cost:
        phase = removeDrift(phase, magn, period=2*np.pi)
    return phase