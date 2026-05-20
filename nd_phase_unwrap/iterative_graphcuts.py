import numpy as np
import maxflow


def computeGraphEdges(shape, cyclic):
    """Flat (i, j) endpoint indices of graph edges, one (i, j) tuple per dim."""
    ndim = len(shape)
    idx = np.arange(int(np.prod(shape))).reshape(shape)
    pairs = []
    for dim in range(ndim):
        if cyclic[dim]:
            ii, jj = idx.ravel(), np.roll(idx, -1, axis=dim).ravel()
        else:
            sl = [slice(None)] * ndim
            sl[dim] = slice(None, -1); ii = idx[tuple(sl)].ravel()
            sl[dim] = slice(1, None);  jj = idx[tuple(sl)].ravel()
        pairs.append((ii.astype(np.int32), jj.astype(np.int32)))
    return pairs


def solveMaxflow(D, V, grid_pairs):
    """Solve a submodular binary MRF via BK max-flow; returns int8 labels in {0,1}."""
    shape = D.shape[1:]
    n = int(np.prod(shape))
    ndim = len(grid_pairs)
    D_flat = D.reshape(2, n)
    V_flat = V.reshape(4, ndim, n)
    src_cap, snk_cap = D_flat[1].copy(), D_flat[0].copy()
    ei, ej, cij, cji = [], [], [], []

    for dim, (ii, jj) in enumerate(grid_pairs):
        Ai, Bi, Ci = V_flat[0, dim, ii], V_flat[1, dim, ii], V_flat[2, dim, ii]
        b, c = Bi - Ai, Ci - Ai  # b+c >= 0 (submodular)
        # Reparameterise into non-negative tedges + directed edges.
        src_cap[ii] += np.maximum(0.0, -b)
        snk_cap[ii] += np.maximum(0.0, -c)
        src_cap[jj] += np.minimum(Ai, Bi)
        snk_cap[jj] += np.minimum(Ai, Ci)
        cap_ij = np.maximum(0.0, b + np.minimum(0.0, c))
        cap_ji = np.maximum(0.0, c + np.minimum(0.0, b))
        mask = (cap_ij > 0) | (cap_ji > 0)
        if np.any(mask):
            ei.append(ii[mask]); ej.append(jj[mask])
            cij.append(cap_ij[mask]); cji.append(cap_ji[mask])

    g = maxflow.GraphFloat()
    nids = g.add_grid_nodes(shape)
    g.add_grid_tedges(nids, src_cap.reshape(shape), snk_cap.reshape(shape))
    if ei:
        g.add_edges(np.concatenate(ei), np.concatenate(ej),
                    np.concatenate(cij), np.concatenate(cji))
    g.maxflow()
    return g.get_grid_segments(nids).astype(np.int8)


def betaJumpMove(p: np.ndarray, beta: float, wD, D, wV, V, grid_pairs):
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
    label = solveMaxflow(D, V, grid_pairs)
    p[label==1] += beta
    return p


def getWeights(magn, pixel_spacing):
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
    wD, wV = getWeights(m, mu)
    cyclic = [False, False, False, True] # which dims are cyclic?
    return getEnergyWithWeights(p, wD, wV, cyclic)


def removeDrift(p, m, period):
    meanPhase = np.mean(m * p) / np.mean(m)
    nDriftedPeriods = int(round(meanPhase/period))
    return p - nDriftedPeriods * period


def unwrap(arr, config):
    magn, phase = np.abs(arr), np.angle(arr)

    wD, wV = getWeights(magn, config.pixel_spacing)
    if not config.data_cost:
        wD *= 0
    
    D = np.zeros((2, *arr.shape))
    V = np.zeros((4, arr.ndim, *arr.shape))
    grid_pairs = computeGraphEdges(phase.shape, config.cyclic)

    minEnergy = getEnergyWithWeights(phase, wD, wV, config.cyclic)
    improved = True
    while improved:
        improved = False
        for beta in [2 * np.pi, -2 * np.pi]: # 2pi jump moves with alternating sign
            phase_updated = betaJumpMove(phase, beta, wD, D, wV, V, grid_pairs)
            energy = getEnergyWithWeights(phase_updated, wD, wV, config.cyclic)
            if energy < minEnergy:
                improved = True
                phase = phase_updated
                minEnergy = energy
    if not config.data_cost:
        phase = removeDrift(phase, magn, period=2*np.pi)
    return phase