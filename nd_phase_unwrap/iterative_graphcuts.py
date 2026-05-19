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
    shape = V.shape[2:]
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


def beta_jump_move(p, beta, wD, wV, grid_pairs):
    pb = p + beta
    D = np.array((wD * p**2, wD * pb**2)) if wD is not None else np.zeros((2, *p.shape))
    V = np.zeros((4, p.ndim, *p.shape))
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


def get_weights(magn, pixel_spacing, data_cost=0):
    M2p = magn**2
    wD = data_cost * M2p if data_cost > 0 else None
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
    grid_pairs = computeGraphEdges(phase.shape, config.cyclic)

    min_energy = get_energy_with_weights(phase, wD, wV, config.cyclic)
    improved = True
    while improved:
        improved = False
        for beta in [2 * np.pi, -2 * np.pi]: # 2pi jump moves with alternating sign
            phase_updated = beta_jump_move(phase, beta, wD, wV, grid_pairs)
            energy = get_energy_with_weights(phase_updated, wD, wV, config.cyclic)
            if energy < min_energy:
                improved = True
                phase = phase_updated
                min_energy = energy
    if not config.data_cost:
        phase = remove_drift(phase, magn, period=2*np.pi)
    return phase