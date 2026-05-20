import numpy as np
import maxflow
from itertools import product


def get_neighbourhood(radius, pixel_spacing):
    bound = np.floor(radius / np.array(pixel_spacing)).astype(int)
    candidates = product(*(range(-b, b + 1) for b in bound))
    neighbours = [ngb for ngb in candidates
                  if ngb > (0,) * len(pixel_spacing) # filter out duplicates in opposite directions and origin
                  and np.linalg.norm(np.array(ngb) * pixel_spacing) <= radius]
    # make sure immediate neighbours are always included:
    neighbours.extend([immediate for immediate in np.eye(len(pixel_spacing), dtype=int) if tuple(immediate) not in neighbours])
    return np.array(neighbours)


def get_neighbour_indices(neighbourhood, shape, cyclic):
    assert neighbourhood.shape[1] == len(shape)
    assert len(shape) == len(cyclic)
    num_ngb = neighbourhood.shape[0]
    num_voxels = np.prod(shape)
    indices = np.arange(num_voxels).reshape(shape)
    ngb_indices = np.empty(shape=(num_ngb, *shape), dtype=int)
    edge_ngb = np.full(shape=(num_ngb, *shape), fill_value=False)
    for ngb in range(num_ngb):
        for axis, shift in enumerate(neighbourhood[ngb]):
            if cyclic[axis] or shift==0:
                continue
            slices = [slice(None)] * indices.ndim
            if shift < 0:
                slices[axis] = slice(0, -shift)
            if shift > 0:
                slices[axis] = slice(-shift, shape[axis])
            edge_ngb[ngb, *tuple(slices)] = True
        ngb_indices[ngb] = np.roll(indices, shift=-neighbourhood[ngb], axis=range(indices.ndim))
    return ngb_indices.reshape(num_ngb, num_voxels), edge_ngb.reshape(num_ngb, num_voxels)


def solve_maxflow(D, V, ngb_indices):
    num_nodes = V.shape[2]
    num_neighbours = len(ngb_indices)
    
    graph = maxflow.GraphFloat()
    graph.add_grid_nodes(num_nodes)

    src_cap, snk_cap = (D[1], D[0]) if D is not None else (np.zeros(num_nodes, dtype=V.dtype), np.zeros(num_nodes, dtype=V.dtype))
    
    ii = np.arange(num_nodes)
    for q in range(num_neighbours): # loop over neighbours
        jj = ngb_indices[q]
        Ai, Bi, Ci = V[0, q, ii], V[1, q, ii], V[2, q, ii]
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
            graph.add_edges(ii[mask], jj[mask], cap_ij[mask], cap_ji[mask])

    graph.add_grid_tedges(ii, src_cap, snk_cap)
    graph.maxflow() # Boykov-Kolmogorov max-flow algorithm
    return graph.get_grid_segments(ii)


def beta_jump_move(phase, beta, wD, wV, ngb_indices):
    p = phase.flatten()
    pb = p + beta
    D = np.array((wD * p**2, wD * pb**2)) if wD is not None else None
    V = wV * [
        (p  -  p[ngb_indices])**2,
        (p  - pb[ngb_indices])**2,
        (pb -  p[ngb_indices])**2,
        (pb - pb[ngb_indices])**2]
    label = solve_maxflow(D, V, ngb_indices)
    p[label==1] += beta
    return p.reshape(phase.shape)


def get_energy(phase, wD, wV, ngb_indices):
    p = phase.flatten()
    energy = np.sum(wD * p**2) if wD is not None else 0.
    energy += np.sum(wV * (p  - p[ngb_indices])**2)
    return energy


def get_weights(magn, pixel_spacing, neighbourhood, ngb_indices, edge_ngb, data_cost=0):
    M2p = magn.flatten()**2
    wD = data_cost * M2p if data_cost > 0 else None
    wV = np.zeros((magn.ndim, *magn.shape))
    with np.errstate(divide='ignore', invalid='ignore'): # suppress divide-by-zero warning
        wV = 1 / (1/M2p + 1/M2p[ngb_indices])
    wV[np.isnan(wV)] = 0.
    distance = np.linalg.norm(neighbourhood * pixel_spacing, axis=1)[:, None]
    wV /= distance * np.sum(1 / distance)
    wV[edge_ngb] = 0
    return wD, wV


def remove_drift(phase, magn, period):
    mean_phase = np.mean(magn * phase) / np.mean(magn)
    mean_period = int(round(mean_phase/period))
    return phase - mean_period * period


def unwrap(arr, config):
    magn, phase = np.abs(arr), np.angle(arr)

    neighbourhood = get_neighbourhood(config.neighbourhood_radius, config.pixel_spacing)
    ngb_indices, edge_ngb = get_neighbour_indices(neighbourhood, arr.shape, config.cyclic)

    wD, wV = get_weights(magn, config.pixel_spacing, neighbourhood, ngb_indices, edge_ngb, config.data_cost)

    min_energy = get_energy(phase, wD, wV, ngb_indices)
    improved = True
    while improved:
        improved = False
        for beta in [2 * np.pi, -2 * np.pi]: # 2pi jump moves with alternating sign
            phase_updated = beta_jump_move(phase, beta, wD, wV, ngb_indices)
            energy = get_energy(phase_updated, wD, wV, ngb_indices)
            if energy < min_energy:
                improved = True
                phase = phase_updated
                min_energy = energy
    if not config.data_cost:
        phase = remove_drift(phase, magn, period=2*np.pi)
    return phase