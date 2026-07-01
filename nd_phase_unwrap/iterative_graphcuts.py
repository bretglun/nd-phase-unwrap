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


def solve_maxflow(unary_costs, binary_costs, ngb_indices):
    num_nodes = binary_costs.shape[2]
    
    graph = maxflow.GraphFloat()
    graph.add_grid_nodes(num_nodes)

    src_cap, snk_cap = (unary_costs[1], unary_costs[0]) if unary_costs is not None else (np.zeros(num_nodes, dtype=binary_costs.dtype), np.zeros(num_nodes, dtype=binary_costs.dtype))
    
    i = np.arange(num_nodes)
    for ngb, j in enumerate(ngb_indices): # loop over neighbours
        # Add edge capacities following Kolmogorov and Zabih 2004
        A, B, C, D = binary_costs[:, ngb]

        src_cap[i[C > A]]  += (C - A)[C > A]
        snk_cap[i[A >= C]] += (A - C)[A >= C]

        snk_cap[j[C > D]]  += (C - D)[C > D]
        src_cap[j[D >= C]] += (D - C)[D >= C]

        edge_cap = B + C - A - D

        graph.add_edges(i, j, edge_cap, np.zeros(num_nodes))

    graph.add_grid_tedges(i, src_cap, snk_cap)
    graph.maxflow() # Boykov-Kolmogorov max-flow algorithm
    return graph.get_grid_segments(i)


def beta_jump_move(phase, beta, unary_weights, binary_weights, ngb_indices):
    p = phase.flatten()
    pb = p + beta
    unary_costs = np.array((unary_weights * p**2, unary_weights * pb**2)) if unary_weights is not None else None
    binary_costs = binary_weights * [
        (p  -  p[ngb_indices])**2,
        (p  - pb[ngb_indices])**2,
        (pb -  p[ngb_indices])**2,
        (pb - pb[ngb_indices])**2]
    label = solve_maxflow(unary_costs, binary_costs, ngb_indices)
    p[label==1] += beta
    return p.reshape(phase.shape)


def get_energy(phase, unary_weights, binary_weights, ngb_indices):
    p = phase.flatten()
    energy = np.sum(unary_weights * p**2) if unary_weights is not None else 0.
    energy += np.sum(binary_weights * (p  - p[ngb_indices])**2)
    return energy


def get_weights(magn, pixel_spacing, neighbourhood, temp_spat_weights, ngb_indices, edge_ngb, data_cost=0):
    M2p = magn.flatten()**2
    unary_weights = data_cost * M2p if data_cost > 0 else None
    binary_weights = np.zeros((magn.ndim, *magn.shape))
    with np.errstate(divide='ignore', invalid='ignore'): # suppress divide-by-zero warning
        binary_weights = 1 / (1/M2p + 1/M2p[ngb_indices])
    binary_weights[np.isnan(binary_weights)] = 0.
    distance = np.linalg.norm(neighbourhood * pixel_spacing, axis=1)[:, None]
    lambda_weights = np.linalg.norm(neighbourhood * temp_spat_weights, axis=1)[:, None]
    binary_weights = (binary_weights / (distance * np.sum(1/distance))) * lambda_weights
    binary_weights[edge_ngb] = 0
    return unary_weights, binary_weights


def remove_drift(phase, magn, period):
    mean_phase = np.mean(magn * phase) / np.mean(magn)
    mean_period = int(round(mean_phase/period))
    return phase - mean_period * period


def unwrap(arr, config):
    magn, phase = np.abs(arr), np.angle(arr)

    neighbourhood = get_neighbourhood(config.neighbourhood_radius, config.pixel_spacing)
    ngb_indices, edge_ngb = get_neighbour_indices(neighbourhood, arr.shape, config.cyclic)

    unary_weights, binary_weights = get_weights(magn, config.pixel_spacing, neighbourhood, config.neighbourhood_weight, ngb_indices, edge_ngb, config.data_cost)

    min_energy = get_energy(phase, unary_weights, binary_weights, ngb_indices)
    improved = True
    while improved:
        improved = False
        for beta in [2 * np.pi, -2 * np.pi]: # 2pi jump moves with alternating sign
            phase_updated = beta_jump_move(phase, beta, unary_weights, binary_weights, ngb_indices)
            energy = get_energy(phase_updated, unary_weights, binary_weights, ngb_indices)
            if energy < min_energy:
                improved = True
                phase = phase_updated
                min_energy = energy
    if not config.data_cost:
        phase = remove_drift(phase, magn, period=2*np.pi)
    return phase