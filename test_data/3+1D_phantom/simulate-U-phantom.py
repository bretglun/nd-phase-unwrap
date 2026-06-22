import numpy as np
from scipy import signal
from nd_phase_unwrap import io
from pathlib import Path

def cardiac_waveform(timepoints):
    """One pulsatile cycle, normalised to [0, 1].

    A triangle wave peaking mid-cycle is the simplest stand-in for a single
    systolic pulse: flow ramps up to a peak and back to rest over the cycle.
    Replace this with a measured aortic flow waveform for something more
    physiological (it just has to be a 1-D array of length ``timepoints``).
    """
    t = np.arange(timepoints)
    # Triangle wave: 0 at t=0, peaks at t=timepoints/2, back to 0 at t=timepoints.
    # return 1.0 - np.abs((2.0 * t / timepoints) - 1.0)

    # More realistic: simple aortic flow model, normalised to [0, 1].
    modelfunc = lambda t: np.sin(np.pi * t / timepoints) * np.exp(-np.pi * t / timepoints)
    return modelfunc(t) / np.max(modelfunc(t))  # normalise to peak of 1.0




def get_upipe_phantom3D(sizex=32, sizey=64, sizez=64, timepoints=32,
                        pipe_radius=None, peak_phase=0.8 * np.pi):
    """Time-resolved complex phantom of a U-bent pipe with pulsatile flow.

    Geometry is a U (two vertical legs joined by a semicircular bend at the
    bottom), meant as a toy aortic phantom for 4D-flow MRI post-processing.

    The U lies in the y-z plane (the large face of the volume) so it fills the
    space: the legs run the full height along z, their separation spans y, and
    the shallow x axis only carries the tube's depth. So x can be the smallest
    dimension while y and z are large.

    Flow is the full 3-D velocity vector. Its speed follows a laminar
    (Poiseuille) parabolic profile and pulses over one cardiac cycle; its
    direction is the unit tangent of the pipe centre line. Each Cartesian
    velocity component is encoded as the phase of its own complex volume,
    exactly as a three-direction velocity-encoded acquisition would store it:

        v_i(voxel, t) = speed * tangent_i ,   speed = profile(r) * waveform(t)
        phase_i       = peak_phase * v_i
        signal_i      = mask * exp(1j * phase_i)

    where
      * profile(r) = 1 - (r / pipe_radius)**2  is 1 on the centre line, 0 at
        the wall;
      * waveform(t) sweeps a single cardiac-like cycle (rest -> peak -> rest);
      * tangent = (tx, ty, tz) is the unit flow direction: -z down one leg,
        +z up the other, rotating through the bend within the y-z plane.
        Because the pipe is planar in y-z, tx is 0 everywhere -- the vx volume
        is therefore all zeros by construction.

    Voxels outside the tube are exactly zero (no signal / no flow).

    Returns a complex array of shape (sizex, sizey, sizez, timepoints, 3).
    The last axis is the velocity-encode direction: index 0 = vx, 1 = vy,
    2 = vz.
    """
    if pipe_radius is None:
        pipe_radius = min(sizex, sizey) / 6.0

    cx, cy = sizex / 2, sizey / 2          # centre of the xy plane
    margin = max(2.0, pipe_radius * 0.5)   # keep a small gap to the boundary

    # Leg separation (centre -> each leg, along y) and the bend's z-level are
    # chosen to make the U fill the y-z plane: legs as far apart as y allows,
    # bend large enough to drop near the floor while leaving room for the legs.
    leg_sep_y = sizey / 2 - pipe_radius - margin           # limited by y extent
    leg_sep_z = 0.55 * sizez - pipe_radius - margin        # leave ~45% for legs
    leg_separation = min(leg_sep_y, leg_sep_z)
    bottom_z = leg_separation + pipe_radius + margin       # bend foot z-level

    # --- coordinate grids (spatial only; time handled by broadcasting) -------
    X, Y, Z = np.meshgrid(np.arange(sizex), np.arange(sizey), np.arange(sizez),
                          indexing='ij')                   # (sizex, sizey, sizez)

    # --- distance from each voxel to the nearest pipe centre line ------------
    # Leg A: axis at (cx, cy - leg_separation), runs z >= bottom_z
    r_legA = np.sqrt((X - cx)**2 + (Y - (cy - leg_separation))**2)
    legA = (r_legA < pipe_radius) & (Z >= bottom_z)

    # Leg B: axis at (cx, cy + leg_separation), runs z >= bottom_z
    r_legB = np.sqrt((X - cx)**2 + (Y - (cy + leg_separation))**2)
    legB = (r_legB < pipe_radius) & (Z >= bottom_z)

    # Semicircular bend: torus whose spine is a semicircle of radius
    # leg_separation in the y-z plane, centred at (cx, cy, bottom_z), keeping
    # only the lower half (Z <= bottom_z). r_bend is the distance from the
    # voxel to that spine (the tube centre line through the bend).
    dist_to_axis_yz = np.sqrt((Y - cy)**2 + (Z - bottom_z)**2)
    r_bend = np.sqrt((dist_to_axis_yz - leg_separation)**2 + (X - cx)**2)
    bend = (r_bend < pipe_radius) & (Z <= bottom_z)

    mask = legA | legB | bend

    # Distance to the centre line, picking whichever segment a voxel belongs to
    r_axis = np.full(mask.shape, np.inf)
    r_axis = np.where(legA, r_legA, r_axis)
    r_axis = np.where(legB, r_legB, r_axis)
    r_axis = np.where(bend, r_bend, r_axis)

    # --- parabolic (Poiseuille) speed profile: 1 at centre line, 0 at wall ---
    profile = np.zeros_like(r_axis)
    profile[mask] = 1.0 - (r_axis[mask] / pipe_radius)**2
    profile = np.clip(profile, 0.0, 1.0)

    # --- unit flow direction (tangent to the centre line) per voxel ----------
    dir_x = np.zeros_like(profile)          # planar pipe (y-z) -> stays zero
    dir_y = np.zeros_like(profile)
    dir_z = np.zeros_like(profile)

    dir_z[legA] = -1.0                      # leg A: flow downward
    dir_z[legB] = +1.0                      # leg B: flow upward

    # Bend tangent (0, sin phi, -cos phi), phi measured at the bend centre
    # (cy, bottom_z) in the y-z plane: phi = 0 -> leg A foot (-z),
    # phi = pi/2 -> very bottom (+y), phi = pi -> leg B foot (+z).
    phi = np.arctan2(bottom_z - Z, cy - Y)
    dir_y[bend] = np.sin(phi)[bend]
    dir_z[bend] = -np.cos(phi)[bend]

    tangent = np.stack([dir_x, dir_y, dir_z], axis=-1)   # (x, y, z, 3)

    # --- temporal modulation: one pulsatile cycle ----------------------------
    waveform = cardiac_waveform(timepoints)              # (timepoints,)

    # --- assemble: speed -> per-component velocity -> per-component phase -----
    speed = profile[..., np.newaxis] * waveform[np.newaxis, np.newaxis,
                                                np.newaxis, :]   # (x,y,z,t)
    phase = (peak_phase
             * speed[..., np.newaxis]                            # (x,y,z,t,1)
             * tangent[:, :, :, np.newaxis, :])                  # (x,y,z,1,3)

    arr = mask[..., np.newaxis, np.newaxis] * np.exp(1j * phase)  # (x,y,z,t,3)
    print(arr.shape)
    return arr


def wrap(arr, factor):
    return np.abs(arr) * np.exp(1j * (np.angle(arr) * factor))


def add_noise(img, VNR):
    std = 1 / VNR
    return img + np.pi * np.random.normal(0, std, img.shape)


path = Path(__file__).parent

truth = get_upipe_phantom3D(sizex=32, sizey=64, sizez=64, timepoints=32, pipe_radius=10, peak_phase=np.pi)  # ground truth
io.save(truth, path / 'truth.npy')

for num_wraps in [0, 1, 2, 3]:
    factor = 1 + 2 * num_wraps
    wrapped = wrap(truth, factor)
    if num_wraps != 0:
        io.save(wrapped, path / f'{num_wraps}wraps.npy')

    for noise_power in [1.0, 2.0, 3.0]:
        VNR = 10**noise_power
        noisy = add_noise(wrapped, VNR)
        io.save(noisy, path / f'{num_wraps}wrapsVNR1e{str(noise_power).replace(".", "_")}.npy')