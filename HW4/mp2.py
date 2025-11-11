import concurrent.futures
import numba
import numpy as np

# Progress bar helper: prefer notebook-friendly tqdm, fall back to console tqdm
try:
    from tqdm.notebook import tqdm
except Exception:
    try:
        from tqdm import tqdm
    except Exception:
        # Fallback no-op progress iterator
        def tqdm(it, total=None, desc=None):
            for x in it:
                yield x

@numba.njit
def acceleration(r,):
    norm = np.linalg.norm(r)
    if norm < 1e-10:  # Add a small threshold to prevent division by zero
        return np.zeros_like(r)
    return -1/norm**3 * r / 4

@numba.njit
def vel_dispersion(v, A=1, B=1):
    return -A/(np.linalg.norm(v)**3 + B) * v

@numba.njit
def rk4_step(r, v, dt, alpha_dispersion=0, A=1, B=1):
    k1_v = acceleration(r) + alpha_dispersion * vel_dispersion(v, A, B)
    k1_r = v

    k2_v = acceleration(r + 0.5 * dt * k1_r) + alpha_dispersion * vel_dispersion(v + 0.5 * dt * k1_v, A, B)
    k2_r = v + 0.5 * dt * k1_v

    k3_v = acceleration(r + 0.5 * dt * k2_r) + alpha_dispersion * vel_dispersion(v + 0.5 * dt * k2_v, A, B)
    k3_r = v + 0.5 * dt * k2_v

    k4_v = acceleration(r + dt * k3_r) + alpha_dispersion * vel_dispersion(v + dt * k3_v, A, B)
    k4_r = v + dt * k3_v

    r_next = r + (dt / 6) * (k1_r + 2 * k2_r + 2 * k3_r + k4_r)
    v_next = v + (dt / 6) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)

    return r_next.astype(np.float64), v_next.astype(np.float64)

@numba.njit
def time_to_schwarzschild(r0, v0, dt, tf, tol=1e-7, alpha_dispersion=0, A=1, B=1):
    num_steps = int(tf / dt)
    r = r0.astype(np.float64) 
    v = v0.astype(np.float64)

    t = 0
    i = 0
    while i < num_steps//2:
        r1, v1 = rk4_step(r, v, dt, alpha_dispersion, A, B)
        r1, v1 = rk4_step(r1, v1, dt, alpha_dispersion, A, B)
        r2, v2 = rk4_step(r, v, 2*dt, alpha_dispersion, A, B)
        norm = np.linalg.norm(r2-r1)
        if norm < 1e-20:  # Add a small threshold to prevent division by zero
            norm = 1e-20
        rho = 30 * dt * tol / norm
        if rho >= 1:
            t += 2 * dt
            r = r1
            v = v1
            i += 1

        dt = min(dt * rho**(1/4), 2 * dt)

        if np.linalg.norm(r) < 1e-7:
            # print(f'{A}, {B} is {t}') # Removed print from Numba func
            return t
        
    return tf  # Return maximum time if Schwarzschild radius not reached

def process_chunk_worker(chunk, r0, v0_values, dt, tf, tol, alpha_dispersion, A_values, B_values):
    """
    Worker that evaluates a chunk (list) of (i,j,k) pairs and returns a list of (i,j,k,t).
    Chunking reduces task overhead when individual evaluations are fast.
    """
    out = []
    for params in chunk:
        i, j, k = params
        # Select the specific v0 vector for this simulation run
        v0_k = v0_values[k] 
        A_i = A_values[i]
        B_j = B_values[j]
        
        t = time_to_schwarzschild(r0, v0_k, dt, tf, tol, alpha_dispersion, A_i, B_j)
        out.append((i, j, k, t))
    return out


# small wrapper for multiprocessing.starmap/imap_unordered compatibility
def _worker_star(args):
    return process_chunk_worker(*args)


def parameter_sweep(r0, v0_values, dt, tf, tol=1e-7, alpha_dispersion=0, A_values=None, B_values=None, parallel=False, max_workers=None, backend='thread', chunk_size=8, show_progress=True):
    """Sweep over A_values, B_values, and v0_values. Returns a 3D results array.

    The output array 'results' will have shape (len(A_values), len(B_values), len(v0_values)).
    results[i, j, k] corresponds to the simulation with A_values[i], B_values[j], and v0_values[k].

    Backends supported:
      - 'thread' : ThreadPoolExecutor (default, notebook-friendly)
      - 'process': ProcessPoolExecutor (concurrent.futures)
      - 'multiprocessing': multiprocessing.Pool
    """
    # Convert A_values/B_values to numpy arrays if needed.
    if A_values is None:
        A_values = np.array([1.0])
    else:
        A_values = np.array(A_values)
    
    if B_values is None:
        B_values = np.array([1.0])
    else:
        B_values = np.array(B_values)
        
    # v0_values MUST be provided, convert to numpy array
    if v0_values is None:
         raise ValueError("v0_values must be provided (e.g., np.array([[0, 1.0, 0]]))")
    
    # Ensure v0_values is a 2D numpy array (an array of vectors)
    v0_values = np.array(v0_values)
    if v0_values.ndim != 2:
        # Try to recover if they passed a single vector
        if v0_values.ndim == 1 and len(v0_values) > 0:
             v0_values = v0_values.reshape(1, -1) # Treat as one vector in a list
        else:
             raise ValueError("v0_values must be a 2D array (list of vectors)")

    # Results array is now 3D
    results = np.zeros((len(A_values), len(B_values), len(v0_values)))

    # Create the full 3D list of parameter indices
    params = [(i, j, k) for i in range(len(A_values)) 
                        for j in range(len(B_values)) 
                        for k in range(len(v0_values))]

    if not parallel:
        # Sequential fill (safe, simple)
        if show_progress:
            it = tqdm(params, total=len(params), desc="Sequential Sweep")
        else:
            it = params
            
        for i, j, k in it:
            results[i, j, k] = time_to_schwarzschild(r0, v0_values[k], dt, tf, tol, alpha_dispersion, A_values[i], B_values[j])
        return results

    # --- Parallel path using chunking ---
    
    # Create chunks
    if chunk_size <= 1:
        chunks = [[p] for p in params]
    else:
        chunks = [params[k:k+chunk_size] for k in range(0, len(params), chunk_size)]

    # Common function to process results from futures
    def process_futures(futures, show_progress, total_chunks):
        if show_progress:
            it = concurrent.futures.as_completed(futures)
            pbar = tqdm(it, total=total_chunks, desc=f"Parallel Sweep ({backend})")
            for future in pbar:
                chunk_out = future.result()
                for (i, j, k, t) in chunk_out:
                    results[i, j, k] = t
        else:
            for future in concurrent.futures.as_completed(futures):
                chunk_out = future.result()
                for (i, j, k, t) in chunk_out:
                    results[i, j, k] = t

    # THREAD backend (fastest to get working inside notebooks)
    if backend == 'thread':
        Executor = concurrent.futures.ThreadPoolExecutor
        worker = process_chunk_worker
        with Executor(max_workers=max_workers) as exe:
            # Pass v0_values to the worker
            futures = [exe.submit(worker, chunk, r0, v0_values, dt, tf, tol, alpha_dispersion, A_values, B_values) for chunk in chunks]
            process_futures(futures, show_progress, len(chunks))
        return results

    # ProcessPoolExecutor backend
    if backend == 'process':
        Executor = concurrent.futures.ProcessPoolExecutor
        # Pass v0_values in the 'args' tuple
        args = [(chunk, r0, v0_values, dt, tf, tol, alpha_dispersion, A_values, B_values) for chunk in chunks]
        with Executor(max_workers=max_workers) as exe:
            futures = [exe.submit(_worker_star, a) for a in args]
            process_futures(futures, show_progress, len(chunks))
        return results

    # Multiprocessing.Pool backend (uses imap_unordered for progress)
    if backend == 'multiprocessing':
        import multiprocessing as mp
        # Pass v0_values in the 'args' tuple
        args = [(chunk, r0, v0_values, dt, tf, tol, alpha_dispersion, A_values, B_values) for chunk in chunks]
        
        with mp.Pool(processes=max_workers) as pool:
            it = pool.imap_unordered(_worker_star, args)
            if show_progress:
                pbar = tqdm(it, total=len(args), desc=f"Parallel Sweep ({backend})")
                for chunk_out in pbar:
                    for (i, j, k, t) in chunk_out:
                        results[i, j, k] = t
            else:
                for chunk_out in it:
                    for (i, j, k, t) in chunk_out:
                        results[i, j, k] = t
        return results

    raise ValueError(f"Unknown backend '{backend}'. Choose 'thread', 'process' or 'multiprocessing'.")