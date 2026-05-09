from mpi4py import MPI
import cv2
import numpy as np
import time
import os
from numba import njit, prange

# --- Nivelul 2: Paralelizare de tip OpenMP folosind Numba ---
@njit(parallel=True)
def compute_local_accumulator(y_idxs, x_idxs, rhos_len, thetas, diagonal):
    """
    Fiecare proces MPI rulează această funcție. Numba folosește mai multe thread-uri 
    (OpenMP în spate) pentru a procesa sub-setul de pixeli primit.
    """
    accumulator = np.zeros((rhos_len, len(thetas)), dtype=np.int32)
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)

    # prange paralelizează această buclă pe mai multe thread-uri
    for i in prange(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]

        for theta_idx in range(len(thetas)):
            rho = int(round(x * cos_t[theta_idx] + y * sin_t[theta_idx]))
            rho_idx = rho + diagonal
            
            # ATENȚIE (Punct de discuție pentru proiect): 
            # Aici poate apărea un 'Race Condition' între thread-urile din Numba 
            # dacă încearcă să modifice aceeași celulă simultan.
            accumulator[rho_idx, theta_idx] += 1

    return accumulator

# --- Nivelul 1: Paralelizare MPI ---
def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    rho_res = 1
    theta_res = np.pi / 180

    image_shape = None
    y_idxs = None
    x_idxs = None

    # Procesul ROOT (Rank 0) citește imaginea și extrage muchiile
    if rank == 0:
        image_path = "data/images/h1.png"
        if not os.path.exists(image_path):
            print(f"[Rank 0] Imaginea nu a fost găsită: {image_path}")
            comm.Abort()

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        edges = cv2.Canny(image, 50, 150)
        
        y_idxs, x_idxs = np.nonzero(edges)
        image_shape = edges.shape
        print(f"[Rank 0] Începem procesarea MPI ({size} procese). Pixeli de muchie: {len(x_idxs)}")

    # 1. ROOT trimite (Broadcast) dimensiunile imaginii tuturor proceselor
    image_shape = comm.bcast(image_shape, root=0)
    height, width = image_shape
    diagonal = int(np.ceil(np.sqrt(height ** 2 + width ** 2)))
    rhos_len = 2 * diagonal + 1
    thetas = np.arange(0, np.pi, theta_res)

    # 2. Împărțim munca (Scatter): ROOT divide pixelii în mod egal pentru fiecare proces MPI
    if rank == 0:
        y_chunks = np.array_split(y_idxs, size)
        x_chunks = np.array_split(x_idxs, size)
    else:
        y_chunks = None
        x_chunks = None

    local_y_idxs = comm.scatter(y_chunks, root=0)
    local_x_idxs = comm.scatter(x_chunks, root=0)

    # Ne asigurăm că toate procesele au ajuns aici înainte de a porni cronometrul
    comm.Barrier()
    start_time = time.time()

    # Fiecare proces MPI își calculează acumulatorul local folosind Numba (Multi-threading)
    local_accumulator = compute_local_accumulator(
        local_y_idxs, local_x_idxs, rhos_len, thetas, diagonal
    )

    # 3. Combinăm rezultatele (Reduce): Însumăm acumulatoarele locale într-unul global pe Rank 0
    global_accumulator = np.zeros_like(local_accumulator)
    comm.Reduce(
        [local_accumulator, MPI.INT],
        [global_accumulator, MPI.INT],
        op=MPI.SUM,
        root=0
    )

    comm.Barrier()
    end_time = time.time()

    # ROOT afișează rezultatele finale
    if rank == 0:
        execution_time = end_time - start_time
        print(f"\n===== Rezultate Hibrid MPI + OpenMP =====")
        print(f"Procese MPI utilizate: {size}")
        print(f"Timp de execuție: {execution_time:.6f} secunde")
        print(f"Voturi maxime în acumulator: {global_accumulator.max()}")

if __name__ == "__main__":
    main()
