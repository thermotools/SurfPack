from surfpack.pcsaft import PC_SAFT
from surfpack import GridSpec
import time
import os, shutil

def test_singlecomp():
    dft = PC_SAFT('NC6')
    grid = GridSpec.Planar(200)
    dft.set_cache_dir('saved_profiles')
    shutil.rmtree('saved_profiles')
    os.makedirs('saved_profiles')

    T = 300
    t0 = time.process_time()
    dft.density_profile_singlecomp(T, grid)
    t1 = time.process_time()
    dft.density_profile_singlecomp(T, grid)
    t2 = time.process_time()

    assert (t2 - t1) < 2 * (t1 - t0) # Lookup is faster than computation

    T = 300.1
    t3 = time.process_time()
    dft.density_profile_singlecomp(T, grid)
    t4 = time.process_time()
    dft.density_profile_singlecomp(T, grid)
    t5 = time.process_time()
    assert (t4 - t3) > 2 * (t2 - t1) # New computation is slower than previous lookup
    assert (t5 - t4) < 2 * (t4 - t3) # Lookup is faster than computation
    assert (t5 - t4) < 2 * (t1 - t0) # Lookup is faster than previous computation

def test_binary():
    dft = PC_SAFT('C2,NC6')
    dft.set_cache_dir('saved_profiles')
    shutil.rmtree('saved_profiles')
    os.makedirs('saved_profiles')
    grid = GridSpec.Planar(200)
    T = 300
    t0 = time.process_time()
    dft.density_profile_tz(T, [0.5, 0.5], grid)
    t1 = time.process_time()
    dft.density_profile_tz(T, [0.5, 0.5], grid)
    t2 = time.process_time()
    assert (t2 - t1) < 2 * (t1 - t0)  # Lookup is faster than computation

    T = 300.1
    t3 = time.process_time()
    dft.density_profile_tz(T, [0.5, 0.5], grid)
    t4 = time.process_time()
    dft.density_profile_tz(T, [0.5, 0.5], grid)
    t5 = time.process_time()
    assert (t4 - t3) > 2 * (t2 - t1)  # New computation is slower than previous lookup
    assert (t5 - t4) < 2 * (t4 - t3)  # Lookup is faster than computation
    assert (t5 - t4) < 2 * (t1 - t0)  # Lookup is faster than previous computation

if __name__ == '__main__':
    test_singlecomp()
    test_binary()