# F-SPEED REPORT — measured (valid peak-rich query)
- reps: 3000, q_peaks: 32, tol: 0.01
- OLD `_search_shift_batch` (serial): 0.017 s (0.006 ms/rep)
- NEW `search_shift_prange` (prange): 0.004 s (0.001 ms/rep)
- speedup: 4.1x
- shapes match: True
- spot-check max |diff| (idx [0, 1500, 2999]): 0.000000
- full-array max |diff|: 0.000000
- nonzero outputs: 3000/3000
- gate (<=6s): PASS
