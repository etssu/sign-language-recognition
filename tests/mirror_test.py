from preprocessing import normalize_row
import numpy as np
import pandas as pd

df = pd.read_csv("../data/landmarks.csv")
row = df.iloc[0]

normal = normalize_row(row, mirror=False)
mirrored = normalize_row(row, mirror=True)

for i in range(21):
    x = normal[i * 3]
    y = normal[i * 3 + 1]
    z = normal[i * 3 + 2]

    mx = mirrored[i * 3]
    my = mirrored[i * 3 + 1]
    mz = mirrored[i * 3 + 2]

    print(
        f"Point {i}: "
        f"X {x:.4f} -> {mx:.4f}, "
        f"Y {y:.4f} -> {my:.4f}, "
        f"Z {z:.4f} -> {mz:.4f}"
    )

    assert np.isclose(mx, -x)
    assert np.isclose(my, y)
    assert np.isclose(mz, z)

print("Mirroring test passed!")