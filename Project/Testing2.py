import numpy as np
Sobel_kernelX = np.array([[1, 3, -1],
                          [2, 8, -2],
                          [1, 25, -1]])

y = Sobel_kernelX**2
print(y)

m = Sobel_kernelX*Sobel_kernelX
print(m)

n = np.matmul(Sobel_kernelX, Sobel_kernelX)
print(n)
