# Soal 1
"""def factorial(n):
    return 1 if (n == 1 or n == 0) else n * factorial(n-1)


num = 5
print("Factorial of", num, "is", factorial(num))"""

# TUGAS MENGGUNAKAN METHODE ITERASI
# Muhammad Sahal Nurdin
# 1IA12
# 51421075

def factorial(n):
    product = 1
    for i in range(n, 0, -1):
        product = product * i
    return product


n = 5
print("Factorial of", n, "is", factorial(n))

"""awal = 11
akhir = 25

for i in range(awal, akhir+1):
    if i > 1:
        for j in range(2, i):
            if(1 % j == 0):
                break
        else:
            print(i)"""

# TUGAS MENCARI BILANGAN PRIMA ATAU BUKAN
# Muhammad Sahal Nurdin
# 1IA12
# 51421075

n = int(input("Masukkan angka:"))
for i in range(2, n):
    if n % i == 0:
        print("Bukan bilangan prima")
        break
else:
    print("Bilangan prima")

"""def Fibonacci(n):
    if n < 0:
        print("Incorrect input")
    # Nilai Fibonacci pertama 0
    elif n == 1:
        return 0
    # Nilai Fibonacci kedua 1
    elif n == 2:
        return 1
    else:
        return Fibonacci(n - 1) + Fibonacci(n-2)"""


# n = 9
# print("Bilangan fibonacci yang ke", n, "adalah", Fibonacci(n))

# TUGAS OUTPUT BERIKAN NILAI SEBELUM 21 TERMASUK ANGKA 21
# Muhammad Sahal Nurdin
# 1IA12
# 51421075

def Fibonacci(n):
    if n < 0:
        print("Incorrect input")
    # Nilai Fibonacci pertama 0
    elif n == 1:
        return 0
    # Nilai Fibonacci kedua 1
    elif n == 2:
        return 1
    else:
        return Fibonacci(n - 1) + Fibonacci(n-2)


n = 9
for n in range(1, 10):  # 10 karena iterasi terakhir tidak dihitung
    print("Bilangan fibonacci yang ke", n, "adalah", Fibonacci(n))


"""import numpy as np
import cupy as cp
import time

# Numpy dan CPU
s = time.time()
x_cpu = np.ones((1000, 1000, 1000))
e = time.time()
print("Waktu yang diperlukan untuk CPU :", e - s)

# CuPy dan GPU
s = time.time()
x_gpu = cp.ones((1000, 1000, 1000))
cp.cuda.Stream.null.synchronize()
e = time.time()
print("Waktu yang diperlukan untuk GPU :", e - s)

# Nilai yang akan dikalikan dengan array
n = 5

# Numpy dan CPU
s = time.time()
x_cpu *= n
e = time.time()
print("Waktu yang diperlukan untuk CPU :", e - s)

### CuPy and GPU
s = time.time()
x_gpu *= n
cp.cuda.Stream.null.synchronize()
e = time.time()
print("Waktu yang diperlukan untuk GPU :", e - s)"""
