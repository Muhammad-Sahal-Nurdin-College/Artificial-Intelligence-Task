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