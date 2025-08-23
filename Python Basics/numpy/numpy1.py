import numpy as np

a=np.array([1,2,3,4,5,6,7,8])
print(a)
print(sum(a))

print(np.zeros((2,3)))
print(np.ones((2,3)))


b=np.arange(0,10,2)
print(b)

c = np.arange(9)
print("Original", c)

d=c.reshape(3,3)
print("Reshaped", d)

print("Mean:",np.mean(c))
print("Median:",np.median(c))
print("Standard Deviation:",np.std(c))
