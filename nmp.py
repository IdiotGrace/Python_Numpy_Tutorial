import numpy as np

# We can initialize numpy arrays from nested Python lists, and access elements using square brackets:
a = np.array([1, 2, 3])
print(type(a))
print(a.shape)
print(a[0], a[1], a[2])
a[0] = 5
print(a)


b = np.array([[1,2,3],[4,5,6]])
print(b.shape)
print(b[0, 0], b[0, 1], b[1, 0])

# Numpy also provides many functions to create arrays:
# Create an array of all zeros
a = np.zeros((2,2))
print(a)

# Create an array of all ones
b = np.ones((1,2))
print(b)

# Create a constant array
c = np.full((2,2), 7)
print(c)

# Create a 2x2 identity matrix
d = np.eye(2)
print(d)

# Create an array filled with random values
e = np.random.random((2,2))
print(e)


# Array indexing
# Slicing: Similar to Python lists, numpy arrays can be sliced. 
# Since arrays may be multidimensional, you must specify a slice for each dimension of the array:
a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
b = a[:2, 1:3]

print(b)
print(a[0,1])
b[0, 0] = 77
print(a[0, 1])

# You can also mix integer indexing with slice indexing. However, doing so will yield an array of lower rank than the original array. 
# Note that this is quite different from the way that MATLAB handles array slicing:
row_r1 = a[1, :]
row_r2 = a[1:2, :]
print(row_r1, row_r1.shape)
print(row_r2, row_r2.shape)

col_r1 = a[:, 1]
col_r2 = a[:, 1:2]
print(col_r1, col_r1.shape)
print(col_r2, col_r2.shape)


# Integer array indexing
a = np.array([[1,2], [3,4], [5,6]])
print(a[[0, 1, 2], [0, 1, 0]])
print(np.array([a[0, 0], a[1, 1], a[2, 0]]))

# When using integer array indexing, you can reuse the same
# element from the source array:
print(a[[0, 0], [1, 1]])
# Equivalent to the previous integer array indexing example
print(np.array([a[0, 1], a[0, 1]]))

# One useful trick with integer array indexing is selecting or mutating one element from each row of a matrix:
a = np.array([[1,2,3], [4,5,6], [7,8,9], [10,11,12]])
print(a)

b = np.array([0, 2, 0, 1])
print(a[np.arange(4), b])
a[np.arange(4), b] += 10
print(a)


# Boolean array indexing
a = np.array([[1,2], [3,4], [5,6]])
bool_idx = (a > 2)
print(bool_idx)
print(a[bool_idx])
print(a[a > 2])

# Datatypes
x = np.array([1, 2])
print(x.dtype)

x = np.array([1.0, 2.0])
print(x.dtype)

x = np.array([1, 2], dtype=np.int64)
print(x.dtype)

# Array math
x = np.array([[1,2],[3,4]], dtype=np.float64)
y = np.array([[5,6],[7,8]], dtype=np.float64)

print(x + y)
print(np.add(x, y))

print(x - y)
print(np.subtract(x, y))

print(x * y)
print(np.multiply(x, y))

print(x / y)
print(np.divide(x, y))

print(np.sqrt(x))

# dot, unlike matlab
x = np.array([[1,2],[3,4]])
y = np.array([[5,6],[7,8]])

v = np.array([9,10])
w = np.array([11, 12])

print(v.dot(w))
print(np.dot(v, w))


print(x.dot(v))
print(np.dot(x, v))


print(x.dot(y))
print(np.dot(x, y))


print(np.sum(x))
print(np.sum(x, axis=0))
print(np.sum(x, axis=1))

print(x.T)

v = np.array([1,2,3])
print(v)
print(v.T)

#Broadcasting
x = np.array([[1,2,3], [4,5,6], [7,8,9], [10,11,12]])
v = np.array([1, 0, 1])
y = np.empty_like(x)

for i in range(4):
	y[i, :] = x[i, :] + v

print(y)


# stack 4 copies of v on top of each other
vv = np.tile(v, (4, 1))
y = x + vv
print(y)

y = x + v
print(y)


v = np.array([1,2,3])
w = np.array([4,5])
print(np.reshape(v, (3, 1)) * w)

x = np.array([[1,2,3], [4,5,6]])
print(x + v)


print((x.T + w).T)
print(x + np.reshape(w, (2, 1)))
print(x * 2)






















