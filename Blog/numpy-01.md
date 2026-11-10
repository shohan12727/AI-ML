## What is numpy? 

Numpy is a python library that is used for computing nemeric calculation. Mostly works with array. It's give me a superpower to the python library.


## Why should we use this ?
There are several of benefit of this.
- it can work with python list very fast 
- it can handle big size data
- it is kind of ovious for in the field of data Science, ml , statics


```python
import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print(a + b)​ ​ # output: [5 7 9]

```



# ndArray and its Attributes 


ndArray is one of the most important python object. 
Here in this section we will talk about ndim, shape, dtype, size

let's get an example: 

```python
import numpy as np

# 1D array
arr1 = np.array([1, 2, 3, 4, 5])

# 2D array
arr2 = np.array([[1, 2, 3],
​ ​ ​ ​ ​ ​ ​ ​ ​ ​ ​ ​ ​ ​ ​ ​ ​ [4, 5, 6]])

# 3D array
arr3 = np.array([
​ ​ ​ ​ [[1, 2, 3], [4, 5, 6]],
​ ​ ​ ​ [[1, 2, 3], [4, 5, 6]],
​ ​ ​ ​ [[1, 2, 3], [4, 5, 6]]
])
```

- 1D is kind of a line
- 2D is kind of a row and column
- 3D is kind of a 2D table

`ndim:` With this attributes we can get to know the dimension of an array

```python
print(arr1.ndim)​ ​ # 1
print(arr2.ndim)​ ​ # 2
print(arr3.ndim)​ ​ # 3
```
`shape:` with this attributes we can get to know the shape of an array like how many row can have of this array.

```python
print(arr1.shape)​ ​ # (5,)
print(arr2.shape)​ ​ # (2, 3)
print(arr3.shape)​ ​ # (3, 2, 3)
```

`dtype:` with this attributes we can get to know the data of an array .


```python
print(arr1.dtype)
```


`size:` with size we can get to know number of element that are exist in python 

```python
print(arr3.size)​ ​ # 18
```


## ndArray Datatype


Things we need to keep in our mind is that in numpy all element muse need to be same data type, like python list we can not have different datatype. 

Let's take a example: 

```python
import numpy as np

arr = np.array([1, 2, 3])
print(arr.dtype)​ ​ ​ ​ ​ # int64

arr = np.array([1.5, 2.4, 3.2])
print(arr.dtype)​ ​ ​ ​ ​ # float64
```

Here all the element are the same datatype, can't be different.


`Upcasting:`

There is concept in numpy called upcasting. In this process, if there is two or  multiple data type in the same array it convert all the array element in one datatype. 

Let me give you few example: 

```python
arr = np.array([1, 2, 3.2])
print(arr.dtype)​ ​ ​ ​ ​ # float64
```

Look at this example, here even through there is two type(float64 and int64) of element in this list we can but it convert into one type of data(float64)

Another example:

```python
arr = np.array([1, 2, 3.2, 'hello'])
print(arr.dtype)​ ​ ​ ​ ​ # <U32 U refers to unicode string 
```
Here all the array convert into unicode32 or U32

`Set  datatype: `

```python
arr = np.array([1, 2, 300], dtype=np.int16)
print(arr.dtype)​ ​ ​ # int16
print(arr)​ ​ ​ ​ ​ ​ ​ ​ ​ # [​ ​ 1​ ​ ​ 2 300]
```
with the dtype keyword we set this data type in int16


`User of astype:`

we can also set the data type with astype() , it create a old copy in new type.

```python
arr = np.array([1,2,3,4,5.5]) # float64
arr2 = arr.astype(np.int32)
print(arr2.dtype)​ ​ ​ # int32
```


`Error:`

If we assign this data as a wrong datatye it will show an error . 

```python
arr = np.array([1, 2, 3.2, 'hello'], dtype=np.float64)
```

