#Problem - 1 
def introduce(name, age):
    introduction_message = f"My name is {name} and I am {age} years old."
    print(introduction_message)
    
# introduce("shohan",20)

#Problem - 2

data = [1, 2, 3, 4, 5, 6, 7, 8]
batch_size = 3

# def batch_generator(data, batch_size):
#     for i in range(0, len(data), batch_size):
#         yield i
    
# batch_generator(data, batch_size)    

def batch_generator(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]


data = [1, 2, 3, 4, 5, 6, 7, 8]
batch_size = 3

for batch in batch_generator(data, batch_size):
    print(batch)


# def demo(): 
#      print ("start")
#      yield 10
#      print ("Continue")
#      yield 20
# g= demo()

# print(next(g))
# print(next(g))



#problem - 3

my_list = [1,2,3,4]
square_all_element_in_list = list(map(lambda x: x*x, my_list))
# print(square_all_element_in_list)



# Problem - 4

def powerFunc(base, exponent = 2):
    return base ** exponent

# print(powerFunc(5))



#problem - 5


list_1 =  [1,2,3,4,5,6]

even_numbers = list(filter(lambda x: x % 2 == 0, list_1))
square_numbers = list(map(lambda x: x *x, even_numbers))
# print(square_numbers)


