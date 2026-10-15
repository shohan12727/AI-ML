#Problem - 1 
def introduce(name, age):
    introduction_message = f"My name is {name} and I am {age} years old."
    print(introduction_message)
    
introduce("shohan",20)

#Problem - 2

def batch_generator(data, batch_size):
    for start in range(0, len(data), batch_size):
        batch = data[start:start + batch_size]
        yield batch

data = [1,2,3,4,5,6,7,7,9,10]
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
square_all_element_in_list = list(map(lambda x: x*x,my_list))
print(square_all_element_in_list)
