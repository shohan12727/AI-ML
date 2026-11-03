user_input = input()


splited_input = user_input.split()


numbers = []

for number in splited_input:
    number = int(number)
    numbers.append(number)
    
 
result = [] 
 
for num in numbers:
    if num % 4 ==0 and num % 8 != 0:
        result.append(num)

print(result)        
        
    

