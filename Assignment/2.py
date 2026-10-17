def compute_ratio(x, y):
    try:
        calculate =  (x + y) / (x * y)
        return calculate
    
    except ZeroDivisionError:
        error_message = "Division by zero"
        return error_message
        
        
        
    
x = int(input("Enter number x: "))
y = int(input("Enter number y: "))    

try:
    result = compute_ratio(x,y)
    print(result)

except Exception as e:
    print(e)