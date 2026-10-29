
def compute_ratio(x, y):
    try:
        result = (x + y) / (x * y)
        return result

    except ZeroDivisionError:
        return "Error: Division not possible"


user_input = input()

split_values = user_input.split()

x = int(split_values[0])
y = int(split_values[1])

result = compute_ratio(x, y)

print(result)
