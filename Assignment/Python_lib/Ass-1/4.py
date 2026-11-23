sentence = input()

words = sentence.split()

vowels = "aeiou"

result = []

for word in words:
    if word[0].lower() in vowels:
        result.append(word)

print(result)