file = open("./sample_data/data.txt", "r")

text = file.read()

file.close()

words = text.split()

longest_word = words[0]

for word in words:
    if len(word) > len(longest_word):
        longest_word = word

print(longest_word)