with open('./ouput.txt') as f:
    content = f.readlines()
content = [x.strip() for x in content]

print(content)
print(len(content))

filtered_list = []
for x in content:
    if x.split()[0] != 'INIT':
        filtered_list.append(x)

content = filtered_list
with open ('./ouput.txt', 'w') as f:
    for x in content:
        f.write(str(x) + '\n')
print('content', content)
total = len(content)
output = [x.split()[0] for x in content]
print(output)
correctness = [x == 'bee' for x in output]
print('correctness', correctness)
accuracy = 0

for x in correctness:
    if x:
        accuracy += 1

print('correct answers', accuracy)
print('total', total)
accuracy = float(accuracy) / float(total)

print('accuracy', accuracy)
