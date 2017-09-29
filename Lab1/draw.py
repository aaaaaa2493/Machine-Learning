import matplotlib.pyplot as plt
from PIL import Image

plt.figure(figsize=(10, 4))

text = sum(list(map(lambda x: list(map(float, x.split())),
                    open('2017-09-28 02_27_43 Wbest 1k.txt').read().split('\n'))), [])

for i in range(10):
    curr_coeff = text[i::10]

    min_coeff = min(curr_coeff)
    max_coeff = max(curr_coeff)

    distance = max_coeff - min_coeff

    normal = [(255-int((i - min_coeff)/distance * 255),)*3 + (255,) for i in curr_coeff]

    image_out = Image.new('RGBA', (28, 28))
    image_out.putdata(normal)
    image_out.save(f'{i}good.png')

print(len(text))


text = open('2017-09-27 22_49_02 7k 28k.txt').read().split('\n')

lambdas = []
successes = []
for i, index in enumerate(range(-6, 6, 1)):

    *curr, success = text[1+18*i:1+18*i+16]

    successes += float(success.split()[-1]),

    ints = [(int(float(i.split()[3])), int(float(i.split()[6]))) for i in curr]

    print([i[1] for i in ints])

    #plt.plot(list(i+1 for i in range(15)), [i[1] for i in ints], color=f'#{hex(min(15, int(2+1.3*i)))[2:]*2}0000')

plt.ylim(0, 1)
plt.xlim(-6, 6)

print(successes)

plt.plot(list(range(-6, 6, 1)), successes)

plt.savefig('plot.png')

