from os import system
from random import randint, shuffle, choice
from tqdm import tqdm

C = 10


def gen_val():
    # return randint(0, 2**64 - 1)
    return randint(0, C)


system("g++ A.cpp -o A -std=c++20 -O2")

for test in tqdm(range(10**4)):
    # print(test, end=" ", flush=True)

    N = 50

    C = randint(0, 20)

    n = randint(1, N)

    with open("test.in", 'w') as f:
        x = set()
        while len(x) < n:
            x.add(randint(0, n + C))
        x = list(x)
        shuffle(x)

        y = [gen_val() for i in x]

        print(1, file=f)
        print(n, file=f)
        for a, b in zip(x, y):
            print(a, b, file=f)

    system("./A < test.in > fuck.out")

    data = open("fuck.out").read()

    with open("cum.in", 'w') as f:
        print(2, file=f)

        print(data, file=f)
        q = randint(0, min(2 * n, N))
        print(q, file=f)

        ar = []
        while len(ar) < q:
            i = randint(0, n - 1)
            ar += [(x[i], y[i])]

        qr = [ar[i][0] for i in range(len(ar))]

        print(*qr, file=f)

    system("./A < cum.in > cock.out")

    cum = open("cock.out").read()
    cum = list(map(int, cum.split()))

    for i in range(q):
        # print(cum[i], ar[i][1])
        assert cum[i] == ar[i][1]
