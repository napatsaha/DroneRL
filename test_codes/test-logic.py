"""
Test logic structure to implement 4 variations of Epsilon-greedy-softmax exploration in DQNPolicy.predict()

"""


def logic(D = None, E = None, PR = None, PG = None):
    if not D and E and not PR:
        print("R: Random")
    elif not D and ((E and PR) or (not E and PG)):
        print("S: Softmax")
    else:
        print("G: Greedy")

if __name__ == "__main__":
    scenarios = [
        [('D', 1)],
        [('D', 0), ('E', 0), ('PG', 0)],
        [('D', 0), ('E', 1), ('PR', 1)],
        [('D', 0), ('E', 0), ('PG', 1)],
        [('D', 0), ('E', 1), ('PR', 0)]
    ]

    for sc in scenarios:
        sc = dict(sc)
        print(sc)
        res = logic(**sc)
        print(res, end="\n\n")