class UnionFindElement:
    def __init__(self, label):
        self.label = label
        self.parent = self
        self.rank = 0

    def Label(self):
        return self.label

    def Parent(self):
        return self.parent

    def Rank(self):
        return self.rank

def Find(element):
    if (element.parent != element):
        element.parent = Find(element.parent)
    return element.parent

def Union(element_one, element_two):
    root_one = Find(element_one)
    root_two = Find(element_two)

    if (root_one == root_two): return

    if (root_one.rank < root_two.rank):
        root_one.parent = root_two
    elif (root_one.rank > root_two.rank):
        root_two.parent = root_one
    else:
        root_two.parent = root_one
        root_one.rank = root_one.rank + 1
