import random

class Banana:
    def __init__(self, parent=None):
        if parent:
            self.name = f'{parent.name} Jr.'
        else:
            self.name = random.choice(['Banana', 'Banani'])

    def __repr__(self):
        return self.name

parent = Banana()
print(parent)
child = Banana(parent)
print(child)
