import sys

class Stack:
    def __init__(self, capacity):
        self.capacity = capacity
        self.top = -1
        self.array = [0] * capacity

    def is_full(self):
        return self.top == self.capacity - 1

    def is_empty(self):
        return self.top == -1

    def push(self, item):
        if not self.is_full():
            self.top += 1
            self.array[self.top] = item

    def pop(self):
        if not self.is_empty():
            top = self.top
            self.top -= 1
            return self.array[top]
        return -sys.maxsize


def move_disks_between_two_poles(src, dest, s, d):
    pole1_top_disk = src.pop()
    pole2_top_disk = dest.pop()

    if pole1_top_disk == -sys.maxsize:
        src.push(pole2_top_disk)
        move_disk(d, s, pole2_top_disk)
    elif pole2_top_disk == -sys.maxsize:
        dest.push(pole1_top_disk)
        move_disk(s, d, pole1_top_disk)
    elif pole1_top_disk > pole2_top_disk:
        src.push(pole1_top_disk)
        src.push(pole2_top_disk)
        move_disk(d, s, pole2_top_disk)
    else:
        dest.push(pole2_top_disk)
        dest.push(pole1_top_disk)
        move_disk(s, d, pole1_top_disk)


def move_disk(from_peg, to_peg, disk):
    print(f"Move the disk {disk} from '{from_peg}' to '{to_peg}'")


def toh_iterative(num_of_disks, src, aux, dest):
    s, d, a = 'S', 'D', 'A'

    if num_of_disks % 2 == 0:
        d, a = a, d

    total_num_of_moves = 2 ** num_of_disks - 1

    for i in range(num_of_disks, 0, -1):
        src.push(i)

    for i in range(1, total_num_of_moves + 1):
        if i % 3 == 1:
            move_disks_between_two_poles(src, dest, s, d)
        elif i % 3 == 2:
            move_disks_between_two_poles(src, aux, s, a)
        else:
            move_disks_between_two_poles(aux, dest, a, d)


if __name__ == '__main__':
    num_of_disks = 3
    print(f'The number of steps: {2 ** num_of_disks - 1}')
    src = Stack(num_of_disks)
    dest = Stack(num_of_disks)
    aux = Stack(num_of_disks)

    toh_iterative(num_of_disks, src, aux, dest)
