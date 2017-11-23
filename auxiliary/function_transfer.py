def loop(func_to_call):
    for i in range(3):
        for j in range(4):
            print(f'{i} {j} ', end='')
            func_to_call()


def print_function():
    def local_function():
        print('This is a called function')
    loop(local_function)



if __name__ == '__main__':
    print_function()