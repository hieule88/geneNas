def funA():
    print('a')
    try:
        print(A)
    except:
        pass
    print('b')
    return 3
print(funA())