import r


def test_py_r_serve():
    print('Testing RServe')

    import pyRserve
    conn = pyRserve.connect()

    math_result = conn.eval('1+1')
    print('According to Rserve, "1+1" = ' + str(math_result))

    pi = conn.eval('pi')
    print('According to Rserve, "PI" = ' + str(pi))

    conn.close()
    print('Finished testing pyRserve')


def main():
    print('Testing R serve API')

    if r.has_connection():
        test_py_r_serve()
        print('Test passed.')
    else:
        print('Failed to establish Rserve connection.')


if __name__ == '__main__':
    main()
