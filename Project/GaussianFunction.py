import math as m


def GaussianFunction(x, y, sigma):
    a = m.sqrt(1/(4*m.pi*sigma**2))
    power = (-(x**2+y**2)/2*sigma**2)
    return a*m.exp(power)


if __name__ == "__main__":
    output = []
    for x in range(-10, 10):
        for y in range(-10, 10):
            output.append(GaussianFunction(x, y, 5))
    print(output)
    print(len(output))
