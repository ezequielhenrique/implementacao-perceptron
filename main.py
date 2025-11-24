def escrever(texto, arquivo="resultado.txt", modo="a"):
    with open(arquivo, modo, encoding="utf-8") as f:
        f.write(str(texto) + "\n")


def perceptron(x, theta, alpha, y_real):
    # Função de ativação (degrau)
    def h(u):
        return 1 if u > 0 else 0
    
    # adiciona o bias no x
    x = [1] + x

    u = 0
    for j in range(len(x)):
        u += theta[j] * x[j]
    
    y_pred = h(u)

    erro = y_real - y_pred

    if erro != 0:
        for j in range(len(theta)):
            theta[j] = theta[j] + alpha * erro * x[j]

    return theta, y_pred, erro


dados = [
    [0, 0, 0],
    [0, 1, 0],
    [1, 0, 1],
    [1, 1, 1]
]

w = [0.0, 0.0, 0.0]   # pesos iniciais
alpha = 0.5

for ciclo in range(6):
    print(f"\n{ciclo + 1}° Ciclo")
    escrever(f"\n{ciclo + 1}° Ciclo")

    for dado in dados:
        x = dado[:-1]
        y_real = dado[-1]

        theta, y_pred, erro = perceptron(x, w, alpha, y_real)
        print(f"x={x}, y_pred={y_pred}, y_real={y_real}, erro={erro}, novos_pesos={w}")
        escrever(f"x={x}, y_pred={y_pred}, y_real={y_real}, erro={erro}, novos_pesos={w}")
