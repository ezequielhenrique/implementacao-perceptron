def escrever(texto, arquivo="resultado.txt", modo="a"):
    with open(arquivo, modo, encoding="utf-8") as f:
        f.write(str(texto) + "\n")


def ativacao(u):
    return 1 if u > 0 else 0


def perceptron(x, w, alpha, y_real):

    x = [1] + x

    u = sum(w[j] * x[j] for j in range(len(x)))

    y_pred = ativacao(u)

    erro = y_real - y_pred

    if erro != 0:
        for j in range(len(w)):
            w[j] = w[j] + alpha * erro * x[j]

    return w, y_pred, erro


dados = [
    [0, 0, 0],
    [0, 1, 0],
    [1, 0, 1],
    [1, 1, 1]
]

w = [0.0, 0.0, 0.0]
alpha = 0.5

escrever("=== Treinamento Perceptron ===", modo="w")

for ciclo in range(6):
    print(f"\n{ciclo+1}° Ciclo")
    escrever(f"\n{ciclo+1}° Ciclo")

    for dado in dados:
        x = dado[:-1]
        y_real = dado[-1]

        w, y_pred, erro = perceptron(x, w, alpha, y_real)

        linha = f"x={x}, y_pred={y_pred}, y_real={y_real}, erro={erro}, novos_pesos={w}"
        print(linha)
        escrever(linha)
