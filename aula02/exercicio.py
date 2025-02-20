# Voce sabe o que uma REGRESSÃO LINEAR?
# Vou te dar um exemplo bem simples!

'''

    Imagina que você tem vários pontos desenhados em um papel, como se fossem estrelinhas espalhadas no céu. 
Agora, você quer encontrar uma linha que passe bem no meio dessas estrelinhas, tentando ficar o mais perto possível de todas elas.
    Essa linha ajuda a gente a entender como uma coisa muda quando outra muda. 
    
    Por exemplo:

🔹 Se você medir sua altura todo ano, pode perceber que, conforme o tempo passa, você cresce. 

A regressão linear ajuda a desenhar uma linha mostrando esse crescimento!

🔹 Se você vender limonada na rua e anotar quantos copos vende em dias quentes e frios, essa linha pode mostrar se 
você vende mais quando faz calor.

Então, a regressão linear é como uma linha mágica que tenta prever o que vai acontecer, baseada no que já aconteceu antes!

'''



# Aqui estão os dados chefe!
# o X contem as váriaveis independentes (ou "entradas" / "preditoes")
# o Y contem as váriaveis dependentes (ou "saidas" / "respostas")

# Esses valores de X e Y são os dados que estamos usando para fazer a regressão linear simples.
x = [3, 2, -1, 4]
y = [7, 5, -1, 9]

# Cálculo das médias de X e Y
# A média de X e Y é importante porque essas médias são usadas para calcular os coeficientes(a relação) da reta de regressão linear.
media_x = sum(x) / len(x)  # Média de x
media_y = sum(y) / len(y)  # Média de y

print(f"Média de x: {media_x}")
print(f"Média de y: {media_y}\n")

# Por que precisamos calcular esses somatórios?
# O numerador mostra como X e Y variam juntos.
# O denominador mostra a variação de X em torno da sua média.
# Eles nos ajudam a ajustar a reta de modo a minimizar o erro entre os valores reais de Y e os valores previstos pela reta.
somatorio_numerador = 0
somatorio_denominador = 0

# Loop para calcularmos todos os valores de X e Y, onde vamos acumular os calculos nas 
# variáveis para fazer o Beta1(coeficiente Angular) e o Beta0(Coeficiente Linear)
for i in range(len(x)):
    somatorio_numerador += (x[i] - media_x) * (y[i] - media_y)
    somatorio_denominador += (x[i] - media_x) ** 2

# Exibir os resultados dos somatórios
print(f"Somatório do numerador: {somatorio_numerador}")
print(f"Somatório do denominador: {somatorio_denominador}\n")

# Cálculo de beta1 (coeficiente angular)
beta1 = somatorio_numerador / somatorio_denominador

# Cálculo de beta0 (coeficiente linear)
beta0 = media_y - beta1 * media_x

# Exibir os resultados
print(f"Coeficiente angular (beta1): {beta1}")
print(f"Coeficiente linear (beta0): {beta0}\n")

# Imprimir a equação da reta
print(f"A equação da reta de regressão é: y = {beta1} * x + {beta0}")
