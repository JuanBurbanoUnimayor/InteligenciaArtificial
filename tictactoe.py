import math # Importa la biblioteca matemática para usar funciones.

# Definir los jugadores
HUMANO = -1 # Representa al jugador humano
COMPUTADORA = 1 # Representa a la computadora

# Crear el tablero vacío
tablero = [
    [0, 0, 0], # Fila 0
    [0, 0, 0], # Fila 1
    [0, 0, 0]  # Fila 2
]

# Función para verificar si hay un ganador
def ganador(tablero, jugador):
    win_state = [
    # Listar todas las combinaciones posibles de victoria
        [tablero[0][0], tablero[0][1], tablero[0][2]], # Fila 0
        [tablero[1][0], tablero[1][1], tablero[1][2]], # Fila 1
        [tablero[2][0], tablero[2][1], tablero[2][2]], # Fila 2
        [tablero[0][0], tablero[1][0], tablero[2][0]], # Columna 0
        [tablero[0][1], tablero[1][1], tablero[2][1]], # Columna 1
        [tablero[0][2], tablero[1][2], tablero[2][2]], # Columna 2
        [tablero[0][0], tablero[1][1], tablero[2][2]], # Diagonal principal
        [tablero[2][0], tablero[1][1], tablero[0][2]], # Diagonal inversa
    ]
    # Verifica si alguna de las combinaciones de victoria coincide con el jugador
    return [jugador, jugador, jugador] in win_state

# Función para verificar si el tablero está lleno
def tablero_lleno(tablero):
    # Revisa cada fila del tablero
    for fila in tablero:
        if 0 in fila: # Si encuentra un espacio vacío (0), retorna False
            return False
    return True # Si no hay espacios vacíos, retorna True

# Evaluar el estado del tablero
def evaluar(tablero):
    # Evalúa si hay un ganador y retorna una puntuación
    if ganador(tablero, COMPUTADORA):
        return 1 # La computadora gana
    elif ganador(tablero, HUMANO):
        return -1 # El humano gana
    else:
        return 0 # Empate

# Algoritmo Minimax
def minimax(tablero, profundidad, jugador):
    # Comprueba si hay un ganador en el estado actual
    if ganador(tablero, COMPUTADORA):
        return 1 # Retorna 1 si la computadora gana
    if ganador(tablero, HUMANO):
        return -1 # Retorna -1 si el humano gana
    if tablero_lleno(tablero):
        return 0 # Retorna 0 si hay un empate

    if jugador == COMPUTADORA: # Si es el turno de la computadora
        mejor = -math.inf # Inicializa la mejor puntuación en negativo infinito
        # Itera sobre cada celda del tablero
        for i in range(3):
            for j in range(3):
                if tablero[i][j] == 0: # Si la celda está vacía
                    tablero[i][j] = COMPUTADORA # Realiza el movimiento de la computadora
                    # Evalúa la jugada recursivamente y actualiza la mejor puntuación
                    mejor = max(mejor, minimax(tablero, profundidad + 1, HUMANO))
                    tablero[i][j] = 0 # Deshace el movimiento
        return mejor # Retorna la mejor puntuación encontrada
    else: # Si es el turno del humano
        peor = math.inf # Inicializa la peor puntuación en positivo infinito
        # Itera sobre cada celda del tablero
        for i in range(3):
            for j in range(3):
                if tablero[i][j] == 0: # Si la celda está vacía
                    tablero[i][j] = HUMANO # Realiza el movimiento del humano
                    # Evalúa la jugada recursivamente y actualiza la peor puntuación
                    peor = min(peor, minimax(tablero, profundidad + 1, COMPUTADORA))
                    tablero[i][j] = 0 # Deshace el movimiento
        return peor # Retorna la peor puntuación encontrada

# Movimiento de la computadora
def movimiento_computadora(tablero):
    mejor_movimiento = None # Inicializa el mejor movimiento como None
    mejor_valor = -math.inf # Inicializa el mejor valor en negativo infinito
    # Itera sobre cada celda del tablero
    for i in range(3): 
        for j in range(3):
            if tablero[i][j] == 0: # Si la celda está vacía
                tablero[i][j] = COMPUTADORA # Realiza el movimiento de la computadora
                # Evalúa el valor de la jugada utilizando minimax
                valor = minimax(tablero, 0, HUMANO)
                tablero[i][j] = 0 # Deshace el movimiento
                # Si el valor de esta jugada es mejor que el anterior
                if valor > mejor_valor:
                    mejor_valor = valor # Actualiza el mejor valor
                    mejor_movimiento = (i, j) # Guarda el mejor movimiento
    return mejor_movimiento # Retorna el mejor movimiento encontrado

# Imprimir el tablero
def imprimir_tablero(tablero):
    # Imprime cada fila del tablero en la consola
    for fila in tablero:
        print(fila)

# Simulación de juego
def juego():
    while True: # Inicia un bucle infinito para el juego
        imprimir_tablero(tablero) # Muestra el estado actual del tablero
        if tablero_lleno(tablero): # Verifica si el tablero está lleno
            print("Empate!") # Informa de un empate
            break # Termina el juego

        # Movimiento del humano
        # Solicita al jugador humano que ingrese su movimiento
        fila = int(input("Introduce la fila (0, 1, 2): "))
        col = int(input("Introduce la columna (0, 1, 2): "))
        if tablero[fila][col] == 0:  # Verifica si el movimiento es válido
            tablero[fila][col] = HUMANO  # Realiza el movimiento del humano
        else:
            print("Movimiento no válido, intenta de nuevo.")  # Informa de un movimiento inválido
            continue  # Vuelve al inicio del ciclo

        # Verifica si el humano ha ganado
        if ganador(tablero, HUMANO):
            imprimir_tablero(tablero)  # Muestra el tablero
            print("¡Has ganado!")  # Informa de la victoria del humano
            break  # Termina el juego

        # Movimiento de la computadora
        movimiento = movimiento_computadora(tablero)  # Calcula el movimiento de la computadora
        if movimiento is None:  # Verifica si hay un movimiento válido
            print("No hay movimientos disponibles. Empate!")  # Informa de un empate
            break  # Termina el juego
        tablero[movimiento[0]][movimiento[1]] = COMPUTADORA  # Realiza el movimiento de la computadora

        # Verifica si la computadora ha ganado
        if ganador(tablero, COMPUTADORA):
            imprimir_tablero(tablero)  # Muestra el tablero
            print("La computadora ha ganado.")  # Informa de la victoria de la computadora
            break  # Termina el juego


# Iniciar el juego
juego()  # Llama a la función juego para comenzar la partida
