import sys
import heapq  # Para implementar la cola de prioridad en A*

class Node():
    def __init__(self, state, parent, action, cost=0, heuristic=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.cost = cost  # Costo acumulado
        self.heuristic = heuristic  # Valor heurístico (distancia estimada al objetivo)

    def __lt__(self, other):
        return (self.cost + self.heuristic) < (other.cost + other.heuristic)


class StackFrontier():
    def __init__(self):
        self.frontier = []

    def add(self, node):
        self.frontier.append(node)

    def contains_state(self, state):
        return any(node.state == state for node in self.frontier)

    def empty(self):
        return len(self.frontier) == 0

    def remove(self):
        if self.empty():
            raise Exception("empty frontier")
        else:
            node = self.frontier[-1]
            self.frontier = self.frontier[:-1]
            return node


class QueueFrontier(StackFrontier):

    def remove(self):
        if self.empty():
            raise Exception("empty frontier")
        else:
            node = self.frontier[0]
            self.frontier = self.frontier[1:]
            return node


class AStarFrontier():
    def __init__(self):
        self.frontier = []

    def add(self, node):
        heapq.heappush(self.frontier, node)

    def contains_state(self, state):
        return any(node.state == state for node in self.frontier)

    def empty(self):
        return len(self.frontier) == 0

    def remove(self):
        if self.empty():
            raise Exception("empty frontier")
        else:
            return heapq.heappop(self.frontier)


class Maze():

    def __init__(self, filename):

        # Leer archivo y configurar dimensiones del laberinto
        with open(filename) as f:
            contents = f.read()

        # Validar si hay un único punto de inicio (A) y objetivo (B)
        if contents.count("A") != 1:
            raise Exception("maze must have exactly one start point")
        if contents.count("B") != 1:
            raise Exception("maze must have exactly one goal")

        # Determinar altura y anchura
        contents = contents.splitlines()
        self.height = len(contents)
        self.width = max(len(line) for line in contents)

        # Crear el mapa de muros
        self.walls = []
        for i in range(self.height):
            row = []
            for j in range(self.width):
                try:
                    if contents[i][j] == "A":
                        self.start = (i, j)
                        row.append(False)
                    elif contents[i][j] == "B":
                        self.goal = (i, j)
                        row.append(False)
                    elif contents[i][j] == " ":
                        row.append(False)
                    else:
                        row.append(True)
                except IndexError:
                    row.append(False)
            self.walls.append(row)

        self.solution = None

    def print(self):
        solution = self.solution[1] if self.solution is not None else None
        print()
        for i, row in enumerate(self.walls):
            for j, col in enumerate(row):
                if col:
                    print("█", end="")
                elif (i, j) == self.start:
                    print("A", end="")
                elif (i, j) == self.goal:
                    print("B", end="")
                elif solution is not None and (i, j) in solution:
                    print("*", end="")
                else:
                    print(" ", end="")
            print()
        print()

    def neighbors(self, state):
        row, col = state
        candidates = [
            ("up", (row - 1, col)),
            ("down", (row + 1, col)),
            ("left", (row, col - 1)),
            ("right", (row, col + 1))
        ]

        result = []
        for action, (r, c) in candidates:
            if 0 <= r < self.height and 0 <= c < self.width and not self.walls[r][c]:
                result.append((action, (r, c)))
        return result

    def heuristic(self, state):
        """Heurística: Distancia Manhattan entre el estado actual y la meta."""
        row1, col1 = state
        row2, col2 = self.goal
        return abs(row1 - row2) + abs(col1 - col2)

    def solve(self, algorithm="DFS"):
        """Encuentra una solución usando el algoritmo seleccionado."""

        # Inicializar la frontera con el nodo inicial
        start = Node(state=self.start, parent=None, action=None)
        
        if algorithm == "DFS":
            frontier = StackFrontier()
        elif algorithm == "BFS":
            frontier = QueueFrontier()
        elif algorithm == "A*":
            frontier = AStarFrontier()
            start.heuristic = self.heuristic(self.start)
        
        frontier.add(start)

        # Seguimiento del número de estados explorados
        self.num_explored = 0

        # Crear el conjunto de estados explorados
        self.explored = set()

        while True:
            # Si la frontera está vacía, no hay solución
            if frontier.empty():
                raise Exception("no solution")

            # Elegir un nodo de la frontera
            node = frontier.remove()
            self.num_explored += 1

            # Si es la meta, se encontró la solución
            if node.state == self.goal:
                actions = []
                cells = []
                while node.parent is not None:
                    actions.append(node.action)
                    cells.append(node.state)
                    node = node.parent
                actions.reverse()
                cells.reverse()
                self.solution = (actions, cells)
                return

            # Marcar el nodo como explorado
            self.explored.add(node.state)

            # Agregar vecinos a la frontera
            for action, state in self.neighbors(node.state):
                if not frontier.contains_state(state) and state not in self.explored:
                    child = Node(state=state, parent=node, action=action)
                    if algorithm == "A*":
                        child.cost = node.cost + 1
                        child.heuristic = self.heuristic(state)
                    frontier.add(child)

    def output_image(self, filename, show_solution=True, show_explored=False):
        from PIL import Image, ImageDraw
        cell_size = 50
        cell_border = 2

        # Crear un lienzo en blanco
        img = Image.new(
            "RGBA",
            (self.width * cell_size, self.height * cell_size),
            "black"
        )
        draw = ImageDraw.Draw(img)

        solution = self.solution[1] if self.solution is not None else None
        for i, row in enumerate(self.walls):
            for j, col in enumerate(row):

                # Muros
                if col:
                    fill = (40, 40, 40)

                # Inicio
                elif (i, j) == self.start:
                    fill = (255, 0, 0)

                # Meta
                elif (i, j) == self.goal:
                    fill = (0, 171, 28)

                # Solución
                elif solution is not None and show_solution and (i, j) in solution:
                    fill = (220, 235, 113)

                # Explorados
                elif solution is not None and show_explored and (i, j) in self.explored:
                    fill = (212, 97, 85)

                # Celda vacía
                else:
                    fill = (237, 240, 252)

                # Dibujar celda
                draw.rectangle(
                    ([(j * cell_size + cell_border, i * cell_size + cell_border),
                      ((j + 1) * cell_size - cell_border, (i + 1) * cell_size - cell_border)]),
                    fill=fill
                )

        img.save(filename)


# Menú interactivo
def menu():
    while True:
        print("\nElige un algoritmo de búsqueda para resolver el laberinto:")
        print("1. DFS (Profundidad)")
        print("2. BFS (Anchura)")
        print("3. A*")
        print("4. Salir")

        option = input("Selecciona una opción: ")

        if option == "1":
            return "DFS"
        elif option == "2":
            return "BFS"
        elif option == "3":
            return "A*"
        elif option == "4":
            sys.exit("Saliendo del programa.")
        else:
            print("Opción no válida. Inténtalo de nuevo.")

# Cargar laberinto y ejecutar la solución con el algoritmo elegido
if __name__ == "__main__":
    filename = input("Introduce el nombre del archivo del laberinto: ")
    m = Maze(filename)
    m.print()
    
    while True:
        algorithm = menu()

        # Reemplazar caracteres no válidos en el nombre del archivo
        algorithm_safe = algorithm.replace("*", "").replace("/", "_").replace("\\", "_")

        print(f"\nResolviendo con el algoritmo {algorithm}...")
        m.solve(algorithm)
        print(f"Estados explorados: {m.num_explored}")
        m.print()

        # Guardar la imagen usando un nombre de archivo válido
        m.output_image(f"maze_{algorithm_safe}.png", show_explored=True)

        if input("\n¿Quieres resolver el laberinto con otro algoritmo? (s/n): ").lower() != 's':
            print("¡Hasta la próxima!")
            break
