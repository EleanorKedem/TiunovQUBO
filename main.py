import numpy as np
from SimCIM import *
from functions import *
import matplotlib
import matplotlib.pyplot as plt
import cProfile
import pstats

datatype = torch.float32
device = 'cpu'

def test():
    """
    Function to test the QUBO optimization process with a predefined matrix and solution vector.

    This function evaluates the performance of the QUBO solver by comparing the energy of
    the obtained solution with the optimal solution and calculates the loss. It also prints
    the convergence iteration and the final solution.

    Steps:
    - Define a QUBO matrix and a solution vector.
    - Compute the optimal solution energy using the matrix and solution vector.
    - Use the QUBOSolver (or alternative solvers) to minimize the QUBO problem.
    - Calculate the loss and print key performance metrics.
    """

    # Define a sample QUBO problem (5x5 matrix and solution vector)
    matrix = [[ 1, -2,  3, -4, 2],
              [-2, -1,  2, -3, 1],
              [ 3,  2,  1, -2, 2],
              [-4, -3, -2, -1, 1],
              [ 2,  1,  2,  1, 1]]
    solVect = [1, 1, 0, 1, 0]

    # Compute the optimal solution energy using the QUBO matrix
    solution = get_value_simcim(solVect, matrix)

    # Minimize the QUBO problem and retrieve the results
    energy, iteration = run_simcim(matrix)

    # Calculate the relative loss compared to the optimal solution
    loss = (energy - solution) / solution

    # Print performance metrics
    print("Solution %f " % solution)
    print("Loss %f" % loss)
    print("Convergence iteration %d, with best route %d" % (iteration, energy))


# fFunction to calculate the Euclidean distance between two points
def calcDistance(x1, y1, x2, y2):
    """
     Calculate the Euclidean distance between two points.

     Args:
         x1, y1 (float): Coordinates of the first point.
         x2, y2 (float): Coordinates of the second point.

     Returns:
         float: Euclidean distance between the two points.
     """
    dist = np.sqrt(pow((x1-x2),2) + pow((y1-y2),2))
    return dist

# Function to create the QUBO matrix from a file
def create_QUBO_matrix(file_name):
    """
    Create a QUBO matrix from a file.

    The file should contain the size of the matrix on the first line,
    followed by rows of entries (i, j, coefficient).

    Args:
        file_name (str): Path to the input file containing the QUBO matrix data.

    Returns:
        np.ndarray: Symmetric QUBO matrix constructed from the file.
    """
    with open(file_name, 'r') as file:
        lines = file.readlines()

        # Read the size of the QUBO matrix from the first line
        size = int(lines[0].strip())

        # Initialize the QUBO matrix with zeros
        QUBO_matrix = np.zeros((size, size), dtype=int)

        # Process each line after the first
        for line in lines[1:]:
            i, j, coefficient = map(int, line.split())
            QUBO_matrix[i, j] = coefficient
            if i != j:
                QUBO_matrix[j, i] = coefficient  # Ensure the matrix is symmetric

        return QUBO_matrix

# Function to benchmark QUBO solutions
def QUBO_benchmark():
    """
    Benchmark the performance of the QUBO solver across multiple cases.

    Iterates through a list of predefined benchmark cases, reads the QUBO matrix
    for each case, and evaluates the solver's performance. Results include loss values,
    convergence iterations, and other metrics.
    """
    print("Begin QUBO using quantum inspired annealing - QUBO benchmark cases")

    data = [(-2098, 'bqp50.1'),(-3702, 'bqp50.2'),(-4626, 'bqp50.3'),(-3544, 'bqp50.4'),(-4012, 'bqp50.5'),(-3693, 'bqp50.6'),(-4520, 'bqp50.7'),(-4216, 'bqp50.8'),(-3780, 'bqp50.9'),(-3507, 'bqp50.10'),
            (-7970, 'bqp100.1'),(-11036, 'bqp100.2'),(-12723, 'bqp100.3'),(-10368, 'bqp100.4'),(-9083, 'bqp100.5'),(-10210, 'bqp100.6'),(-10125, 'bqp100.7'),(-11435, 'bqp100.8'),(-11455, 'bqp100.9'),(-12565, 'bqp100.10'),
            (-45607, 'bqp250.1'),(-44810, 'bqp250.2'),(-49037, 'bqp250.3'),(-41274, 'bqp250.4'),(-47961, 'bqp250.5'),(-41014, 'bqp250.6'),(-46757, 'bqp250.7'),(-35726, 'bqp250.8'),(-48916, 'bqp250.9'),(-40442, 'bqp250.10'),
            (-116586, 'bqp500.1'),(-128223, 'bqp500.2'),(-130812, 'bqp500.3'),(-130097, 'bqp500.4'),(-125487, 'bqp500.5'),(-121772, 'bqp500.6'),(-122201, 'bqp500.7'),(-123559, 'bqp500.8'),(-120798, 'bqp500.9'),(-130619, 'bqp500.10'),
            (-371438, 'bqp1000.1'),(-354932, 'bqp1000.2'),(-371236, 'bqp1000.3'),(-370675, 'bqp1000.4'),(-352760, 'bqp1000.5'),(-359629, 'bqp1000.6'),(-371193, 'bqp1000.7'),(-351994, 'bqp1000.8'),(-349337, 'bqp1000.9'),(-351415, 'bqp1000.10'),
            (-1515944, 'bqp2500.1'),(-1471392, 'bqp2500.2'),(-1414192, 'bqp2500.3'),(-1507701, 'bqp2500.4'),(-1491816, 'bqp2500.5'),(-1469162, 'bqp2500.6'),(-1479040, 'bqp2500.7'),(-1484199, 'bqp2500.8'),(-1482413, 'bqp2500.9'),(-1483355, 'bqp2500.10'),
            (-3931583, 'p3000.1'),(-5193073, 'p3000.2'),(-5111533, 'p3000.3'),(-5761822, 'p3000.4'),(-5675625, 'p3000.5'),
            (-6181830, 'p4000.1'),(-7801355, 'p4000.2'), (-7741685, 'p4000.3'), (-8711822, 'p4000.4'), (-8908979, 'p4000.5'),
            (-8559680, 'p5000.1'), (-10836019, 'p5000.2'), (-10489137, 'p5000.3'), (-12252318, 'p5000.4'), (-12731803, 'p5000.5'),
            (-11384976, 'p6000.1'), (-14333855, 'p6000.2'), (-16132915, 'p6000.3'),
            (-14478676, 'p7000.1'), (-18249948, 'p7000.2'), (-20446407, 'p7000.3'),
            (-285149199, 'torusg3-15'), (-41684814, 'torusg3-8'), (-3014, 'toruspm3-15-50'), (-458, 'toruspm3-8-50')]


    loss_values = []
    convergence = []
    n = 0
    sum_iteration = sum_loss = 0
    for file in range(len(data)):
        n += 1
        optSol = data[file][0]
        print("problem %d, optimal solution %d" % (n, optSol))

        matrix = create_QUBO_matrix(data[file][1])

        # Initialize the QUBO solver
        energy, iteration = run_simcim(matrix)

        # Calculate the relative loss compared to the optimal solution
        loss = (energy - optSol) / optSol

        # Collect results for statistics across different problems
        loss_values.append(loss)
        convergence.append(iteration)
        sum_iteration += iteration
        sum_loss += loss

        # Print performance metrics
        print("Loss %f" % loss)  # Relative loss in energy
        print("Convergence iteration %d, with best solution %d" % (iteration, energy))  # Convergence stats
        print("Average iteration %d, average loss %f" % ((sum_iteration / n), (sum_loss / n)))  # Statistics across different problems


def run_simcim(matrix):
    """
    Run the SimCIM solver on a given QUBO matrix.

    Args:
        matrix (list of lists): QUBO matrix to be solved.

    Returns:
        tuple: (solution value, number of iterations).
    """
    J, b = get_Jh(matrix)  # Convert QUBO matrix to J and h

    simcim = Simcim(J, b, device, datatype)

    # Initialize the profiler for stats
    profiler = cProfile.Profile()
    profiler.enable()

    # Minimize the QUBO problem and retrieve the results
    c_current, c_evol, iteration = simcim.evolve()

    profiler.disable()

    # Print out the stats
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumtime').print_stats(10)
    s_cur = torch.sign(c_current)
    E = energy(J, b, s_cur)

    print('Evolution of amplitudes - QUBO')
    fig, ax = plt.subplots()#figsize=(5, 5))
    plt.title("Evolution of amplitudes - QUBO")
    for i in range(J.size(0)):
        ax.plot(c_evol[i, :].cpu().numpy())
    ax.set_xlabel('iteration')
    ax.set_ylabel('amplitudes')
    fig.tight_layout()
    fig.show()

    s_min = s_cur[:, torch.argmin(E)]
    solution = get_value_simcim(s_min, matrix)
    print(solution)
    print('Best among ' + str(simcim.params_disc['attempt_num']) + ' runs')

    return solution, int(iteration)

def main():
    """
    Main function to execute the QUBO benchmark and other evaluations.
    """
    QUBO_benchmark()

if __name__ == "__main__":
    main()

# def national_TSP():
#     print("\nBegin TSP using quantum inspired annealing - National TSP")
#
#     """Read a National TSP file and return a list of coordinates."""
#     data = [(27603, "wi29.tsp"),
#             (6656, "dj38.tsp"),
#             (9352, "qa194.tsp"),
#             (79114, "uy734.tsp"),
#             (95345, "zi929.tsp"),
#             (11340, "lu980.tsp"),
#             (26051, "rw1621.tsp"),
#             (86891, "mu1979.tsp"),
#             (96132, "nu3496.tsp"),
#             (1290319, "ca4663.tsp"),
#             (394543, "tz6117.tsp"),
#             (172350, "eg7146.tsp"),
#             (238242, "ym7663.tsp"),
#             (114831, "pm8079.tsp"),
#             (206128, "ei8246.tsp"),
#             (837377, "ar9152.tsp"),
#             (491869, "ja9847.tsp"),
#             (300876, "gr9882.tsp"),
#             (1061387, "kz9976.tsp"),
#             (520383, "fi10639.tsp"),
#             (427246, "mo14185.tsp"),
#             (176940, "ho14473.tsp"),
#             (557274, "it16862.tsp"),
#             (569115, "vm22775.tsp"),
#             (855331,"sw24978.tsp"),
#             (959011, "bm33708.tsp"),
#             (4565452, "ch71009.tsp")]
#     loss_values = []
#     convergence = []
#     n = 0
#     sum_iteration = sum_loss = 0
#
#     for file in range(len(data)):
#         n += 1
#         optDist = data[file][0]
#         print("place %d, optimal tour %d" % (n, optDist))
#         with open(data[file][1], 'r') as file:
#             lines = file.readlines()
#         # Extracting coordinates
#         coordinates = []
#         for line in lines:
#             parts = line.strip().split()
#             if parts[0].isdigit():  # This checks if the line starts with a node number
#                 x, y = float(parts[1]), float(parts[2])
#                 coordinates.append((x, y))
#         TSPSize = len(coordinates)
#         print("TSP size %d" % TSPSize)
#
#         # Iterating through the points two at a time
#         matrix = np.empty((TSPSize, TSPSize))
#         for i in range(TSPSize):
#             for j in range(i, TSPSize):
#                 matrix[i][j] = matrix[j][i] = round(calcDistance(coordinates[i][0], coordinates[i][1], coordinates[j][0], coordinates[j][1]))
#
#         path, energy, iteration = run_simcim(matrix.numpy())
#         loss = (energy - optDist) / optDist
#         loss_values.append(loss)
#         convergence.append(iteration)
#         sum_iteration += iteration
#         sum_loss += loss
#         print("Loss %f" % loss)
#         print("Convergence iteration %d, with best route %d" % (iteration, energy))
#         print("Average iteration %d, average loss %f" % ((sum_iteration / n), (sum_loss / n)))