import pandas as pd
import heapq
from collections import defaultdict
import math

coordinates_data = {
    'star': ['Sun', 'Proxima Centauri', 'YZ Ceti', 'Upsilon Andromedae', '61 Virginis'],
    'x': [0, 176, -280, 512, 102],
    'y': [0, -406, 1568, -623, -201],
    'z': [0, -49, 40, 133, 144]
}

distances_data = {
    'source': ['Sun', 'Proxima Centauri', 'Sun', 'Proxima Centauri', 'YZ Ceti'],
    'destination': ['Proxima Centauri', 'YZ Ceti', 'YZ Ceti', 'Upsilon Andromedae', '61 Virginis'],
    'distance': [401, 2028, 2200, 2273, 2100]
}

coordinates_df = pd.DataFrame(coordinates_data)
distances_df = pd.DataFrame(distances_data)

coordinates = {}
for index, row in coordinates_df.iterrows():
    star = row['star']
    coordinates[star] = (row['x'], row['y'], row['z'])

graph = defaultdict(list)
for index, row in distances_df.iterrows():
    source = row['source']
    destination = row['destination']
    distance = row['distance']
    graph[source].append((destination, distance))
    graph[destination].append((source, distance))

def dijkstra(graph, start, end):
    queue = [(0, start)]
    distances = {star: float('inf') for star in graph}
    distances[start] = 0
    previous_nodes = {star: None for star in graph}
    iterations = 0

    while queue:
        iterations += 1
        current_distance, current_star = heapq.heappop(queue)

        if current_star == end:
            break

        for neighbor, weight in graph[current_star]:
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous_nodes[neighbor] = current_star
                heapq.heappush(queue, (distance, neighbor))

    path = []
    current_star = end
    while previous_nodes[current_star] is not None:
        path.append(current_star)
        current_star = previous_nodes[current_star]
    path.append(start)
    path.reverse()

    return distances[end], path, iterations

def euclidean_distance(star1, star2):
    (x1, y1, z1) = coordinates[star1]
    (x2, y2, z2) = coordinates[star2]
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2) ** 0.5

def a_star(graph, start, end):
    queue = [(0, start)]
    g_scores = {star: float('inf') for star in graph}
    g_scores[start] = 0
    f_scores = {star: float('inf') for star in graph}
    f_scores[start] = euclidean_distance(start, end)
    previous_nodes = {star: None for star in graph}
    iterations = 0

    while queue:
        iterations += 1
        current_f_score, current_star = heapq.heappop(queue)

        if current_star == end:
            break

        for neighbor, weight in graph[current_star]:
            tentative_g_score = g_scores[current_star] + weight
            if tentative_g_score < g_scores[neighbor]:
                g_scores[neighbor] = tentative_g_score
                f_scores[neighbor] = tentative_g_score + euclidean_distance(neighbor, end)
                previous_nodes[neighbor] = current_star
                heapq.heappush(queue, (f_scores[neighbor], neighbor))

    path = []
    current_star = end
    while previous_nodes[current_star] is not None:
        path.append(current_star)
        current_star = previous_nodes[current_star]
    path.append(start)
    path.reverse()

    return g_scores[end], path, iterations

def main():
    start_star = 'Sun'
    end_star1 = 'Upsilon Andromedae'
    end_star2 = '61 Virginis'
    test_cases = [(start_star, end_star1), (start_star, end_star2)]

    for start, end in test_cases:
        print(f"Testing path from {start} to {end}...")

        dijkstra_distance, dijkstra_path, dijkstra_iterations = dijkstra(graph, start, end)
        print("Dijkstra's algorithm:")
        print(f"Distance: {dijkstra_distance}")
        print(f"Path: {dijkstra_path}")
        print(f"Iterations: {dijkstra_iterations}")

        a_star_distance, a_star_path, a_star_iterations = a_star(graph, start, end)
        print("A* algorithm:")
        print(f"Distance: {a_star_distance}")
        print(f"Path: {a_star_path}")
        print(f"Iterations: {a_star_iterations}")

        print("Comparison:")
        print(f"Dijkstra's Distance: {dijkstra_distance}, Path: {dijkstra_path}, Iterations: {dijkstra_iterations}")
        print(f"A* Distance: {a_star_distance}, Path: {a_star_path}, Iterations: {a_star_iterations}")
        print()

if __name__ == "__main__":
    main()