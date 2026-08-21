import heapq
graph = {
    "A": [("B", 2), ("E", 3)],
    "B": [("A", 2), ("C", 1), ("G", 9)],
    "C": [("B", 1)],
    "E": [("A", 3), ("D", 6)],
    "D": [("E", 6), ("G", 1)],
    "G": [("B", 9), ("D", 1)]
}
heuristic = {
    "A": 11,
    "B": 6,
    "C": 99,
    "E": 7,
    "D": 1,
    "G": 0
}

def a_star(graph, heuristic, start, goal):
    priority_queue = []
    heapq.heappush(priority_queue, (heuristic[start], 0, start))

    came_from = {start: None}
    g_cost = {node: float("inf") for node in graph}
    g_cost[start] = 0

    while priority_queue:
        f_cost, current_g, current = heapq.heappop(priority_queue)

        if current_g > g_cost[current]:
            continue

        if current == goal:
            path = []
            while current is not None:
                path.append(current)
                current = came_from[current]
            return path[::-1], g_cost[goal]

        for neighbour, edge_cost in graph[current]:
            new_g = g_cost[current] + edge_cost

            if new_g < g_cost[neighbour]:
                g_cost[neighbour] = new_g
                f_cost = new_g + heuristic[neighbour]

                came_from[neighbour] = current
                heapq.heappush(
                    priority_queue,
                    (f_cost, new_g, neighbour)
                )

    return None, float("inf")


path, cost = a_star(graph, heuristic, "A", "G")

print("Shortest path:", " -> ".join(path))
print("Total cost:", cost)