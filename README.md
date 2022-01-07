# Path-planning with visualization
In this repo, I have implemented all the search algorithms I learnt in AIFA course.
I have also added visuals, so as to provide better understanding to the inner working of the algorithms.

## Algorithms 
  - DFS
  - BFS
  - A star
  - DFBB
  - RRT 

## Inferences worth noting

  - DFS 
    Very fast algorithm but paths obtained are not optimal.
  - BFS
    Same as DFS
  - A star
    Most widely used path planning algorithm. Optimal solution is obtained in reasonable time

    But results and performance heavily depend on heuristic algorithm used.
  - DFBB
    Takes a lot of time, even more than A star as it does quite a lot of unnecessary exploration. But the path found it optimal.
  - RRT
    Rapidly exploring Random Tree, as the name suggests, randomly builds an exploration tree untill it gets to the proximity of the goal, from where it undergoes straight path to the goal.

    It is an iterative method and thus speed can be acheived as per requirement, at the cost of quality.

    As per my thinking, we can use it to generate low quality approximate path and later generate detailed path using A star between consecutive nodes.
