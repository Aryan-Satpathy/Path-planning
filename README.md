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
  
    <img src="https://user-images.githubusercontent.com/86613790/148540269-65e1a78a-f997-48d4-baf2-98f59845d4b0.png" width=33% height=33%>
    
    Very fast algorithm but paths obtained are not optimal.
  - BFS
  
    <img src="https://user-images.githubusercontent.com/86613790/148539683-e4e4ddd4-1b08-47ce-b7e7-248caf255b0a.png" width=33% height=33%>  
  
    Relatively slower and more optimal path
  - A star
    
    <p float="left">
    <img src="https://user-images.githubusercontent.com/86613790/148538728-acae1b08-d38e-427d-938b-024732dca4ce.png" width=32% height=32%>  
    <img src="https://user-images.githubusercontent.com/86613790/148539092-ed48d922-e607-4c29-992a-dfb456643b19.png" width=32% height=32%>
    <img src="https://user-images.githubusercontent.com/86613790/148539408-f4db3f9e-847e-43d8-b2b1-7906d4365ee7.png" width=32% height=32%>
    </p>

    Most widely used path planning algorithm. Optimal solution is obtained in reasonable time

    But results and performance heavily depend on heuristic algorithm used.
  - DFBB

    <img src="https://user-images.githubusercontent.com/86613790/148540655-08f216f7-1665-4a94-a915-071d624f1442.png" width=33% height=33%>
    
    Takes a lot of time, even more than A star as it does quite a lot of unnecessary exploration. But the path found it optimal.
  - RRT

    <img src="https://user-images.githubusercontent.com/86613790/148540819-f73e98cd-8b7c-4430-ad72-44172e10ae50.png" width=33% height=33%>

    Rapidly exploring Random Tree, as the name suggests, randomly builds an exploration tree untill it gets to the proximity of the goal, from where it undergoes straight path to the goal.

    It is an iterative method and thus speed can be acheived as per requirement, at the cost of quality.

    As per my thinking, we can use it to generate low quality approximate path and later generate detailed path using A star between consecutive nodes.

    **Bonus :**
    
      Big mistake coming here. Have convolution now
      
      ![image](https://user-images.githubusercontent.com/86613790/148541104-f249f06f-f781-48cf-a946-b406757ae031.png)
