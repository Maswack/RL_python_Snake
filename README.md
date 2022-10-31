# WAITING FOR MODEL TO TRAIN
# StableBaselines3-python-Snake #

## setup ##
**Enviroment (game) was made by TheAILearner** https://github.com/TheAILearner

Required Libraries for running libraries imported to this project:
1. pytorch
2. box2d
3. swig (to make box2d work properly)
4. All that are imported into *snakeEnv.py*

Also use anaconda because otherwise It would probably
not work

## reward ##
For reward I checked the change of distace from snakes head to the apple and collection of apple.

```python
        euclidean_dist_to_apple = np.linalg.norm(np.array(self.snake_head) - np.array(self.apple_position))


        #total reward from distance
        self.total_distance_reward = (250 - euclidean_dist_to_apple) / 25
        
        
        #calculate change in distance to apple
        self.distance_reward = self.total_distance_reward - self.prev_distance_reward
        
        
        #save prev as current, and calculate end reward
        self.prev_distance_reward = self.distance_reward
        self.reward = self.distance_reward + apple_reward
  
```




