# Spatiotemporal Trajectory Prediction for Traffic Agents

### Can we infer the future positions of traffic agents?

# Exploring the Data

## Animate a scene
For starters, let's create an animation consisting of the most basic information being provided: 
- Lanes (constructed from their positions and norms) 
- Agent positions and velocities (represented as points and arrows, respectively).

![A gif showing an animated scene](visualize/images/simple_animation.gif)

At this level, certain details emerge: 
- What the heck is our coordinate frame? Why are our points plotted between such weird ranges
- We can clearly make out an intersection. 
- There is some interesting behaviors apparent in this one scene:
    - The red agent waits for the olive agent to pass before performing a right-turn onto the main road.
    - The olive agent changes lanes to overcome towards the left right as the input data stops and the output data begins.
- The velocities are rather noisy; if we look at the bottom three agents, their velocity go all over the place despite them staying relatively still. If we are going to use velocity at all, we may need to filter it in some way.

Let's try to discover the coordinate frame by plotting all of the values of p_in on a single plot:

![All of p_in plotted.](visualize/images/naive_positions.png)

We get two cities, which must be why there is the `city` field for each scene. More importantly, if we were to naively use these positions as inputs, the equivalent would be:
- *Driving with a GPS zoomed all the way out* 
- *Driving while oriented towards north* 

Maybe someone could do these things, but it's not practical, we don't do that,  and it doesn't make sense for us to impose these limitations on our network. Instead, let's put the world relative to us.

### Position
We can do this by transforming the positions of the agents to be relative to the ego agent. This is done by subtracting the ego agent's position from all of the other agents' positions.

### Orientation

Notes:
- Due to the erratic velocity we saw earlier, we aren't going to rely on it for orientation, even though conceptually that's an extremely strong option. 
- Even the positions are somewhat noisy, so without filtering, we might get a noisy orientation.
- We can use the closest lane, but this may fail in the case of intersections (If you look at the olive agent, we can see that his orientation would be quite erratic, especially due to his lane change).
- Perhaps we can use something like a kalman filter?

In light of all of this, we'll start with orienting the agents based on the angle between the beginning input position and the ending input position. This is a very naive approach, but it's a start and it'll give stable results. Furthermore, it'll make the network's job easier by providing a stable orientation (we only have to worry about the positional change of the ego agent). It also makes it easier to update lanes, since we only have to rotate them once.


![A gif showing an animated scene](visualize/images/translated_to_agent.gif)

The data is now much more intuitive. Asides from the static angle, we can see that this is not too different from how we might percieve the world while driving.  

One major note is that we need to transition from predicting positions to predicting displacements. As shown in the animation, the ego agent is always at the origin, so p_in becomes nothing but 0s. We can't infer anything from that, so we need to predict the change in position (which will still give us our movement in the world).

### Model considerations

First thing's first. Let's validate our training process by using a simple DNN.

### Lane Relevancy

Now that we're including lanes, we should ask ourselves: "How relevant are lanes to the task at hand?"

At first glance, the answer is "very," but, looking at the animation centered on the ego agent, we can see that not all lanes are created equal. In fact, the intersection is somewhat confusing in terms of which direction agents should be heading. As such, we'll perform an experiment where we filter out lanes positions where the normal is not sufficiently aligned with the ego agent's orientation.
