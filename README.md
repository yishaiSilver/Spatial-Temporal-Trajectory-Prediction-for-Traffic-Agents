# Spatiotemporal Trajectory Prediction for Traffic Agents

### Can we infer the future positions of traffic agents?

- [Introduction](#introduction)
- [Exploring The Dataset: Argoverse](#exploring-the-dataset-argoverse)
    - [Explore a Scene](#explore-a-scene)
    - [Transforming the POV](#transforming-the-pov)
        - [Position](#position)
        - [Orientation](#orientation)
    <!-- - [Lane Relevancy](#lane-relevancy) -->
- [Model considerations](#model-considerations)

# Introduction

This is an interesting problem first introduced to me in an undergrad class, and, frankly, it was my first real introduction to ML (asides from training YOLO models like a script-kiddy). Despite being a top-performer in the class, my code was quite terrible, consisting mostly of some poorly documented jupyter notebooks. As such, I've decided to revisit the problem, clean it up, and see if I can take it further. Moreover, the spatio-temporal nature of the problem is applicable to many other fields (Ad ranking, NLP, robotics, etc.), so it's a good problem to have a solid understanding of.

# Exploring the Dataset: Argoverse

The description for the dataset is gives as follows:
```
    city- a string for the city the data was taken from. Can either be 'PIT' or 'MIA'
    scene_idx- a unique non-zero integer identifying the scene
    track_id- tracked maximum 60 agents with 32 digits ID in the form of "00000000-0000-0000-0000-0000XXXXXXXX"
    agent_id- The track id for the agent to be predicted in this scene
    p_in- the (x,y) position input in the first two seconds (19 time steps) of max 60 tracked agents
    v_in-the (x,y) velocity input in the first two seconds (19 time steps) of max 60 tracked agents
    p_out- the (x,y) position output in the next three seconds (30 time steps) of max 60 tracked agents
    v_out - the (x,y) velocity output in the next three seconds (30 time steps) of max 60 tracked agents
    car_mask- boolean index for the real car, we need to align the car numbers. We track maximum 60 agents. Mask equals 1 means the agent appears in the scene, and 0 otherwise
    lane- a (k, 2) numpy array of floats where 'k' is the number of lanes in the scene, representing (x,y) position for centerline nodes in this scene
    lane_norm- a (k, 2) numpy array of floats where 'k' is the number of lanes in the scene, representing the direction of each lane node
```

<details><summary>Amalgamated Data.</summary>
For starters, we can see that the data is given in a global coordinate frame, which can generally be seen through the amalgamation of all lane positions:

![All of p_in plotted.](visualize/images/naive_positions.png)

This is a bit confusing and it's hard to make out what's going on. Let's zoom in to the details of a specific scene.

</details>

<details><summary>Single Scene</summary>

Looking at the data contained in a specifc scene, things begin to make more sense.

![A gif showing an animated scene](visualize/images/simple_animation.gif)

We can see that, although the data is in a global coordinate frame, we can clearly make out lanes and the agents. We can even see how some of the agents are interacting with eachother.

Things are still odd, though. For starters, we're looking at this from the perspective of the olive agent that drives by the intersection, but that's not quickly apparent. For one, we're driving w.r.t. to a global coordinate frame with some arbitrary orientation. Let's try to transform the data to be more intuitive.
</details>

<details><summary>Using Coordinate Transformations to Make Data More Intuitive</summary>

**Position:**

This is done easily by subtracting the ego agent's position from all of the other scene information, thus centering the data on the agent. A more simple approach like subtracting the scene's minimum position from all scene positions may also work, but this would not be as intuitive; who among us drives by considering our positions relative to the street corner?

**Orientation:**

First Notes:

- Due to the erratic velocities we saw earlier, we aren't going to rely on it for orientation, even though conceptually that's an extremely strong option. 
- Even the positions are somewhat noisy, so without filtering, we might get a noisy orientation.
- We can use the closest lane, but this may fail in the case of intersections (If you look at the olive agent, we can see that his orientation would be quite erratic, especially due to his lane change).
- Perhaps we can use something like a kalman filter? Maybe a submodel that predicts the orientation based on the positions of the agents?

To begin with, we'll use the angle between the first and last input positions. This approach will give stable results and is reasonable enough. It'll also make the network's job easier by providing a stable orientation (we only have to worry about the positional change of the ego agent). It also makes it easier to update lanes, since we only have to rotate them once.

Another benefit of this is that it normalizes the vast majority of our ego agent's movements to be in a single direction, making the task easier and improving model explainability.

</details>


### Results

After having explored our data and transformed it to be more intuitive, we can clearly make out what's going on.

![A gif showing an animated scene](visualize/images/translated_to_agent.gif)

One major note is that we need to transition from predicting positions to predicting displacements. As shown in the animation, the ego agent is always at the origin, so p_in becomes nothing but 0s. We can't infer anything about the agent's movement from that, so we need to predict the change in position.

# Model considerations

There's a couple challenges that we need to address:
- Variable inputs:
    - The number of agents in the scene can vary.
    - The number of lane positions can vary.
- Spatial considerations

Possible solutions (certainly not extensive, but we'll add as we go):
- Variable input length:
    - Zero-Padding
    - RNNs
    - Recursion
- Spatial considerations:
    - CNNs
    - Positional Embeddings (I.e. NeRF stuff)
    - Attention
    - Graph Neural Networks

### Pipeline Philosophy

There's a bunch of models that we can use, each with a varying amount of input. Each model may need to have its inputs transformed/organized in a different way. As an example, a CNN may need images, an RNN may need sequences, and an MLP may need a flattened vector. Once we add in the different ways these formats need to be assembled depending on what information is being included, the requirements become quite complex.

In order to keep things simple and organized, as well as to make it easier to swap models, we'll add a special kind of transformation layer that will take in the raw data and encapsulate model's needs as a tuple of matrices. The model will then be able to take in this tuple and do its thing, knowing that the data is in the correct format.

As such, the pipeline will look something like this:

Dataset -> Transformation -> Per-Model Preprocessing -> Collation -> Model Inference -> Data Postprocessing -> Loss -> Backpropagation

### MLP

To start, let's validate our pipeline with a simple MLP. Moreover, we'll do a simple ablation study to validate our prior conclusions.

<details><summary>Ablation Study</summary>

| Experiment Performed                                                                                    | Val. RMSE | Train RMSE   |
|--------------------------------------------------------------------------------------------------------------|----------|--------|
| MLP on just the (naive) positions of the ego agent                                                         | NaN     | NaN   |
| Transformed coordinates                          | 2.776 | 2.751 |
| Transformed coordinates, 3 closest agents        | 3.562 | 3.517 |
| Transformed coordinates, lanes                   | 3.107 | 3.100 |
| Transformed coordinates, lanes, 3 closest agents | 3.339 | 3.437 |
| The following experiments use positional embeddings (where L=3)   |||
| Transformed coordinates                          | 3.175 | 3.451 |
| Transformed coordinates, 3 closest agents        | 2.951 | 3.446 |
| Transformed coordinates, lanes                   | 2.879 | 3.228 |
| Transformed coordinates, lanes, 3 closest agents | 2.704 | 3.443 |

</details>

- We can see that some transformation is required; we cannot simple use GPS-esque coordinates. The loss is just too large, even for floats. 
- Another important thing to note is that the lanes outperform agents as supplementary data.
- The positional embeddings seem to help. 
    - This certainly justifies investing efforts into other ways of handling spatial data.
    - Confusingly, the experiment with non-embedded transformed coordinates seemed to outperform almost every other experiment.
    - Another important thing to note is that the ratio between training loss and validation loss is much more balanced without embeddings, which leads me to believe that the embeddings are overfitting the training data and may benefit from more regularization.

While the results of our simple MLP and data-engineering are great, we can see that it falls short in a few key areas: 

<details><summary>Disapparance of Spatial Information</summary>
One key drawback of the MLP is that -- due to predicting all output movements in one step -- we can only provide a single image of the lanes. This is not ideal as the lanes eprovide key information regarding where the agents are going at specific timesteps. Consider the following example:

![A gif showing an animated scene in which the agent turns](visualize/images/failed_turn.gif)

We can see that the output is generally in the correct direction, but since the network only has a vague idea of where the lanes are, it fails to correctly predict the agent's turn.

A similar , more complicated challenge is that the positions of non-target agents are not given in the output. Lane positions can easily be inferred from the current/predicted position of the agent, but the positions of other agents cannot. 
</details>

### RNN



<!-- ### Lane Relevancy

Now that we're including lanes, we should ask ourselves: "How relevant are lanes to the task at hand?"

At first glance, the answer is "very," but, looking at the animation centered on the ego agent, we can see that not all lanes are created equal. In fact, the intersection is somewhat confusing in terms of which direction agents should be heading. As such, we'll perform an experiment where we filter out lanes positions where the normal is not sufficiently aligned with the ego agent's orientation. -->