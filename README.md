# Kobuki Project

This code is done as apart of the onboarding task that I did for the Univeristy of Glasgow driverless team. This code has been edited since the competition in 2023 with the addition of a cost function and better control code.

## Background

This project was done in a attempt to integrate into UGRDV easier allowing us to get to grips with ros2 and in general making a driverless system. In this project I was in charge of making the Path Planning and control code. This taught me alot about how these system take data that is given by perception and tells the car how to drive.

## The code

This code implementes a Delaunay triangluation and beam search to generate paths for the robot to follow. From here a cost fucntion is then run on these paths ensure that paths staying within the bounds are chosen over others that may go off or produce very bad lines.