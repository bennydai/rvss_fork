# Workshop Overview Notes - Tutor Use

## Schedule

The workshop will run during the following sessions:

1. Monday 16:00 - 18:00
2. Tuesday 16:00 - 18:00
3. Wednesday 19:30 - 21:00
4. Thursday 16:00 - 18:00
5. Thursday 19:30 - 21:00

Sessions 1 and 2 will feature tutorials at the start of the workshop, and the rest of the sessions will be entirely free-form.
That is, during the remainder of sessions 1 and 2, and during sessions 3-5, the workshop demonstrators will not teach content directly but will instead assist the teams of students having trouble.

## Work For Students

There are five main areas of work for students to complete.
They are as follows.

### Calibration

The students will need to calibrate the wheel and camera parameters of their robots.
A script will be provided to assist with both, however the quality will depend on the effort students put in.
The wheels are calibrated by driving the robot at a fixed velocity for a fixed distance, and then measuring the time elapsed.
The cameras are calibrated using a Charuco board.
Especially camera calibration is highly sensitive to having a good collection of images.

### SLAM

The students do **not** have to develop a SLAM algorithm from scratch.
A straightforward 2D SLAM EKF solution is provided.
However, the performance of the SLAM algorithm can be greatly improved by choosing sensible covariance values, and by developing some outlier rejection methods.
Another choice the students need to make is what trajectory to drive their robot to improve their mapping outcomes.

### Neural Network

The students do **not** have to train a new neural network.
A network is provided to classify the animals in the course.
However, options such as the fineness of the animal detection, how to threshold detections, etc. are left to the students.
Moreover, the students will need to find a way to convert outputs from the network into bearings of the animals with respect to the robot in 2D.

### Triangulation & Data Fusion

The students will be provided code that computes the optimal SE(2) transform between two given maps **without** taking covariance into account.
They are encouraged to improve this estimate by dealing with the covariance.

Students will also need to combine many measurements of animal bearings in order to triangulate the animal positions in the world.
Instructions on how to do this with and without covariance information are provided.

### Data collection

Students need to collect data in order to run their experiments efficiently.
Scripts are provided to assist with data collection and storage as well as playback.

### Control

Students can optionally improve the keyboard control of their robots.
Doing this can greatly improve the robot's efficiency but may have drawbacks, such as motion blur at high speeds.

### Wrangling

For the final demonstration, the teams of students will be split into pilots and wranglers.
Since each team has 3 members, they will have to choose whether to have 2 pilots or 2 wranglers.
The pilots will not be able to see where their robots are placed in the course.
The only information the pilots will have available is what is transmitted by their robot and computed by their algorithms.
The wranglers, on the other hand, will not be able to see the algorithm output from anything being run on the pilots' laptops, but will be able to see the entire course and the locations of the robots therein.
Both the pilots and wranglers are encouraged to share information by speaking to each other, although neither will be able to see what the other sees.

There will be considerable challenge in completing this task optimally.

## TODOs

- Add zip folder with all code to workshop site
- Improve workshop overview on website using Rob's notes.
