import Slam
import Measurements
import Robot
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

class simulator:
    def __init__(self, robot, n):
        self.landmarks = np.random.rand(2,n)*8 - 4
        self.robot = robot
        self.n = n
        self.tags = [i+1 for i in range(n)]

        self.range = 3

    def step(self, drive_meas):
        self.robot.drive(drive_meas)
    
    def measure(self):
        idx_list = []
        lm_bff = self.robot.measure(self.landmarks, range(self.n))

        measurements = []
        for i in range(self.n):
            lm_bff_i = lm_bff[:, i:i+1]
            if np.linalg.norm(lm_bff_i) > self.range or lm_bff_i[0,0] <= 0:
                continue
            
            lm = Measurements.MarkerMeasurement(lm_bff_i, self.tags[i])
            measurements.append(lm)
        
        return measurements
    
    def draw_sim_state(self, ax):
        # Draw landmarks
        ax.plot(self.landmarks[0,:], self.landmarks[1,:], 'm.')
        for i in range(self.n):
            ax.text(self.landmarks[0,i], self.landmarks[1,i], str(i+1))

        # Draw robot
        arrow_scale = 0.4
        ax.arrow(self.robot.state[0,0], self.robot.state[1,0],
                 arrow_scale * np.cos(self.robot.state[2,0]), arrow_scale * np.sin(self.robot.state[2,0]),
                 color='b', head_width=0.3*arrow_scale)
        
        
        ax.axis('equal')
        ax.set_xlim(-5,5)
        ax.set_ylim(-5,5)


pibot = Robot.Robot(0.15, 0.01, np.eye(3), [0,0,0,0,0])
slammer = Slam.Slam(pibot)

trubot = Robot.Robot(0.15, 0.01, np.eye(3), [0,0,0,0,0])
simmer = simulator(trubot, 10)

# Set robot velocity
left_speed, right_speed = 10, 12
dt = 0.5
drive_meas = Measurements.DriveMeasurement(left_speed, right_speed, dt)

fig, ax = plt.subplots()

for step in range(500):
    # Measure and Update
    measurements = simmer.measure()

    slammer.add_landmarks(measurements)
    slammer.update(measurements)
    
    # Drive and predict
    simmer.step(drive_meas)
    
    slammer.predict(drive_meas)

    print("Processed step {}.".format(step+1))

    plt.cla()
    slammer.draw_slam_state(ax)
    simmer.draw_sim_state(ax)
    plt.draw()
    plt.pause(0.02)

plt.show()

