from pynput.keyboard import Key, Listener, KeyCode

class Keyboard:
    wheel_vel_forward = 25
    wheel_vel_rotation = 10

    def __init__(self, ppi=None):
        # Storage for key presses
        self.directions = [False for _ in range(4)]
        self.signal_net = False
        self.signal_slam = False
        self.signal_stop = False

        # Connection to PenguinPi robot
        self.ppi = ppi
        self.wheel_vels = [0, 0]

        self.listener = Listener(on_press=self.on_press).start()

    def on_press(self, key):
        print(key)
        # handle steering
        if key == Key.up:
            self.directions[0] = True
        elif key == Key.down:
            self.directions[1] = True
        elif key == Key.left:
            self.directions[2] = True
        elif key == Key.right:
            self.directions[3] = True
        elif key == Key.space: 
            self.signal_stop = True
        # handle signalling

        if isinstance(key, KeyCode):
            if key.char == 'p':
                self.signal_net = True
            elif key.char == 'o':
                self.signal_slam = True

        self.send_drive_signal()
        
    def get_drive_signal(self):
        if (self.signal_stop):
            self.directions = [False for _ in range(4)]
            self.signal_stop = False
        
        if (self.directions[0] and self.directions[1]):
            self.directions[0] = False
            self.directions[1] = False
        
        if (self.directions[2] and self.directions[3]):
            self.directions[2] = False
            self.directions[3] = False
            
        drive_forward = (self.directions[0] - self.directions[1]) * self.wheel_vel_forward
        drive_rotate = (self.directions[2] - self.directions[3]) * self.wheel_vel_rotation

        left_speed = drive_forward - drive_rotate
        right_speed = drive_forward + drive_rotate

        return left_speed, right_speed
    
    def send_drive_signal(self):
        if not self.ppi is None:
            lv, rv = self.get_drive_signal()
            lv, rv = self.ppi.set_velocity(lv, rv)
            self.wheel_vels = [lv, rv]
            
    def latest_drive_signal(self):
        return self.wheel_vels
    
    def get_net_signal(self):
        if self.signal_net:
            self.signal_net = False
            return True
        else:
            return False
    
    def get_slam_signal(self):
        if self.signal_slam:
            self.signal_slam = False
            return True
        else:
            return False

if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, "{}/integration".format(os.getcwd()))
    sys.path.insert(0, "../integration")
    import penguinPiC
    ppi = penguinPiC.PenguinPi()

    keyboard_control = Keyboard(ppi)
    while True:
        continue