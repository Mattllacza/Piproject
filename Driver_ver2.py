import rospy
import numpy as np
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32, Float32MultiArray

class ShoeChaserLegsFinal:
    def __init__(self):
        rospy.init_node('shoe_chaser_legs_final')

        # publishers / subs
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.scan_sub = rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        self.shoe_sub = rospy.Subscriber('/shoe_dets', Float32MultiArray, self.shoe_callback)
        self.clap_sub = rospy.Subscriber('/clap_doa_angle', Float32, self.clap_callback)

        # speeds
        self.wander_speed = 0.22
        self.chase_speed = 0.18

        # distances
        self.stop_dist = 0.18
        self.victory_dist = 0.35
        self.target_width = 0.35

        # states
        self.state = "SEARCH"
        self.last_shoe_time = rospy.Time(0)
        self.shoe_center = 0.5
        self.shoe_width = 0.0

        # clap vars
        self.last_clap_time = rospy.Time(0)
        self.last_clap_angle = None
        self.clap_turn_start = None
        self.clap_turn_dir = 0.0

        # clap tuning
        self.clap_max_age = 1.5
        self.clap_turn_speed = 0.8
        self.clap_turn_duration = 1.0

        rospy.loginfo("Driver: DOA Enabled")

    def shoe_callback(self, msg):
        # shoe data update
        if len(msg.data) >= 3:
            self.shoe_center = msg.data[0]
            self.shoe_width = msg.data[2]
            self.last_shoe_time = rospy.Time.now()

    def clap_callback(self, msg):
        # clap angle received
        self.last_clap_time = rospy.Time.now()
        ang = msg.data

        if self.state != "SEARCH":
            return

        turn_dir = 0.0

        # right sector
        if ang >= 290 or ang <= 80:
            turn_dir = -1.0
        # left sector
        elif 100 <= ang <= 260:
            turn_dir = 1.0
        else:
            return

        self.last_clap_angle = ang
        self.clap_turn_dir = turn_dir
        self.clap_turn_start = rospy.Time.now()
        self.state = "TURN_TO_CLAP"

    def preprocess_lidar(self, ranges):
        # lidar cleanup
        r = np.array(ranges)
        r[np.isnan(r)] = 3.5
        r[np.isinf(r)] = 3.5
        r[r == 0.0] = 3.5
        return r

    def scan_callback(self, msg):
        cmd = Twist()
        ranges = self.preprocess_lidar(msg.ranges)
        front_sector = np.concatenate((ranges[-20:], ranges[0:20]))
        min_front = np.min(front_sector)

        now = rospy.Time.now()
        t_shoe = (now - self.last_shoe_time).to_sec()
        t_clap = (now - self.last_clap_time).to_sec()

        if self.state == "FINISHED":
            return

        # obstacle check
        if min_front < self.stop_dist and self.state != "CELEBRATE":
            if t_shoe < 1.0:
                self.celebrate()
                return
            else:
                self.state = "SEARCH"

        # automatic mode switch
        if self.state not in ["CELEBRATE", "TURN_TO_CLAP"]:
            self.state = "CHASE" if t_shoe < 1.0 else "SEARCH"

        # chase mode
        if self.state == "CHASE":
            if self.shoe_width > self.target_width and min_front < self.victory_dist:
                self.celebrate()
                return

            cmd.linear.x = self.chase_speed
            cmd.angular.z = (0.5 - self.shoe_center) * 2.0

            if min_front < 0.4:
                cmd.linear.x = 0.1

        # turn toward clap
        elif self.state == "TURN_TO_CLAP":
            if self.clap_turn_start is None or t_clap > self.clap_max_age:
                self.state = "SEARCH"
            else:
                elapsed = (now - self.clap_turn_start).to_sec()
                if elapsed > self.clap_turn_duration or min_front < self.stop_dist:
                    self.state = "SEARCH"
                    cmd.angular.z = 0.0
                else:
                    cmd.linear.x = 0.0
                    cmd.angular.z = self.clap_turn_dir * self.clap_turn_speed

        # wander mode
        elif self.state == "SEARCH":
            left = np.sum(ranges[0:60])
            right = np.sum(ranges[-60:])
            cmd.linear.x = self.wander_speed
            cmd.angular.z = np.clip((left - right) * 0.01, -0.8, 0.8)
            if min_front < 0.5:
                cmd.angular.z = 1.0

        self.cmd_pub.publish(cmd)

    def celebrate(self):
        self.state = "CELEBRATE"
        t = Twist()
        self.cmd_pub.publish(t)

        print("\n" * 5)
        print("############ TARGET ACQUIRED ############")
        print("\n" * 5)

        t.angular.z = 2.5
        end = rospy.Time.now() + rospy.Duration(2.0)
        while rospy.Time.now() < end:
            self.cmd_pub.publish(t)
            rospy.sleep(0.1)

        t.angular.z = 0.0
        self.cmd_pub.publish(t)
        self.state = "FINISHED"

if __name__ == '__main__':
    try:
        node = ShoeChaserLegsFinal()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
