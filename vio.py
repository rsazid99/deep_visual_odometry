import rclpy
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Image
from message_filters import Subscriber, ApproximateTimeSynchronizer
from cv_bridge import CvBridge
import cv2
import matplotlib.pyplot as plt
from nav_msgs.msg import Odometry as OdometryMsg, Path
from geometry_msgs.msg import PoseStamped, TransformStamped
from tf2_ros import TransformBroadcaster
from sensor_msgs.msg import Imu
from collections import deque
import copy


def rotmat_to_quat(R):
    # x, y, z, w
    qw = np.sqrt(max(0.0, 1.0 + R[0,0] + R[1,1] + R[2,2])) / 2.0
    qx = (R[2,1] - R[1,2]) / (4.0*qw + 1e-12)
    qy = (R[0,2] - R[2,0]) / (4.0*qw + 1e-12)
    qz = (R[1,0] - R[0,1]) / (4.0*qw + 1e-12)
    return float(qx), float(qy), float(qz), float(qw)


class Odometry(Node):
    def __init__(self):
        super().__init__('odometry')
        self.bridge = CvBridge()
        self.rgb_image = Subscriber(self, Image, '/camera/camera/color/image_raw')
        self.depth_image = Subscriber(self, Image, '/camera/camera/aligned_depth_to_color/image_raw')
        self.ts = ApproximateTimeSynchronizer([self.rgb_image, self.depth_image], queue_size=15, slop=0.01)
        self.ts.registerCallback(self.synced_image_callback)

        self.prev_image = None
        self.current_image = None
        self.current_depth = None
        self.prev_depth = None
        self.prev_stamp = None
        self.prev_pose = None

        # camera intrinsics (fill in with your values!)
        self.fx = 385.57672119140625
        self.fy = 385.14483642578125
        self.cx = 324.1756896972656
        self.cy = 242.3677215576172
        self.K = np.array([[self.fx, 0, self.cx],
                        [0, self.fy, self.cy],
                        [0,       0,      1]], dtype=np.float32)


        # Relative pose (prev -> curr) and global pose (world -> curr)
        self.T_prev_curr = np.eye(4, dtype=np.float32)
        self.T_w_c       = np.eye(4, dtype=np.float32)
        self.optical_to_robot_tf = np.array([
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)

        # Feature extractor / matcher
        self.orb = cv2.ORB_create(2000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.max_matches = 500   # how many best matches to keep

        


        self.odom_pub = self.create_publisher(OdometryMsg, '/vio/odom', 10)
        self.path_pub = self.create_publisher(Path, '/vio/path', 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.path_msg = Path()
        self.path_msg.header.frame_id = 'odom'
        self.body_frame  = 'camera_color_optical_frame'




    def publish_odometry(self, stamp, T_odom_base):
        """
        T_odom_base: 4x4 transform (odom -> base_link), np.ndarray (float64)
        """
        R_wb = T_odom_base[:3, :3]
        t_wb = T_odom_base[:3, 3]
        qx, qy, qz, qw = rotmat_to_quat(R_wb)

        odom = OdometryMsg()
        odom.header.stamp = stamp
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"

        # Pose
        odom.pose.pose.position.x = float(t_wb[0])
        odom.pose.pose.position.y = float(t_wb[1])
        odom.pose.pose.position.z = float(t_wb[2])
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw


        # Twist (body-frame) from finite differences
        if self.prev_pose is not None and self.prev_stamp is not None:
            t_now = stamp.sec + stamp.nanosec * 1e-9
            dt = max(1e-6, t_now - self.prev_stamp)

            # Relative transform: base_prev -> base_now
            T_rel = np.linalg.inv(self.prev_pose) @ T_odom_base
            R_rel = T_rel[:3, :3]
            p_rel = T_rel[:3, 3]

            # Express twist in current body frame (base_link)
            v_body = (R_rel.T @ p_rel) / dt

            # Angular velocity from Rodrigues inverse (axis-angle) divided by dt
            # cv.Rodrigues returns rvec with magnitude = angle [rad], direction = axis
            rvec, _ = cv2.Rodrigues(R_rel)
            omega_prev = rvec.ravel() / dt           # in prev frame
            omega_body = R_rel.T @ omega_prev        # rotate into current body frame

            odom.twist.twist.linear.x  = float(v_body[0])
            odom.twist.twist.linear.y  = float(v_body[1])
            odom.twist.twist.linear.z  = float(v_body[2])
            odom.twist.twist.angular.x = float(omega_body[0])
            odom.twist.twist.angular.y = float(omega_body[1])
            odom.twist.twist.angular.z = float(omega_body[2])

        # (optional) covariances
        odom.pose.covariance[0]  = 0.05**2
        odom.pose.covariance[7]  = 0.05**2
        odom.pose.covariance[14] = 0.10**2
        odom.pose.covariance[21] = 0.05**2
        odom.pose.covariance[28] = 0.05**2
        odom.pose.covariance[35] = 0.10**2

        self.odom_pub.publish(odom)

        # === Append to and publish Path ===
        pose_stamped = PoseStamped()
        pose_stamped.header.stamp = stamp
        pose_stamped.header.frame_id = "odom"
        pose_stamped.pose = odom.pose.pose  # reuse same pose as odometry

        self.path_msg.header.stamp = stamp
        self.path_msg.poses.append(pose_stamped)

        # Optional: keep path from growing forever (memory)
        MAX_PATH_LENGTH = 5000
        if len(self.path_msg.poses) > MAX_PATH_LENGTH:
            self.path_msg.poses = self.path_msg.poses[-MAX_PATH_LENGTH:]

        self.path_pub.publish(self.path_msg)

        # TF: odom -> base_link
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = "odom"
        t.child_frame_id = "base_link"
        t.transform.translation.x = float(t_wb[0])
        t.transform.translation.y = float(t_wb[1])
        t.transform.translation.z = float(t_wb[2])
        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw
        self.tf_broadcaster.sendTransform(t)

    def synced_image_callback(self, img1_msg, img2_msg):
        cv_image = self.bridge.imgmsg_to_cv2(img1_msg, desired_encoding='bgr8')
        depth_image = self.bridge.imgmsg_to_cv2(img2_msg, desired_encoding='passthrough').astype(np.float32)
        depth_image = depth_image.astype(np.float32) / 1000.0
        depth_image[(depth_image <= 0.3) | (depth_image >= 4.0) | ~np.isfinite(depth_image)] = np.nan
        t_curr = img1_msg.header.stamp.sec + img1_msg.header.stamp.nanosec * 1e-9

        if self.prev_image is None and self.prev_depth is None:
            self.prev_image = copy.deepcopy(cv_image)
            self.prev_depth = copy.deepcopy(depth_image)
            self.prev_stamp = t_curr
            return

        self.current_image = cv_image
        self.current_depth = depth_image

        # ========== SIMPLE 2-FRAME RGB-D ODOMETRY ==========

        gray_prev = cv2.cvtColor(self.prev_image, cv2.COLOR_BGR2GRAY)
        gray_curr = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)

        # 1) Detect ORB features
        kp1, des1 = self.orb.detectAndCompute(gray_prev, None)
        kp2, des2 = self.orb.detectAndCompute(gray_curr, None)

        if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
            self.prev_image = copy.deepcopy(self.current_image)
            self.prev_depth = copy.deepcopy(self.current_depth)
            self.prev_stamp = t_curr
            return

        # 2) Match features
        matches = self.bf.match(des1, des2)
        matches = sorted(matches, key=lambda m: m.distance)
        matches = matches[:200]   # 200 was default value

        # 3) Build 3D–2D correspondences
        pts3d = []
        pts2d = []
        valid_matches = []

        for m in matches:
            u1, v1 = kp1[m.queryIdx].pt
            u2, v2 = kp2[m.trainIdx].pt

            z = self.prev_depth[int(round(v1)), int(round(u1))]
            if np.isnan(z) or z <= 0:
                continue

            x = (u1 - self.cx) * z / self.fx
            y = (v1 - self.cy) * z / self.fy

            pts3d.append([x, y, z])
            pts2d.append([u2, v2])
            valid_matches.append(m)

        pts3d = np.asarray(pts3d, dtype=np.float32)
        pts2d = np.asarray(pts2d, dtype=np.float32)

        if len(pts3d) >= 6:
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                pts3d,
                pts2d,
                self.K,
                None,
                iterationsCount=100,
                reprojectionError=3.0,
                confidence=0.99,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
        else:
            success = False
            inliers = None

        if success:
            # 4) Build relative pose T_prev_curr (prev -> curr)
            R, _ = cv2.Rodrigues(rvec)
            self.T_prev_curr = np.eye(4, dtype=np.float32)
            self.T_prev_curr[:3, :3] = R
            self.T_prev_curr[:3, 3] = tvec.ravel()

            # 5) Integrate into global pose T_w_c
            # (world->curr = world->prev @ prev->curr)
            self.T_prev_curr = self.optical_to_robot_tf@self.T_prev_curr@np.linalg.inv(self.optical_to_robot_tf)
            self.T_w_c = self.T_w_c @ self.T_prev_curr

            # 6) Publish odometry in world frame
            self.publish_odometry(img1_msg.header.stamp, self.T_w_c)

            # 7) Visualize inlier matches
            if inliers is not None and len(inliers) > 0:
                inlier_matches = [valid_matches[i[0]] for i in inliers]
            else:
                inlier_matches = valid_matches

            match_vis = cv2.drawMatches(
                self.prev_image, kp1,
                self.current_image, kp2,
                inlier_matches,
                None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )
            cv2.imshow("RGB-D VO matches", match_vis)
            cv2.waitKey(1)

        # 8) Update previous frame
        self.prev_image = copy.deepcopy(self.current_image)
        self.prev_depth = copy.deepcopy(self.current_depth)
        self.prev_stamp = t_curr
        self.prev_pose = copy.deepcopy(self.T_w_c)



def main(args=None):
    rclpy.init(args=args)
    odom = Odometry()
    rclpy.spin(odom)
    odom.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
