#!/usr/bin/env python3
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from builtin_interfaces.msg import Duration

try:
    from cv_bridge import CvBridge
    CV_BRIDGE_AVAILABLE = True
except ImportError:
    CV_BRIDGE_AVAILABLE = False

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


# Disaster-relevant COCO classes i think this is only what we care about
DISASTER_RELEVANT_CLASSES = {
    0:  ('person',    (0,   255, 0)),    # survivor — GREEN
    39: ('bottle',    (255, 165, 0)),    # supply item — ORANGE
    41: ('cup',       (255, 165, 0)),
    56: ('chair',     (128, 128, 128)),  # furniture debris — GREY
    57: ('couch',     (128, 128, 128)),
    60: ('dining table', (128, 128, 128)),
    63: ('laptop',    (0, 165, 255)),    # valuables — BLUE
    67: ('cell phone',(0, 165, 255)),
    73: ('book',      (200, 200, 0)),
    76: ('scissors',  (0, 0, 255)),      # hazard — RED
    77: ('teddy bear',(180, 105, 255)),  # indicates child — PURPLE
}

# Colour scheme for annotated image overlay
BOX_THICKNESS  = 2
FONT           = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE     = 0.55
FONT_THICKNESS = 2


class CameraProcessor(Node):
    def __init__(self):
        super().__init__('camera_processor')

        # ── Parameters ─────────────────────────────────────────────────
        self.declare_parameter('model_path',     'yolov8n.pt')
        self.declare_parameter('confidence',     0.35)
        self.declare_parameter('iou_threshold',  0.45)
        self.declare_parameter('image_topic',    '/camera/image_raw')
        self.declare_parameter('device',         'cpu')   # 'cpu' or '0' for GPU
        self.declare_parameter('publish_annotated', True)
        self.declare_parameter('smoke_detection',   True)
        self.declare_parameter('inference_rate',    5.0)  # Hz — limit GPU/CPU load

        self.model_path         = self.get_parameter('model_path').value
        self.confidence         = self.get_parameter('confidence').value
        self.iou_threshold      = self.get_parameter('iou_threshold').value
        self.image_topic        = self.get_parameter('image_topic').value
        self.device             = self.get_parameter('device').value
        self.publish_annotated  = self.get_parameter('publish_annotated').value
        self.smoke_detection    = self.get_parameter('smoke_detection').value
        inference_rate          = self.get_parameter('inference_rate').value

        self.min_inference_interval = 1.0 / inference_rate
        self.last_inference_time    = 0.0

        # cv_bridge
        if not CV_BRIDGE_AVAILABLE:
            self.get_logger().error(
                'cv_bridge not found!'
            )
            raise RuntimeError('cv_bridge required')
        self.bridge = CvBridge()

        # load yolo
        self.model = None
        self.model_type = 'none'
        self._load_model()

        # qos
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5
        )

        self.img_sub = self.create_subscription(
            Image, self.image_topic,
            self.image_callback, sensor_qos
        )

        # Publishers
        self.annotated_pub = self.create_publisher(
            Image, '/camera/annotated_image', 10
        )
        self.markers_pub = self.create_publisher(
            MarkerArray, '/camera/detections', 10
        )
        self.stats_pub = self.create_publisher(
            String, '/camera/detection_stats', 10
        )

        # Counters
        self.frames_processed  = 0
        self.total_detections  = 0
        self.person_detections = 0
        self.inference_times   = []

        self.get_logger().info(
            f'CameraProcessor ready | model={self.model_type} | '
            f'conf={self.confidence} | device={self.device} | '
            f'rate={inference_rate:.1f}Hz'
        )

    # load model
    def _load_model(self):
        if not YOLO_AVAILABLE:
            self.get_logger().error(
                'ultralytics not installed!\n'
                'Run: pip install ultralytics'
            )
            return

        # Try custom disaster model
        custom_path = Path(self.model_path)
        if custom_path.exists() and custom_path.suffix == '.pt':
            try:
                self.model = YOLO(str(custom_path))
                self.model_type = f'custom:{custom_path.name}'
                self.get_logger().info(f' Loaded custom model: {custom_path}')
                return
            except Exception as e:
                self.get_logger().warn(f'Custom model failed: {e}. Falling back to yolov8n.pt')

        # Fall back
        try:
            self.model = YOLO('yolov8n.pt')
            self.model_type = 'yolov8n-coco'
            self.get_logger().info(
                ' Loaded yolov8n.pt (COCO pretrained)\n'
                '   detects: person, furniture, items\n'
                '   for disaster classes: train on AIDER dataset and pass model_path'
            )
        except Exception as e:
            self.get_logger().error(f'Failed to load any YOLO model: {e}')


    # method to detect smoke
    # generated from gpt
    def _detect_smoke(self, frame_bgr):
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # Smoke mask: low saturation + grey-white value range
        smoke_mask = (
            (s < 50) &         # low colour saturation
            (v > 80) &         # not black
            (v < 230)          # not pure white (glare)
        ).astype(np.uint8) * 255

        # Local variance — smoke has texture variation
        v_float = v.astype(np.float32)
        v_mean  = cv2.blur(v_float, (15, 15))
        v_sq    = cv2.blur(v_float ** 2, (15, 15))
        v_var   = v_sq - v_mean ** 2
        variance_mask = (v_var > 20).astype(np.uint8) * 255

        combined = cv2.bitwise_and(smoke_mask, variance_mask)

        # Morphological cleanup
        kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        cleaned = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned,  cv2.MORPH_OPEN,  kernel)

        contours, _ = cv2.findContours(
            cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        smoke_boxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 3000:   # minimum smoke region size (px²)
                x, y, w, h = cv2.boundingRect(cnt)
                # Confidence heuristic based on area
                conf = min(0.95, area / (frame_bgr.shape[0] * frame_bgr.shape[1]) * 10)
                smoke_boxes.append((x, y, w, h, conf))

        return smoke_boxes


    # check darkness
    def _assess_lighting(self, frame_bgr):
        grey       = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(grey)) / 255.0
        is_dark    = brightness < 0.25
        return brightness, is_dark

    def _run_detection(self, frame_bgr):
        
        results = {
            'yolo': [],
            'smoke': [],
            'lighting': {},
            'inference_ms': 0.0,
        }

        h, w = frame_bgr.shape[:2]

        # check Lighting  
        brightness, is_dark = self._assess_lighting(frame_bgr)
        results['lighting'] = {
            'brightness': round(brightness, 3),
            'is_dark':    is_dark,
        }

        if self.model is not None:
            t0 = time.time()
            yolo_results = self.model.predict(
                source=frame_bgr,
                conf=self.confidence,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False,
            )
            inference_ms = (time.time() - t0) * 1000
            results['inference_ms'] = round(inference_ms, 1)
            self.inference_times.append(inference_ms)

            if yolo_results and len(yolo_results) > 0:
                for det in yolo_results[0].boxes:
                    cls_id  = int(det.cls[0])
                    conf    = float(det.conf[0])
                    xyxy    = det.xyxy[0].tolist()
                    x1, y1, x2, y2 = [int(v) for v in xyxy]

                    # Normalised centre for depth estimation
                    cx_norm = ((x1 + x2) / 2) / w
                    cy_norm = ((y1 + y2) / 2) / h
                    box_area_frac = ((x2 - x1) * (y2 - y1)) / (w * h)

                    # Class name — prefer disaster-specific, fall back to COCO
                    if cls_id in DISASTER_RELEVANT_CLASSES:
                        class_name, colour = DISASTER_RELEVANT_CLASSES[cls_id]
                        is_relevant = True
                    else:
                        class_name = self.model.names.get(cls_id, f'class_{cls_id}')
                        colour     = (180, 180, 180)
                        is_relevant = False

                    # Priority flag [person detection is highest priority]
                    is_person = (cls_id == 0)

                    results['yolo'].append({
                        'class_id':       cls_id,
                        'class_name':     class_name,
                        'confidence':     round(conf, 3),
                        'bbox':           [x1, y1, x2, y2],
                        'cx_norm':        round(cx_norm, 3),
                        'cy_norm':        round(cy_norm, 3),
                        'box_area_frac':  round(box_area_frac, 4),
                        'colour':         colour,
                        'is_relevant':    is_relevant,
                        'is_person':      is_person,
                    })

                    if is_person:
                        self.person_detections += 1

        # Smoke detection 
        if self.smoke_detection:
            smoke_boxes = self._detect_smoke(frame_bgr)
            for x, y, ww, hh, conf in smoke_boxes:
                results['smoke'].append({
                    'class_name':    'smoke',
                    'confidence':    round(conf, 3),
                    'bbox':          [x, y, x + ww, y + hh],
                    'cx_norm':       round((x + ww / 2) / w, 3),
                    'cy_norm':       round((y + hh / 2) / h, 3),
                    'box_area_frac': round((ww * hh) / (w * h), 4),
                    'colour':        (80, 80, 80),
                    'is_relevant':   True,
                    'is_person':     False,
                })

        return results

    # Draw bounding boxes
    def _annotate_frame(self, frame_bgr, detection_results):
        annotated = frame_bgr.copy()
        h, w = annotated.shape[:2]

        # Lighting warning 
        if detection_results['lighting']['is_dark']:
            cv2.putText(annotated, '⚠ LOW LIGHT', (10, 30),
                        FONT, 0.7, (0, 100, 255), 2)

        # YOLOv8 detections
        for det in detection_results['yolo']:
            x1, y1, x2, y2 = det['bbox']
            colour = det['colour']

            # Box (thicker for persons)
            thickness = BOX_THICKNESS + (2 if det['is_person'] else 0)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, thickness)

            # Label background
            label = f"{det['class_name']} {det['confidence']:.2f}"
            (lw, lh), _ = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICKNESS)
            cv2.rectangle(annotated,
                          (x1, y1 - lh - 8), (x1 + lw + 4, y1),
                          colour, -1)
            cv2.putText(annotated, label,
                        (x1 + 2, y1 - 4), FONT, FONT_SCALE,
                        (0, 0, 0), FONT_THICKNESS)

            # Person alert
            if det['is_person']:
                cv2.putText(annotated, '!! SURVIVOR !!',
                            (x1, y2 + 20), FONT, 0.6, (0, 255, 0), 2)

        # Smoke detections
        for det in detection_results['smoke']:
            x1, y1, x2, y2 = det['bbox']
            # Dashed rectangle effect (draw partial lines)
            dash_len = 15
            for i in range(x1, x2, dash_len * 2):
                cv2.line(annotated, (i, y1), (min(i + dash_len, x2), y1), (80, 80, 80), 2)
                cv2.line(annotated, (i, y2), (min(i + dash_len, x2), y2), (80, 80, 80), 2)
            for i in range(y1, y2, dash_len * 2):
                cv2.line(annotated, (x1, i), (x1, min(i + dash_len, y2)), (80, 80, 80), 2)
                cv2.line(annotated, (x2, i), (x2, min(i + dash_len, y2)), (80, 80, 80), 2)

            label = f"smoke {det['confidence']:.2f}"
            cv2.putText(annotated, label, (x1 + 2, y1 - 6),
                        FONT, FONT_SCALE, (80, 80, 80), FONT_THICKNESS)

        # Stats HUD (bottom-left)
        yolo_count  = len(detection_results['yolo'])
        smoke_count = len(detection_results['smoke'])
        inf_ms      = detection_results['inference_ms']
        hud_lines = [
            f"YOLO: {yolo_count} obj | Smoke: {smoke_count}",
            f"Inference: {inf_ms:.0f}ms | Frame: {self.frames_processed}",
            f"Total persons: {self.person_detections}",
        ]
        for i, line in enumerate(hud_lines):
            y_pos = h - 15 - i * 22
            cv2.putText(annotated, line, (8, y_pos),
                        FONT, 0.45, (0, 255, 255), 1)

        return annotated

    # Publish detection results as RViz MarkerArray
    def _publish_markers(self, all_detections, stamp):
        marker_array = MarkerArray()
        marker_id    = 0

        all_dets = all_detections['yolo'] + all_detections['smoke']

        for det in all_dets:
            # Rough depth estimate
            # larger box area = closer object
            approx_depth = max(0.3, 2.0 / (det['box_area_frac'] ** 0.5 + 0.01))
            approx_depth = min(approx_depth, 8.0)

            m              = Marker()
            m.header.stamp = stamp
            m.header.frame_id = 'camera_rgb_optical_frame'
            m.ns           = 'disaster_detections'
            m.id           = marker_id
            m.type         = Marker.TEXT_VIEW_FACING
            m.action       = Marker.ADD
            m.lifetime     = Duration(sec=1, nanosec=0)  # make sure to  expire after 1s

            # Position in camera frame: 
            # x=right, y=down, z=forward
            cx = (det['cx_norm'] - 0.5) * approx_depth * 1.2
            cy = (det['cy_norm'] - 0.5) * approx_depth
            m.pose.position = Point(x=cx, y=cy, z=float(approx_depth))
            m.pose.orientation.w = 1.0

            r, g, b = [c / 255.0 for c in det['colour']]
            m.color.r = r
            m.color.g = g
            m.color.b = b
            m.color.a = 0.9
            m.scale.z = 0.2

            label = f"{det['class_name']}\n{det['confidence']:.2f}"
            if det['is_person']:
                label = f" SURVIVOR found wow, \n{det['confidence']:.2f}"
            m.text = label

            marker_array.markers.append(m)
            marker_id += 1

            # Sphere at detection location
            sphere           = Marker()
            sphere.header    = m.header
            sphere.ns        = 'disaster_spheres'
            sphere.id        = marker_id
            sphere.type      = Marker.SPHERE
            sphere.action    = Marker.ADD
            sphere.lifetime  = Duration(sec=1, nanosec=0)
            sphere.pose.position = Point(x=cx, y=cy, z=float(approx_depth))
            sphere.pose.orientation.w = 1.0
            sphere.scale.x   = 0.15
            sphere.scale.y   = 0.15
            sphere.scale.z   = 0.15
            sphere.color.r   = r
            sphere.color.g   = g
            sphere.color.b   = b
            sphere.color.a   = 0.7
            marker_array.markers.append(sphere)
            marker_id += 1

        self.markers_pub.publish(marker_array)


    def image_callback(self, msg: Image):
        # for Rate limiting
        # don't run YOLOv8 on every frame
        now = time.time()
        if now - self.last_inference_time < self.min_inference_interval:
            return
        self.last_inference_time = now

        # Convert ROS image to OpenCV
        try:
            frame_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge conversion failed: {e}')
            return

        self.frames_processed += 1

        # Run detection pipeline
        detection_results = self._run_detection(frame_bgr)
        n_detected = len(detection_results['yolo']) + len(detection_results['smoke'])
        self.total_detections += n_detected

        # Publish annotated image
        if self.publish_annotated:
            annotated = self._annotate_frame(frame_bgr, detection_results)
            try:
                ann_msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
                ann_msg.header = msg.header
                self.annotated_pub.publish(ann_msg)
            except Exception as e:
                self.get_logger().error(f'Failed to publish annotated image: {e}')

        # Publish RViz markers
        self._publish_markers(detection_results, msg.header.stamp)

        # Publish stats JSON
        stats = {
            'frame':          self.frames_processed,
            'n_yolo':         len(detection_results['yolo']),
            'n_smoke':        len(detection_results['smoke']),
            'n_persons':      sum(1 for d in detection_results['yolo'] if d['is_person']),
            'inference_ms':   detection_results['inference_ms'],
            'lighting':       detection_results['lighting'],
            'detections': [
                {
                    'class': d['class_name'],
                    'conf':  d['confidence'],
                    'cx':    d['cx_norm'],
                    'cy':    d['cy_norm'],
                }
                for d in (detection_results['yolo'] + detection_results['smoke'])
            ],
        }
        stats_msg = String()
        stats_msg.data = json.dumps(stats)
        self.stats_pub.publish(stats_msg)

        # Log to see when something is detected
        persons_this_frame = [d for d in detection_results['yolo'] if d['is_person']]
        if persons_this_frame:
            self.get_logger().warn(
                f' SURVIVOR DETECTED! '
                f'conf={persons_this_frame[0]["confidence"]:.2f} | '
                f'frame={self.frames_processed}',
                throttle_duration_sec=2.0
            )

        if self.frames_processed % 30 == 0:
            mean_inf = (sum(self.inference_times[-30:]) / 30
                        if self.inference_times else 0)
            self.get_logger().info(
                f'Frame {self.frames_processed} | '
                f'avg_inference={mean_inf:.0f}ms | '
                f'total_detections={self.total_detections} | '
                f'persons_seen={self.person_detections}'
            )


def main(args=None):
    rclpy.init(args=args)
    node = CameraProcessor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info(
            f'\n CameraProcessor Summary \n'
            f'  Frames processed:    {node.frames_processed}\n'
            f'  Total detections:    {node.total_detections}\n'
            f'  Persons detected:    {node.person_detections}\n'
            f'  Model used:          {node.model_type}\n'
        )
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
