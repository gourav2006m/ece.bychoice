import cv2
import mediapipe as mp
import serial
import time
import os
import math
from collections import Counter


# ============================================================
# ESP32 SETTINGS
# ============================================================

ESP32_PORT = "COM8"       # <-- CHANGE THIS TO YOUR ESP32 PORT
BAUD_RATE = 115200


# ============================================================
# CONNECT TO ESP32
# ============================================================

try:
    esp32 = serial.Serial(
        ESP32_PORT,
        BAUD_RATE,
        timeout=1
    )

    time.sleep(2)

    print("ESP32 connected")

except Exception as e:

    print("Could not connect to ESP32")
    print(e)
    exit()


# ============================================================
# MEDIAPIPE MODEL
# ============================================================

SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    SCRIPT_DIR,
    "hand_landmarker.task"
)


if not os.path.exists(MODEL_PATH):

    print()
    print("ERROR: hand_landmarker.task not found!")
    print()
    print("Put hand_landmarker.task in:")
    print(SCRIPT_DIR)
    print()

    esp32.close()
    exit()


# ============================================================
# MEDIAPIPE SETUP
# ============================================================

BaseOptions = mp.tasks.BaseOptions

HandLandmarker = mp.tasks.vision.HandLandmarker

HandLandmarkerOptions = (
    mp.tasks.vision.HandLandmarkerOptions
)

VisionRunningMode = (
    mp.tasks.vision.RunningMode
)


options = HandLandmarkerOptions(

    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),

    running_mode=VisionRunningMode.IMAGE,

    num_hands=1,

    min_hand_detection_confidence=0.65,

    min_hand_presence_confidence=0.65,

    min_tracking_confidence=0.65
)


# ============================================================
# GEOMETRY FUNCTIONS
# ============================================================

def distance(p1, p2):

    return math.sqrt(
        (p1.x - p2.x) ** 2 +
        (p1.y - p2.y) ** 2 +
        (p1.z - p2.z) ** 2
    )


def angle(a, b, c):

    """
    Calculates angle ABC.

    a = first point
    b = middle point
    c = third point
    """

    ba = [
        a.x - b.x,
        a.y - b.y,
        a.z - b.z
    ]

    bc = [
        c.x - b.x,
        c.y - b.y,
        c.z - b.z
    ]

    dot_product = (
        ba[0] * bc[0] +
        ba[1] * bc[1] +
        ba[2] * bc[2]
    )

    magnitude_ba = math.sqrt(
        ba[0] ** 2 +
        ba[1] ** 2 +
        ba[2] ** 2
    )

    magnitude_bc = math.sqrt(
        bc[0] ** 2 +
        bc[1] ** 2 +
        bc[2] ** 2
    )

    if magnitude_ba == 0 or magnitude_bc == 0:
        return 0

    cosine = dot_product / (
        magnitude_ba * magnitude_bc
    )

    cosine = max(
        -1,
        min(1, cosine)
    )

    return math.degrees(
        math.acos(cosine)
    )


# ============================================================
# FINGER DETECTION
# ============================================================

def finger_extended(
    landmarks,
    mcp,
    pip,
    dip,
    tip
):

    """
    Determines whether a finger is extended.

    Uses joint angles instead of simply checking Y coordinates.
    """

    angle_pip = angle(
        landmarks[mcp],
        landmarks[pip],
        landmarks[dip]
    )

    angle_dip = angle(
        landmarks[pip],
        landmarks[dip],
        landmarks[tip]
    )

    # Straight finger usually has angles
    # close to 180 degrees.

    if angle_pip > 150 and angle_dip > 145:

        return True

    return False


# ============================================================
# THUMB DETECTION
# ============================================================

def thumb_extended(
    landmarks,
    handedness
):

    thumb_mcp_angle = angle(
        landmarks[2],
        landmarks[3],
        landmarks[4]
    )

    # Distance between thumb tip and index MCP
    thumb_index_distance = distance(
        landmarks[4],
        landmarks[5]
    )

    # Distance between wrist and thumb tip
    wrist_thumb_distance = distance(
        landmarks[0],
        landmarks[4]
    )

    # Distance between wrist and index MCP
    wrist_index_distance = distance(
        landmarks[0],
        landmarks[5]
    )

    # A spread/extended thumb should have
    # a relatively large distance from index base.

    if thumb_mcp_angle > 130:

        if thumb_index_distance > (
            wrist_index_distance * 0.45
        ):

            return True

    return False


# ============================================================
# COUNT FINGERS
# ============================================================

def count_fingers(
    landmarks,
    handedness
):

    fingers = []

    # --------------------------------------------------------
    # THUMB
    # --------------------------------------------------------

    thumb = thumb_extended(
        landmarks,
        handedness
    )

    fingers.append(thumb)


    # --------------------------------------------------------
    # INDEX
    # --------------------------------------------------------

    index = finger_extended(
        landmarks,
        5,      # MCP
        6,      # PIP
        7,      # DIP
        8       # TIP
    )

    fingers.append(index)


    # --------------------------------------------------------
    # MIDDLE
    # --------------------------------------------------------

    middle = finger_extended(
        landmarks,
        9,
        10,
        11,
        12
    )

    fingers.append(middle)


    # --------------------------------------------------------
    # RING
    # --------------------------------------------------------

    ring = finger_extended(
        landmarks,
        13,
        14,
        15,
        16
    )

    fingers.append(ring)


    # --------------------------------------------------------
    # LITTLE
    # --------------------------------------------------------

    little = finger_extended(
        landmarks,
        17,
        18,
        19,
        20
    )

    fingers.append(little)


    return sum(fingers), fingers


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("ERROR: Could not open camera")

    esp32.close()

    exit()


# Camera resolution

cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    1280
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    720
)


# ============================================================
# SMOOTHING
# ============================================================

history = []

HISTORY_SIZE = 10

last_sent = -1


# ============================================================
# START MEDIAPIPE
# ============================================================

with HandLandmarker.create_from_options(
    options
) as landmarker:

    while True:

        # ----------------------------------------------------
        # READ CAMERA
        # ----------------------------------------------------

        success, frame = cap.read()

        if not success:
            continue


        # Mirror image

        frame = cv2.flip(
            frame,
            1
        )


        # ----------------------------------------------------
        # CONVERT IMAGE
        # ----------------------------------------------------

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )


        # ----------------------------------------------------
        # DETECT HAND
        # ----------------------------------------------------

        result = landmarker.detect(
            mp_image
        )


        finger_count = 0

        detected = False


        # ----------------------------------------------------
        # HAND FOUND
        # ----------------------------------------------------

        if result.hand_landmarks:

            detected = True

            landmarks = result.hand_landmarks[0]

            handedness = (
                result.handedness[0][0].category_name
            )


            # Count fingers

            raw_count, finger_states = count_fingers(
                landmarks,
                handedness
            )


            finger_count = raw_count


            # ------------------------------------------------
            # DRAW LANDMARKS
            # ------------------------------------------------

            h, w, _ = frame.shape

            for landmark in landmarks:

                x = int(
                    landmark.x * w
                )

                y = int(
                    landmark.y * h
                )

                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (0, 255, 0),
                    -1
                )


            # ------------------------------------------------
            # DRAW FINGER STATUS
            # ------------------------------------------------

            labels = [
                "Thumb",
                "Index",
                "Middle",
                "Ring",
                "Little"
            ]


            for i in range(5):

                status = (
                    "OPEN"
                    if finger_states[i]
                    else "CLOSED"
                )

                text = (
                    labels[i] +
                    ": " +
                    status
                )

                cv2.putText(
                    frame,
                    text,
                    (30, 150 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0)
                    if finger_states[i]
                    else (0, 0, 255),
                    2
                )


        # ====================================================
        # NO HAND
        # ====================================================

        if not detected:

            finger_count = 0

            cv2.putText(
                frame,
                "NO HAND",
                (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.4,
                (0, 0, 255),
                3
            )


        # ====================================================
        # SMOOTH FINGER COUNT
        # ====================================================

        history.append(
            finger_count
        )


        if len(history) > HISTORY_SIZE:

            history.pop(0)


        # Get most common result

        stable_count = Counter(
            history
        ).most_common(1)[0][0]


        # ====================================================
        # SEND TO ESP32
        # ====================================================

        if (
            len(history) >= HISTORY_SIZE
            and stable_count != last_sent
        ):

            message = (
                str(stable_count)
                + "\n"
            )

            esp32.write(
                message.encode()
            )

            last_sent = stable_count

            print(
                "Fingers detected:",
                stable_count
            )


        # ====================================================
        # DISPLAY MAIN COUNT
        # ====================================================

        cv2.putText(
            frame,
            f"Fingers: {stable_count}",
            (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (255, 255, 0),
            3
        )


        # ====================================================
        # DISPLAY ESP32 STATUS
        # ====================================================

        cv2.putText(
            frame,
            f"ESP32: {ESP32_PORT}",
            (30, 340),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )


        cv2.putText(
            frame,
            "Press Q to exit",
            (30, 375),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )


        # ====================================================
        # SHOW CAMERA
        # ====================================================

        cv2.imshow(
            "ESP32 Hand Gesture Controller",
            frame
        )


        # ====================================================
        # EXIT
        # ====================================================

        if cv2.waitKey(1) & 0xFF == ord("q"):

            break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

esp32.close()

print("Program stopped.")