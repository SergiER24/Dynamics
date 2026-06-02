#include <Arduino.h>
#include <ESP32Servo.h>

// ============================================================================
// Taller 2 trajectory with updated real-servo calibration
//
// Physical mapping confirmed by calibration:
//   GPIO 27 -> Servo 1 -> theta1
//   GPIO 26 -> Servo 2 -> theta0
//   GPIO 25 -> Gripper
//
// Mechanical zero for both arm servos is write(90).
//
// Final calibration used here:
//   servo1_command = 90 - theta1_model
//   servo2_command = theta0_model
//
// Gripper calibration used here:
//   start = 45
//   open  = 90
//   close = 45
//
// Waypoints come from the current Taller 2 notebook solution.
//
// Requested geometric changes:
//   longer tool model -> +25 mm
//   pickup           -> 25 mm lower than the original statement point
//   pre_release      -> high approach over the basket
//   release          -> gripper centered over the basket before opening
//
// Python-side theoretical bounds used to obtain the new waypoints:
//   theta0_model in [45, 135] deg
//   theta1_model in [-45, 70] deg
//
// The extra 5 g tool mass belongs to the dynamics/notebook model.
// This ESP32 sketch only replays the updated kinematic waypoints.
// ============================================================================

// --- Pins -------------------------------------------------------------------
const int PIN_SERVO1 = 27;   // Servo 1 -> theta1
const int PIN_SERVO2 = 26;   // Servo 2 -> theta0
const int PIN_GRIPPER = 25;  // Gripper

// --- PWM limits -------------------------------------------------------------
const int PULSE_MIN_US = 500;
const int PULSE_MAX_US = 2400;

// --- Gripper commands -------------------------------------------------------
const float GRIPPER_OPEN_CMD = 90.0f;
const float GRIPPER_CLOSED_CMD = 45.0f;
const float GRIPPER_START_CMD = GRIPPER_CLOSED_CMD;
const float GRIPPER_RETURN_OPEN_CMD = 180.0f;  // fully open for the return motion

// --- Motion tuning ----------------------------------------------------------
const unsigned int STEP_MS = 20;                  // 50 Hz update rate
const unsigned long STARTUP_HOLD_MS = 1500UL;     // hold the calibrated zero pose
const unsigned long STARTUP_APPROACH_MS = 2500UL; // calibrated start -> home
const unsigned long STARTUP_GRIPPER_OPEN_MS = 600UL;
const unsigned long RETURN_GRIPPER_OPEN_MS = 400UL;
const unsigned long RETURN_TO_HOME_MS = 1800UL;   // release -> home for a clean loop
const unsigned long LOOP_PAUSE_MS = 1000UL;       // pause before the next pickup cycle
const bool RUN_ONLY_ONCE = false;

// --- Servo objects ----------------------------------------------------------
Servo servo1;
Servo servo2;
Servo gripper;

float currentGripperCmd = GRIPPER_START_CMD;

// --- Waypoint structure -----------------------------------------------------
struct Waypoint {
  const char* name;
  float theta0;  // model angle [deg]
  float theta1;  // model angle [deg]
};

// --- Physical startup pose in model coordinates -----------------------------
// servo1 = 90 -> theta1 = 0
// servo2 = 90 -> theta0 = 90
const Waypoint WP_PHYSICAL_START = {"physical_start", 90.0f, 0.0f};

// --- Taller 2 waypoints from the notebook -----------------------------------
const Waypoint WP_HOME        = {"home",        110.000000f, 30.000000f};
const Waypoint WP_PRE_PICK    = {"pre_pick",     71.778460f, 11.528455f};
const Waypoint WP_PICKUP      = {"pickup",      107.353554f, 68.354278f};
const Waypoint WP_PRE_RELEASE = {"pre_release",  97.115528f, -11.256500f};
const Waypoint WP_RELEASE     = {"release",     104.247190f, -10.413857f};

// --- Trajectory segments ----------------------------------------------------
struct Segment {
  const char* name;
  const Waypoint* from;
  const Waypoint* to;
  unsigned long durationMs;
  bool isDwell;
};

const Segment TRAJECTORY[] = {
  {"home -> pre_pick",        &WP_HOME,        &WP_PRE_PICK,    1400UL, false},
  {"pre_pick -> pickup",      &WP_PRE_PICK,    &WP_PICKUP,       800UL, false},
  {"pickup dwell",            &WP_PICKUP,      &WP_PICKUP,       400UL, true},
  {"pickup -> pre_pick",      &WP_PICKUP,      &WP_PRE_PICK,     900UL, false},
  {"pre_pick -> pre_release", &WP_PRE_PICK,    &WP_PRE_RELEASE, 2000UL, false},
  {"pre_release -> release",  &WP_PRE_RELEASE, &WP_RELEASE,     1000UL, false},
  {"release dwell",           &WP_RELEASE,     &WP_RELEASE,      400UL, true},
};

const size_t NUM_SEGMENTS = sizeof(TRAJECTORY) / sizeof(TRAJECTORY[0]);

// --- Utility ----------------------------------------------------------------
float clampf(float value, float lower, float upper) {
  if (value < lower) return lower;
  if (value > upper) return upper;
  return value;
}

// Quintic rest-to-rest scalar // smooth position, velocity and acceleration
float quinticRestToRest(float u) {
  u = clampf(u, 0.0f, 1.0f);
  return (10.0f * u * u * u) - (15.0f * u * u * u * u) + (6.0f * u * u * u * u * u);
}

// Servo 1 calibration // write(90) is the mechanical zero for theta1 = 0
int theta1ModelToServo1(float theta1Deg) {
  float cmd = 90.0f - theta1Deg;
  cmd = clampf(cmd, 0.0f, 180.0f);
  return (int)lroundf(cmd);
}

// Servo 2 calibration // write(90) is the mechanical zero for theta0 = 90
int theta0ModelToServo2(float theta0Deg) {
  float cmd = theta0Deg;
  cmd = clampf(cmd, 0.0f, 180.0f);
  return (int)lroundf(cmd);
}

int gripperCmdToWrite(float cmdDeg) {
  float cmd = clampf(cmdDeg, 0.0f, 180.0f);
  return (int)lroundf(cmd);
}

// Write one mechanism pose to the real robot
void writePose(const Waypoint& pose) {
  servo1.write(theta1ModelToServo1(pose.theta1)); // GPIO 27
  servo2.write(theta0ModelToServo2(pose.theta0)); // GPIO 26
}

void writeGripper(float cmdDeg) {
  currentGripperCmd = clampf(cmdDeg, 0.0f, 180.0f);
  gripper.write(gripperCmdToWrite(currentGripperCmd));
}

void printPose(const char* prefix, const Waypoint& pose) {
  Serial.printf(
      "%s %s | theta0(model)=%.3f deg | theta1(model)=%.3f deg\n",
      prefix,
      pose.name,
      pose.theta0,
      pose.theta1
  );
  Serial.printf(
      "   servo2(theta0)=%d deg | servo1(theta1)=%d deg\n",
      theta0ModelToServo2(pose.theta0),
      theta1ModelToServo1(pose.theta1)
  );
}

// Smooth gripper motion // used only at pickup and release
void moveGripperSmooth(float targetCmd, unsigned long durationMs, unsigned int stepMs = STEP_MS) {
  float startCmd = currentGripperCmd;
  float endCmd = clampf(targetCmd, 0.0f, 180.0f);
  int steps = max(1, (int)(durationMs / stepMs));

  for (int i = 1; i <= steps; ++i) {
    float u = (float)i / (float)steps;
    float s = quinticRestToRest(u);
    float cmd = startCmd + s * (endCmd - startCmd);
    gripper.write(gripperCmdToWrite(cmd));
    delay(stepMs);
  }

  writeGripper(endCmd);
}

// Smooth arm motion between two waypoints
void moveArmQuintic(const Waypoint& from, const Waypoint& to, unsigned long durationMs, unsigned int stepMs = STEP_MS) {
  int steps = max(1, (int)(durationMs / stepMs));

  for (int i = 1; i <= steps; ++i) {
    float u = (float)i / (float)steps;
    float s = quinticRestToRest(u);

    float theta0 = from.theta0 + s * (to.theta0 - from.theta0);
    float theta1 = from.theta1 + s * (to.theta1 - from.theta1);

    servo1.write(theta1ModelToServo1(theta1));
    servo2.write(theta0ModelToServo2(theta0));
    delay(stepMs);
  }

  writePose(to);  // force exact final pose
}

// Run one segment
void runSegment(const Segment& segment, size_t idx, size_t totalSegments) {
  Serial.printf(
      "[%u/%u] %s (%lu ms)\n",
      (unsigned)(idx + 1),
      (unsigned)totalSegments,
      segment.name,
      segment.durationMs
  );

  if (segment.isDwell) {
    writePose(*segment.to);

    if (segment.to == &WP_PICKUP) {
      unsigned long actionMs = min(300UL, segment.durationMs);
      Serial.printf("   -> Closing gripper to %.1f deg\n", GRIPPER_CLOSED_CMD);
      moveGripperSmooth(GRIPPER_CLOSED_CMD, actionMs);
      delay(segment.durationMs - actionMs);
    } else if (segment.to == &WP_RELEASE) {
      unsigned long actionMs = min(300UL, segment.durationMs);
      Serial.printf("   -> Opening gripper to %.1f deg\n", GRIPPER_OPEN_CMD);
      moveGripperSmooth(GRIPPER_OPEN_CMD, actionMs);
      delay(segment.durationMs - actionMs);
    } else {
      delay(segment.durationMs);
    }

    return;
  }

  moveArmQuintic(*segment.from, *segment.to, segment.durationMs);
}

void runTrajectory() {
  for (size_t i = 0; i < NUM_SEGMENTS; ++i) {
    runSegment(TRAJECTORY[i], i, NUM_SEGMENTS);
  }
}

// Attach servos and move them to their calibrated zero
void attachServos() {
  pinMode(PIN_SERVO1, OUTPUT);
  pinMode(PIN_SERVO2, OUTPUT);
  pinMode(PIN_GRIPPER, OUTPUT);

  digitalWrite(PIN_SERVO1, LOW);
  digitalWrite(PIN_SERVO2, LOW);
  digitalWrite(PIN_GRIPPER, LOW);
  delay(200);

  servo1.setPeriodHertz(50);
  servo2.setPeriodHertz(50);
  gripper.setPeriodHertz(50);

  servo1.attach(PIN_SERVO1, PULSE_MIN_US, PULSE_MAX_US);
  delay(120);
  servo2.attach(PIN_SERVO2, PULSE_MIN_US, PULSE_MAX_US);
  delay(120);
  gripper.attach(PIN_GRIPPER, PULSE_MIN_US, PULSE_MAX_US);
  delay(250);

  // Calibrated zero pose before running the notebook trajectory
  servo1.write(90);
  servo2.write(90);
  gripper.write((int)GRIPPER_START_CMD);
  currentGripperCmd = GRIPPER_START_CMD;
}

// --- Setup ------------------------------------------------------------------
void setup() {
  Serial.begin(115200);
  delay(500);

  ESP32PWM::allocateTimer(0);
  ESP32PWM::allocateTimer(1);
  ESP32PWM::allocateTimer(2);

  attachServos();

  printPose("Physical startup pose:", WP_PHYSICAL_START);
  delay(STARTUP_HOLD_MS);

  // Start physically closed, then open before approaching the ball.
  Serial.println("Opening gripper before the pickup sequence...");
  moveGripperSmooth(GRIPPER_OPEN_CMD, STARTUP_GRIPPER_OPEN_MS);
  delay(200);

  printPose("Approaching home pose:", WP_HOME);
  moveArmQuintic(WP_PHYSICAL_START, WP_HOME, STARTUP_APPROACH_MS);

  Serial.println("=== Taller 2 trajectory ready ===");
  printPose("Home pose:", WP_HOME);
}

// --- Main loop --------------------------------------------------------------
void loop() {
  runTrajectory();

  if (RUN_ONLY_ONCE) {
    Serial.println("Trajectory finished. Holding final pose.");
    while (true) {
      delay(1000);
    }
  }

  Serial.println("Opening gripper to the maximum before returning...");
  moveGripperSmooth(GRIPPER_RETURN_OPEN_CMD, RETURN_GRIPPER_OPEN_MS);
  delay(150);

  Serial.println("Returning to home pose...");
  moveArmQuintic(WP_RELEASE, WP_HOME, RETURN_TO_HOME_MS);

  Serial.println("Home reached. Waiting before the next cycle...");
  delay(LOOP_PAUSE_MS);
}
