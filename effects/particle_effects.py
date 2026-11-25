"""
Particle Effects Engine for NEON HAND AI.
Generates glowing fingertip particle trails with velocity-aware spawning,
drag physics, alpha/size fade-out, hard particle caps for stable FPS,
and multi-hand support.
"""

from collections import deque
from dataclasses import dataclass
import math
import random
import time
from typing import Deque, Dict, Optional, Sequence, Tuple, Union
import cv2
import numpy as np

from configuration.settings import NeonColors
from tracking.hand_detector import DetectedHand
from tracking.landmark_processor import FINGERTIP_INDICES, ProcessedHand

HandLike = Union[ProcessedHand, DetectedHand]


@dataclass
class ParticleConfig:
    """Configurable parameters for fingertip particle trails."""
    enabled: bool = True
    max_particles: int = 180
    spawn_per_tip: int = 2
    min_speed_for_burst: float = 3.0
    lifetime_sec: float = 0.55
    base_radius: int = 4
    drag: float = 0.90
    drift_speed: float = 40.0
    trail_length: int = 12
    glow_intensity: float = 0.8
    blur_downscale: float = 0.5


@dataclass
class Particle:
    """Single glowing particle with position, velocity, and lifetime state."""
    x: float
    y: float
    vx: float
    vy: float
    age: float
    lifetime: float
    radius: float
    color: Tuple[int, int, int]


def draw_glowing_particle(
    glow_canvas: np.ndarray,
    core_canvas: np.ndarray,
    particle: Particle,
) -> None:
    """Reusable function to draw a single fading glowing particle."""
    life_ratio = max(0.0, 1.0 - (particle.age / particle.lifetime))
    if life_ratio <= 0.0:
        return

    fade = life_ratio ** 1.4
    radius = max(1, int(round(particle.radius * life_ratio)))
    faded_color = (
        int(particle.color[0] * fade),
        int(particle.color[1] * fade),
        int(particle.color[2] * fade),
    )
    center = (int(round(particle.x)), int(round(particle.y)))
    cv2.circle(glow_canvas, center, radius + 2, faded_color, -1, cv2.LINE_AA)

    core_val = int(255 * fade)
    cv2.circle(
        core_canvas,
        center,
        max(1, radius - 1),
        (core_val, core_val, core_val),
        -1,
        cv2.LINE_AA,
    )


def draw_fingertip_ribbon(
    glow_canvas: np.ndarray,
    core_canvas: np.ndarray,
    points: Sequence[Tuple[int, int]],
    color: Tuple[int, int, int],
) -> None:
    """Reusable function to draw a fading neon ribbon trail behind a fingertip."""
    n = len(points)
    if n < 2:
        return

    for i in range(1, n):
        t = i / float(n - 1)  # 0 (oldest) -> 1 (newest)
        fade = t * t
        seg_color = (
            int(color[0] * fade),
            int(color[1] * fade),
            int(color[2] * fade),
        )
        thickness = max(1, int(round(4 * t)))
        cv2.line(glow_canvas, points[i - 1], points[i], seg_color, thickness + 2, cv2.LINE_AA)
        if t > 0.45:
            core_v = int(220 * fade)
            cv2.line(
                core_canvas,
                points[i - 1],
                points[i],
                (core_v, core_v, core_v),
                max(1, thickness - 1),
                cv2.LINE_AA,
            )


class ParticleSystem:
    """
    Manages particle spawning, physics, lifetime expiration, and multi-hand
    fingertip ribbon trails with strict FPS protection.
    """

    def __init__(self, config: Optional[ParticleConfig] = None) -> None:
        self.config = config or ParticleConfig()
        self._particles: Deque[Particle] = deque(maxlen=self.config.max_particles)
        self._trails: Dict[str, Deque[Tuple[int, int]]] = {}
        self._prev_tips: Dict[str, Tuple[int, int]] = {}
        self._last_time: float = time.perf_counter()
        self._glow_buf: Optional[np.ndarray] = None
        self._core_buf: Optional[np.ndarray] = None

    def _ensure_buffers(self, shape: Tuple[int, int, int]) -> None:
        if self._glow_buf is None or self._glow_buf.shape != shape:
            self._glow_buf = np.zeros(shape, dtype=np.uint8)
            self._core_buf = np.zeros(shape, dtype=np.uint8)
        else:
            self._glow_buf.fill(0)
            self._core_buf.fill(0)

    def _step_physics(self, dt: float) -> None:
        """Advances particle positions, applies drag, and removes expired particles."""
        if not self._particles:
            return

        drag_factor = self.config.drag ** (dt * 60.0)
        alive: Deque[Particle] = deque(maxlen=self.config.max_particles)

        for p in self._particles:
            p.age += dt
            if p.age >= p.lifetime:
                continue
            p.x += p.vx * dt
            p.y += p.vy * dt
            p.vx *= drag_factor
            p.vy *= drag_factor
            alive.append(p)

        self._particles = alive

    def _spawn_for_hand(self, hand_key: str, hand: HandLike) -> None:
        """Updates fingertip trails and spawns particles based on fingertip motion."""
        pts = hand.landmarks_px
        if len(pts) < 21:
            return

        is_left = getattr(hand, "handedness", "Right").lower() == "left"
        tip_color = NeonColors.ELECTRIC_PURPLE if is_left else NeonColors.CYAN
        extended_map: Dict[str, bool] = getattr(hand, "extended_fingers", {})

        for finger_name, tip_idx in FINGERTIP_INDICES.items():
            tip_pt = pts[tip_idx]
            trail_key = f"{hand_key}_{finger_name}"

            if trail_key not in self._trails:
                self._trails[trail_key] = deque(maxlen=self.config.trail_length)
            self._trails[trail_key].append(tip_pt)

            prev_pt = self._prev_tips.get(trail_key, tip_pt)
            dx = tip_pt[0] - prev_pt[0]
            dy = tip_pt[1] - prev_pt[1]
            speed = math.hypot(dx, dy)
            self._prev_tips[trail_key] = tip_pt

            # Skip spawning on curled fingers or stationary frames to protect FPS
            if not extended_map.get(finger_name, True):
                continue

            spawn_count = (
                self.config.spawn_per_tip
                if speed >= self.config.min_speed_for_burst
                else 1
            )
            # Respect hard particle cap
            available_slots = self.config.max_particles - len(self._particles)
            spawn_count = min(spawn_count, max(0, available_slots))

            for _ in range(spawn_count):
                angle = random.uniform(0.0, 2.0 * math.pi)
                drift = random.uniform(0.3, 1.0) * self.config.drift_speed
                vx = -dx * 2.5 + math.cos(angle) * drift
                vy = -dy * 2.5 + math.sin(angle) * drift
                self._particles.append(
                    Particle(
                        x=float(tip_pt[0]),
                        y=float(tip_pt[1]),
                        vx=vx,
                        vy=vy,
                        age=0.0,
                        lifetime=self.config.lifetime_sec * random.uniform(0.75, 1.15),
                        radius=float(self.config.base_radius),
                        color=tip_color if random.random() > 0.35 else NeonColors.ELECTRIC_BLUE,
                    )
                )

    def update(self, hands: Sequence[HandLike], dt: Optional[float] = None) -> None:
        """Updates particle physics and spawns new particles for all detected hands."""
        if not self.config.enabled:
            return

        now = time.perf_counter()
        step_dt = dt if dt is not None else min(0.05, max(0.001, now - self._last_time))
        self._last_time = now

        self._step_physics(step_dt)

        active_prefixes = set()
        seen_counts: Dict[str, int] = {}
        for hand in hands:
            label = getattr(hand, "handedness", "Right")
            idx = seen_counts.get(label, 0)
            seen_counts[label] = idx + 1
            hand_key = f"{label}_{idx}"
            active_prefixes.add(hand_key)
            self._spawn_for_hand(hand_key, hand)

        # Prune trails for hands no longer in the frame
        stale_trails = [
            k for k in self._trails
            if not any(k.startswith(f"{prefix}_") for prefix in active_prefixes)
        ]
        for k in stale_trails:
            del self._trails[k]
            self._prev_tips.pop(k, None)

    @property
    def particle_count(self) -> int:
        """Returns the current number of active particles."""
        return len(self._particles)

    def render_to_buffers(self, glow_canvas: np.ndarray, core_canvas: np.ndarray) -> None:
        """Draws all active ribbons and particles onto provided off-screen buffers."""
        if not self.config.enabled:
            return

        for trail_key, pts in self._trails.items():
            color = NeonColors.ELECTRIC_PURPLE if "Left" in trail_key else NeonColors.CYAN
            draw_fingertip_ribbon(glow_canvas, core_canvas, list(pts), color)

        for particle in self._particles:
            draw_glowing_particle(glow_canvas, core_canvas, particle)

    def render(
        self,
        frame: np.ndarray,
        hands: Sequence[HandLike],
        dt: Optional[float] = None,
    ) -> np.ndarray:
        """Updates particles and composites glowing trails onto the webcam frame."""
        if not self.config.enabled or frame is None or frame.size == 0:
            return frame

        self.update(hands, dt=dt)
        if not self._particles and not self._trails:
            return frame

        self._ensure_buffers(frame.shape)
        assert self._glow_buf is not None and self._core_buf is not None

        self.render_to_buffers(self._glow_buf, self._core_buf)

        h, w = frame.shape[:2]
        if self.config.blur_downscale < 1.0:
            sw = max(1, int(w * self.config.blur_downscale))
            sh = max(1, int(h * self.config.blur_downscale))
            small = cv2.resize(self._glow_buf, (sw, sh), interpolation=cv2.INTER_LINEAR)
            blurred = cv2.GaussianBlur(small, (7, 7), 0)
            bloom = cv2.resize(blurred, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            bloom = cv2.GaussianBlur(self._glow_buf, (13, 13), 0)

        out = cv2.addWeighted(frame, 1.0, bloom, self.config.glow_intensity, 0)
        cv2.add(out, self._glow_buf, dst=out)
        cv2.add(out, self._core_buf, dst=out)
        return out

