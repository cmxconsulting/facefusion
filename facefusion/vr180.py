"""
VR180 : traitement des visages dans une vue plate (perspective) plutôt que dans l'image équirectangulaire ou fisheye.

Pour chaque œil :
1. détection grossière des visages sur 9 vues perspective qui couvrent les 180° de l'œil ;
2. pour chaque visage, création d'une vue perspective centrée sur lui (le visage n'y est plus déformé) ;
3. les processors « visage » tournent sur cette vue (cible et images de référence reprojetées de la même façon) ;
4. seuls les pixels modifiés sont reprojetés dans l'œil, avec un masque fondu.
Les processors qui travaillent sur l'image entière (frame_enhancer, frame_colorizer, background_remover)
tournent ensuite sur l'image complète.

Repère : x vers la droite, y vers le bas, z devant. Longitude (yaw) et latitude (pitch) dans [-pi/2, pi/2].
"""
from functools import lru_cache
from typing import Any, Callable, List, Optional, Tuple

import cv2
import numpy

from facefusion import state_manager
from facefusion.face_detector import detect_faces, detect_faces_by_angle
from facefusion.types import Mask, VisionFrame

FRAME_PROCESSORS = [ 'background_remover', 'frame_colorizer', 'frame_enhancer' ]
DETECT_VIEW_ANGLES = [ -55.0, 0.0, 55.0 ]
DETECT_VIEW_FOV = 90.0
DETECT_VIEW_SIZE = 640
FACE_VIEW_FACTOR = 2.6
FACE_VIEW_FOV_RANGE = (25.0, 110.0)
FACE_VIEW_SIZE_RANGE = (512, 2048)
CHANGE_THRESHOLD = 2.0

Region = Tuple[int, int, int, int]
ProcessView = Callable[[VisionFrame, Mask, List[VisionFrame]], Tuple[VisionFrame, Mask]]


def is_enabled() -> bool:
	return state_manager.get_item('vr180_projection') in [ 'equirect', 'fisheye' ]


def is_frame_processor(processor_module : Any) -> bool:
	return any(processor_module.__name__.endswith('.' + processor + '.core') for processor in FRAME_PROCESSORS)


def get_eye_regions(height : int, width : int) -> List[Region]:
	vr180_layout = state_manager.get_item('vr180_layout')

	if vr180_layout == 'sbs':
		half_width = width // 2
		return [ (0, 0, half_width, height), (half_width, 0, width - half_width, height) ]
	if vr180_layout == 'tb':
		half_height = height // 2
		return [ (0, 0, width, half_height), (0, half_height, width, height - half_height) ]
	return [ (0, 0, width, height) ]


def get_eye(vision_frame : numpy.ndarray, eye_index : int) -> numpy.ndarray:
	x, y, w, h = get_eye_regions(vision_frame.shape[0], vision_frame.shape[1])[eye_index]
	return vision_frame[y:y + h, x:x + w]


def direction_to_eye_pixel(directions : numpy.ndarray, eye_width : int, eye_height : int) -> Tuple[numpy.ndarray, numpy.ndarray]:
	x, y, z = directions[..., 0], directions[..., 1], directions[..., 2]

	if state_manager.get_item('vr180_projection') == 'fisheye':
		theta = numpy.arccos(numpy.clip(z, -1.0, 1.0))
		phi = numpy.arctan2(y, x)
		radius = theta / (numpy.pi / 2) * min(eye_width, eye_height) / 2
		return eye_width / 2 + radius * numpy.cos(phi) - 0.5, eye_height / 2 + radius * numpy.sin(phi) - 0.5

	longitude = numpy.arctan2(x, z)
	latitude = numpy.arcsin(numpy.clip(y, -1.0, 1.0))
	return (longitude / numpy.pi + 0.5) * eye_width - 0.5, (latitude / numpy.pi + 0.5) * eye_height - 0.5


def eye_pixel_to_direction(pixel_x : numpy.ndarray, pixel_y : numpy.ndarray, eye_width : int, eye_height : int) -> numpy.ndarray:
	if state_manager.get_item('vr180_projection') == 'fisheye':
		offset_x = pixel_x + 0.5 - eye_width / 2
		offset_y = pixel_y + 0.5 - eye_height / 2
		theta = numpy.hypot(offset_x, offset_y) / (min(eye_width, eye_height) / 2) * (numpy.pi / 2)
		phi = numpy.arctan2(offset_y, offset_x)
		return numpy.stack([ numpy.sin(theta) * numpy.cos(phi), numpy.sin(theta) * numpy.sin(phi), numpy.cos(theta) ], axis = -1)

	longitude = ((pixel_x + 0.5) / eye_width - 0.5) * numpy.pi
	latitude = ((pixel_y + 0.5) / eye_height - 0.5) * numpy.pi
	return numpy.stack([ numpy.cos(latitude) * numpy.sin(longitude), numpy.sin(latitude), numpy.cos(latitude) * numpy.cos(longitude) ], axis = -1)


def create_rotation(yaw : float, pitch : float) -> numpy.ndarray:
	rotation_yaw = numpy.array([ [ numpy.cos(yaw), 0, numpy.sin(yaw) ], [ 0, 1, 0 ], [ -numpy.sin(yaw), 0, numpy.cos(yaw) ] ])
	rotation_pitch = numpy.array([ [ 1, 0, 0 ], [ 0, numpy.cos(pitch), numpy.sin(pitch) ], [ 0, -numpy.sin(pitch), numpy.cos(pitch) ] ])
	return rotation_yaw @ rotation_pitch


def calculate_focal(fov : float, view_size : int) -> float:
	return view_size / 2 / numpy.tan(numpy.radians(fov) / 2)


def view_pixel_to_direction(pixel_x : numpy.ndarray, pixel_y : numpy.ndarray, rotation : numpy.ndarray, focal : float, view_size : int) -> numpy.ndarray:
	rays = numpy.stack([ (pixel_x + 0.5 - view_size / 2) / focal, (pixel_y + 0.5 - view_size / 2) / focal, numpy.ones_like(pixel_x) ], axis = -1)
	rays /= numpy.linalg.norm(rays, axis = -1, keepdims = True)
	return rays @ rotation.T


@lru_cache(maxsize = 32)
def create_detect_view_maps(yaw : float, pitch : float, eye_width : int, eye_height : int) -> Tuple[numpy.ndarray, numpy.ndarray]:
	return create_view_maps(yaw, pitch, DETECT_VIEW_FOV, DETECT_VIEW_SIZE, eye_width, eye_height)


def create_view_maps(yaw : float, pitch : float, fov : float, view_size : int, eye_width : int, eye_height : int) -> Tuple[numpy.ndarray, numpy.ndarray]:
	grid_y, grid_x = numpy.mgrid[0:view_size, 0:view_size].astype(numpy.float32)
	directions = view_pixel_to_direction(grid_x, grid_y, create_rotation(yaw, pitch), calculate_focal(fov, view_size), view_size)
	map_x, map_y = direction_to_eye_pixel(directions, eye_width, eye_height)
	return map_x.astype(numpy.float32), map_y.astype(numpy.float32)


def render_view(eye_frame : numpy.ndarray, yaw : float, pitch : float, fov : float, view_size : int) -> numpy.ndarray:
	map_x, map_y = create_view_maps(yaw, pitch, fov, view_size, eye_frame.shape[1], eye_frame.shape[0])
	return remap_view(eye_frame, map_x, map_y)


def render_detect_view(eye_frame : numpy.ndarray, yaw : float, pitch : float) -> numpy.ndarray:
	map_x, map_y = create_detect_view_maps(yaw, pitch, eye_frame.shape[1], eye_frame.shape[0])
	return remap_view(eye_frame, map_x, map_y)


def remap_view(eye_frame : numpy.ndarray, map_x : numpy.ndarray, map_y : numpy.ndarray) -> numpy.ndarray:
	return cv2.remap(eye_frame, map_x, map_y, cv2.INTER_LINEAR, borderMode = cv2.BORDER_REPLICATE)


def detect_face_directions(eye_frame : VisionFrame) -> List[Tuple[float, float, float, float]]:
	eye_height, eye_width = eye_frame.shape[:2]
	focal = calculate_focal(DETECT_VIEW_FOV, DETECT_VIEW_SIZE)
	candidates : List[Tuple[float, float, float, float]] = []

	for pitch in DETECT_VIEW_ANGLES:
		for yaw in DETECT_VIEW_ANGLES:
			rotation = create_rotation(numpy.radians(yaw), numpy.radians(pitch))
			view_frame = render_detect_view(eye_frame, float(numpy.radians(yaw)), float(numpy.radians(pitch)))

			for face_detector_angle in state_manager.get_item('face_detector_angles'):
				if face_detector_angle == 0:
					bounding_boxes, face_scores, _ = detect_faces(view_frame)
				else:
					bounding_boxes, face_scores, _ = detect_faces_by_angle(view_frame, face_detector_angle)

				for bounding_box, face_score in zip(bounding_boxes, face_scores):
					if face_score < state_manager.get_item('face_detector_score'):
						continue
					center_x = numpy.array((bounding_box[0] + bounding_box[2]) / 2)
					center_y = numpy.array((bounding_box[1] + bounding_box[3]) / 2)
					direction = view_pixel_to_direction(center_x, center_y, rotation, focal, DETECT_VIEW_SIZE)
					face_extent = max(bounding_box[2] - bounding_box[0], bounding_box[3] - bounding_box[1])
					face_angle = 2 * numpy.arctan(face_extent / 2 / focal)
					face_yaw = float(numpy.arctan2(direction[0], direction[2]))
					face_pitch = float(numpy.arcsin(numpy.clip(direction[1], -1.0, 1.0)))

					if eye_pixel_inside(direction, eye_width, eye_height):
						candidates.append((face_yaw, face_pitch, float(face_angle), float(face_score)))

	return merge_face_directions(candidates)


def eye_pixel_inside(direction : numpy.ndarray, eye_width : int, eye_height : int) -> bool:
	pixel_x, pixel_y = direction_to_eye_pixel(direction, eye_width, eye_height)
	return direction[2] > 0 and 0 <= pixel_x < eye_width and 0 <= pixel_y < eye_height


def merge_face_directions(candidates : List[Tuple[float, float, float, float]]) -> List[Tuple[float, float, float, float]]:
	faces : List[Tuple[float, float, float, float]] = []

	for candidate in sorted(candidates, key = lambda item: item[3], reverse = True):
		candidate_direction = create_rotation(candidate[0], candidate[1])[:, 2]
		is_duplicate = False

		for face in faces:
			face_direction = create_rotation(face[0], face[1])[:, 2]
			distance = numpy.arccos(numpy.clip(numpy.dot(candidate_direction, face_direction), -1.0, 1.0))

			if distance < max(candidate[2], face[2]) / 2:
				is_duplicate = True
				break

		if not is_duplicate:
			faces.append(candidate)

	return faces


def resolve_face_view(face_angle : float, eye_width : int) -> Tuple[float, int]:
	fov = float(numpy.clip(numpy.degrees(face_angle) * FACE_VIEW_FACTOR, *FACE_VIEW_FOV_RANGE))
	native_focal = eye_width / numpy.pi
	view_size = int(numpy.clip(2 * numpy.tan(numpy.radians(fov) / 2) * native_focal, *FACE_VIEW_SIZE_RANGE))
	return fov, view_size - view_size % 2


def calculate_paste_region(yaw : float, pitch : float, fov : float, view_size : int, eye_width : int, eye_height : int) -> Optional[Region]:
	border = numpy.linspace(0, view_size - 1, 64, dtype = numpy.float32)
	edge_x = numpy.concatenate([ border, border, numpy.zeros_like(border), numpy.full_like(border, view_size - 1), [ view_size / 2 ] ])
	edge_y = numpy.concatenate([ numpy.zeros_like(border), numpy.full_like(border, view_size - 1), border, border, [ view_size / 2 ] ])
	directions = view_pixel_to_direction(edge_x, edge_y, create_rotation(yaw, pitch), calculate_focal(fov, view_size), view_size)
	pixel_x, pixel_y = direction_to_eye_pixel(directions, eye_width, eye_height)
	x1 = int(numpy.clip(numpy.floor(pixel_x.min()) - 2, 0, eye_width))
	y1 = int(numpy.clip(numpy.floor(pixel_y.min()) - 2, 0, eye_height))
	x2 = int(numpy.clip(numpy.ceil(pixel_x.max()) + 2, 0, eye_width))
	y2 = int(numpy.clip(numpy.ceil(pixel_y.max()) + 2, 0, eye_height))

	if x2 - x1 < 2 or y2 - y1 < 2:
		return None
	return x1, y1, x2 - x1, y2 - y1


def create_inverse_maps(region : Region, yaw : float, pitch : float, fov : float, view_size : int, eye_width : int, eye_height : int) -> Tuple[numpy.ndarray, numpy.ndarray]:
	x, y, w, h = region
	grid_y, grid_x = numpy.mgrid[y:y + h, x:x + w].astype(numpy.float32)
	directions = eye_pixel_to_direction(grid_x, grid_y, eye_width, eye_height)
	camera = directions @ create_rotation(yaw, pitch)
	focal = calculate_focal(fov, view_size)
	in_front = camera[..., 2] > 1e-3
	safe_z = numpy.where(in_front, camera[..., 2], 1.0)
	map_x = numpy.where(in_front, camera[..., 0] / safe_z * focal + view_size / 2 - 0.5, -1.0)
	map_y = numpy.where(in_front, camera[..., 1] / safe_z * focal + view_size / 2 - 0.5, -1.0)
	return map_x.astype(numpy.float32), map_y.astype(numpy.float32)


def create_change_mask(original_view : VisionFrame, processed_view : VisionFrame) -> Mask:
	view_size = original_view.shape[0]
	difference = numpy.abs(processed_view.astype(numpy.float32) - original_view.astype(numpy.float32)).max(axis = 2)
	change_mask = (difference > CHANGE_THRESHOLD).astype(numpy.uint8)
	kernel_size = max(3, view_size // 64) | 1
	change_mask = cv2.morphologyEx(change_mask, cv2.MORPH_CLOSE, numpy.ones((kernel_size, kernel_size), numpy.uint8))
	change_mask = cv2.dilate(change_mask, numpy.ones((kernel_size, kernel_size), numpy.uint8))
	change_mask = cv2.GaussianBlur(change_mask.astype(numpy.float32), (0, 0), kernel_size / 2)
	edge_fade = numpy.clip(numpy.minimum(numpy.arange(view_size), numpy.arange(view_size)[::-1]) / (view_size * 0.05), 0, 1).astype(numpy.float32)
	return numpy.clip(change_mask * edge_fade[:, None] * edge_fade[None, :], 0, 1)


def process_eye(temp_eye : VisionFrame, temp_eye_mask : Mask, target_eyes : List[VisionFrame], process_view : ProcessView) -> Tuple[VisionFrame, Mask]:
	eye_height, eye_width = temp_eye.shape[:2]

	for face_yaw, face_pitch, face_angle, _ in detect_face_directions(temp_eye):
		fov, view_size = resolve_face_view(face_angle, eye_width)
		region = calculate_paste_region(face_yaw, face_pitch, fov, view_size, eye_width, eye_height)

		if not region:
			continue

		original_view = render_view(temp_eye, face_yaw, face_pitch, fov, view_size)
		original_view_mask = render_view(temp_eye_mask, face_yaw, face_pitch, fov, view_size)
		target_views = [ render_view(target_eye, face_yaw, face_pitch, fov, view_size) for target_eye in target_eyes ]
		processed_view, processed_view_mask = process_view(original_view.copy(), original_view_mask.copy(), target_views)
		change_mask = create_change_mask(original_view, processed_view)

		x, y, w, h = region
		map_x, map_y = create_inverse_maps(region, face_yaw, face_pitch, fov, view_size, eye_width, eye_height)
		paste_frame = cv2.remap(processed_view, map_x, map_y, cv2.INTER_LINEAR, borderMode = cv2.BORDER_REPLICATE)
		paste_mask = cv2.remap(processed_view_mask, map_x, map_y, cv2.INTER_LINEAR, borderMode = cv2.BORDER_REPLICATE)
		paste_alpha = cv2.remap(change_mask, map_x, map_y, cv2.INTER_LINEAR, borderMode = cv2.BORDER_CONSTANT, borderValue = 0)

		temp_eye = temp_eye.copy()
		temp_eye_mask = temp_eye_mask.copy()
		eye_area = temp_eye[y:y + h, x:x + w].astype(numpy.float32)
		temp_eye[y:y + h, x:x + w] = (eye_area * (1 - paste_alpha[..., None]) + paste_frame.astype(numpy.float32) * paste_alpha[..., None]).round().clip(0, 255).astype(temp_eye.dtype)
		mask_area = temp_eye_mask[y:y + h, x:x + w].astype(numpy.float32)
		temp_eye_mask[y:y + h, x:x + w] = (mask_area * (1 - paste_alpha) + paste_mask.astype(numpy.float32) * paste_alpha).round().clip(0, 255).astype(temp_eye_mask.dtype)

	return temp_eye, temp_eye_mask


def process_frame(temp_vision_frame : VisionFrame, temp_vision_mask : Mask, target_vision_frames : List[VisionFrame], process_view : ProcessView) -> Tuple[VisionFrame, Mask]:
	temp_vision_frame = temp_vision_frame.copy()
	temp_vision_mask = temp_vision_mask.copy()
	frame_height, frame_width = temp_vision_frame.shape[:2]

	for eye_index, (x, y, w, h) in enumerate(get_eye_regions(frame_height, frame_width)):
		target_eyes = [ resize_eye(get_eye(target_vision_frame, eye_index), w, h) for target_vision_frame in target_vision_frames ]
		temp_eye, temp_eye_mask = process_eye(temp_vision_frame[y:y + h, x:x + w], temp_vision_mask[y:y + h, x:x + w], target_eyes, process_view)
		temp_vision_frame[y:y + h, x:x + w] = temp_eye
		temp_vision_mask[y:y + h, x:x + w] = temp_eye_mask

	return temp_vision_frame, temp_vision_mask


def resize_eye(eye_frame : VisionFrame, width : int, height : int) -> VisionFrame:
	if eye_frame.shape[1] == width and eye_frame.shape[0] == height:
		return eye_frame
	return cv2.resize(eye_frame, (width, height), interpolation = cv2.INTER_AREA)
