(define (problem roverprob69862) (:domain Rover)
(:objects
	general - Lander
	colour high_res low_res - Mode
	rover0 rover1 - Rover
	rover0store rover1store - Store
	waypoint0 waypoint1 - Waypoint
	camera0 camera1 camera2 camera3 camera4 - Camera
	objective0 objective1 objective2 objective3 objective4 - Objective
	)
(:init
	(visible waypoint0 waypoint1)
	(visible waypoint1 waypoint0)
	(at_lander general waypoint1)
	(channel_free general)
	(at rover0 waypoint0)
	(available rover0)
	(store_of rover0store rover0)
	(empty rover0store)
	(equipped_for_rock_analysis rover0)
	(equipped_for_imaging rover0)
	(can_traverse rover0 waypoint0 waypoint1)
	(can_traverse rover0 waypoint1 waypoint0)
	(at rover1 waypoint0)
	(available rover1)
	(store_of rover1store rover1)
	(empty rover1store)
	(equipped_for_soil_analysis rover1)
	(equipped_for_imaging rover1)
	(can_traverse rover1 waypoint0 waypoint1)
	(can_traverse rover1 waypoint1 waypoint0)
	(on_board camera0 rover1)
	(calibration_target camera0 objective1)
	(supports camera0 colour)
	(supports camera0 low_res)
	(on_board camera1 rover0)
	(calibration_target camera1 objective1)
	(supports camera1 low_res)
	(on_board camera2 rover0)
	(calibration_target camera2 objective1)
	(supports camera2 colour)
	(supports camera2 high_res)
	(supports camera2 low_res)
	(on_board camera3 rover1)
	(calibration_target camera3 objective1)
	(supports camera3 high_res)
	(on_board camera4 rover0)
	(calibration_target camera4 objective1)
	(supports camera4 colour)
	(visible_from objective0 waypoint1)
	(visible_from objective1 waypoint0)
	(visible_from objective2 waypoint0)
	(visible_from objective3 waypoint1)
	(visible_from objective4 waypoint0)
	(visible_from objective4 waypoint1)
)

(:goal (and
(communicated_image_data objective4 high_res)
(communicated_image_data objective1 colour)
(communicated_image_data objective3 low_res)
(communicated_image_data objective4 low_res)
	)
)
)
