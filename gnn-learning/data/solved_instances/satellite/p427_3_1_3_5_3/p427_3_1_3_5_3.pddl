(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	satellite2 - satellite
	instrument2 - instrument
	infrared2 - mode
	image1 - mode
	spectrograph0 - mode
	Star0 - direction
	GroundStation2 - direction
	GroundStation4 - direction
	GroundStation1 - direction
	Star3 - direction
	Phenomenon5 - direction
	Planet6 - direction
	Planet7 - direction
)
(:init
	(supports instrument0 image1)
	(supports instrument0 spectrograph0)
	(calibration_target instrument0 GroundStation4)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 GroundStation4)
	(supports instrument1 spectrograph0)
	(supports instrument1 infrared2)
	(calibration_target instrument1 GroundStation1)
	(on_board instrument1 satellite1)
	(power_avail satellite1)
	(pointing satellite1 Star0)
	(supports instrument2 spectrograph0)
	(calibration_target instrument2 Star3)
	(on_board instrument2 satellite2)
	(power_avail satellite2)
	(pointing satellite2 GroundStation1)
)
(:goal (and
	(pointing satellite0 GroundStation4)
	(pointing satellite2 GroundStation4)
	(have_image Phenomenon5 image1)
	(have_image Planet6 spectrograph0)
	(have_image Planet7 spectrograph0)
))

)
