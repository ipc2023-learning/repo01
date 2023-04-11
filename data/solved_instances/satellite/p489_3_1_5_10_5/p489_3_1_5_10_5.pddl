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
	spectrograph4 - mode
	image1 - mode
	spectrograph0 - mode
	image3 - mode
	Star2 - direction
	GroundStation3 - direction
	Star4 - direction
	Star5 - direction
	Star6 - direction
	GroundStation8 - direction
	Star9 - direction
	Star1 - direction
	GroundStation0 - direction
	GroundStation7 - direction
	Star10 - direction
	Star11 - direction
	Phenomenon12 - direction
	Phenomenon13 - direction
	Planet14 - direction
)
(:init
	(supports instrument0 image1)
	(supports instrument0 infrared2)
	(calibration_target instrument0 Star1)
	(calibration_target instrument0 Star9)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 Star9)
	(supports instrument1 image1)
	(supports instrument1 spectrograph4)
	(calibration_target instrument1 GroundStation0)
	(calibration_target instrument1 GroundStation7)
	(on_board instrument1 satellite1)
	(power_avail satellite1)
	(pointing satellite1 GroundStation8)
	(supports instrument2 image3)
	(supports instrument2 spectrograph0)
	(calibration_target instrument2 GroundStation7)
	(on_board instrument2 satellite2)
	(power_avail satellite2)
	(pointing satellite2 GroundStation8)
)
(:goal (and
	(pointing satellite0 Phenomenon13)
	(pointing satellite1 Phenomenon13)
	(pointing satellite2 Star10)
	(have_image Star10 image3)
	(have_image Star11 infrared2)
	(have_image Phenomenon12 spectrograph4)
	(have_image Phenomenon13 spectrograph4)
	(have_image Planet14 spectrograph4)
))

)
