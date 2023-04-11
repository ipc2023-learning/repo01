(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	instrument1 - instrument
	satellite1 - satellite
	instrument2 - instrument
	thermograph0 - mode
	spectrograph2 - mode
	infrared1 - mode
	Star1 - direction
	GroundStation2 - direction
	Star3 - direction
	Star5 - direction
	GroundStation4 - direction
	GroundStation0 - direction
	GroundStation6 - direction
	Planet7 - direction
	Phenomenon8 - direction
	Star9 - direction
	Star10 - direction
	Planet11 - direction
)
(:init
	(supports instrument0 infrared1)
	(supports instrument0 thermograph0)
	(supports instrument0 spectrograph2)
	(calibration_target instrument0 GroundStation4)
	(supports instrument1 infrared1)
	(calibration_target instrument1 GroundStation0)
	(on_board instrument0 satellite0)
	(on_board instrument1 satellite0)
	(power_avail satellite0)
	(pointing satellite0 Phenomenon8)
	(supports instrument2 thermograph0)
	(supports instrument2 infrared1)
	(calibration_target instrument2 GroundStation6)
	(on_board instrument2 satellite1)
	(power_avail satellite1)
	(pointing satellite1 Star9)
)
(:goal (and
	(pointing satellite0 Planet11)
	(pointing satellite1 Star9)
	(have_image Planet7 infrared1)
	(have_image Phenomenon8 spectrograph2)
	(have_image Star9 thermograph0)
	(have_image Star10 spectrograph2)
	(have_image Planet11 spectrograph2)
))

)
