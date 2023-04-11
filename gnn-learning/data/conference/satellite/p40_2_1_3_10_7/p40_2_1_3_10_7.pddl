(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	infrared1 - mode
	thermograph0 - mode
	spectrograph2 - mode
	GroundStation0 - direction
	Star1 - direction
	GroundStation2 - direction
	Star3 - direction
	GroundStation4 - direction
	Star5 - direction
	Star9 - direction
	Star8 - direction
	GroundStation6 - direction
	GroundStation7 - direction
	Planet10 - direction
	Phenomenon11 - direction
	Star12 - direction
	Phenomenon13 - direction
	Star14 - direction
	Star15 - direction
	Planet16 - direction
)
(:init
	(supports instrument0 infrared1)
	(calibration_target instrument0 GroundStation6)
	(calibration_target instrument0 Star8)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 Planet16)
	(supports instrument1 infrared1)
	(supports instrument1 thermograph0)
	(supports instrument1 spectrograph2)
	(calibration_target instrument1 GroundStation7)
	(on_board instrument1 satellite1)
	(power_avail satellite1)
	(pointing satellite1 GroundStation2)
)
(:goal (and
	(pointing satellite0 Phenomenon11)
	(pointing satellite1 Star1)
	(have_image Planet10 infrared1)
	(have_image Phenomenon11 infrared1)
	(have_image Star12 thermograph0)
	(have_image Phenomenon13 spectrograph2)
	(have_image Star14 thermograph0)
	(have_image Star15 thermograph0)
	(have_image Planet16 infrared1)
))

)
