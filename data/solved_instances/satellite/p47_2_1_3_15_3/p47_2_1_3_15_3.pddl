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
	GroundStation4 - direction
	Star5 - direction
	GroundStation6 - direction
	Star8 - direction
	Star9 - direction
	GroundStation10 - direction
	GroundStation11 - direction
	Star12 - direction
	GroundStation13 - direction
	Star14 - direction
	GroundStation2 - direction
	GroundStation7 - direction
	Star3 - direction
	Planet15 - direction
	Star16 - direction
	Phenomenon17 - direction
)
(:init
	(supports instrument0 spectrograph2)
	(supports instrument0 infrared1)
	(supports instrument0 thermograph0)
	(calibration_target instrument0 GroundStation2)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 GroundStation6)
	(supports instrument1 spectrograph2)
	(supports instrument1 infrared1)
	(supports instrument1 thermograph0)
	(calibration_target instrument1 Star3)
	(calibration_target instrument1 GroundStation7)
	(on_board instrument1 satellite1)
	(power_avail satellite1)
	(pointing satellite1 GroundStation11)
)
(:goal (and
	(pointing satellite0 GroundStation0)
	(have_image Planet15 spectrograph2)
	(have_image Star16 thermograph0)
	(have_image Phenomenon17 thermograph0)
))

)
