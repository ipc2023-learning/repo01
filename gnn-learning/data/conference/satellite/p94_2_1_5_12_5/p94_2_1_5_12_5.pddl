(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	spectrograph3 - mode
	infrared1 - mode
	thermograph0 - mode
	spectrograph2 - mode
	thermograph4 - mode
	Star1 - direction
	Star3 - direction
	Star6 - direction
	GroundStation7 - direction
	GroundStation8 - direction
	GroundStation11 - direction
	Star4 - direction
	GroundStation2 - direction
	GroundStation5 - direction
	GroundStation9 - direction
	GroundStation10 - direction
	GroundStation0 - direction
	Star12 - direction
	Phenomenon13 - direction
	Star14 - direction
	Phenomenon15 - direction
	Planet16 - direction
)
(:init
	(supports instrument0 spectrograph3)
	(supports instrument0 spectrograph2)
	(supports instrument0 thermograph0)
	(supports instrument0 infrared1)
	(calibration_target instrument0 GroundStation10)
	(calibration_target instrument0 GroundStation5)
	(calibration_target instrument0 GroundStation2)
	(calibration_target instrument0 Star4)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 Phenomenon13)
	(supports instrument1 thermograph4)
	(calibration_target instrument1 GroundStation0)
	(calibration_target instrument1 GroundStation10)
	(calibration_target instrument1 GroundStation9)
	(on_board instrument1 satellite1)
	(power_avail satellite1)
	(pointing satellite1 GroundStation2)
)
(:goal (and
	(have_image Star12 thermograph4)
	(have_image Phenomenon13 infrared1)
	(have_image Star14 thermograph0)
	(have_image Phenomenon15 spectrograph3)
	(have_image Planet16 spectrograph2)
))

)
