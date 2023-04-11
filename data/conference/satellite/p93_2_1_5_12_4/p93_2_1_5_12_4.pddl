(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	thermograph4 - mode
	thermograph0 - mode
	spectrograph2 - mode
	spectrograph3 - mode
	infrared1 - mode
	GroundStation0 - direction
	Star1 - direction
	GroundStation2 - direction
	Star3 - direction
	GroundStation5 - direction
	GroundStation8 - direction
	GroundStation11 - direction
	Star6 - direction
	Star4 - direction
	GroundStation10 - direction
	GroundStation7 - direction
	GroundStation9 - direction
	Star12 - direction
	Phenomenon13 - direction
	Star14 - direction
	Phenomenon15 - direction
)
(:init
	(supports instrument0 spectrograph3)
	(calibration_target instrument0 Star4)
	(calibration_target instrument0 Star6)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 Star12)
	(supports instrument1 infrared1)
	(supports instrument1 thermograph0)
	(supports instrument1 thermograph4)
	(supports instrument1 spectrograph2)
	(calibration_target instrument1 GroundStation9)
	(calibration_target instrument1 GroundStation7)
	(calibration_target instrument1 GroundStation10)
	(on_board instrument1 satellite1)
	(power_avail satellite1)
	(pointing satellite1 Star14)
)
(:goal (and
	(pointing satellite1 GroundStation7)
	(have_image Star12 thermograph4)
	(have_image Phenomenon13 infrared1)
	(have_image Star14 thermograph0)
	(have_image Phenomenon15 spectrograph3)
))

)
