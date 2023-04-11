(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	instrument2 - instrument
	instrument3 - instrument
	thermograph0 - mode
	infrared1 - mode
	GroundStation0 - direction
	GroundStation3 - direction
	GroundStation5 - direction
	GroundStation6 - direction
	Star9 - direction
	Star4 - direction
	GroundStation7 - direction
	GroundStation2 - direction
	GroundStation8 - direction
	GroundStation1 - direction
	Phenomenon10 - direction
	Planet11 - direction
	Planet12 - direction
	Phenomenon13 - direction
)
(:init
	(supports instrument0 infrared1)
	(supports instrument0 thermograph0)
	(calibration_target instrument0 Star4)
	(calibration_target instrument0 GroundStation7)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 Phenomenon10)
	(supports instrument1 infrared1)
	(supports instrument1 thermograph0)
	(calibration_target instrument1 GroundStation7)
	(calibration_target instrument1 GroundStation8)
	(supports instrument2 thermograph0)
	(calibration_target instrument2 GroundStation1)
	(calibration_target instrument2 GroundStation2)
	(supports instrument3 infrared1)
	(calibration_target instrument3 GroundStation1)
	(calibration_target instrument3 GroundStation8)
	(on_board instrument1 satellite1)
	(on_board instrument2 satellite1)
	(on_board instrument3 satellite1)
	(power_avail satellite1)
	(pointing satellite1 Star9)
)
(:goal (and
	(pointing satellite0 GroundStation0)
	(pointing satellite1 GroundStation5)
	(have_image Phenomenon10 thermograph0)
	(have_image Planet11 infrared1)
	(have_image Planet12 thermograph0)
	(have_image Phenomenon13 thermograph0)
))

)
