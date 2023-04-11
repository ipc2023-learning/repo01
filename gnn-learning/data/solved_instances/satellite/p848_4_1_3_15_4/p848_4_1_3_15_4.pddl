(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	satellite1 - satellite
	instrument1 - instrument
	satellite2 - satellite
	instrument2 - instrument
	satellite3 - satellite
	instrument3 - instrument
	thermograph2 - mode
	image1 - mode
	thermograph0 - mode
	GroundStation3 - direction
	Star6 - direction
	Star7 - direction
	GroundStation8 - direction
	GroundStation11 - direction
	GroundStation13 - direction
	Star9 - direction
	GroundStation0 - direction
	GroundStation12 - direction
	Star2 - direction
	GroundStation14 - direction
	Star5 - direction
	GroundStation4 - direction
	Star10 - direction
	GroundStation1 - direction
	Planet15 - direction
	Planet16 - direction
	Phenomenon17 - direction
	Star18 - direction
)
(:init
	(supports instrument0 image1)
	(supports instrument0 thermograph2)
	(calibration_target instrument0 GroundStation12)
	(calibration_target instrument0 GroundStation14)
	(calibration_target instrument0 GroundStation0)
	(calibration_target instrument0 Star9)
	(calibration_target instrument0 GroundStation13)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 Star18)
	(supports instrument1 thermograph0)
	(calibration_target instrument1 Star2)
	(calibration_target instrument1 GroundStation14)
	(calibration_target instrument1 GroundStation4)
	(on_board instrument1 satellite1)
	(power_avail satellite1)
	(pointing satellite1 GroundStation0)
	(supports instrument2 thermograph0)
	(calibration_target instrument2 GroundStation1)
	(calibration_target instrument2 GroundStation4)
	(calibration_target instrument2 Star5)
	(calibration_target instrument2 GroundStation14)
	(on_board instrument2 satellite2)
	(power_avail satellite2)
	(pointing satellite2 Star7)
	(supports instrument3 image1)
	(supports instrument3 thermograph2)
	(supports instrument3 thermograph0)
	(calibration_target instrument3 GroundStation1)
	(calibration_target instrument3 Star10)
	(on_board instrument3 satellite3)
	(power_avail satellite3)
	(pointing satellite3 Star2)
)
(:goal (and
	(pointing satellite2 GroundStation12)
	(have_image Planet15 thermograph2)
	(have_image Planet16 image1)
	(have_image Phenomenon17 thermograph2)
	(have_image Star18 thermograph0)
))

)
